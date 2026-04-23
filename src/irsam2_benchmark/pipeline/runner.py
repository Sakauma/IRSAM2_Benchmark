"""pipeline 命令调度器。

Author: Egor Izmaylov

这里是 CLI 与 benchmark 运行时之间的桥梁：
- 负责根据命令类型执行 stage 或 baseline；
- 负责生成 benchmark spec；
- 负责聚合 per-seed 结果；
- 负责写出 artifact manifest 和 reference snapshot。
"""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from ..baselines import build_baseline_registry
from ..config import AppConfig
from ..core.interfaces import ArtifactRecord, InferenceMode, PipelineStage, PromptPolicy, PromptSource, PromptType
from ..data import build_dataset_adapter
from ..evaluation.reporting import write_results
from ..evaluation.runner import evaluate_method
from .stages import run_adapt_stage, run_distill_stage, run_quantize_stage, run_transfer_stage


def _effective_prompt_policy(config: AppConfig, inference_mode: InferenceMode) -> Dict[str, Any]:
    """根据推理模式返回真正写入 spec 的 prompt policy。"""
    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        # no-prompt 模式不应复用图像 prompt policy，否则 benchmark spec 会误导。
        return PromptPolicy(
            name="no_prompt_auto_mask",
            prompt_type=PromptType.NONE,
            prompt_source=PromptSource.NONE,
            prompt_budget=0,
            refresh_interval=None,
            multi_mask=True,
            notes="Automatic mask generation without external prompts.",
        ).to_dict()
    return config.evaluation.prompt_policy.to_dict()


def _build_benchmark_spec(config: AppConfig, inference_mode: InferenceMode) -> Dict[str, Any]:
    """构建冻结版 benchmark spec。"""
    return {
        "benchmark_version": config.evaluation.benchmark_version,
        "split_version": config.evaluation.split_version,
        "prompt_policy_version": config.evaluation.prompt_policy_version,
        "metric_schema_version": config.evaluation.metric_schema_version,
        "reference_result_version": config.evaluation.reference_result_version,
        "track": config.evaluation.track,
        "protocol": config.evaluation.protocol,
        "inference_mode": inference_mode.value,
        "prompt_policy": _effective_prompt_policy(config, inference_mode),
        "config_path": str(config.config_path),
    }


def _seed_result(seed: int, aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """把 seed 写回聚合指标，形成单次运行记录。"""
    payload = {"seed": seed}
    payload.update({k: v for k, v in aggregate.items() if not isinstance(v, list)})
    return payload


def _std_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算 per-seed 数值字段的样本标准差。"""
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "seed" or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            numeric.setdefault(key, []).append(float(value))
    return {
        key: (0.0 if len(values) < 2 else _sample_std(values))
        for key, values in numeric.items()
    }


def _sample_std(values: List[float]) -> float:
    """手工实现样本标准差，避免额外引入统计依赖。"""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _snapshot_reference_outputs(config: AppConfig, baseline_name: str, output_dir: Any) -> None:
    """把 baseline 输出复制到 reference_results 下，作为未来回归基线。"""
    snapshot_dir = config.reference_results_root / baseline_name / config.runtime.output_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        target = snapshot_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def run_command(config: AppConfig, command: str, baseline_name: Optional[str] = None) -> None:
    """统一执行入口。"""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_spec = _build_benchmark_spec(config, config.inference_mode)

    if command == "transfer":
        result = run_transfer_stage(config)
        write_results(output_dir, benchmark_spec=benchmark_spec, artifact_manifest={"records": [result.record.to_dict()]}, summary={"command": command}, results=[], eval_rows=[])
        return
    if command == "adapt":
        transfer = run_transfer_stage(config)
        adapt = run_adapt_stage(config)
        write_results(output_dir, benchmark_spec=benchmark_spec, artifact_manifest={"records": [transfer.record.to_dict(), adapt.record.to_dict()]}, summary={"command": command}, results=[], eval_rows=[])
        return
    if command == "distill":
        distill = run_distill_stage(config)
        write_results(output_dir, benchmark_spec=benchmark_spec, artifact_manifest={"records": [distill.record.to_dict()]}, summary={"command": command}, results=[], eval_rows=[])
        return
    if command == "quantize":
        quant = run_quantize_stage(config)
        write_results(output_dir, benchmark_spec=benchmark_spec, artifact_manifest={"records": [quant.record.to_dict()]}, summary={"command": command}, results=[], eval_rows=[])
        return
    if command == "pipeline":
        transfer = run_transfer_stage(config)
        adapt = run_adapt_stage(config)
        distill = run_distill_stage(config)
        quant = run_quantize_stage(config)
        write_results(
            output_dir,
            benchmark_spec=benchmark_spec,
            artifact_manifest={"records": [transfer.record.to_dict(), adapt.record.to_dict(), distill.record.to_dict(), quant.record.to_dict()]},
            summary={"command": command},
            results=[],
            eval_rows=[],
        )
        return
    if command == "ablation-grid":
        write_results(
            output_dir,
            benchmark_spec=benchmark_spec,
            artifact_manifest={"records": []},
            summary={"command": command, "ablations": config.ablations},
            results=[],
            eval_rows=[],
        )
        return

    dataset = build_dataset_adapter(config).load(config)
    baselines = build_baseline_registry(config)
    if command == "baseline":
        if baseline_name is None:
            raise RuntimeError("baseline_name is required for the baseline command.")
        method = baselines[baseline_name]
        benchmark_spec = _build_benchmark_spec(config, method.inference_mode)
    elif command == "evaluate":
        method = baselines["sam2_zero_shot"]
        benchmark_spec = _build_benchmark_spec(config, method.inference_mode)
    else:
        raise ValueError(f"Unknown command: {command}")

    artifact_records = [
        ArtifactRecord(
            stage=PipelineStage.EVALUATE.value,
            artifact_dir=str(output_dir),
            artifact_name=f"{command}_{baseline_name or 'default'}",
            metadata={
                "command": command,
                "baseline_name": baseline_name or "sam2_zero_shot",
                "model_id": config.model.model_id,
                "dataset_id": config.dataset.dataset_id,
                "track": config.evaluation.track,
                # 这里明确记录方法自身的推理模式，而不是盲目复用配置默认值。
                "inference_mode": method.inference_mode.value,
            },
        ).to_dict()
    ]

    results = []
    eval_rows: List[Dict[str, Any]] = []
    for seed in config.runtime.seeds:
        aggregate, rows = evaluate_method(
            method=method,
            samples=dataset.samples,
            config=config,
            track_name=config.evaluation.track,
            inference_mode=method.inference_mode,
        )
        results.append(_seed_result(seed, aggregate))
        eval_rows.extend([{**row, "seed": seed} for row in rows])

    summary = {
        "command": command,
        "baseline_name": baseline_name if command == "baseline" else "sam2_zero_shot",
        "dataset_manifest": dataset.manifest.to_dict(),
        "mean": _mean_numeric(results),
        "std": _std_numeric(results),
        "per_seed": results,
    }

    write_results(
        output_dir,
        benchmark_spec=benchmark_spec,
        artifact_manifest={
            "records": artifact_records
            + [
                ArtifactRecord(
                    stage=PipelineStage.EVALUATE.value,
                    artifact_dir=str(path.parent),
                    artifact_name=name,
                    metadata={"path": str(path)},
                ).to_dict()
                for name, path in {
                    "benchmark_spec": output_dir / "benchmark_spec.json",
                    "artifact_manifest": output_dir / "artifact_manifest.json",
                    "summary": output_dir / "summary.json",
                    "results": output_dir / "results.json",
                    "eval_rows": output_dir / "eval_reports" / "rows.json",
                }.items()
            ]
        },
        summary=summary,
        results=results,
        eval_rows=eval_rows,
    )
    if command == "baseline" and baseline_name is not None:
        _snapshot_reference_outputs(config, baseline_name, output_dir)


def _mean_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算 per-seed 结果的均值。"""
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "seed" or not isinstance(value, (int, float)):
                continue
            numeric.setdefault(key, []).append(float(value))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in numeric.items()}
