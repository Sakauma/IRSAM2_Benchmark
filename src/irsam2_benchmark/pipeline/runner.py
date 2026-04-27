from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..baselines import build_baseline_registry
from ..config import AppConfig
from ..core.interfaces import ArtifactRecord, InferenceMode, PromptPolicy, PromptSource, PromptType
from ..data import build_dataset_adapter
from ..data.masks import sample_mask_array, sample_mask_or_zeros
from ..data.views import build_image_view
from ..evaluation.reporting import write_results
from ..evaluation.runner import _merge_error_context, _reset_error_log, _write_sample_error, align_mask_to_sample, evaluate_method
from ..evaluation.visualization import save_visualizations


def _effective_prompt_policy(config: AppConfig, inference_mode: InferenceMode) -> Dict[str, Any]:
    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
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
    # benchmark_spec 是每个 run 的协议快照。
    # 它随结果一起落盘，后续可以在不读取原始 YAML 的情况下判断结果口径。
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
        "method": config.method,
        "config_path": str(config.config_path),
    }


def _run_text(command: List[str], cwd: Any) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_info(config: AppConfig) -> Dict[str, Any]:
    status = _run_text(["git", "status", "--short"], config.root)
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"], config.root),
        "branch": _run_text(["git", "branch", "--show-current"], config.root),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _torch_info() -> Dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() else ""
        return {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", ""),
            "gpu_name": gpu_name,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Any) -> Dict[str, Any]:
    path = path if hasattr(path, "exists") else Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": _sha256(path),
    }


def _sam2_info(config: AppConfig) -> Dict[str, Any]:
    repo = config.sam2_repo
    return {
        "repo": str(repo),
        "commit": _run_text(["git", "rev-parse", "HEAD"], repo) if repo.exists() else "",
        "dirty": bool(_run_text(["git", "status", "--short"], repo)) if repo.exists() else False,
    }


def _checkpoint_path(config: AppConfig) -> Path:
    path = Path(config.model.ckpt)
    return path if path.is_absolute() else config.sam2_repo / path


def _write_run_metadata(
    *,
    output_dir: Any,
    config: AppConfig,
    command: str,
    baseline_name: str | None,
    benchmark_spec: Dict[str, Any],
    dataset_manifest: Dict[str, Any] | None = None,
    output_paths: Dict[str, Any] | None = None,
) -> None:
    # run_metadata 记录可复现性信息：项目/SAM2 commit、Python/Torch、checkpoint hash 和输出路径。
    metadata = {
        "command": command,
        "baseline_name": baseline_name,
        "config_path": str(config.config_path),
        "benchmark_spec": benchmark_spec,
        "git": _git_info(config),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": _torch_info(),
        "sam2": _sam2_info(config),
        "checkpoint": _file_info(_checkpoint_path(config)),
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "root": str(config.dataset_root),
            "manifest": dataset_manifest or {},
        },
        "outputs": {name: str(path) for name, path in (output_paths or {}).items()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_error_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _validate_evaluation_outputs(
    *,
    command: str,
    config: AppConfig,
    sample_count: int,
    eval_rows: List[Dict[str, Any]],
    error_log_path: Path,
) -> None:
    # baseline 至少应产生一行评估结果。
    # 如果全失败但进程正常返回，这里会把 silent failure 提升成显式错误。
    if command != "baseline":
        return
    if sample_count <= 0:
        raise RuntimeError(f"{command} loaded zero samples for dataset_id={config.dataset.dataset_id!r}.")
    if eval_rows:
        return
    error_count = _count_error_records(error_log_path)
    message = (
        f"{command} produced zero evaluation rows for dataset_id={config.dataset.dataset_id!r} "
        f"with sample_count={sample_count} and error_count={error_count}."
    )
    if error_log_path.exists():
        message += f" See error log: {error_log_path}"
    raise RuntimeError(message)


def _seed_result(seed: int, aggregate: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"seed": seed}
    payload.update({k: v for k, v in aggregate.items() if not isinstance(v, list)})
    return payload


def _std_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
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
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _snapshot_reference_outputs(config: AppConfig, baseline_name: str, output_dir: Any) -> None:
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


def _visual_gt_mask(sample) -> np.ndarray:
    return sample_mask_or_zeros(sample)


def _build_visual_records(
    *,
    method,
    samples,
    inference_mode: InferenceMode,
    visual_limit: int,
    config: AppConfig | None = None,
    track_name: str = "",
    error_context: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    # 可视化只抽取前 visual_limit 个案例，避免完整 benchmark 生成过多图片。
    # no-prompt 模式按 image 展示，prompted 模式按 instance 展示。
    if visual_limit <= 0:
        return []

    records: List[Dict[str, Any]] = []
    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        image_view = build_image_view(samples)
        for _, items in image_view.items():
            representative = items[0]
            group_sample_ids = [item.sample_id for item in items]
            try:
                prediction = method.predict_sample(representative)
                gt_instances = []
                for item in items:
                    gt_mask = sample_mask_array(item)
                    if gt_mask is not None:
                        gt_instances.append({"mask": gt_mask})
                pred_instances = []
                for instance in prediction.get("instances", []):
                    aligned_mask, _ = align_mask_to_sample(instance["mask"], representative)
                    pred_instances.append({**instance, "mask": aligned_mask})
                records.append(
                    {
                        "sample": representative,
                        "gt_instances": gt_instances,
                        "pred_instances": pred_instances,
                    }
                )
            except Exception as exc:
                _write_sample_error(
                    config=config,
                    method=method,
                    sample=representative,
                    exc=exc,
                    context=_merge_error_context(
                        error_context,
                        {
                            "track_name": track_name,
                            "stage": "visual_auto_mask",
                            "group_sample_ids": group_sample_ids,
                        },
                    ),
                )
            if len(records) >= visual_limit:
                break
        return records

    for item in samples[:visual_limit]:
        try:
            prediction = method.predict_sample(item)
            pred_mask, _ = align_mask_to_sample(prediction["mask"], item)
            records.append(
                {
                    "sample": item,
                    "pred_mask": pred_mask,
                    "gt_mask": _visual_gt_mask(item),
                }
            )
        except Exception as exc:
            _write_sample_error(
                config=config,
                method=method,
                sample=item,
                exc=exc,
                context=_merge_error_context(error_context, {"track_name": track_name, "stage": "visual_prompted"}),
            )
    return records


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_method(config: AppConfig, command: str, baseline_name: Optional[str]):
    baselines = build_baseline_registry(config)
    if command != "baseline":
        raise ValueError(f"Unknown command: {command}")
    if baseline_name is None:
        raise RuntimeError("baseline_name is required for the baseline command.")
    if baseline_name not in baselines:
        valid = ", ".join(sorted(baselines))
        raise RuntimeError(f"Unknown baseline {baseline_name!r}. Valid baselines: {valid}")
    return baseline_name, baselines[baseline_name]


def run_command(config: AppConfig, command: str, baseline_name: Optional[str] = None) -> None:
    # run_command 是 main.py 的执行核心：
    # 先解析配置和数据集，再按 seed 重建 method，最后写 results/summary/rows/metadata。
    if command != "baseline":
        raise ValueError(f"Unknown command: {command}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset_adapter(config).load(config)
    resolved_baseline_name, template_method = _resolve_method(config, command, baseline_name)
    benchmark_spec = _build_benchmark_spec(config, template_method.inference_mode)

    artifact_records = [
        ArtifactRecord(
            stage="evaluate",
            artifact_dir=str(output_dir),
            artifact_name=f"{command}_{resolved_baseline_name}",
            metadata={
                "command": command,
                "baseline_name": resolved_baseline_name,
                "model_id": config.model.model_id,
                "dataset_id": config.dataset.dataset_id,
                "track": config.evaluation.track,
                "inference_mode": template_method.inference_mode.value,
            },
        ).to_dict()
    ]
    results = []
    eval_rows: List[Dict[str, Any]] = []
    _reset_error_log(config)
    for seed in config.runtime.seeds:
        # 每个 seed 都重新构造 method，避免 SAM2 predictor 或随机状态在 seed 间泄漏。
        set_global_seed(seed)
        _, method = _resolve_method(config, command, baseline_name)
        error_context = {
            "command": command,
            "baseline_name": resolved_baseline_name,
            "seed": seed,
            "output_dir": str(output_dir),
        }
        aggregate, rows = evaluate_method(
            method=method,
            samples=dataset.samples,
            config=config,
            track_name=config.evaluation.track,
            inference_mode=method.inference_mode,
            error_context=error_context,
        )
        results.append(_seed_result(seed, aggregate))
        eval_rows.extend([{**row, "seed": seed} for row in rows])
        if config.runtime.save_visuals:
            # 可视化失败不会影响评估行生成；错误会进入 error_log.jsonl。
            visual_records = _build_visual_records(
                method=method,
                samples=dataset.samples,
                inference_mode=method.inference_mode,
                visual_limit=max(0, int(config.runtime.visual_limit)),
                config=config,
                track_name=config.evaluation.track,
                error_context=error_context,
            )
            visual_paths = save_visualizations(
                output_dir=output_dir,
                visual_records=visual_records,
                inference_mode=method.inference_mode,
                method_name=resolved_baseline_name,
                seed=seed,
            )
            if visual_paths:
                visual_dir = output_dir / "visuals" / resolved_baseline_name / f"seed_{seed}"
                artifact_records.append(
                    ArtifactRecord(
                        stage="evaluate",
                        artifact_dir=str(visual_dir),
                        artifact_name=f"visuals_seed_{seed}",
                        metadata={"count": len(visual_paths), "paths": [str(path) for path in visual_paths]},
                    ).to_dict()
                )
    summary = {
        "command": command,
        "baseline_name": resolved_baseline_name,
        "dataset_manifest": dataset.manifest.to_dict(),
        "mean": _mean_numeric(results),
        "std": _std_numeric(results),
        "per_seed": results,
    }
    output_artifacts = {
        "benchmark_spec": output_dir / "benchmark_spec.json",
        "artifact_manifest": output_dir / "artifact_manifest.json",
        "summary": output_dir / "summary.json",
        "results": output_dir / "results.json",
        "eval_rows": output_dir / "eval_reports" / "rows.json",
    }
    error_log_path = output_dir / "eval_reports" / "error_log.jsonl"
    if error_log_path.exists():
        output_artifacts["error_log"] = error_log_path
    output_paths = write_results(
        output_dir,
        benchmark_spec=benchmark_spec,
        artifact_manifest={
            "records": artifact_records
            + [
                ArtifactRecord(
                    stage="evaluate",
                    artifact_dir=str(path.parent),
                    artifact_name=name,
                    metadata={"path": str(path)},
                ).to_dict()
                for name, path in output_artifacts.items()
            ]
        },
        summary=summary,
        results=results,
        eval_rows=eval_rows,
    )
    if error_log_path.exists():
        output_paths["error_log"] = error_log_path
    _write_run_metadata(
        output_dir=output_dir,
        config=config,
        command=command,
        baseline_name=resolved_baseline_name,
        benchmark_spec=benchmark_spec,
        dataset_manifest=dataset.manifest.to_dict(),
        output_paths=output_paths,
    )
    _validate_evaluation_outputs(
        command=command,
        config=config,
        sample_count=len(dataset.samples),
        eval_rows=eval_rows,
        error_log_path=error_log_path,
    )
    if command == "baseline" and config.runtime.update_reference_results:
        _snapshot_reference_outputs(config, resolved_baseline_name, output_dir)


def _mean_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "seed" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric.setdefault(key, []).append(float(value))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in numeric.items()}
