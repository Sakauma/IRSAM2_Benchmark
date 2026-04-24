from __future__ import annotations

import random
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from ..baselines import build_baseline_registry
from ..config import AppConfig
from ..core.interfaces import ArtifactRecord, InferenceMode, PipelineStage, PromptPolicy, PromptSource, PromptType
from ..data import build_dataset_adapter
from ..evaluation.reporting import write_results
from ..evaluation.runner import evaluate_method
from .stages import run_adapt_stage, run_distill_stage, run_quantize_stage, run_transfer_stage


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
    if command == "baseline":
        if baseline_name is None:
            raise RuntimeError("baseline_name is required for the baseline command.")
        return baseline_name, baselines[baseline_name]
    if command == "evaluate":
        return "sam2_zero_shot", baselines["sam2_zero_shot"]
    raise ValueError(f"Unknown command: {command}")


def run_command(config: AppConfig, command: str, baseline_name: Optional[str] = None) -> None:
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
    resolved_baseline_name, template_method = _resolve_method(config, command, baseline_name)
    benchmark_spec = _build_benchmark_spec(config, template_method.inference_mode)

    artifact_records = [
        ArtifactRecord(
            stage=PipelineStage.EVALUATE.value,
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
    for seed in config.runtime.seeds:
        set_global_seed(seed)
        _, method = _resolve_method(config, command, baseline_name)
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
        "baseline_name": resolved_baseline_name,
        "dataset_manifest": dataset.manifest.to_dict(),
        "mean": _mean_numeric(results),
        "std": _std_numeric(results),
        "per_seed": results,
    }
    output_paths = write_results(
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
    if command == "baseline":
        _snapshot_reference_outputs(config, resolved_baseline_name, output_dir)


def _mean_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "seed" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric.setdefault(key, []).append(float(value))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in numeric.items()}
