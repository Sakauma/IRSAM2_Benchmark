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
from ..core.interfaces import ArtifactRecord, InferenceMode, PipelineStage, PromptPolicy, PromptSource, PromptType
from ..data import build_dataset_adapter
from ..data.views import build_image_view, build_track_view
from ..evaluation.reporting import write_results
from ..evaluation.runner import evaluate_method
from ..evaluation.visualization import save_visualizations
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
        "method": config.method,
        "modules": config.modules,
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
    if sample.mask_array is not None:
        return np.asarray(sample.mask_array, dtype=np.float32)
    return np.zeros((sample.height, sample.width), dtype=np.float32)


def _build_visual_records(
    *,
    method,
    samples,
    inference_mode: InferenceMode,
    visual_limit: int,
) -> List[Dict[str, Any]]:
    if visual_limit <= 0:
        return []

    records: List[Dict[str, Any]] = []
    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        image_view = build_image_view(samples)
        for _, items in image_view.items():
            representative = items[0]
            prediction = method.predict_sample(representative)
            gt_instances = [{"mask": item.mask_array} for item in items if item.mask_array is not None]
            records.append(
                {
                    "sample": representative,
                    "gt_instances": gt_instances,
                    "pred_instances": prediction.get("instances", []),
                }
            )
            if len(records) >= visual_limit:
                break
        return records

    if inference_mode == InferenceMode.VIDEO_PROPAGATION:
        track_view = build_track_view(samples)
        for _, items in track_view.items():
            predictions = method.predict_sequence(items)
            for item in items:
                pred_mask = np.asarray(predictions[item.sample_id], dtype=np.float32)
                records.append(
                    {
                        "sample": item,
                        "pred_mask": pred_mask,
                        "gt_mask": _visual_gt_mask(item),
                    }
                )
                if len(records) >= visual_limit:
                    return records
        return records

    for item in samples[:visual_limit]:
        prediction = method.predict_sample(item)
        pred_mask = np.asarray(prediction["mask"], dtype=np.float32)
        records.append(
            {
                "sample": item,
                "pred_mask": pred_mask,
                "gt_mask": _visual_gt_mask(item),
            }
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
        if config.runtime.save_visuals:
            visual_records = _build_visual_records(
                method=method,
                samples=dataset.samples,
                inference_mode=method.inference_mode,
                visual_limit=max(0, int(config.runtime.visual_limit)),
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
                        stage=PipelineStage.EVALUATE.value,
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
    _write_run_metadata(
        output_dir=output_dir,
        config=config,
        command=command,
        baseline_name=resolved_baseline_name,
        benchmark_spec=benchmark_spec,
        dataset_manifest=dataset.manifest.to_dict(),
        output_paths=output_paths,
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
