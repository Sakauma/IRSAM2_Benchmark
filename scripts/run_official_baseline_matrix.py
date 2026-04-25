#!/usr/bin/env python

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
REQUIRED_RUN_OUTPUTS = ("summary.json", "results.json", "eval_reports/rows.json", "run_metadata.json")
TRACKED_ENV_VARS = (
    "ARTIFACT_ROOT",
    "CHECKPOINT_ROOT",
    "CUDA_VISIBLE_DEVICES",
    "DATASET_ROOT",
    "MATRIX_DATASETS",
    "MATRIX_MODELS",
    "MATRIX_MODES",
    "MATRIX_RESUME",
    "MATRIX_SEEDS",
    "MULTIMODAL_DATASET_ROOT",
    "PYTHONPATH",
    "PYTHON_BIN",
    "RBGT_DATASET_ROOT",
    "SAM2_CKPT_ROOT",
    "SAM2_REPO",
    "VISUAL_LIMIT",
)
SUMMARY_FIELDNAMES = [
    "status",
    "dataset",
    "model",
    "baseline",
    "output_dir",
    "mIoU",
    "Dice",
    "BoundaryF1",
    "LatencyMs",
    "BBoxIoU",
    "instance_f1",
    "instance_precision",
    "instance_recall",
    "mIoU_std",
    "Dice_std",
    "BoundaryF1_std",
    "LatencyMs_std",
]
FAILURE_FIELDNAMES = [
    "dataset",
    "model",
    "baseline",
    "output_dir",
    "error_type",
    "returncode",
    "message",
    "started_at",
    "completed_at",
    "duration_seconds",
]
CHECKPOINT_METADATA_CACHE: dict[str, dict[str, Any]] = {}

MODELS = [
    {
        "alias": "tiny",
        "model_id": "sam2.1_hiera_tiny",
        "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_tiny.pt",
    },
    {
        "alias": "small",
        "model_id": "sam2.1_hiera_small",
        "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_small.pt",
    },
    {
        "alias": "base_plus",
        "model_id": "sam2.1_hiera_base_plus",
        "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_base_plus.pt",
    },
    {
        "alias": "large",
        "model_id": "sam2.1_hiera_large",
        "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_large.pt",
    },
]

BASELINES = [
    {
        "name": "sam2_zero_shot",
        "alias": "box",
        "track": "track_a_image_prompted",
        "inference_mode": "box",
        "prompt_policy": {
            "name": "box_prompt",
            "prompt_type": "box",
            "prompt_source": "gt",
            "prompt_budget": 1,
            "refresh_interval": 10,
            "multi_mask": False,
            "notes": "Official matrix box prompt baseline.",
        },
    },
    {
        "name": "sam2_zero_shot_point",
        "alias": "point",
        "track": "track_a_image_prompted",
        "inference_mode": "point",
        "prompt_policy": {
            "name": "point_prompt",
            "prompt_type": "point",
            "prompt_source": "gt",
            "prompt_budget": 1,
            "refresh_interval": 10,
            "multi_mask": False,
            "notes": "Official matrix point prompt baseline.",
        },
    },
    {
        "name": "sam2_zero_shot_box_point",
        "alias": "box_point",
        "track": "track_a_image_prompted",
        "inference_mode": "box+point",
        "prompt_policy": {
            "name": "box_point_prompt",
            "prompt_type": "box+point",
            "prompt_source": "gt",
            "prompt_budget": 2,
            "refresh_interval": 10,
            "multi_mask": False,
            "notes": "Official matrix box+point prompt baseline.",
        },
    },
    {
        "name": "sam2_no_prompt_auto_mask",
        "alias": "no_prompt",
        "track": "track_b_auto_mask",
        "inference_mode": "no_prompt_auto_mask",
        "prompt_policy": {
            "name": "no_prompt_auto_mask",
            "prompt_type": "none",
            "prompt_source": "none",
            "prompt_budget": 0,
            "refresh_interval": None,
            "multi_mask": True,
            "notes": "Official matrix no-prompt automatic mask baseline.",
        },
    },
]

DATASETS = [
    {
        "alias": "multimodal",
        "config_path": PROJECT_ROOT / "configs" / "benchmark_v1.yaml",
        "dataset_root_env": "MULTIMODAL_DATASET_ROOT",
        "dataset_root_default": "/root/autodl-tmp/datasets/MultiModalCOCOClean",
    },
    {
        "alias": "rbgt",
        "config_path": PROJECT_ROOT / "configs" / "benchmark_v1_rbgt_tiny.yaml",
        "dataset_root_env": "RBGT_DATASET_ROOT",
        "dataset_root_default": "/root/autodl-tmp/datasets/RBGT-Tiny",
    },
]


def _split_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_rows(rows: list[dict], aliases: list[str], key: str) -> list[dict]:
    if not aliases:
        return rows
    allowed = set(aliases)
    return [row for row in rows if row[key] in allowed]


def _checkpoint_root() -> Path:
    explicit = os.environ.get("SAM2_CKPT_ROOT") or os.environ.get("CHECKPOINT_ROOT")
    if explicit:
        return Path(explicit)
    sam2_repo = Path(os.environ.get("SAM2_REPO", "/root/sam2"))
    return sam2_repo / "checkpoints"


def _resolve_model_ckpt(model: dict) -> str:
    raw = Path(model["ckpt"])
    if raw.is_absolute():
        if not raw.exists():
            raise FileNotFoundError(f"Checkpoint not found: {raw}")
        return str(raw)

    candidate = _checkpoint_root() / raw.name
    if candidate.exists():
        return str(candidate)

    sam2_repo_candidate = Path(os.environ.get("SAM2_REPO", "/root/sam2")) / raw
    if sam2_repo_candidate.exists():
        return str(sam2_repo_candidate)

    raise FileNotFoundError(
        "Official SAM2 checkpoint not found for matrix run.\n"
        f"  model alias: {model['alias']}\n"
        f"  expected under checkpoint root: {_checkpoint_root() / raw.name}\n"
        f"  fallback repo-relative path: {sam2_repo_candidate}\n"
        "Please set SAM2_CKPT_ROOT (or CHECKPOINT_ROOT) to the directory that contains the official SAM2.1 .pt files."
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_text(args: list[str], *, cwd: Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    except Exception:
        return None
    output = (result.stdout or result.stderr or "").strip()
    return output or None


def _git_metadata() -> dict[str, Any]:
    status = _run_text(["git", "status", "--porcelain"]) or ""
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote_origin": _run_text(["git", "remote", "get-url", "origin"]),
        "dirty": bool(status.strip()),
    }


def _python_version(python_bin: str) -> str | None:
    return _run_text([python_bin, "--version"])


def _gpu_metadata() -> list[dict[str, str]]:
    output = _run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if output is None:
        return []
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return rows


def _collect_common_metadata(python_bin: str) -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "matrix_runner": sys.version.replace("\n", " "),
            "target_python": _python_version(python_bin),
            "python_bin": python_bin,
        },
        "gpu": _gpu_metadata(),
    }


def _checkpoint_metadata(path: str) -> dict[str, Any]:
    checkpoint = Path(path)
    metadata: dict[str, Any] = {"path": str(checkpoint), "exists": checkpoint.exists()}
    if checkpoint.exists():
        stat = checkpoint.stat()
        cache_key = f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        if cache_key in CHECKPOINT_METADATA_CACHE:
            return dict(CHECKPOINT_METADATA_CACHE[cache_key])
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata.update(
            {
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": digest.hexdigest(),
            }
        )
        CHECKPOINT_METADATA_CACHE[cache_key] = dict(metadata)
    return metadata


def _env_snapshot(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in TRACKED_ENV_VARS if key in env}


def _run_output_name(dataset: dict, model: dict, baseline: dict) -> str:
    return f"official_baseline_matrix/{dataset['alias']}/{model['alias']}/{baseline['alias']}"


def _is_completed_run(output_dir: Path) -> bool:
    for relative in REQUIRED_RUN_OUTPUTS:
        if not (output_dir / relative).exists():
            return False
    try:
        metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("status") == "completed"


def _summary_row(dataset: dict, model: dict, baseline: dict, output_dir: Path, summary: dict, status: str) -> dict[str, Any]:
    mean_metrics = summary.get("mean", {})
    std_metrics = summary.get("std", {})
    return {
        "status": status,
        "dataset": dataset["alias"],
        "model": model["alias"],
        "baseline": baseline["alias"],
        "output_dir": str(output_dir),
        "mIoU": mean_metrics.get("mIoU"),
        "Dice": mean_metrics.get("Dice"),
        "BoundaryF1": mean_metrics.get("BoundaryF1"),
        "LatencyMs": mean_metrics.get("LatencyMs"),
        "BBoxIoU": mean_metrics.get("BBoxIoU"),
        "instance_f1": mean_metrics.get("instance_f1"),
        "instance_precision": mean_metrics.get("instance_precision"),
        "instance_recall": mean_metrics.get("instance_recall"),
        "mIoU_std": std_metrics.get("mIoU"),
        "Dice_std": std_metrics.get("Dice"),
        "BoundaryF1_std": std_metrics.get("BoundaryF1"),
        "LatencyMs_std": std_metrics.get("LatencyMs"),
    }


def _base_run_metadata(
    *,
    dataset: dict,
    model: dict,
    baseline: dict,
    payload: dict,
    output_dir: Path,
    command: list[str],
    env: dict[str, str],
    dataset_root: str,
    resolved_ckpt: str | None,
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "running",
        "dataset": dataset["alias"],
        "model": model["alias"],
        "baseline": baseline["alias"],
        "output_dir": str(output_dir),
        "command": command,
        "config": payload,
        "dataset_root": dataset_root,
        "checkpoint": _checkpoint_metadata(resolved_ckpt) if resolved_ckpt else {"path": model.get("ckpt"), "exists": False},
        "environment": _env_snapshot(env),
        **common_metadata,
    }


def _failure_record(
    *,
    dataset: dict,
    model: dict,
    baseline: dict,
    output_dir: Path,
    error: BaseException,
    started_at: str | None,
    completed_at: str,
    duration_seconds: float | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset["alias"],
        "model": model["alias"],
        "baseline": baseline["alias"],
        "output_dir": str(output_dir),
        "error_type": type(error).__name__,
        "returncode": getattr(error, "returncode", None),
        "message": str(error),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
    }


def _write_matrix_outputs(matrix_root: Path, summary_rows: list[dict], failures: list[dict]) -> None:
    _write_json(matrix_root / "matrix_summary.json", summary_rows)
    _write_csv(matrix_root / "matrix_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    _write_json(matrix_root / "matrix_failures.json", failures)
    _write_csv(matrix_root / "matrix_failures.csv", failures, FAILURE_FIELDNAMES)


def main() -> int:
    python_bin = os.environ.get("PYTHON_BIN", sys.executable or "python")
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", str(PROJECT_ROOT / "artifacts")))
    matrix_root = artifact_root / "official_baseline_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)

    selected_model_aliases = _split_csv_env("MATRIX_MODELS", [row["alias"] for row in MODELS])
    selected_dataset_aliases = _split_csv_env("MATRIX_DATASETS", [row["alias"] for row in DATASETS])
    selected_baseline_aliases = _split_csv_env("MATRIX_MODES", [row["alias"] for row in BASELINES])
    selected_seeds = [int(value) for value in _split_csv_env("MATRIX_SEEDS", ["42"])]
    visual_limit = int(os.environ.get("VISUAL_LIMIT", "24"))
    resume_enabled = _bool_env("MATRIX_RESUME", True)
    common_metadata = _collect_common_metadata(python_bin)

    selected_models = _selected_rows(MODELS, selected_model_aliases, "alias")
    selected_datasets = _selected_rows(DATASETS, selected_dataset_aliases, "alias")
    selected_baselines = _selected_rows(BASELINES, selected_baseline_aliases, "alias")

    summary_rows: list[dict] = []
    failures: list[dict] = []
    run_count = len(selected_models) * len(selected_datasets) * len(selected_baselines)
    completed = 0

    with tempfile.TemporaryDirectory(prefix="irsam2_matrix_") as temp_dir:
        temp_root = Path(temp_dir)
        for dataset in selected_datasets:
            base_config = _load_yaml(dataset["config_path"])
            dataset_root = os.environ.get(dataset["dataset_root_env"], dataset["dataset_root_default"])
            for model in selected_models:
                for baseline in selected_baselines:
                    completed += 1
                    print(
                        f"[{completed}/{run_count}] dataset={dataset['alias']} model={model['alias']} mode={baseline['alias']}",
                        flush=True,
                    )
                    output_dir = artifact_root / _run_output_name(dataset, model, baseline)
                    if resume_enabled and _is_completed_run(output_dir):
                        print(f"[skip] completed run found at {output_dir}", flush=True)
                        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
                        summary_rows.append(_summary_row(dataset, model, baseline, output_dir, summary, status="skipped"))
                        continue

                    started_at = _utc_now()
                    started_perf = time.perf_counter()
                    metadata: dict[str, Any] | None = None
                    try:
                        payload = json.loads(json.dumps(base_config))
                        resolved_ckpt = _resolve_model_ckpt(model)
                        payload["model"]["model_id"] = model["model_id"]
                        payload["model"]["cfg"] = model["cfg"]
                        payload["model"]["ckpt"] = resolved_ckpt
                        payload["runtime"]["save_visuals"] = True
                        payload["runtime"]["visual_limit"] = visual_limit
                        payload["runtime"]["update_reference_results"] = False
                        payload["runtime"]["seeds"] = selected_seeds
                        payload["runtime"]["output_name"] = _run_output_name(dataset, model, baseline)
                        payload["evaluation"]["benchmark_version"] = "irsam2-benchmark-v1-official-matrix"
                        payload["evaluation"]["track"] = baseline["track"]
                        payload["evaluation"]["inference_mode"] = baseline["inference_mode"]
                        payload["evaluation"]["prompt_policy"] = baseline["prompt_policy"]

                        temp_config = temp_root / f"{dataset['alias']}_{model['alias']}_{baseline['alias']}.yaml"
                        _write_yaml(temp_config, payload)

                        env = os.environ.copy()
                        env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
                            os.pathsep
                        )
                        env["DATASET_ROOT"] = dataset_root
                        env["ARTIFACT_ROOT"] = str(artifact_root)
                        command = [
                            python_bin,
                            str(MAIN_PY),
                            "run",
                            "baseline",
                            "--config",
                            str(temp_config),
                            "--baseline",
                            baseline["name"],
                        ]
                        metadata = _base_run_metadata(
                            dataset=dataset,
                            model=model,
                            baseline=baseline,
                            payload=payload,
                            output_dir=output_dir,
                            command=command,
                            env=env,
                            dataset_root=dataset_root,
                            resolved_ckpt=resolved_ckpt,
                            common_metadata=common_metadata,
                        )
                        metadata["started_at"] = started_at
                        _write_json(output_dir / "run_metadata.json", metadata)

                        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

                        completed_at = _utc_now()
                        duration_seconds = round(time.perf_counter() - started_perf, 3)
                        summary_path = output_dir / "summary.json"
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        metadata.update(
                            {
                                "status": "completed",
                                "completed_at": completed_at,
                                "duration_seconds": duration_seconds,
                                "summary_path": str(summary_path),
                                "mean_metrics": summary.get("mean", {}),
                                "std_metrics": summary.get("std", {}),
                            }
                        )
                        _write_json(output_dir / "run_metadata.json", metadata)
                        summary_rows.append(_summary_row(dataset, model, baseline, output_dir, summary, status="completed"))
                    except Exception as error:
                        completed_at = _utc_now()
                        duration_seconds = round(time.perf_counter() - started_perf, 3)
                        failure = _failure_record(
                            dataset=dataset,
                            model=model,
                            baseline=baseline,
                            output_dir=output_dir,
                            error=error,
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_seconds=duration_seconds,
                        )
                        failures.append(failure)
                        if metadata is None:
                            metadata = {
                                "status": "failed",
                                "dataset": dataset["alias"],
                                "model": model["alias"],
                                "baseline": baseline["alias"],
                                "output_dir": str(output_dir),
                                **common_metadata,
                            }
                        metadata.update({"status": "failed", "completed_at": completed_at, "duration_seconds": duration_seconds, "failure": failure})
                        _write_json(output_dir / "run_metadata.json", metadata)
                        print(f"[failed] dataset={dataset['alias']} model={model['alias']} mode={baseline['alias']}: {error}", flush=True)
                        continue

    _write_matrix_outputs(matrix_root, summary_rows, failures)
    print(f"[done] matrix summary written to {matrix_root / 'matrix_summary.json'}")
    print(f"[done] matrix csv written to {matrix_root / 'matrix_summary.csv'}")
    print(f"[done] failure list written to {matrix_root / 'matrix_failures.json'}")
    if failures:
        print(f"[failed] {len(failures)} matrix run(s) failed. See {matrix_root / 'matrix_failures.json'}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
