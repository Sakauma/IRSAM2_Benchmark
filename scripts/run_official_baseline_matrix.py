#!/usr/bin/env python
# Author: Egor Izmaylov
#
# Run the full official SAM2 baseline matrix:
# - 4 official SAM2.1 checkpoints
# - 2 datasets (MultiModalCOCOClean / RBGT-Tiny)
# - 4 inference modes (box / point / box+point / no-prompt auto-mask)
#
# Each run writes its own frozen benchmark directory and this script also
# aggregates a matrix_summary.json/csv for quick comparison.

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"

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


def _selected_rows(rows: list[dict], aliases: list[str], key: str) -> list[dict]:
    if not aliases:
        return rows
    allowed = set(aliases)
    return [row for row in rows if row[key] in allowed]


def _checkpoint_root() -> Path:
    """Resolve the preferred official-checkpoint root for matrix runs.

    Priority:
    1. SAM2_CKPT_ROOT
    2. CHECKPOINT_ROOT
    3. SAM2_REPO/checkpoints
    """
    explicit = os.environ.get("SAM2_CKPT_ROOT") or os.environ.get("CHECKPOINT_ROOT")
    if explicit:
        return Path(explicit)
    sam2_repo = Path(os.environ.get("SAM2_REPO", "/root/sam2"))
    return sam2_repo / "checkpoints"


def _resolve_model_ckpt(model: dict) -> str:
    """Resolve one official SAM2 checkpoint to an absolute path.

    We fail early with a helpful message so the user does not have to dig
    through the downstream SAM2 loading stack to see which file is missing.
    """
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

    selected_models = _selected_rows(MODELS, selected_model_aliases, "alias")
    selected_datasets = _selected_rows(DATASETS, selected_dataset_aliases, "alias")
    selected_baselines = _selected_rows(BASELINES, selected_baseline_aliases, "alias")

    summary_rows: list[dict] = []
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
                    print(f"[{completed}/{run_count}] dataset={dataset['alias']} model={model['alias']} mode={baseline['alias']}", flush=True)
                    payload = json.loads(json.dumps(base_config))
                    payload["model"]["model_id"] = model["model_id"]
                    payload["model"]["cfg"] = model["cfg"]
                    payload["model"]["ckpt"] = _resolve_model_ckpt(model)
                    payload["runtime"]["save_visuals"] = True
                    payload["runtime"]["visual_limit"] = visual_limit
                    payload["runtime"]["seeds"] = selected_seeds
                    payload["runtime"]["output_name"] = f"official_baseline_matrix/{dataset['alias']}/{model['alias']}/{baseline['alias']}"
                    payload["evaluation"]["benchmark_version"] = "irsam2-benchmark-v1-official-matrix"
                    payload["evaluation"]["track"] = baseline["track"]
                    payload["evaluation"]["inference_mode"] = baseline["inference_mode"]
                    payload["evaluation"]["prompt_policy"] = baseline["prompt_policy"]

                    temp_config = temp_root / f"{dataset['alias']}_{model['alias']}_{baseline['alias']}.yaml"
                    _write_yaml(temp_config, payload)

                    env = os.environ.copy()
                    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
                    env["DATASET_ROOT"] = dataset_root
                    env["ARTIFACT_ROOT"] = str(artifact_root)

                    subprocess.run(
                        [python_bin, str(MAIN_PY), "run", "baseline", "--config", str(temp_config), "--baseline", baseline["name"]],
                        cwd=PROJECT_ROOT,
                        env=env,
                        check=True,
                    )

                    output_dir = artifact_root / payload["runtime"]["output_name"]
                    summary_path = output_dir / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    mean_metrics = summary.get("mean", {})
                    std_metrics = summary.get("std", {})
                    summary_rows.append(
                        {
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
                    )

    summary_json = matrix_root / "matrix_summary.json"
    summary_csv = matrix_root / "matrix_summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
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
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] matrix summary written to {summary_json}")
    print(f"[done] matrix csv written to {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
