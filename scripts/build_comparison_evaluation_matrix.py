#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(os.environ.get("IRSAM2_DATASET_ROOT", "/home/sakauma/dataset"))
IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

PUBLIC_DATASETS = {
    "nuaa_sirst": {
        "name": "NUAA-SIRST",
        "external_id": "NUAA-SIRST",
        "root_name": "NUAA-SIRST",
        "images_dir": "images",
        "masks_dir": "masks",
    },
    "nudt_sirst": {
        "name": "NUDT-SIRST",
        "external_id": "NUDT-SIRST",
        "root_name": "NUDT-SIRST",
        "images_dir": "images",
        "masks_dir": "masks",
    },
    "irstd_1k": {
        "name": "IRSTD-1K",
        "external_id": "IRSTD-1K",
        "root_name": "IRSTD-1k",
        "alt_root_names": ["IRSTD-1K"],
        "images_dir": "images",
        "masks_dir": "masks",
    },
}

DEFAULT_OURS_ANALYSES = {
    "M5-small-target-best": PROJECT_ROOT
    / "artifacts"
    / "sam2_ir_qd_m5_small_target_best_v1"
    / "analysis"
    / "checkpoint_sweep_summary.csv",
    "M6-promptnet-v2": PROJECT_ROOT
    / "artifacts"
    / "sam2_ir_qd_m6_promptnet_v2_v1"
    / "analysis"
    / "checkpoint_sweep_summary.csv",
}

MODEL_FAMILIES = {
    "bgm": "ir_supervised_segmentation",
    "dnanet": "ir_supervised_segmentation",
    "drpcanet": "ir_supervised_segmentation",
    "hdnet": "ir_supervised_segmentation",
    "mshnet": "ir_supervised_segmentation",
    "pconv_mshnet_p43": "ir_supervised_segmentation",
    "pconv_yolov8n_p2_p43_boxmask": "ir_detection_boxmask",
    "rpcanet_pp": "ir_supervised_segmentation",
    "sctransnet": "ir_supervised_segmentation",
    "serankdet": "ir_supervised_segmentation",
    "uiu_net": "ir_supervised_segmentation",
    "edge_sam": "generic_sam_family",
    "efficient_sam_vitt": "generic_sam_family",
    "fastsam_s": "generic_sam_family",
    "fastsam_x": "generic_sam_family",
    "hq_sam_vit_b": "generic_sam_family",
    "mobile_sam": "generic_sam_family",
    "sam_vit_b": "generic_sam_family",
    "sam2_unet_cod": "sam2_task_specific_transfer",
}

MODEL_LABELS = {
    "bgm": "BGM",
    "dnanet": "DNANet",
    "drpcanet": "DRPCANet",
    "edge_sam": "EdgeSAM",
    "efficient_sam_vitt": "EfficientSAM-ViT-T",
    "fastsam_s": "FastSAM-S",
    "fastsam_x": "FastSAM-X",
    "hdnet": "HDNet",
    "hq_sam_vit_b": "HQ-SAM-ViT-B",
    "mobile_sam": "MobileSAM",
    "mshnet": "MSHNet",
    "pconv_mshnet_p43": "PConv-MSHNet-P43",
    "pconv_yolov8n_p2_p43_boxmask": "PConv-YOLOv8n-P2-P43-BoxMask",
    "rpcanet_pp": "RPCANet++",
    "sam2_unet_cod": "SAM2-UNet-COD",
    "sam_vit_b": "SAM-ViT-B",
    "sctransnet": "SCTransNet",
    "serankdet": "SeRankDet",
    "uiu_net": "UIU-Net",
}

METRIC_COLUMNS = [
    "mIoU",
    "Dice",
    "WholeMaskIoU25Rate",
    "WholeMaskIoU50Rate",
    "FalseAlarmPixelsPerMP",
    "PredAreaPixels",
    "GTAreaPixels",
]

OURS_METRIC_MAP = {
    "mIoU": "mIoU_mean",
    "Dice": "Dice_mean",
    "TargetRecallIoU25": "TargetRecallIoU25_mean",
    "TargetRecallIoU50": "TargetRecallIoU50_mean",
    "FalseAlarmPixelsPerMP": "FalseAlarmPixelsPerMP_mean",
    "PromptHitRate": "PromptHitRate_mean",
    "PromptBoxCoverage": "PromptBoxCoverage_mean",
    "PromptTopKHitRate": "PromptTopKHitRate_mean",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build comparison tables between current SAM2-IR-QD runs and imported third-party masks."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--external-root", type=Path, default=PROJECT_ROOT / "artifacts" / "external_predictions")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "comparison_evaluation_matrix_latest")
    parser.add_argument(
        "--ours-analysis",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Additional or replacement analysis CSV. May be repeated. Defaults to M5 and M6 if present.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated external model directory names. Empty means all models with all public datasets.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(PUBLIC_DATASETS),
        help="Comma-separated dataset keys. Default: public IR3.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    dataset_keys = _split_csv_arg(args.datasets)
    unknown_datasets = [key for key in dataset_keys if key not in PUBLIC_DATASETS]
    if unknown_datasets:
        raise ValueError(f"Unknown dataset keys: {unknown_datasets}")

    output_root = args.output_root.resolve()
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_inventory, gt_by_dataset = load_ground_truth(args.dataset_root.resolve(), dataset_keys)
    external_models = discover_external_models(args.external_root.resolve(), dataset_keys, _split_csv_arg(args.models))
    external_dataset_rows = evaluate_external_models(
        external_root=args.external_root.resolve(),
        models=external_models,
        gt_by_dataset=gt_by_dataset,
        threshold=args.threshold,
    )
    external_macro_rows = macro_by_model(external_dataset_rows)

    analysis_inputs = parse_ours_analysis_args(args.ours_analysis)
    ours_dataset_rows = collect_ours_analysis_rows(analysis_inputs, dataset_keys)
    ours_macro_rows = macro_by_model(ours_dataset_rows)
    comparison_macro_rows = sorted(
        external_macro_rows + ours_macro_rows,
        key=lambda row: (_as_float(row.get("mIoU")) is not None, _as_float(row.get("mIoU")) or -1.0),
        reverse=True,
    )

    write_csv(tables_dir / "dataset_inventory.csv", dataset_inventory)
    write_csv(tables_dir / "external_public_dataset.csv", external_dataset_rows)
    write_csv(tables_dir / "external_public_macro.csv", external_macro_rows)
    write_csv(tables_dir / "ours_public_dataset.csv", ours_dataset_rows)
    write_csv(tables_dir / "ours_public_macro.csv", ours_macro_rows)
    write_csv(tables_dir / "comparison_public_macro.csv", comparison_macro_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "external_root": str(args.external_root.resolve()),
        "output_root": str(output_root),
        "dataset_keys": dataset_keys,
        "external_models": external_models,
        "ours_analysis_inputs": {label: str(path) for label, path in analysis_inputs.items()},
        "threshold": args.threshold,
        "notes": [
            "External TargetRecall columns are whole-mask IoU threshold proxy rates.",
            "Ours rows are read from benchmark analysis CSV files and keep benchmark TargetRecall metrics.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_root / "comparison-report.md", dataset_inventory, comparison_macro_rows, ours_macro_rows)

    print(f"Wrote comparison matrix to {output_root}")
    print(f"External models: {len(external_models)}")
    print(f"Ours analysis files: {len(analysis_inputs)}")
    return 0


def _split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_root(dataset_root: Path, spec: dict[str, Any]) -> Path:
    candidates = [dataset_root / str(spec["root_name"])]
    candidates.extend(dataset_root / name for name in spec.get("alt_root_names", []))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _mask_key(path: Path) -> set[str]:
    stem = path.stem
    normalized = re.sub(r"_pixels\d+$", "", stem)
    return {stem, stem.lower(), normalized, normalized.lower()}


def _mask_map(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in _image_files(root):
        for key in _mask_key(path):
            output.setdefault(key, path)
    return output


def _load_binary_mask(path: Path, *, size: tuple[int, int] | None = None, threshold: float = 0.5) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.NEAREST)
        arr = np.asarray(image, dtype=np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return arr > threshold


def load_ground_truth(dataset_root: Path, dataset_keys: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    inventory: list[dict[str, Any]] = []
    gt_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset_key in dataset_keys:
        spec = PUBLIC_DATASETS[dataset_key]
        root = _dataset_root(dataset_root, spec)
        image_dir = root / str(spec["images_dir"])
        mask_dir = root / str(spec["masks_dir"])
        masks = _mask_map(mask_dir)
        samples: list[dict[str, Any]] = []
        missing_masks = 0
        skipped_size_mismatch = 0
        for image_path in _image_files(image_dir):
            mask_path = masks.get(image_path.stem) or masks.get(image_path.stem.lower())
            if mask_path is None:
                missing_masks += 1
                continue
            with Image.open(image_path) as image:
                size = image.size
            with Image.open(mask_path) as mask_image:
                mask_size = mask_image.size
            if mask_size != size:
                skipped_size_mismatch += 1
                continue
            gt_mask = _load_binary_mask(mask_path)
            samples.append(
                {
                    "dataset": dataset_key,
                    "dataset_name": spec["name"],
                    "external_id": spec["external_id"],
                    "frame_id": image_path.stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "size": size,
                    "gt_mask": gt_mask,
                    "GTAreaPixels": float(gt_mask.sum()),
                }
            )
        gt_by_dataset[spec["external_id"]] = samples
        inventory.append(
            {
                "dataset": dataset_key,
                "dataset_name": spec["name"],
                "external_id": spec["external_id"],
                "root": str(root),
                "image_count": len(_image_files(image_dir)),
                "sample_count": len(samples),
                "missing_masks": missing_masks,
                "skipped_size_mismatch": skipped_size_mismatch,
            }
        )
    return inventory, gt_by_dataset


def discover_external_models(external_root: Path, dataset_keys: list[str], requested: list[str]) -> list[str]:
    required_ids = {PUBLIC_DATASETS[key]["external_id"] for key in dataset_keys}
    if requested:
        models = requested
    else:
        models = [path.name for path in sorted(external_root.iterdir()) if path.is_dir()]
    output = []
    for model in models:
        if model.endswith("_smoke") or model == "local_adapt_smoke":
            continue
        model_root = external_root / model
        if all((model_root / dataset_id).exists() for dataset_id in required_ids):
            output.append(model)
    return output


def _prediction_path(prediction_dir: Path, frame_id: str) -> Path | None:
    direct = prediction_dir / f"{frame_id}.png"
    if direct.exists():
        return direct
    matches = sorted(path for path in prediction_dir.glob(f"{frame_id}.*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return matches[0] if matches else None


def evaluate_external_models(
    *,
    external_root: Path,
    models: list[str],
    gt_by_dataset: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for external_id, samples in gt_by_dataset.items():
            prediction_dir = external_root / model / external_id
            sample_metrics: list[dict[str, float]] = []
            missing = 0
            for sample in samples:
                prediction_path = _prediction_path(prediction_dir, str(sample["frame_id"]))
                if prediction_path is None:
                    missing += 1
                    continue
                pred_mask = _load_binary_mask(prediction_path, size=sample["size"], threshold=threshold)
                sample_metrics.append(mask_metrics(pred_mask, sample["gt_mask"]))
            rows.append(
                {
                    "model": model,
                    "label": MODEL_LABELS.get(model, model),
                    "family": MODEL_FAMILIES.get(model, "external_prediction"),
                    "protocol": "external_mask_import_fast_v1",
                    "dataset": external_id,
                    "sample_count": len(sample_metrics),
                    "run_count": 1,
                    "seed_count": "",
                    "missing_predictions": missing,
                    **aggregate_metric_rows(sample_metrics),
                    "notes": "WholeMaskIoU25/50 are proxy rates; benchmark TargetRecall requires full evaluator rows.",
                }
            )
    return rows


def mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred_b = pred_mask > 0.5
    gt_b = gt_mask > 0.5
    inter = float(np.logical_and(pred_b, gt_b).sum())
    pred_sum = float(pred_b.sum())
    gt_sum = float(gt_b.sum())
    union = pred_sum + gt_sum - inter
    miou = 1.0 if union <= 0.0 else inter / union
    dice = 1.0 if pred_sum + gt_sum <= 0.0 else (2.0 * inter) / (pred_sum + gt_sum)
    false_alarm = float(np.logical_and(pred_b, np.logical_not(gt_b)).sum()) / max(float(pred_b.size) / 1_000_000.0, 1e-6)
    return {
        "mIoU": miou,
        "Dice": dice,
        "WholeMaskIoU25Rate": 1.0 if miou >= 0.25 else 0.0,
        "WholeMaskIoU50Rate": 1.0 if miou >= 0.50 else 0.0,
        "FalseAlarmPixelsPerMP": false_alarm,
        "PredAreaPixels": pred_sum,
        "GTAreaPixels": gt_sum,
    }


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for metric in METRIC_COLUMNS:
        values = [row[metric] for row in rows if metric in row]
        payload[metric] = mean(values) if values else None
        payload[f"{metric}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else None)
    return payload


def parse_ours_analysis_args(values: list[str]) -> dict[str, Path]:
    if values:
        output: dict[str, Path] = {}
        for value in values:
            if "=" not in value:
                raise ValueError(f"--ours-analysis must use LABEL=CSV, got {value!r}")
            label, raw_path = value.split("=", 1)
            output[label.strip()] = Path(raw_path).expanduser().resolve()
        return output
    return {label: path for label, path in DEFAULT_OURS_ANALYSES.items() if path.exists()}


def collect_ours_analysis_rows(analysis_inputs: dict[str, Path], dataset_keys: list[str]) -> list[dict[str, Any]]:
    dataset_set = set(dataset_keys)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for label, path in analysis_inputs.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = str(row.get("dataset", ""))
                if dataset not in dataset_set:
                    continue
                method = str(row.get("method", ""))
                groups[(label, method, dataset)].append(row)

    output: list[dict[str, Any]] = []
    for (label, method, dataset), rows in sorted(groups.items()):
        payload: dict[str, Any] = {
            "model": f"{label}:{method}",
            "label": f"{label} {method}",
            "family": "ours_sam2_ir_qd",
            "protocol": "benchmark_analysis_summary",
            "dataset": PUBLIC_DATASETS[dataset]["external_id"],
            "sample_count": _max_int(rows, "sample_count"),
            "run_count": len(rows),
            "seed_count": _seed_count(rows),
            "missing_predictions": "",
            "notes": "Read from benchmark analysis CSV; TargetRecall metrics are benchmark target-level metrics.",
        }
        for metric, column in OURS_METRIC_MAP.items():
            values = [_as_float(row.get(column)) for row in rows]
            valid = [value for value in values if value is not None]
            payload[metric] = mean(valid) if valid else None
            payload[f"{metric}_std"] = stdev(valid) if len(valid) > 1 else (0.0 if valid else None)
        output.append(payload)
    return output


def _max_int(rows: list[dict[str, str]], key: str) -> int:
    values = []
    for row in rows:
        value = _as_float(row.get(key))
        if value is not None:
            values.append(int(value))
    return max(values) if values else 0


def _seed_count(rows: list[dict[str, str]]) -> int:
    seeds = set()
    for row in rows:
        seed = row.get("TrainSeed_mean") or row.get("PromptTrainSeed_mean")
        if seed not in {None, ""}:
            seeds.add(str(seed))
    return len(seeds) if seeds else len(rows)


def macro_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, group_rows in grouped.items():
        payload: dict[str, Any] = {
            "model": model,
            "label": group_rows[0].get("label", model),
            "family": group_rows[0].get("family", ""),
            "protocol": group_rows[0].get("protocol", ""),
            "dataset": "public_ir3_macro",
            "dataset_count": len(group_rows),
            "sample_count": sum(int(row.get("sample_count") or 0) for row in group_rows),
            "run_count": sum(int(row.get("run_count") or 0) for row in group_rows),
            "seed_count": _max_maybe_numeric(group_rows, "seed_count"),
            "missing_predictions": _sum_maybe_numeric(group_rows, "missing_predictions"),
            "notes": group_rows[0].get("notes", ""),
        }
        metric_names = sorted(
            {
                key
                for row in group_rows
                for key in row
                if key
                not in {
                    "model",
                    "label",
                    "family",
                    "protocol",
                    "dataset",
                    "sample_count",
                    "run_count",
                    "seed_count",
                    "missing_predictions",
                    "notes",
                }
                and not key.endswith("_std")
            }
        )
        for metric in metric_names:
            values = [_as_float(row.get(metric)) for row in group_rows]
            valid = [value for value in values if value is not None]
            payload[metric] = mean(valid) if valid else None
            payload[f"{metric}_std"] = stdev(valid) if len(valid) > 1 else (0.0 if valid else None)
        output.append(payload)
    return sorted(output, key=lambda row: _as_float(row.get("mIoU")) or -1.0, reverse=True)


def _sum_maybe_numeric(rows: list[dict[str, Any]], key: str) -> Any:
    values = [_as_float(row.get(key)) for row in rows]
    valid = [value for value in values if value is not None]
    if len(valid) != len(values):
        return ""
    return int(sum(valid))


def _max_maybe_numeric(rows: list[dict[str, Any]], key: str) -> Any:
    values = [_as_float(row.get(key)) for row in rows]
    valid = [value for value in values if value is not None]
    if not valid:
        return ""
    return int(max(valid))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    dataset_inventory: list[dict[str, Any]],
    comparison_macro_rows: list[dict[str, Any]],
    ours_macro_rows: list[dict[str, Any]],
) -> None:
    top_rows = comparison_macro_rows[:15]
    best_ours = next((row for row in comparison_macro_rows if row.get("family") == "ours_sam2_ir_qd"), None)
    best_external = next((row for row in comparison_macro_rows if row.get("family") != "ours_sam2_ir_qd"), None)

    lines = [
        "# Comparison Evaluation Matrix",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "This matrix compares imported third-party masks with current SAM2-IR-QD analysis summaries on the public IR3 datasets: NUAA-SIRST, NUDT-SIRST, and IRSTD-1K.",
        "",
        "External rows are computed directly from saved prediction masks. Their `WholeMaskIoU25Rate` and `WholeMaskIoU50Rate` are proxy recall rates, not benchmark target-level recall. Ours rows are read from benchmark analysis CSV files and keep benchmark target-level recall fields.",
        "",
        "## Dataset Inventory",
        "",
        "| Dataset | Samples | Missing Masks | Skipped Size Mismatch |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in dataset_inventory:
        lines.append(
            f"| {row['dataset_name']} | {row['sample_count']} | {row['missing_masks']} | {row['skipped_size_mismatch']} |"
        )

    lines.extend(
        [
            "",
            "## Public IR3 Macro Ranking",
            "",
            "| Rank | Family | Model | mIoU | Dice | TR/IoU25 | TR/IoU50 | FApx/MP |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(top_rows, start=1):
        tr25 = _metric_display(row, "TargetRecallIoU25", fallback="WholeMaskIoU25Rate")
        tr50 = _metric_display(row, "TargetRecallIoU50", fallback="WholeMaskIoU50Rate")
        lines.append(
            "| "
            f"{index} | {row.get('family', '')} | {row.get('label', row.get('model', ''))} | "
            f"{_metric_display(row, 'mIoU')} | {_metric_display(row, 'Dice')} | "
            f"{tr25} | {tr50} | {_metric_display(row, 'FalseAlarmPixelsPerMP', digits=1)} |"
        )

    lines.extend(["", "## Current Decision", ""])
    if best_external and best_ours:
        external_miou = _as_float(best_external.get("mIoU")) or 0.0
        ours_miou = _as_float(best_ours.get("mIoU")) or 0.0
        lines.append(
            f"- Best external public-IR3 method: `{best_external.get('label')}` with mIoU `{external_miou:.4f}`."
        )
        lines.append(f"- Best current SAM2-IR-QD method: `{best_ours.get('label')}` with mIoU `{ours_miou:.4f}`.")
        lines.append(
            "- The current method should not be positioned as supervised IR small-target segmentation SOTA on public IR3."
        )
    elif ours_macro_rows:
        lines.append("- Ours rows were found, but no comparable external rows were available.")
    else:
        lines.append("- No current SAM2-IR-QD analysis rows were found.")
    lines.append(
        "- The stronger paper direction is SAM2 automatic prompt transfer for infrared small targets, with MultiModal-small proposal recall as the next bottleneck to attack."
    )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tables/external_public_dataset.csv`",
            "- `tables/external_public_macro.csv`",
            "- `tables/ours_public_dataset.csv`",
            "- `tables/ours_public_macro.csv`",
            "- `tables/comparison_public_macro.csv`",
            "- `manifest.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_display(row: dict[str, Any], key: str, *, fallback: str | None = None, digits: int = 4) -> str:
    value = _as_float(row.get(key))
    if value is None and fallback is not None:
        value = _as_float(row.get(fallback))
    if value is None:
        return ""
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
