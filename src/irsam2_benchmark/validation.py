from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .config import AppConfig
from .core.interfaces import InferenceMode
from .data import build_dataset_adapter
from .data.masks import sample_mask_array
from .data.sample import Sample

REQUIRED_ARTIFACT_FILES = (
    "benchmark_spec.json",
    "run_metadata.json",
    "summary.json",
    "results.json",
    "eval_reports/rows.json",
)

REQUIRED_HEALTH_FIELDS = (
    "expected_sample_count",
    "expected_eval_units",
    "expected_row_count",
    "row_count",
    "error_count",
    "missing_row_count",
    "failure_rate",
    "failure_rate_threshold",
)

REQUIRED_ROW_FIELDS = ("sample_id", "frame_id", "sequence_id", "eval_unit")


def _counter_dict(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _area_summary(areas: List[float]) -> Dict[str, Any]:
    if not areas:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(areas),
        "min": float(min(areas)),
        "max": float(max(areas)),
        "mean": float(sum(areas) / len(areas)),
    }


def _sample_mask_area(sample: Sample) -> float | None:
    mask = sample_mask_array(sample)
    if mask is None:
        return None
    return float(np.asarray(mask).astype(bool).sum())


def _add_warning_summary(report: Dict[str, Any], warning_messages: List[str]) -> Dict[str, Any]:
    report["warning_count"] = len(warning_messages)
    report["size_mismatch_warning_count"] = len([item for item in warning_messages if "image/mask size mismatch" in item])
    report["warning_examples"] = warning_messages[:5]
    return report


def _multimodal_polygon_anomalies(root: Path) -> Dict[str, Any]:
    label_dir = root / "label"
    report = {
        "total_instances": 0,
        "invalid_polygon_instances": 0,
        "multi_polygon_instances": 0,
        "json_error_count": 0,
    }
    if not label_dir.exists():
        return report
    for label_path in sorted(label_dir.glob("*.json")):
        try:
            data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except Exception:
            report["json_error_count"] += 1
            continue
        instances = data.get("detection", {}).get("instances", [])
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            report["total_instances"] += 1
            masks = instance.get("mask", [])
            polygons = masks if isinstance(masks, list) else []
            valid_polygons = [
                polygon
                for polygon in polygons
                if isinstance(polygon, list) and len(polygon) >= 6
            ]
            if not valid_polygons:
                report["invalid_polygon_instances"] += 1
            if len(valid_polygons) > 1:
                report["multi_polygon_instances"] += 1
    return report


def preflight_dataset(config: AppConfig) -> Dict[str, Any]:
    errors: List[str] = []
    warning_messages: List[str] = []
    root = config.dataset_root
    report: Dict[str, Any] = {
        "valid": False,
        "errors": errors,
        "warnings": warning_messages,
        "dataset_id": config.dataset.dataset_id,
        "adapter_name": "",
        "root": str(root),
        "sample_count": 0,
        "image_count": 0,
        "sequence_count": 0,
        "category_count": 0,
        "category_counts": {},
        "target_scale_counts": {},
        "annotation_protocol_counts": {},
        "supervision_type_counts": {},
        "empty_mask_count": 0,
        "missing_mask_count": 0,
        "missing_bbox_count": 0,
        "missing_point_count": 0,
        "warning_count": 0,
        "size_mismatch_warning_count": 0,
        "warning_examples": [],
        "area_pixels": _area_summary([]),
        "multimodal_polygon_anomalies": _multimodal_polygon_anomalies(root),
    }

    if not root.exists():
        errors.append(f"Dataset root does not exist: {root}")
        return _add_warning_summary(report, warning_messages)

    try:
        adapter = build_dataset_adapter(config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = adapter.load(config)
        warning_messages.extend(str(item.message) for item in caught)
    except Exception as exc:
        errors.append(f"Dataset load failed: {exc}")
        return _add_warning_summary(report, warning_messages)

    samples = loaded.samples
    areas: List[float] = []
    missing_mask_count = 0
    empty_mask_count = 0
    for sample in samples:
        area = _sample_mask_area(sample)
        if area is None:
            missing_mask_count += 1
            continue
        if area <= 0.0:
            empty_mask_count += 1
        areas.append(area)

    anomalies = report["multimodal_polygon_anomalies"]
    if anomalies["invalid_polygon_instances"] > 0:
        warning_messages.append(f"Invalid MultiModal polygon instances: {anomalies['invalid_polygon_instances']}")
    if anomalies["multi_polygon_instances"] > 0:
        warning_messages.append(f"MultiModal multi-polygon instances use the first polygon: {anomalies['multi_polygon_instances']}")
    if anomalies["json_error_count"] > 0:
        warning_messages.append(f"Unreadable MultiModal label JSON files: {anomalies['json_error_count']}")

    if not samples:
        errors.append(f"Dataset loaded zero samples for dataset_id={config.dataset.dataset_id!r}.")

    report.update(
        {
            "valid": not errors,
            "adapter_name": loaded.manifest.adapter_name,
            "root": loaded.manifest.root,
            "sample_count": loaded.manifest.sample_count,
            "image_count": loaded.manifest.image_count,
            "sequence_count": loaded.manifest.sequence_count,
            "category_count": loaded.manifest.category_count,
            "category_counts": _counter_dict(sample.category for sample in samples),
            "target_scale_counts": _counter_dict(sample.target_scale for sample in samples),
            "annotation_protocol_counts": _counter_dict(sample.annotation_protocol_flag for sample in samples),
            "supervision_type_counts": _counter_dict(sample.supervision_type for sample in samples),
            "empty_mask_count": empty_mask_count,
            "missing_mask_count": missing_mask_count,
            "missing_bbox_count": len([sample for sample in samples if not sample.bbox_tight and not sample.bbox_loose]),
            "missing_point_count": len([sample for sample in samples if not sample.point_prompt]),
            "area_pixels": _area_summary(areas),
        }
    )
    return _add_warning_summary(report, warning_messages)


def _read_json(path: Path, errors: List[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Invalid JSON in {path}: {exc}")
        return None


def _find_nonfinite_numbers(value: Any, path: str = "$") -> List[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [] if math.isfinite(float(value)) else [path]
    if isinstance(value, dict):
        paths: List[str] = []
        for key, item in value.items():
            paths.extend(_find_nonfinite_numbers(item, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_find_nonfinite_numbers(item, f"{path}[{index}]"))
        return paths
    return []


def validate_run_artifacts(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    errors: List[str] = []
    warnings_list: List[str] = []
    missing_files = [relative for relative in REQUIRED_ARTIFACT_FILES if not (run_dir / relative).exists()]
    errors.extend(f"Missing required artifact file: {relative}" for relative in missing_files)

    payloads: Dict[str, Any] = {}
    for relative in REQUIRED_ARTIFACT_FILES:
        path = run_dir / relative
        if path.exists():
            payloads[relative] = _read_json(path, errors)

    summary = payloads.get("summary.json")
    results = payloads.get("results.json")
    rows = payloads.get("eval_reports/rows.json")
    benchmark_spec = payloads.get("benchmark_spec.json")

    if not isinstance(summary, dict):
        errors.append("summary.json must contain a JSON object.")
        summary = {}
    mean_metrics = summary.get("mean", {})
    if not isinstance(mean_metrics, dict) or not mean_metrics:
        errors.append("summary.json must contain a non-empty mean metric object.")
    for field in REQUIRED_HEALTH_FIELDS:
        if field not in summary:
            errors.append(f"summary.json is missing health field: {field}")
    failure_rate = summary.get("failure_rate")
    failure_rate_threshold = summary.get("failure_rate_threshold")
    if isinstance(failure_rate, (int, float)) and isinstance(failure_rate_threshold, (int, float)):
        if float(failure_rate) > float(failure_rate_threshold):
            errors.append(
                f"summary.json failure_rate={float(failure_rate):.4f} exceeds "
                f"failure_rate_threshold={float(failure_rate_threshold):.4f}."
            )

    if not isinstance(results, list):
        errors.append("results.json must contain a JSON array.")
        results = []

    if not isinstance(rows, list):
        errors.append("eval_reports/rows.json must contain a JSON array.")
        rows = []
    if not rows:
        errors.append("eval_reports/rows.json must contain at least one row.")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {index} must be a JSON object.")
            continue
        for field in REQUIRED_ROW_FIELDS:
            if field not in row:
                errors.append(f"Row {index} is missing field: {field}")

    if isinstance(benchmark_spec, dict):
        expected_eval_unit = (
            "image"
            if benchmark_spec.get("inference_mode") == InferenceMode.NO_PROMPT_AUTO_MASK.value
            else "instance"
        )
        for index, row in enumerate(rows):
            if isinstance(row, dict) and row.get("eval_unit") != expected_eval_unit:
                errors.append(
                    f"Row {index} eval_unit={row.get('eval_unit')!r} does not match expected {expected_eval_unit!r}."
                )
    elif benchmark_spec is not None:
        errors.append("benchmark_spec.json must contain a JSON object.")

    for name, payload in (("summary.json", summary), ("results.json", results), ("eval_reports/rows.json", rows)):
        for number_path in _find_nonfinite_numbers(payload):
            errors.append(f"{name} contains a non-finite numeric value at {number_path}.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings_list,
        "run_dir": str(run_dir),
        "required_files": list(REQUIRED_ARTIFACT_FILES),
        "row_count": len(rows) if isinstance(rows, list) else 0,
        "mean_metric_count": len(mean_metrics) if isinstance(mean_metrics, dict) else 0,
    }
