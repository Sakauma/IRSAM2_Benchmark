"""Evaluation runner.
Author: Egor Izmaylov

This module executes image-level, instance-set, and sequence-aware evaluation.
It now also:
- shows tqdm progress bars when tqdm is installed
- evaluates no-prompt auto-mask at image level instead of per-instance duplication
- returns qualitative visualization payloads for downstream PNG export
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.sample import Sample
from ..data.views import build_image_view, build_sequence_view
from .image_metrics import bbox_iou, boundary_f1, dice_score, mask_iou
from .instance_metrics import greedy_match_instances
from .temporal_metrics import compute_temporal_metrics

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _progress(iterable, *, desc: str):
    """Wrap iterables with tqdm when available, otherwise return them unchanged."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    aggregate: Dict[str, Any] = {}
    for key, values in numeric.items():
        aggregate[key] = float(sum(values) / len(values)) if values else 0.0
    return aggregate


def _image_level_label(items: List[Sample], attr: str) -> str:
    """Aggregate image-level categorical metadata for multi-instance rows.

    No-prompt automatic mask runs are evaluated once per image rather than once
    per instance. For grouped reports we still need stable image-level labels,
    but an image may contain several categories or target scales. In that case
    we explicitly mark the field as mixed instead of leaking the first instance
    label into the report.
    """
    values = sorted({str(getattr(item, attr, "unknown")) for item in items if getattr(item, attr, None) is not None})
    if not values:
        return "unknown"
    if len(values) == 1:
        return values[0]
    return "mixed"


def evaluate_method(
    *,
    method,
    samples: List[Sample],
    config: AppConfig,
    track_name: str,
    inference_mode: InferenceMode,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    visual_records: List[Dict[str, Any]] = []
    visual_limit = max(0, int(config.runtime.visual_limit))

    def should_capture_visual() -> bool:
        return bool(config.runtime.save_visuals) and (visual_limit == 0 or len(visual_records) < visual_limit)

    if inference_mode == InferenceMode.VIDEO_PROPAGATION:
        sequence_view = build_sequence_view(samples)
        sequence_metrics = []
        for sequence_id, items in _progress(sequence_view.items(), desc=f"{method.name} sequences"):
            start = time.perf_counter()
            predictions = method.predict_sequence(items)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(items))
            sequence_rows = []
            for item in items:
                pred_mask = np.asarray(predictions[item.sample_id], dtype=np.float32)
                gt_mask = np.asarray(item.mask_array, dtype=np.float32) if item.mask_array is not None else np.zeros_like(pred_mask)
                row = build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms)
                row["track"] = track_name
                sequence_rows.append({**row, "pred_mask": pred_mask})
                rows.append(row)
                if should_capture_visual():
                    visual_records.append({"sample": item, "pred_mask": pred_mask, "gt_mask": gt_mask})
            temporal = compute_temporal_metrics(sequence_rows)
            temporal["sequence_id"] = sequence_id
            sequence_metrics.append(temporal)
        summary = _aggregate(rows)
        if sequence_metrics:
            summary.update(_aggregate(sequence_metrics))
        return summary, rows, visual_records

    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        image_view = build_image_view(samples)
        for image_id, items in _progress(image_view.items(), desc=f"{method.name} images"):
            representative = items[0]
            start = time.perf_counter()
            pred = method.predict_sample(representative)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            gt_instances = [{"mask": item.mask_array} for item in items if item.mask_array is not None]
            instance_metrics = greedy_match_instances(pred.get("instances", []), gt_instances)
            row = {
                "sample_id": image_id,
                "frame_id": representative.image_path.as_posix(),
                "sequence_id": representative.sequence_id,
                "category_name": _image_level_label(items, "category"),
                "target_scale": _image_level_label(items, "target_scale"),
                "device_source": _image_level_label(items, "device_source"),
                "annotation_protocol_flag": _image_level_label(items, "annotation_protocol_flag"),
                "track": track_name,
                "LatencyMs": elapsed_ms,
                "num_gt_instances": len(gt_instances),
                **instance_metrics,
            }
            rows.append(row)
            if should_capture_visual():
                visual_records.append(
                    {
                        "sample": representative,
                        "pred_instances": pred.get("instances", []),
                        "gt_instances": gt_instances,
                    }
                )
        return _aggregate(rows), rows, visual_records

    for item in _progress(samples, desc=f"{method.name} samples"):
        start = time.perf_counter()
        pred = method.predict_sample(item)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        pred_mask = np.asarray(pred["mask"], dtype=np.float32)
        gt_mask = np.asarray(item.mask_array, dtype=np.float32) if item.mask_array is not None else np.zeros_like(pred_mask)
        row = build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms)
        row["track"] = track_name
        rows.append(row)
        if should_capture_visual():
            visual_records.append({"sample": item, "pred_mask": pred_mask, "gt_mask": gt_mask})
    return _aggregate(rows), rows, visual_records


def build_segmentation_row(item: Sample, pred_mask: np.ndarray, gt_mask: np.ndarray, elapsed_ms: float) -> Dict[str, Any]:
    pred_area = float((pred_mask > 0.5).sum())
    gt_area = float((gt_mask > 0.5).sum())
    pred_box = item.bbox_loose if pred_area > 0.0 else None
    gt_box = item.bbox_tight
    height = max(1, item.height)
    width = max(1, item.width)
    return {
        "sample_id": item.sample_id,
        "frame_id": item.frame_id,
        "sequence_id": item.sequence_id,
        "category_name": item.category,
        "target_scale": item.target_scale,
        "device_source": item.device_source,
        "annotation_protocol_flag": item.annotation_protocol_flag,
        "mIoU": mask_iou(pred_mask, gt_mask),
        "Dice": dice_score(pred_mask, gt_mask),
        "BoundaryF1": boundary_f1(pred_mask, gt_mask),
        "LatencyMs": elapsed_ms,
        "BBoxIoU": bbox_iou(pred_box, gt_box),
        "TightBoxMaskIoU": mask_iou(box_to_area(item.bbox_tight, item.height, item.width), gt_mask) if item.bbox_tight else 0.0,
        "LooseBoxMaskIoU": mask_iou(box_to_area(item.bbox_loose, item.height, item.width), gt_mask) if item.bbox_loose else 0.0,
        "PredAreaRatio": pred_area / float(height * width),
        "GTAreaRatio": gt_area / float(height * width),
    }


def box_to_area(box: list[float] | None, height: int, width: int) -> np.ndarray:
    if box is None:
        return np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1.0
    return mask
