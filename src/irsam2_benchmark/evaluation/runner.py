from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.sample import Sample
from ..data.views import build_sequence_view
from .image_metrics import bbox_iou, boundary_f1, dice_score, mask_iou
from .instance_metrics import greedy_match_instances
from .temporal_metrics import compute_temporal_metrics


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


def evaluate_method(
    *,
    method,
    samples: List[Sample],
    config: AppConfig,
    track_name: str,
    inference_mode: InferenceMode,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    if inference_mode == InferenceMode.VIDEO_PROPAGATION:
        sequence_view = build_sequence_view(samples)
        sequence_metrics = []
        for sequence_id, items in sequence_view.items():
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
            temporal = compute_temporal_metrics(
                sequence_rows
            )
            temporal["sequence_id"] = sequence_id
            sequence_metrics.append(temporal)
        summary = _aggregate(rows)
        if sequence_metrics:
            summary.update(_aggregate(sequence_metrics))
        return summary, rows

    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        for item in samples:
            start = time.perf_counter()
            pred = method.predict_sample(item)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            gt_instances = [{"mask": item.mask_array}] if item.mask_array is not None else []
            instance_metrics = greedy_match_instances(pred.get("instances", []), gt_instances)
            row = {
                "sample_id": item.sample_id,
                "frame_id": item.frame_id,
                "sequence_id": item.sequence_id,
                "category_name": item.category,
                "target_scale": item.target_scale,
                "device_source": item.device_source,
                "annotation_protocol_flag": item.annotation_protocol_flag,
                "LatencyMs": elapsed_ms,
                **instance_metrics,
            }
            rows.append(row)
        return _aggregate(rows), rows

    for item in samples:
        start = time.perf_counter()
        pred = method.predict_sample(item)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        pred_mask = np.asarray(pred["mask"], dtype=np.float32)
        gt_mask = np.asarray(item.mask_array, dtype=np.float32) if item.mask_array is not None else np.zeros_like(pred_mask)
        rows.append(build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms))
    return _aggregate(rows), rows


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
