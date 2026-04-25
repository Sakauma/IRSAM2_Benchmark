from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.prompt_synthesis import mask_to_tight_box
from ..data.sample import Sample
from ..data.views import build_image_view, build_track_view
from .image_metrics import bbox_iou, boundary_f1, boundary_f1_tolerance, dice_score, mask_iou
from .instance_metrics import greedy_match_instances
from .prompt_metrics import prompt_metrics
from .small_target_metrics import small_target_metrics
from .temporal_metrics import compute_temporal_metrics


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    aggregate: Dict[str, Any] = {}
    for key, values in numeric.items():
        aggregate[key] = float(sum(values) / len(values)) if values else 0.0
    return aggregate


def _summarize_values(values: Iterable[str]) -> str:
    unique = sorted({value for value in values if value})
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "multiple"


def _mask_box(mask: np.ndarray) -> list[float] | None:
    if float((mask > 0.5).sum()) <= 0.0:
        return None
    return mask_to_tight_box(mask)


def _image_level_row(items: List[Sample], elapsed_ms: float, instance_metrics: Dict[str, float], modality: str) -> Dict[str, Any]:
    representative = items[0]
    row = {
        "sample_id": representative.frame_id,
        "frame_id": representative.frame_id,
        "sequence_id": representative.sequence_id,
        "eval_unit": "image",
        "modality": modality,
        "supervision_type": _summarize_values(item.supervision_type for item in items),
        "category_name": _summarize_values(item.category for item in items),
        "target_scale": _summarize_values(item.target_scale for item in items),
        "device_source": _summarize_values(item.device_source for item in items),
        "annotation_protocol_flag": _summarize_values(item.annotation_protocol_flag for item in items),
        "LatencyMs": elapsed_ms,
        **instance_metrics,
    }
    track_value = _summarize_values(item.track_id for item in items if item.track_id is not None)
    if track_value != "unknown":
        row["track_id"] = track_value
    return row


def evaluate_method(
    *,
    method,
    samples: List[Sample],
    config: AppConfig,
    track_name: str,
    inference_mode: InferenceMode,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    modality = getattr(getattr(config, "dataset", None), "modality", "ir")

    rows: List[Dict[str, Any]] = []
    if inference_mode == InferenceMode.VIDEO_PROPAGATION:
        track_view = build_track_view(samples)
        sequence_metrics = []
        for (sequence_id, track_id), items in track_view.items():
            frame_ids = {item.frame_id for item in items}
            if len(frame_ids) != len(items):
                raise RuntimeError(
                    f"Video propagation expects one sample per frame within a track, got duplicates for sequence_id={sequence_id!r}, track_id={track_id!r}."
                )
            start = time.perf_counter()
            predictions = method.predict_sequence(items)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(items))
            sequence_rows = []
            for item in items:
                pred_mask = np.asarray(predictions[item.sample_id], dtype=np.float32)
                gt_mask = np.asarray(item.mask_array, dtype=np.float32) if item.mask_array is not None else np.zeros_like(pred_mask)
                row = build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms, modality=modality)
                row["track"] = track_name
                row["track_id"] = track_id
                sequence_rows.append({**row, "pred_mask": pred_mask})
                rows.append(row)
            temporal = compute_temporal_metrics(sequence_rows)
            temporal["sequence_id"] = sequence_id
            temporal["track_id"] = track_id
            sequence_metrics.append(temporal)
        summary = _aggregate(rows)
        if sequence_metrics:
            summary.update(_aggregate(sequence_metrics))
        return summary, rows

    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        image_view = build_image_view(samples)
        for _, items in image_view.items():
            representative = items[0]
            start = time.perf_counter()
            pred = method.predict_sample(representative)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            gt_instances = [{"mask": item.mask_array} for item in items if item.mask_array is not None]
            pred_instances = pred.get("instances", [])
            if gt_instances:
                instance_metrics = greedy_match_instances(pred_instances, gt_instances)
            else:
                instance_metrics = {"num_pred_instances": float(len(pred_instances)), "num_gt_instances": 0.0}
            rows.append(_image_level_row(items, elapsed_ms, instance_metrics, modality=modality))
        return _aggregate(rows), rows

    for item in samples:
        start = time.perf_counter()
        pred = method.predict_sample(item)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        pred_mask = np.asarray(pred["mask"], dtype=np.float32)
        gt_mask = np.asarray(item.mask_array, dtype=np.float32) if item.mask_array is not None else np.zeros_like(pred_mask)
        rows.append(build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms, modality=modality, prompt=pred.get("prompt")))
    return _aggregate(rows), rows


def build_segmentation_row(
    item: Sample,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    elapsed_ms: float,
    modality: str = "ir",
    prompt: Dict[str, object] | None = None,
) -> Dict[str, Any]:
    pred_area = float((pred_mask > 0.5).sum())
    gt_area = float((gt_mask > 0.5).sum())
    pred_box = _mask_box(pred_mask)
    gt_box = item.bbox_tight
    height = max(1, item.height)
    width = max(1, item.width)
    has_mask_gt = item.mask_array is not None and item.supervision_type == "mask"
    prompt_metadata: Dict[str, Any] = {}
    if prompt:
        prompt_metadata = {
            "PromptProtocol": prompt.get("protocol", "unknown"),
            "PromptSource": prompt.get("source", "unknown"),
            "PromptBoxVariant": prompt.get("box_variant", "none"),
            "PromptTightBoxRule": prompt.get("tight_box_rule", "unknown"),
            "PromptLooseBoxRule": prompt.get("loose_box_rule", "unknown"),
            "PromptLooseBoxPadRatio": prompt.get("loose_box_pad_ratio"),
            "PromptLooseBoxMinPad": prompt.get("loose_box_min_pad"),
            "PromptLooseBoxMinSide": prompt.get("loose_box_min_side"),
            "PromptLooseBoxMaxSideMultiplier": prompt.get("loose_box_max_side_multiplier"),
            "PromptPointRule": prompt.get("point_rule", "unknown"),
        }
    elif item.metadata.get("prompt_generation"):
        prompt_generation = item.metadata["prompt_generation"]
        prompt_metadata = {
            "PromptProtocol": prompt_generation.get("protocol", "unknown"),
            "PromptSource": prompt_generation.get("source", "unknown"),
            "PromptBoxVariant": "unused",
            "PromptTightBoxRule": prompt_generation.get("tight_box_rule", "unknown"),
            "PromptLooseBoxRule": prompt_generation.get("loose_box_rule", "unknown"),
            "PromptLooseBoxPadRatio": prompt_generation.get("loose_box_pad_ratio"),
            "PromptLooseBoxMinPad": prompt_generation.get("loose_box_min_pad"),
            "PromptLooseBoxMinSide": prompt_generation.get("loose_box_min_side"),
            "PromptLooseBoxMaxSideMultiplier": prompt_generation.get("loose_box_max_side_multiplier"),
            "PromptPointRule": prompt_generation.get("point_rule", "unknown"),
        }
    row = {
        "sample_id": item.sample_id,
        "frame_id": item.frame_id,
        "sequence_id": item.sequence_id,
        "eval_unit": "instance",
        "modality": modality,
        "supervision_type": item.supervision_type,
        "category_name": item.category,
        "target_scale": item.target_scale,
        "device_source": item.device_source,
        "annotation_protocol_flag": item.annotation_protocol_flag,
        "LatencyMs": elapsed_ms,
        "BBoxIoU": bbox_iou(pred_box, gt_box),
        "PredAreaRatio": pred_area / float(height * width),
        "PredAreaPixels": pred_area,
        **prompt_metadata,
    }
    if has_mask_gt:
        exact_boundary_f1 = boundary_f1(pred_mask, gt_mask)
        row.update(
            {
                "mIoU": mask_iou(pred_mask, gt_mask),
                "Dice": dice_score(pred_mask, gt_mask),
                "BoundaryF1": exact_boundary_f1,
                "BoundaryF1Exact": exact_boundary_f1,
                "BoundaryF1Tol1": boundary_f1_tolerance(pred_mask, gt_mask, radius=1),
                "TightBoxMaskIoU": mask_iou(box_to_area(item.bbox_tight, item.height, item.width), gt_mask) if item.bbox_tight else 0.0,
                "LooseBoxMaskIoU": mask_iou(box_to_area(item.bbox_loose, item.height, item.width), gt_mask) if item.bbox_loose else 0.0,
                "GTAreaRatio": gt_area / float(height * width),
                **small_target_metrics(pred_mask, gt_mask),
                **prompt_metrics(prompt, gt_mask),
            }
        )
    elif prompt:
        prompt_box = prompt.get("box")
        prompt_point = prompt.get("point")
        if isinstance(prompt_box, list):
            row["PromptBoxBBoxIoU"] = bbox_iou([float(value) for value in prompt_box[:4]], item.bbox_tight)
        if isinstance(prompt_point, list) and len(prompt_point) >= 2 and item.bbox_tight is not None:
            x, y = float(prompt_point[0]), float(prompt_point[1])
            x1, y1, x2, y2 = item.bbox_tight
            row["PromptPointInBBox"] = 1.0 if x1 <= x <= x2 and y1 <= y <= y2 else 0.0
    if item.track_id is not None:
        row["track_id"] = item.track_id
    return row


def box_to_area(box: list[float] | None, height: int, width: int) -> np.ndarray:
    if box is None:
        return np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1.0
    return mask
