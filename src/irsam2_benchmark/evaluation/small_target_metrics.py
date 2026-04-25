from __future__ import annotations

from typing import Dict

import numpy as np

from ..data.prompt_synthesis import connected_components
from .image_metrics import mask_iou


def _components(mask: np.ndarray) -> list[np.ndarray]:
    return connected_components((mask > 0.5).astype(np.float32))


def _target_recall(pred_components: list[np.ndarray], gt_components: list[np.ndarray], threshold: float) -> float:
    if not gt_components:
        return 1.0 if not pred_components else 0.0
    recalled = 0
    for gt_component in gt_components:
        best_iou = 0.0
        for pred_component in pred_components:
            best_iou = max(best_iou, mask_iou(pred_component, gt_component))
        if best_iou >= threshold:
            recalled += 1
    return float(recalled) / float(len(gt_components))


def small_target_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred_b = (pred_mask > 0.5).astype(np.float32)
    gt_b = (gt_mask > 0.5).astype(np.float32)
    pred_components = _components(pred_b)
    gt_components = _components(gt_b)
    false_alarm = np.logical_and(pred_b > 0.5, gt_b <= 0.5)
    megapixels = max(float(pred_b.size) / 1_000_000.0, 1e-6)
    false_alarm_components = 0
    for component in pred_components:
        if float((component * gt_b).sum()) <= 0.0:
            false_alarm_components += 1
    return {
        "TargetRecallIoU10": _target_recall(pred_components, gt_components, 0.10),
        "TargetRecallIoU25": _target_recall(pred_components, gt_components, 0.25),
        "TargetRecallIoU50": _target_recall(pred_components, gt_components, 0.50),
        "FalseAlarmPixelsPerMP": float(false_alarm.sum()) / megapixels,
        "FalseAlarmComponents": float(false_alarm_components),
        "GTAreaPixels": float(gt_b.sum()),
        "PredAreaPixels": float(pred_b.sum()),
    }

