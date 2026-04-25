from __future__ import annotations

from typing import Dict

import numpy as np


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _box_coverage(box: list[float] | None, gt_mask: np.ndarray) -> float:
    if box is None:
        return 0.0
    gt_area = float((gt_mask > 0.5).sum())
    if gt_area <= 0.0:
        return 0.0
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    h, w = gt_mask.shape
    crop = gt_mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    return float((crop > 0.5).sum()) / gt_area


def prompt_metrics(prompt: Dict[str, object] | None, gt_mask: np.ndarray) -> Dict[str, float]:
    if not prompt:
        return {}
    point = prompt.get("point")
    box = prompt.get("box")
    centroid = _centroid(gt_mask)
    metrics: Dict[str, float] = {}
    if isinstance(point, list) and len(point) >= 2:
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        h, w = gt_mask.shape
        hit = 0.0
        if 0 <= y < h and 0 <= x < w and gt_mask[y, x] > 0.5:
            hit = 1.0
        metrics["PromptHitRate"] = hit
        if centroid is not None:
            cx, cy = centroid
            metrics["PromptDistanceToCentroid"] = float(((float(point[0]) - cx) ** 2 + (float(point[1]) - cy) ** 2) ** 0.5)
    if isinstance(box, list):
        metrics["PromptBoxCoverage"] = _box_coverage(box, gt_mask)
    return metrics

