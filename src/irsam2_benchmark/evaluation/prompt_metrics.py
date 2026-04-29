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
    if "candidate_score" in prompt:
        metrics["AutoPromptCandidateScore"] = float(prompt.get("candidate_score") or 0.0)
    points = prompt.get("points")
    labels = prompt.get("point_labels")
    if isinstance(points, list):
        metrics["AutoPromptNumPoints"] = float(len(points))
    if isinstance(labels, list):
        negative_count = sum(1 for label in labels if int(label) == 0)
        metrics["AutoPromptNegativePointCount"] = float(negative_count)
        if isinstance(points, list) and negative_count > 0:
            h, w = gt_mask.shape
            negative_hits = 0
            for candidate_point, label in zip(points, labels):
                if int(label) != 0 or not isinstance(candidate_point, list) or len(candidate_point) < 2:
                    continue
                x, y = int(round(float(candidate_point[0]))), int(round(float(candidate_point[1])))
                if 0 <= y < h and 0 <= x < w and gt_mask[y, x] > 0.5:
                    negative_hits += 1
            metrics["NegativePromptInGtRate"] = float(negative_hits) / float(negative_count)
    if "fallback" in prompt:
        metrics["AutoPromptFallback"] = 1.0 if bool(prompt.get("fallback")) else 0.0
    return metrics
