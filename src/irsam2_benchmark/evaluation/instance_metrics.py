from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .image_metrics import mask_iou


def greedy_match_instances(pred_instances: List[Dict[str, object]], gt_instances: List[Dict[str, object]], iou_threshold: float = 0.5) -> Dict[str, float]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_instances):
        pred_mask = np.asarray(pred["mask"], dtype=np.float32)
        for gt_idx, gt in enumerate(gt_instances):
            gt_mask = np.asarray(gt["mask"], dtype=np.float32)
            iou = mask_iou(pred_mask, gt_mask)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_iou: list[float] = []
    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matched_iou.append(iou)
    pred_count = len(pred_instances)
    gt_count = len(gt_instances)
    match_count = len(matched_iou)
    precision = match_count / pred_count if pred_count else 1.0
    recall = match_count / gt_count if gt_count else 1.0
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "num_pred_instances": float(pred_count),
        "num_matched_instances": float(match_count),
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "instance_f1": float(f1),
        "matched_instance_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
    }
