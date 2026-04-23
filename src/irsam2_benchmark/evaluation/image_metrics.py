"""图像级评估指标。

Author: Egor Izmaylov
"""

from __future__ import annotations

import numpy as np


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """计算二值 mask 的 IoU。"""
    pred_b = pred > 0.5
    target_b = target > 0.5
    union = float(np.logical_or(pred_b, target_b).sum())
    if union <= 0.0:
        # 两边都为空时，按完全一致处理。
        return 1.0
    return float(np.logical_and(pred_b, target_b).sum()) / union


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """计算 Dice 系数。"""
    pred_b = pred > 0.5
    target_b = target > 0.5
    denom = float(pred_b.sum() + target_b.sum())
    if denom <= 0.0:
        return 1.0
    return 2.0 * float(np.logical_and(pred_b, target_b).sum()) / denom


def _boundary(mask: np.ndarray) -> np.ndarray:
    """提取一个近似的 4 邻域边界图。"""
    mask_b = mask > 0.5
    up = np.pad(mask_b[1:, :], ((0, 1), (0, 0)))
    down = np.pad(mask_b[:-1, :], ((1, 0), (0, 0)))
    left = np.pad(mask_b[:, 1:], ((0, 0), (0, 1)))
    right = np.pad(mask_b[:, :-1], ((0, 0), (1, 0)))
    same = up & down & left & right & mask_b
    return (mask_b ^ same).astype(np.float32)


def boundary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    """计算边界 F1。

    这是一个简化实现，但足够稳定地反映边界贴合程度。
    """
    pred_b = _boundary(pred) > 0.5
    target_b = _boundary(target) > 0.5
    tp = float(np.logical_and(pred_b, target_b).sum())
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    if pred_sum <= 0.0 and target_sum <= 0.0:
        return 1.0
    if pred_sum <= 0.0 or target_sum <= 0.0:
        return 0.0
    precision = tp / pred_sum
    recall = tp / target_sum
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    """计算两个轴对齐框的 IoU。"""
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return intersection / max(1.0, area_a + area_b - intersection)
