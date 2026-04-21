from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List

import cv2
import numpy as np
import torch


def compute_miou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    inter = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    inter = np.logical_and(pred_bin, target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()
    return float((2.0 * inter) / total) if total > 0 else 1.0


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pred_u8 = (pred > 0.5).astype(np.uint8)
    target_u8 = (target > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.morphologyEx(pred_u8, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target_u8, cv2.MORPH_GRADIENT, kernel)
    tp = np.logical_and(pred_boundary, target_boundary).sum()
    fp = np.logical_and(pred_boundary, 1 - target_boundary).sum()
    fn = np.logical_and(1 - pred_boundary, target_boundary).sum()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return float(2 * precision * recall / max(1e-6, precision + recall))


def compute_latency_ms(callable_fn: Callable[[], np.ndarray]):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def primary_metric_from_metrics(metrics: Dict[str, float]) -> float:
    return float((1.0 - metrics["mIoU"]) + 0.001 * metrics["LatencyMs"])


def safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(np.mean(values_list))


def infer_target_scale(box, image_height: int, image_width: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in box]
    image_area = max(1.0, float(image_height * image_width))
    box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
    if box_area_ratio < 0.01:
        return "small"
    if box_area_ratio < 0.05:
        return "medium"
    return "large"


def summarize_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "sample_count": int(len(rows)),
        "mIoU": safe_mean(row["mIoU"] for row in rows),
        "Dice": safe_mean(row["Dice"] for row in rows),
        "BoundaryF1": safe_mean(row["BoundaryF1"] for row in rows),
        "LatencyMs": safe_mean(row["LatencyMs"] for row in rows),
        "BBoxIoU": safe_mean(row["BBoxIoU"] for row in rows),
        "TightBoxMaskIoU": safe_mean(float(row.get("TightBoxMaskIoU", 0.0)) for row in rows),
        "LooseBoxMaskIoU": safe_mean(float(row.get("LooseBoxMaskIoU", 0.0)) for row in rows),
        "PredAreaRatio": safe_mean(row["PredAreaRatio"] for row in rows),
        "GTAreaRatio": safe_mean(row["GTAreaRatio"] for row in rows),
    }
