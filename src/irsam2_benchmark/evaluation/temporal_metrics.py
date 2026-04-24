from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .image_metrics import boundary_f1, mask_iou


def compute_temporal_metrics(sequence_rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not sequence_rows:
        return {
            "temporal_iou_mean": 0.0,
            "temporal_boundary_f1": 0.0,
            "mask_jitter_score": 0.0,
            "propagation_decay": 0.0,
            "track_recall": 0.0,
            "track_precision": 0.0,
            "identity_switch_count": 0.0,
        }

    frame_ious = [float(row.get("mIoU", 0.0)) for row in sequence_rows]
    boundary_scores = [float(row.get("BoundaryF1", 0.0)) for row in sequence_rows]
    presence = [1.0 if float(row.get("PredAreaRatio", 0.0)) > 0.0 else 0.0 for row in sequence_rows]
    gt_presence = [1.0 if float(row.get("GTAreaRatio", 0.0)) > 0.0 else 0.0 for row in sequence_rows]

    jitter = []
    for prev_row, next_row in zip(sequence_rows[:-1], sequence_rows[1:]):
        prev_mask = np.asarray(prev_row["pred_mask"], dtype=np.float32)
        next_mask = np.asarray(next_row["pred_mask"], dtype=np.float32)
        jitter.append(1.0 - mask_iou(prev_mask, next_mask))

    first_quarter = frame_ious[: max(1, len(frame_ious) // 4)]
    last_quarter = frame_ious[-max(1, len(frame_ious) // 4) :]
    track_tp = sum(1.0 for pred, gt in zip(presence, gt_presence) if pred > 0.0 and gt > 0.0)
    track_precision = track_tp / max(1.0, sum(presence))
    track_recall = track_tp / max(1.0, sum(gt_presence))

    return {
        "temporal_iou_mean": float(np.mean(frame_ious)),
        "temporal_boundary_f1": float(np.mean(boundary_scores)),
        "mask_jitter_score": float(np.mean(jitter)) if jitter else 0.0,
        "propagation_decay": float(np.mean(first_quarter) - np.mean(last_quarter)),
        "track_recall": float(track_recall),
        "track_precision": float(track_precision),
        "identity_switch_count": 0.0,
    }
