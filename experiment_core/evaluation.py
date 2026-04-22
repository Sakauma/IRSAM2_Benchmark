"""评估逻辑。

这里负责把方法预测结果转成：
1. 聚合指标
2. 逐样本行记录
3. 代表性可视化
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .data import InfraredDataset, Sample, build_box_prior
from .methods import BaseMethod
from .metrics import (
    compute_boundary_f1,
    compute_dice,
    compute_latency_ms,
    compute_miou,
    infer_target_scale,
    summarize_metric_rows,
)


@dataclass(frozen=True)
class EvaluationOutput:
    """单次评估调用的统一返回结构。"""
    aggregate_metrics: Dict[str, float]
    metric_rows: List[Dict[str, object]]
    visual_path: Optional[str]


def save_visual(output_dir: Path, image_rgb: np.ndarray, target: np.ndarray, pred: np.ndarray, prefix: str) -> str:
    """保存一张代表性三联图：原图 / GT / 预测。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_rgb)
    axes[1].imshow(target, cmap="gray")
    axes[2].imshow(pred > 0.5, cmap="gray")
    for axis, title in zip(axes, ["image", "gt", "pred"]):
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    out_path = output_dir / f"{prefix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def evaluate_samples(
    method: BaseMethod,
    samples: List[Sample],
    output_dir: Path,
    prefix: str,
    save_artifacts: bool,
) -> EvaluationOutput:
    """逐样本执行预测并生成指标行。"""
    dataset = InfraredDataset(samples, require_mask=True)
    metric_rows: List[Dict[str, object]] = []
    visual_path: Optional[str] = None
    for item in dataset:
        # latency 直接包在预测调用外层，保证和最终输出保持一致。
        pred, latency = compute_latency_ms(lambda: method.predict(item))
        target = item["mask"]
        image_height, image_width = target.shape
        bbox = item["bbox"]
        box_mask = build_box_prior(bbox, image_height, image_width)
        tight_box = item.get("bbox_tight")
        loose_box = item.get("bbox_loose")
        tight_box_mask = None if tight_box is None else build_box_prior(tight_box, image_height, image_width)
        loose_box_mask = None if loose_box is None else build_box_prior(loose_box, image_height, image_width)
        gt_area_ratio = float((target > 0.5).sum() / max(1, image_height * image_width))
        pred_area_ratio = float((pred > 0.5).sum() / max(1, image_height * image_width))
        bbox_area_ratio = float(box_mask.sum() / max(1, image_height * image_width))
        metric_rows.append(
            {
                # 这些字段会直接写入 eval report，便于做分组分析和问题排查。
                "sample_id": item["sample_id"],
                "category_name": item["category_name"],
                "device_source": item["device_source"],
                "annotation_protocol_flag": item["annotation_protocol_flag"],
                "target_scale": infer_target_scale(bbox, image_height, image_width),
                "bbox": [float(v) for v in bbox.tolist()],
                "bbox_tight": None if tight_box is None else [float(v) for v in tight_box.tolist()],
                "bbox_loose": None if loose_box is None else [float(v) for v in loose_box.tolist()],
                "mIoU": compute_miou(pred, target),
                "Dice": compute_dice(pred, target),
                "BoundaryF1": compute_boundary_f1(pred, target),
                "LatencyMs": float(latency),
                "BBoxIoU": compute_miou(pred, box_mask),
                "TightBoxMaskIoU": 0.0 if tight_box_mask is None else compute_miou(tight_box_mask, target),
                "LooseBoxMaskIoU": 0.0 if loose_box_mask is None else compute_miou(loose_box_mask, target),
                "GTAreaRatio": gt_area_ratio,
                "PredAreaRatio": pred_area_ratio,
                "BBoxAreaRatio": bbox_area_ratio,
            }
        )
        if save_artifacts and visual_path is None:
            # 每次评估只保存第一张代表性可视化，避免输出爆炸。
            visual_path = save_visual(output_dir, item["image_rgb"], target, pred, prefix)
    aggregate_metrics = summarize_metric_rows(metric_rows)
    return EvaluationOutput(
        aggregate_metrics=aggregate_metrics,
        metric_rows=metric_rows,
        visual_path=visual_path,
    )
