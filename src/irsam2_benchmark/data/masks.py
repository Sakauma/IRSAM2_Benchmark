from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .sample import Sample

MASK_SOURCE_KEY = "mask_source"


def polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
    """把 MultiModal/COCO polygon rasterize 成二值 float32 mask。"""
    canvas = Image.new("L", (width, height), 0)
    xy = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
    ImageDraw.Draw(canvas).polygon(xy, outline=1, fill=1)
    return np.array(canvas, dtype=np.float32)


def _mask_path_to_array(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)


def sample_mask_array(sample: Sample) -> np.ndarray | None:
    """按统一优先级读取样本 GT mask。

    优先级为 eager mask -> mask_path -> lazy metadata source。
    这样评估代码不需要关心某个数据集是直接保存 mask、从文件读 mask，还是按需从 polygon 解码。
    """
    if sample.mask_array is not None:
        return np.asarray(sample.mask_array, dtype=np.float32)
    if sample.mask_path is not None:
        return _mask_path_to_array(sample.mask_path)

    source = sample.metadata.get(MASK_SOURCE_KEY)
    if not isinstance(source, dict):
        return None
    if source.get("type") != "polygon":
        return None

    points = source.get("points")
    if not isinstance(points, list) or len(points) < 6:
        return None
    height = int(source.get("height", sample.height))
    width = int(source.get("width", sample.width))
    return polygon_to_mask(points, height=height, width=width)


def sample_mask_or_zeros(sample: Sample) -> np.ndarray:
    """返回可直接参与指标计算的 GT mask；无 GT mask 时返回同尺寸全零 mask。"""
    mask = sample_mask_array(sample)
    if mask is not None:
        return mask
    return np.zeros((sample.height, sample.width), dtype=np.float32)
