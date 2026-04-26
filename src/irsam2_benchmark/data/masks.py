from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .sample import Sample

MASK_SOURCE_KEY = "mask_source"


def polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
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
    mask = sample_mask_array(sample)
    if mask is not None:
        return mask
    return np.zeros((sample.height, sample.width), dtype=np.float32)
