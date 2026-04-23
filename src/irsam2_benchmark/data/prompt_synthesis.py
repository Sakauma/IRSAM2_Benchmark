from __future__ import annotations

import math
from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


def clamp_box_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), max(0.0, width - 1.0))
    y1 = min(max(0.0, y1), max(0.0, height - 1.0))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def mask_to_tight_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_box_xyxy(box: Iterable[float], width: int, height: int, pad_ratio: float = 0.15, min_pad: float = 2.0, min_side: float = 12.0) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(min_pad, bw * pad_ratio)
    pad_y = max(min_pad, bh * pad_ratio)
    loose = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    if loose[2] - loose[0] < min_side:
        extra = 0.5 * (min_side - (loose[2] - loose[0]))
        loose[0] -= extra
        loose[2] += extra
    if loose[3] - loose[1] < min_side:
        extra = 0.5 * (min_side - (loose[3] - loose[1]))
        loose[1] -= extra
        loose[3] += extra
    return clamp_box_xyxy(loose, width, height)


def mask_to_point_prompt(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[np.ndarray] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            coords: list[Tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or binary[ny, nx] == 0:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            component = np.zeros_like(mask, dtype=np.float32)
            for cy, cx in coords:
                component[cy, cx] = 1.0
            components.append(component)
    return components
