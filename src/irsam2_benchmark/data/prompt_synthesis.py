"""由 mask 合成 prompt 的工具函数。

Author: Egor Izmaylov

这一层很关键，因为平台要求：
- mask-only 数据集无需事先转 VOC/COCO；
- 但 benchmark 仍需要稳定的 box / point prompt。

因此这里提供确定性的 prompt synthesis，而不是带随机性的在线增强。
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


def clamp_box_xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    """把 xyxy 框裁剪到图像边界内。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), max(0.0, width - 1.0))
    y1 = min(max(0.0, y1), max(0.0, height - 1.0))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def mask_to_tight_box(mask: np.ndarray) -> list[float]:
    """从 mask 生成最紧外接轴对齐框。"""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        # 空 mask 时返回最小合法框，避免后续流程崩溃。
        return [0.0, 0.0, 1.0, 1.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_box_xyxy(
    box: Iterable[float],
    width: int,
    height: int,
    pad_ratio: float = 0.15,
    min_pad: float = 2.0,
    min_side: float = 12.0,
) -> list[float]:
    """从 tight box 生成 canonical loose box。

    这里的设计目标不是几何最优，而是得到稳定、可复现、难度适中的 benchmark prompt。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(min_pad, bw * pad_ratio)
    pad_y = max(min_pad, bh * pad_ratio)
    loose = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]

    # 对小目标做额外保护，防止扩框后仍然过小导致 prompt 几乎退化成点。
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
    """从 mask 生成中心点 prompt。"""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """对 mask 做四连通域分解。

    主要用于 class-index mask，把同一类别下的多个实例拆成独立样本。
    """
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
