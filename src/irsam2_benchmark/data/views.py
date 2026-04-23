"""多视图数据组织工具。

Author: Egor Izmaylov

同一份 Sample 列表会被不同模块按不同粒度访问：
- 实例级；
- 图像级；
- 序列级。
这里统一提供这些视图转换。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .sample import Sample


def build_instance_view(samples: Iterable[Sample]) -> List[Sample]:
    """实例视图直接返回顺序列表。"""
    return list(samples)


def build_image_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    """按 frame_id 聚合成图像视图。"""
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.frame_id].append(sample)
    return dict(grouped)


def build_sequence_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    """按 sequence_id 聚合成序列视图，并在序列内稳定排序。"""
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sequence_id].append(sample)
    for sequence_id, items in grouped.items():
        # 这里固定用 frame_index -> temporal_key -> sample_id 排序，
        # 保证时序评估口径不会受原始文件枚举顺序影响。
        items.sort(key=lambda item: (item.frame_index, item.temporal_key, item.sample_id))
    return dict(grouped)
