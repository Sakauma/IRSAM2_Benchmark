from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .sample import Sample


def build_instance_view(samples: Iterable[Sample]) -> List[Sample]:
    # prompted segmentation 的默认 eval unit：一个 Sample 就是一个目标实例。
    return list(samples)


def build_image_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    # no-prompt auto-mask 需要按图片聚合多个 GT instance，再和预测 instance 做匹配。
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.frame_id].append(sample)
    return dict(grouped)


def build_sequence_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    # 时序任务按 sequence 排序；frame_index/temporal_key 保证传播顺序稳定。
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sequence_id].append(sample)
    for sequence_id, items in grouped.items():
        items.sort(key=lambda item: (item.frame_index, item.temporal_key, item.sample_id))
    return dict(grouped)


def build_track_view(samples: Iterable[Sample]) -> Dict[Tuple[str, str], List[Sample]]:
    # video propagation 按 sequence+track 分组，要求每条 track 在每帧最多对应一个目标。
    grouped: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.track_id is None:
            raise RuntimeError(
                f"Track-aware evaluation requires track_id for sample_id={sample.sample_id!r} in sequence_id={sample.sequence_id!r}."
            )
        grouped[(sample.sequence_id, sample.track_id)].append(sample)
    for key, items in grouped.items():
        items.sort(key=lambda item: (item.frame_index, item.temporal_key, item.sample_id))
    return dict(grouped)
