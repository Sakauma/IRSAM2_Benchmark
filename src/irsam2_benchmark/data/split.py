"""数据切分策略。

Author: Egor Izmaylov

这里默认使用 group-aware deterministic split，而不是随机图像切分。
这样可以避免同一序列/设备的近邻帧泄漏到不同集合中。
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

from .sample import Sample


def _group_key(sample: Sample) -> str:
    """定义切分组键。

    当前默认把 sequence_id 和 device_source 绑定成一个组，
    这样同一序列、同一来源的数据不会被拆散。
    """
    return f"{sample.sequence_id}::{sample.device_source}"


def deterministic_group_split(
    samples: Sequence[Sample],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """按组做确定性 train/val/test 切分。"""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[_group_key(sample)].append(sample)

    ordered_keys = sorted(grouped)
    total = len(ordered_keys)
    test_count = max(1, int(round(total * test_ratio))) if total >= 3 else min(1, total)
    val_count = max(1, int(round(total * val_ratio))) if total - test_count >= 2 else min(1, max(0, total - test_count))

    # 通过固定排序后的尾部切分，保证在相同数据清单下结果恒定可复现。
    test_keys = set(ordered_keys[-test_count:])
    val_keys = set(ordered_keys[max(0, len(ordered_keys) - test_count - val_count) : len(ordered_keys) - test_count])
    train_keys = [key for key in ordered_keys if key not in test_keys and key not in val_keys]

    train = [sample for key in train_keys for sample in grouped[key]]
    val = [sample for key in ordered_keys if key in val_keys for sample in grouped[key]]
    test = [sample for key in ordered_keys if key in test_keys for sample in grouped[key]]
    return train, val, test
