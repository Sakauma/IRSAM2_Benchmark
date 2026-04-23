from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

from .sample import Sample


def _group_key(sample: Sample) -> str:
    return f"{sample.sequence_id}::{sample.device_source}"


def deterministic_group_split(
    samples: Sequence[Sample],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[_group_key(sample)].append(sample)

    ordered_keys = sorted(grouped)
    total = len(ordered_keys)
    test_count = max(1, int(round(total * test_ratio))) if total >= 3 else min(1, total)
    val_count = max(1, int(round(total * val_ratio))) if total - test_count >= 2 else min(1, max(0, total - test_count))

    test_keys = set(ordered_keys[-test_count:])
    val_keys = set(ordered_keys[max(0, len(ordered_keys) - test_count - val_count) : len(ordered_keys) - test_count])
    train_keys = [key for key in ordered_keys if key not in test_keys and key not in val_keys]

    train = [sample for key in train_keys for sample in grouped[key]]
    val = [sample for key in ordered_keys if key in val_keys for sample in grouped[key]]
    test = [sample for key in ordered_keys if key in test_keys for sample in grouped[key]]
    return train, val, test
