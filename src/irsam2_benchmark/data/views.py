from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .sample import Sample


def build_instance_view(samples: Iterable[Sample]) -> List[Sample]:
    return list(samples)


def build_image_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.frame_id].append(sample)
    return dict(grouped)


def build_sequence_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sequence_id].append(sample)
    for sequence_id, items in grouped.items():
        items.sort(key=lambda item: (item.frame_index, item.temporal_key, item.sample_id))
    return dict(grouped)
