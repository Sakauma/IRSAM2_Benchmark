"""Multi-view dataset organization helpers.
Author: Egor Izmaylov

The same normalized ``Sample`` list needs to be consumed at different granularities:
- instance view for prompt-conditioned image baselines
- image view for no-prompt automatic mask generation
- sequence view for video propagation and temporal evaluation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .sample import Sample


def build_instance_view(samples: Iterable[Sample]) -> List[Sample]:
    """Return the plain instance-level list in its incoming order."""
    return list(samples)


def build_image_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    """Group samples into image-level buckets using the normalized image path.

    We intentionally key the image view by ``image_path`` rather than ``frame_id``.
    Some datasets store per-object identifiers in ``frame_id``, which would make a
    no-prompt run evaluate the same image multiple times. Grouping by the concrete
    image path ensures all instances from one image are evaluated together.
    """
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.image_path.as_posix()].append(sample)
    return dict(grouped)


def build_sequence_view(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    """Group samples by sequence and sort them deterministically within each one."""
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sequence_id].append(sample)
    for sequence_id, items in grouped.items():
        # Freeze temporal order so sequence metrics do not depend on filesystem
        # enumeration order or adapter-specific insertion order.
        items.sort(key=lambda item: (item.frame_index, item.temporal_key, item.sample_id))
    return dict(grouped)
