from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

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
