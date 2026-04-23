from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Sample:
    image_path: Path
    sample_id: str
    frame_id: str
    sequence_id: str
    frame_index: int
    temporal_key: str
    width: int
    height: int
    category: str
    target_scale: str
    device_source: str
    annotation_protocol_flag: str
    supervision_type: str
    bbox_tight: Optional[list[float]] = None
    bbox_loose: Optional[list[float]] = None
    point_prompt: Optional[list[float]] = None
    mask_array: Optional[np.ndarray] = None
    mask_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_item_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "frame_id": self.frame_id,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "temporal_key": self.temporal_key,
            "width": self.width,
            "height": self.height,
            "category": self.category,
            "target_scale": self.target_scale,
            "device_source": self.device_source,
            "annotation_protocol_flag": self.annotation_protocol_flag,
            "supervision_type": self.supervision_type,
            "bbox_tight": self.bbox_tight,
            "bbox_loose": self.bbox_loose,
            "point_prompt": self.point_prompt,
            "metadata": self.metadata,
        }

    def has_mask(self) -> bool:
        return self.mask_array is not None or self.mask_path is not None
