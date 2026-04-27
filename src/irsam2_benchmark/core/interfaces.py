from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class InferenceMode(str, Enum):
    PROMPTED = "prompted"
    BOX = "box"
    POINT = "point"
    BOX_POINT = "box+point"
    SINGLE_MASK = "single_mask"
    MULTI_MASK = "multi_mask"
    NO_PROMPT_AUTO_MASK = "no_prompt_auto_mask"
    VIDEO_PROPAGATION = "video_propagation"
    ADAPTED_TEACHER = "adapted_teacher"
    DISTILLED_STUDENT = "distilled_student"
    QUANTIZED_STUDENT = "quantized_student"


class PipelineStage(str, Enum):
    TRANSFER = "transfer"
    ADAPT = "adapt"
    DISTILL = "distill"
    QUANTIZE = "quantize"
    EVALUATE = "evaluate"
    PIPELINE = "pipeline"


class Track(str, Enum):
    IMAGE_PROMPTED = "track_a_image_prompted"
    AUTO_MASK = "track_b_auto_mask"
    VIDEO_PROPAGATION = "track_c_video_propagation"
    FULL_PIPELINE = "track_d_full_pipeline"


class PromptType(str, Enum):
    BOX = "box"
    POINT = "point"
    BOX_POINT = "box+point"
    NONE = "none"


class PromptSource(str, Enum):
    GT = "gt"
    SYNTHESIZED = "synthesized"
    NONE = "none"


@dataclass(frozen=True)
class PromptPolicy:
    name: str
    prompt_type: PromptType
    prompt_source: PromptSource
    prompt_budget: int
    refresh_interval: Optional[int] = None
    multi_mask: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCapabilities:
    supports_box_prompt: bool = True
    supports_point_prompt: bool = True
    supports_box_point_prompt: bool = True
    supports_single_mask: bool = True
    supports_multi_mask: bool = True
    supports_auto_mask: bool = False
    supports_video_propagation: bool = False
    supports_transfer: bool = True
    supports_adapt: bool = False
    supports_distill_teacher: bool = False
    supports_quant_export: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    stage: str
    artifact_dir: str
    artifact_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "artifact_dir": self.artifact_dir,
            "artifact_name": self.artifact_name,
            "metadata": self.metadata,
        }
