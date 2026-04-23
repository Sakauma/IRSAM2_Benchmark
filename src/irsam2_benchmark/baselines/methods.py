"""Baseline 与参考方法实现。

Author: Egor Izmaylov

这个模块负责定义 benchmark 中的第一公民方法：
- 几何 baseline；
- SAM2 zero-shot baseline；
- 无 prompt automatic mask baseline；
- 视频传播 baseline；
- 以及后续 full pipeline 会替换成真实 artifact 的 reference placeholders。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.prompt_synthesis import clamp_box_xyxy
from ..data.sample import Sample
from ..models import SAM2ModelAdapter, load_image_rgb


def box_to_mask(box: list[float], height: int, width: int) -> np.ndarray:
    """把 xyxy 框直接 rasterize 成矩形 mask。

    这是最简单也最重要的几何 baseline：它回答“仅凭提示框本身能到什么水平”。
    """
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_xyxy(box, width, height)]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class BaseMethod:
    """所有 benchmark 方法的统一最小接口。"""

    name = "base"
    family = "base"
    inference_mode = InferenceMode.ZERO_SHOT

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        """对单样本做预测。"""
        raise NotImplementedError

    def predict_sequence(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        """默认序列预测退化为逐帧调用。

        真正具备时序能力的方法会覆盖这个实现。
        """
        return {sample.sample_id: self.predict_sample(sample)["mask"] for sample in samples}


class BBoxRectMaskBaseline(BaseMethod):
    """直接把提示框当作预测结果的几何 baseline。"""

    name = "BBoxRectMaskBaseline"
    family = "baseline"
    inference_mode = InferenceMode.BOX

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        # 优先使用 loose box，因为 benchmark 里 canonical box 就是它。
        box = sample.bbox_loose or sample.bbox_tight or [0.0, 0.0, 1.0, 1.0]
        return {"mask": box_to_mask(box, sample.height, sample.width), "score": 1.0}


class ZeroShotSAM2(BaseMethod):
    """SAM2 零样本 baseline。

    这个类负责把 benchmark 的统一 Sample prompt 字段翻译成 SAM2 predictor 输入。
    """

    name = "ZeroShotSAM2"
    family = "sam2"

    def __init__(self, adapter: SAM2ModelAdapter, prompt_mode: InferenceMode):
        self.adapter = adapter
        self.inference_mode = prompt_mode

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)

        # multimask 只在明确多 mask 模式时打开；其余情况默认返回单个最佳 mask。
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        if self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = sample.bbox_loose or sample.bbox_tight
        elif self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array([sample.point_prompt], dtype=np.float32) if sample.point_prompt else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if sample.point_prompt else None
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = sample.bbox_loose or sample.bbox_tight
            kwargs["points"] = np.array([sample.point_prompt], dtype=np.float32) if sample.point_prompt else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if sample.point_prompt else None
        elif self.inference_mode in {InferenceMode.SINGLE_MASK, InferenceMode.MULTI_MASK}:
            # 当前平台对 single/multi mask 的默认初始化仍然基于 box。
            kwargs["box"] = sample.bbox_loose or sample.bbox_tight
        else:
            raise ValueError(f"Unsupported zero-shot prompt mode: {self.inference_mode.value}")

        result = self.adapter.predict_image(image_rgb, **kwargs)
        masks = result["masks"]
        scores = result["scores"]

        # 这里明确选择分数最高的候选 mask，保证 benchmark 口径稳定。
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx])}


class NoPromptAutoMaskSAM2(BaseMethod):
    """SAM2 automatic mask generation baseline。"""

    name = "NoPromptAutoMaskSAM2"
    family = "sam2"
    inference_mode = InferenceMode.NO_PROMPT_AUTO_MASK

    def __init__(self, adapter: SAM2ModelAdapter):
        self.adapter = adapter

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)
        raw_masks = self.adapter.predict_auto_masks(image_rgb)
        instances = []
        for idx, item in enumerate(raw_masks):
            segmentation = item.get("segmentation")
            if segmentation is None:
                continue
            mask = np.asarray(segmentation, dtype=np.float32)
            instances.append(
                {
                    # 预测实例 id 采用稳定命名，便于后续排查实例级结果。
                    "instance_id": f"{sample.sample_id}::pred::{idx}",
                    "mask": mask,
                    "score": float(item.get("predicted_iou", item.get("stability_score", 1.0))),
                }
            )
        return {"instances": instances}


class ZeroShotSAM2VideoPropagation(BaseMethod):
    """SAM2 时序传播 baseline。"""

    name = "ZeroShotSAM2VideoPropagation"
    family = "sam2"
    inference_mode = InferenceMode.VIDEO_PROPAGATION

    def __init__(self, adapter: SAM2ModelAdapter, prompt_policy):
        self.adapter = adapter
        self.prompt_policy = prompt_policy

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise RuntimeError("Video propagation is sequence-only. Use predict_sequence().")

    def predict_sequence(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        # 真正的序列推理交给 adapter 层统一封装，baseline 这里只保留协议入口。
        return self.adapter.predict_video_sequence(samples, self.prompt_policy)


class ArtifactBackedReferenceMethod(BaseMethod):
    """引用型占位方法。

    这些方法用于冻结平台接口与 CLI 名称。
    后续当真实 adaptation / student / quantized artifact 实现完成后，
    可以在不改 benchmark 外部协议的前提下把这里替换掉。
    """

    def __init__(self, name: str, family: str, inference_mode: InferenceMode):
        self.name = name
        self.family = family
        self.inference_mode = inference_mode

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise RuntimeError(f"{self.name} is a reference-stage placeholder. Plug in a trained artifact before using it.")


def build_baseline_registry(config: AppConfig) -> Dict[str, BaseMethod]:
    """构造 baseline registry。

    这里集中定义平台公开的 baseline 名称到实现实例的映射。
    """
    adapter = SAM2ModelAdapter(config)
    return {
        "bbox_rect": BBoxRectMaskBaseline(),
        "sam2_zero_shot": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_zero_shot_point": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_zero_shot_box_point": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_no_prompt_auto_mask": NoPromptAutoMaskSAM2(adapter),
        "sam2_video_propagation": ZeroShotSAM2VideoPropagation(adapter, prompt_policy=config.evaluation.prompt_policy),
        "reference_adaptation": ArtifactBackedReferenceMethod("ReferenceAdaptation", "reference", InferenceMode.ADAPTED_TEACHER),
        "reference_pseudo": ArtifactBackedReferenceMethod("ReferencePseudo", "reference", InferenceMode.ADAPTED_TEACHER),
        "reference_student": ArtifactBackedReferenceMethod("ReferenceStudent", "reference", InferenceMode.DISTILLED_STUDENT),
        "reference_quantized_student": ArtifactBackedReferenceMethod("ReferenceQuantizedStudent", "reference", InferenceMode.QUANTIZED_STUDENT),
    }
