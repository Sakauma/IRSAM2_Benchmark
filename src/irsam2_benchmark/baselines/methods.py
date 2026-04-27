from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.prompt_synthesis import (
    MASK_DERIVED_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL,
    clamp_box_xyxy,
)
from ..data.sample import Sample
from ..models import SAM2ModelAdapter, load_image_rgb
from ..priors import PriorFactory
from ..prompts import PromptFactory


def box_to_mask(box: list[float], height: int, width: int) -> np.ndarray:
    # BBox baseline 把 prompt box 直接当作预测 mask，用来衡量“框形状本身”的上限/偏差。
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_xyxy(box, width, height)]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class BaseMethod:
    # 所有 baseline 都暴露统一接口：单样本、批量样本、视频序列。
    # runner 会根据 inference_mode 选择对应评估协议。
    name = "base"
    family = "base"
    inference_mode = InferenceMode.ZERO_SHOT

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        return {sample.sample_id: self.predict_sample(sample) for sample in samples}

    def predict_sequence(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        return {sample.sample_id: self.predict_sample(sample)["mask"] for sample in samples}


class BBoxRectMaskBaseline(BaseMethod):
    name = "BBoxRectMaskBaseline"
    family = "baseline"
    inference_mode = InferenceMode.BOX

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        box = sample.bbox_loose or sample.bbox_tight or [0.0, 0.0, 1.0, 1.0]
        return {"mask": box_to_mask(box, sample.height, sample.width), "score": 1.0}


class ZeroShotSAM2(BaseMethod):
    name = "ZeroShotSAM2"
    family = "sam2"

    def __init__(self, adapter: SAM2ModelAdapter, prompt_mode: InferenceMode, box_variant: str = "loose"):
        self.adapter = adapter
        self.inference_mode = prompt_mode
        self.box_variant = box_variant

    def _sample_box(self, sample: Sample) -> list[float] | None:
        # 默认使用 loose box；tight box 仅用于 protocol ablation。
        if self.box_variant == "tight":
            return sample.bbox_tight or sample.bbox_loose
        return sample.bbox_loose or sample.bbox_tight

    def _prompt_payload(self, sample: Sample, *, box: list[float] | None = None, point: list[float] | None = None) -> Dict[str, Any]:
        # prompt 元数据会写入每行结果，保证论文表能追溯 box/point 是如何从 GT mask 派生的。
        prompt_generation = dict(sample.metadata.get("prompt_generation", {}))
        if box is not None and point is not None:
            protocol = (
                MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL
                if self.box_variant == "tight"
                else MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL
            )
        elif box is not None:
            protocol = (
                MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL
                if self.box_variant == "tight"
                else MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL
            )
        elif point is not None:
            protocol = MASK_DERIVED_CENTROID_POINT_PROTOCOL
        else:
            protocol = prompt_generation.get("protocol", "unknown")
        payload: Dict[str, Any] = {
            **prompt_generation,
            "protocol": protocol,
            "box_variant": self.box_variant if box is not None else None,
            "box": box,
            "point": point,
        }
        return payload

    def _predict_kwargs_and_prompt(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # sam2_zero_shot 不是 no-prompt；它按 inference_mode 传入 box、point 或 box+point。
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        box = self._sample_box(sample)
        point = sample.point_prompt
        if self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = box
            prompt = self._prompt_payload(sample, box=box)
        elif self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array([point], dtype=np.float32) if point else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if point else None
            prompt = self._prompt_payload(sample, point=point)
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = box
            kwargs["points"] = np.array([point], dtype=np.float32) if point else None
            kwargs["point_labels"] = np.array([1], dtype=np.int32) if point else None
            prompt = self._prompt_payload(sample, box=box, point=point)
        elif self.inference_mode in {InferenceMode.SINGLE_MASK, InferenceMode.MULTI_MASK}:
            kwargs["box"] = box
            prompt = self._prompt_payload(sample, box=box)
        else:
            raise ValueError(f"Unsupported zero-shot prompt mode: {self.inference_mode.value}")
        return kwargs, prompt

    def _prediction_from_result(self, result: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
        # SAM2 可能返回多个候选 mask；论文主表取分数最高的候选，保持确定性。
        masks = result["masks"]
        scores = result["scores"]
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx]), "prompt": prompt}

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        image_rgb = load_image_rgb(sample.image_path)
        kwargs, prompt = self._predict_kwargs_and_prompt(sample)
        result = self.adapter.predict_image(image_rgb, **kwargs)
        return self._prediction_from_result(result, prompt)

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        # 批量路径只减少 set_image/predict 调用开销，不改变每个 sample 的 prompt 内容。
        if not samples:
            return {}
        images = [load_image_rgb(sample.image_path) for sample in samples]
        kwargs_and_prompts = [self._predict_kwargs_and_prompt(sample) for sample in samples]
        kwargs_list = [item[0] for item in kwargs_and_prompts]
        prompts = [item[1] for item in kwargs_and_prompts]
        results = self.adapter.predict_images(
            images,
            boxes=[kwargs.get("box") for kwargs in kwargs_list],
            points=[kwargs.get("points") for kwargs in kwargs_list],
            point_labels=[kwargs.get("point_labels") for kwargs in kwargs_list],
            multimask_output=self.inference_mode == InferenceMode.MULTI_MASK,
        )
        if len(results) != len(samples):
            raise RuntimeError(f"Batch prediction returned {len(results)} results for {len(samples)} samples.")
        return {
            sample.sample_id: self._prediction_from_result(result, prompt)
            for sample, result, prompt in zip(samples, results, prompts)
        }


class NoPromptAutoMaskSAM2(BaseMethod):
    name = "NoPromptAutoMaskSAM2"
    family = "sam2"
    inference_mode = InferenceMode.NO_PROMPT_AUTO_MASK

    def __init__(self, adapter: SAM2ModelAdapter):
        self.adapter = adapter

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        # 自动掩码模式不接受外部 prompt；runner 会按 image-level 与 GT instances 做匹配。
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
                    "instance_id": f"{sample.frame_id}::pred::{idx}",
                    "mask": mask,
                    "score": float(item.get("predicted_iou", item.get("stability_score", 1.0))),
                }
            )
        runtime = getattr(getattr(self.adapter, "config", None), "runtime", None)
        points_per_batch = int(getattr(runtime, "auto_mask_points_per_batch", 64))
        return {"instances": instances, "auto_mask_points_per_batch": points_per_batch}


def load_image_tensor(path) -> torch.Tensor:
    # physics auto-prompt 只需要单通道亮度张量；SAM2 推理仍走归一化后的 RGB 伪三通道。
    raw = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    min_v = float(raw.min())
    max_v = float(raw.max())
    norm = (raw.astype(np.float32) - min_v) / max(1e-6, max_v - min_v)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)


class PhysicsAutoPromptSAM2(BaseMethod):
    name = "PhysicsAutoPromptSAM2"
    family = "ir_prompted_sam2"
    inference_mode = InferenceMode.BOX_POINT

    def __init__(self, config: AppConfig, adapter: SAM2ModelAdapter):
        self.config = config
        self.adapter = adapter
        prior_config = config.modules.get(
            "prior",
            {
                "name": "prior_fusion",
                "enabled": ["local_contrast", "multi_scale_top_hat", "snr_like"],
                "scales": [7, 15, 31],
                "weights": {"local_contrast": 0.4, "multi_scale_top_hat": 0.4, "snr_like": 0.2},
            },
        )
        prompt_config = config.modules.get(
            "prompt",
            {
                "name": "heuristic_physics",
                "type": "box+point",
                "percentile": 99.5,
                "top_k": 1,
                "min_component_area": 2,
                "box_pad_ratio": 0.25,
            },
        )
        self.prior = PriorFactory.build(prior_config)
        self.prompt_generator = PromptFactory.build(prompt_config)

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        # 先从红外先验图生成候选 prompt，再把 prompt 交给 SAM2 分割。
        image_tensor = load_image_tensor(sample.image_path)
        with torch.inference_mode():
            prior_maps = self.prior(image_tensor)
            prompt = self.prompt_generator(image_tensor, prior_maps)
        image_rgb = load_image_rgb(sample.image_path)
        kwargs: Dict[str, Any] = {"multimask_output": False}
        if prompt.get("box") is not None:
            kwargs["box"] = prompt["box"]
        if prompt.get("point") is not None:
            kwargs["points"] = np.array([prompt["point"]], dtype=np.float32)
            kwargs["point_labels"] = np.array([1], dtype=np.int32)
        result = self.adapter.predict_image(image_rgb, **kwargs)
        masks = result["masks"]
        scores = result["scores"]
        best_idx = int(np.argmax(scores))
        return {"mask": masks[best_idx].astype(np.float32), "score": float(scores[best_idx]), "prompt": prompt}


class ZeroShotSAM2VideoPropagation(BaseMethod):
    name = "ZeroShotSAM2VideoPropagation"
    family = "sam2"
    inference_mode = InferenceMode.VIDEO_PROPAGATION

    def __init__(self, adapter: SAM2ModelAdapter, prompt_policy):
        self.adapter = adapter
        self.prompt_policy = prompt_policy

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise RuntimeError("Video propagation is sequence-only. Use predict_sequence().")

    def predict_sequence(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        # 视频传播只在 sequence/track view 中调用，不能逐 instance 独立评估。
        return self.adapter.predict_video_sequence(samples, self.prompt_policy)


class ArtifactBackedReferenceMethod(BaseMethod):
    """引用方法先稳定平台接口；真正训练逻辑后续可无缝替换。"""

    def __init__(self, name: str, family: str, inference_mode: InferenceMode):
        self.name = name
        self.family = family
        self.inference_mode = inference_mode

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise RuntimeError(f"{self.name} is a reference-stage placeholder. Plug in a trained artifact before using it.")


def build_baseline_registry(config: AppConfig) -> Dict[str, BaseMethod]:
    adapter = SAM2ModelAdapter(config)
    return {
        "bbox_rect": BBoxRectMaskBaseline(),
        "sam2_zero_shot": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_zero_shot_tight_box": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.BOX, box_variant="tight"),
        "sam2_zero_shot_point": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_zero_shot_box_point": ZeroShotSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_no_prompt_auto_mask": NoPromptAutoMaskSAM2(adapter),
        "sam2_physics_auto_prompt": PhysicsAutoPromptSAM2(config, adapter),
        "sam2_video_propagation": ZeroShotSAM2VideoPropagation(adapter, prompt_policy=config.evaluation.prompt_policy),
        "reference_adaptation": ArtifactBackedReferenceMethod("ReferenceAdaptation", "reference", InferenceMode.ADAPTED_TEACHER),
        "reference_pseudo": ArtifactBackedReferenceMethod("ReferencePseudo", "reference", InferenceMode.ADAPTED_TEACHER),
        "reference_student": ArtifactBackedReferenceMethod("ReferenceStudent", "reference", InferenceMode.DISTILLED_STUDENT),
        "reference_quantized_student": ArtifactBackedReferenceMethod("ReferenceQuantizedStudent", "reference", InferenceMode.QUANTIZED_STUDENT),
    }
