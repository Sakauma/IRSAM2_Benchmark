from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import BenchmarkMethodProtocol, InferenceMode
from ..data.prompt_synthesis import (
    MASK_DERIVED_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL,
    clamp_box_xyxy,
)
from ..data.sample import Sample
from ..models import SAM2ModelAdapter, load_image_rgb


def box_to_mask(box: list[float], height: int, width: int) -> np.ndarray:
    # BBox baseline 把 prompt box 直接当作预测 mask，用来衡量“框形状本身”的上限/偏差。
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_xyxy(box, width, height)]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class BaseMethod:
    # 所有 baseline 都暴露统一接口：单样本和批量样本。
    # runner 会根据 inference_mode 选择对应评估协议。
    name = "base"
    family = "base"
    inference_mode = InferenceMode.PROMPTED

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_samples(self, samples: List[Sample]) -> Dict[str, Dict[str, Any]]:
        return {sample.sample_id: self.predict_sample(sample) for sample in samples}


class BBoxRectMaskBaseline(BaseMethod):
    name = "BBoxRectMaskBaseline"
    family = "baseline"
    inference_mode = InferenceMode.BOX

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        box = sample.bbox_loose or sample.bbox_tight or [0.0, 0.0, 1.0, 1.0]
        return {"mask": box_to_mask(box, sample.height, sample.width), "score": 1.0}


class PretrainedPromptedSAM2(BaseMethod):
    name = "PretrainedPromptedSAM2"
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
        # pretrained prompted SAM2 按 inference_mode 传入 box、point 或 box+point；它不是 no-prompt。
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
            raise ValueError(f"Unsupported pretrained prompted SAM2 mode: {self.inference_mode.value}")
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


CANONICAL_BASELINE_NAMES = (
    "bbox_rect",
    "sam2_pretrained_box_prompt",
    "sam2_pretrained_tight_box_prompt",
    "sam2_pretrained_point_prompt",
    "sam2_pretrained_box_point_prompt",
    "sam2_no_prompt_auto_mask",
)


def build_baseline_registry(config: AppConfig) -> Dict[str, BenchmarkMethodProtocol]:
    adapter = SAM2ModelAdapter(config)
    return {
        "bbox_rect": BBoxRectMaskBaseline(),
        "sam2_pretrained_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_pretrained_tight_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX, box_variant="tight"),
        "sam2_pretrained_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_pretrained_box_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_no_prompt_auto_mask": NoPromptAutoMaskSAM2(adapter),
    }
