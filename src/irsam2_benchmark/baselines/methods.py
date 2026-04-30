from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import BenchmarkMethodProtocol, InferenceMode
from ..data.auto_prompt import HEURISTIC_IR_AUTO_PROMPT_PROTOCOL, generate_heuristic_ir_auto_prompt_from_path
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


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"External prediction frame_id must be a safe relative path, got {value!r}.")
    return path


def _resolve_prediction_root(config: AppConfig, raw_root: str) -> Path:
    env_root = os.environ.get("EXTERNAL_PREDICTION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    path = Path(raw_root).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        config.root / path,
        config.root.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class ExternalPredictionMaskBaseline(BaseMethod):
    name = "ExternalPredictionMaskBaseline"
    family = "external_prediction"
    inference_mode = InferenceMode.SINGLE_MASK

    def __init__(self, config: AppConfig):
        self._configuration_error: str | None = None
        raw_root = config.method.get("prediction_root")
        if not raw_root:
            self._configuration_error = "external_prediction_mask requires method.prediction_root."
            self.prediction_root = Path()
        else:
            self.prediction_root = _resolve_prediction_root(config, str(raw_root))
        self.dataset_id = str(config.method.get("prediction_dataset_id") or config.dataset.dataset_id)
        self.prediction_suffix = str(config.method.get("prediction_suffix", ".png"))
        if not self.prediction_suffix.startswith("."):
            self.prediction_suffix = f".{self.prediction_suffix}"
        self.threshold = float(config.method.get("prediction_threshold", 0.5))
        self.model_name = str(config.method.get("external_model_name") or config.method.get("name") or "external_prediction")
        if self._configuration_error:
            self._latency_by_frame_id: Dict[str, float] = {}
            self._latency_by_sample_id: Dict[str, float] = {}
            self._prediction_by_sample_id: Dict[str, Path] = {}
        else:
            self._latency_by_frame_id, self._latency_by_sample_id, self._prediction_by_sample_id = self._load_prediction_manifest()

    @property
    def _dataset_dir(self) -> Path:
        return self.prediction_root / self.dataset_id

    def _resolve_manifest_prediction_path(self, value: object) -> Path | None:
        if value is None:
            return None
        raw = Path(str(value))
        if raw.is_absolute():
            return raw
        candidate = self._dataset_dir / raw
        if candidate.exists():
            return candidate
        return (Path.cwd() / raw).resolve()

    def _load_prediction_manifest(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Path]]:
        manifest_path = self._dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            return {}, {}, {}
        latency_by_frame_id: Dict[str, float] = {}
        latency_by_sample_id: Dict[str, float] = {}
        prediction_by_sample_id: Dict[str, Path] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                frame_id = record.get("frame_id")
                sample_id = record.get("sample_id")
                latency_ms = record.get("latency_ms")
                if frame_id is not None and latency_ms is not None:
                    latency_by_frame_id[str(frame_id)] = float(latency_ms)
                if sample_id is not None:
                    if latency_ms is not None:
                        latency_by_sample_id[str(sample_id)] = float(latency_ms)
                    prediction_path = self._resolve_manifest_prediction_path(record.get("prediction_path"))
                    if prediction_path is not None:
                        prediction_by_sample_id[str(sample_id)] = prediction_path
        return latency_by_frame_id, latency_by_sample_id, prediction_by_sample_id

    def _prediction_path(self, frame_id: str, sample_id: str | None = None) -> Path:
        if sample_id is not None:
            manifest_path = self._prediction_by_sample_id.get(sample_id)
            if manifest_path is not None and manifest_path.exists():
                return manifest_path
        rel = _safe_relative_path(frame_id)
        appended = self._dataset_dir / Path(f"{rel.as_posix()}{self.prediction_suffix}")
        replaced = self._dataset_dir / rel.with_suffix(self.prediction_suffix)
        candidates = [appended]
        if replaced != appended:
            candidates.append(replaced)
        if rel.suffix:
            candidates.append(self._dataset_dir / rel)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        candidate_list = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"External prediction mask not found for frame_id={frame_id!r}. Tried: {candidate_list}")

    def predict_sample(self, sample: Sample) -> Dict[str, Any]:
        if self._configuration_error:
            raise ValueError(self._configuration_error)
        prediction_path = self._prediction_path(sample.frame_id, sample.sample_id)
        with Image.open(prediction_path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
        if arr.size and float(arr.max()) > 1.0:
            arr = arr / 255.0
        mask = (arr > self.threshold).astype(np.float32)
        payload: Dict[str, Any] = {
            "mask": mask,
            "score": 1.0,
            "prompt": {
                "protocol": "external_prediction_import_v1",
                "source": "external_prediction",
                "box_variant": "none",
            },
            "metadata": {
                "ExternalPredictionModel": self.model_name,
                "ExternalPredictionDatasetId": self.dataset_id,
                "ExternalPredictionPath": str(prediction_path),
                "ExternalPredictionRoot": str(self.prediction_root),
                "ExternalPredictionThreshold": self.threshold,
            },
        }
        latency_ms = self._latency_by_sample_id.get(sample.sample_id, self._latency_by_frame_id.get(sample.frame_id))
        if latency_ms is not None:
            payload["LatencyMs"] = latency_ms
        return payload


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
        # 同一图像内的多实例 prompt 共享一次 SAM2 image embedding。
        if not samples:
            return {}
        kwargs_and_prompts = [self._predict_kwargs_and_prompt(sample) for sample in samples]
        image_groups: Dict[str, List[int]] = {}
        for idx, sample in enumerate(samples):
            image_groups.setdefault(str(sample.image_path), []).append(idx)

        predictions: Dict[str, Dict[str, Any]] = {}
        for indices in image_groups.values():
            group_samples = [samples[idx] for idx in indices]
            group_kwargs = [kwargs_and_prompts[idx][0] for idx in indices]
            group_prompts = [kwargs_and_prompts[idx][1] for idx in indices]
            image_rgb = load_image_rgb(group_samples[0].image_path)
            results = self.adapter.predict_prompts_for_image(
                image_rgb,
                boxes=[kwargs.get("box") for kwargs in group_kwargs],
                points=[kwargs.get("points") for kwargs in group_kwargs],
                point_labels=[kwargs.get("point_labels") for kwargs in group_kwargs],
                multimask_output=self.inference_mode == InferenceMode.MULTI_MASK,
            )
            if len(results) != len(group_samples):
                raise RuntimeError(f"Prompt prediction returned {len(results)} results for {len(group_samples)} samples.")
            for sample, result, prompt in zip(group_samples, results, group_prompts):
                predictions[sample.sample_id] = self._prediction_from_result(result, prompt)
        return predictions


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


class HeuristicAutoPromptedSAM2(BaseMethod):
    name = "HeuristicAutoPromptedSAM2"
    family = "sam2"

    def __init__(
        self,
        adapter: SAM2ModelAdapter,
        prompt_mode: InferenceMode,
        *,
        use_negative_ring: bool = False,
        top_k: int = 1,
    ):
        self.adapter = adapter
        self.inference_mode = prompt_mode
        self.use_negative_ring = use_negative_ring
        self.top_k = top_k

    def _auto_prompt(self, sample: Sample) -> Dict[str, Any]:
        auto_prompt = generate_heuristic_ir_auto_prompt_from_path(
            sample.image_path,
            top_k=self.top_k,
            negative_ring=self.use_negative_ring,
        )
        prompt = {
            **auto_prompt.metadata,
            "protocol": HEURISTIC_IR_AUTO_PROMPT_PROTOCOL,
            "box_variant": "heuristic",
            "point_rule": "highest_scoring_ir_response_peak",
            "box_rule": "connected_response_component_around_primary_peak",
        }
        return prompt

    def _predict_kwargs_and_prompt(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._auto_prompt(sample)
        kwargs: Dict[str, Any] = {"multimask_output": self.inference_mode == InferenceMode.MULTI_MASK}
        if self.inference_mode == InferenceMode.POINT:
            kwargs["points"] = np.array([prompt["point"]], dtype=np.float32)
            kwargs["point_labels"] = np.array([1], dtype=np.int32)
        elif self.inference_mode == InferenceMode.BOX:
            kwargs["box"] = prompt["box"]
        elif self.inference_mode == InferenceMode.BOX_POINT:
            kwargs["box"] = prompt["box"]
            kwargs["points"] = np.array(prompt["points"], dtype=np.float32)
            kwargs["point_labels"] = np.array(prompt["point_labels"], dtype=np.int32)
        else:
            raise ValueError(f"Unsupported heuristic auto prompt mode: {self.inference_mode.value}")
        return kwargs, prompt

    def _prediction_from_result(self, result: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
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


CANONICAL_BASELINE_NAMES = (
    "bbox_rect",
    "external_prediction_mask",
    "sam2_pretrained_box_prompt",
    "sam2_pretrained_tight_box_prompt",
    "sam2_pretrained_point_prompt",
    "sam2_pretrained_box_point_prompt",
    "sam2_no_prompt_auto_mask",
    "sam2_heuristic_auto_point_prompt",
    "sam2_heuristic_auto_box_prompt",
    "sam2_heuristic_auto_box_point_prompt",
    "sam2_heuristic_auto_box_point_neg_prompt",
)


def build_baseline_registry(config: AppConfig) -> Dict[str, BenchmarkMethodProtocol]:
    adapter = SAM2ModelAdapter(config)
    return {
        "bbox_rect": BBoxRectMaskBaseline(),
        "external_prediction_mask": ExternalPredictionMaskBaseline(config),
        "sam2_pretrained_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_pretrained_tight_box_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX, box_variant="tight"),
        "sam2_pretrained_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_pretrained_box_point_prompt": PretrainedPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_no_prompt_auto_mask": NoPromptAutoMaskSAM2(adapter),
        "sam2_heuristic_auto_point_prompt": HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.POINT),
        "sam2_heuristic_auto_box_prompt": HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX),
        "sam2_heuristic_auto_box_point_prompt": HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT),
        "sam2_heuristic_auto_box_point_neg_prompt": HeuristicAutoPromptedSAM2(
            adapter,
            prompt_mode=InferenceMode.BOX_POINT,
            use_negative_ring=True,
        ),
    }
