from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import ModelCapabilities, PromptPolicy, PromptType
from ..data.sample import Sample


def _require_module(module_name: str, package_name: str):
    # 依赖缺失时给出面向 benchmark 环境的错误，而不是暴露底层 import 栈。
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Required SAM2 runtime dependency is not available: {package_name}. "
            f"Install the benchmark runtime dependencies and verify your PyTorch/CUDA environment."
        ) from exc


def check_sam2_runtime(repo: Path) -> None:
    # 这里只检查运行 SAM2 必需的 Python 依赖和本地 repo 路径；CUDA 可用性由 PyTorch 推理时判断。
    _require_module("torch", "torch>=2.5.1")
    _require_module("torchvision", "torchvision>=0.20.1")
    _require_module("hydra", "hydra-core>=1.3.2")
    _require_module("iopath", "iopath>=0.1.10")
    if not repo.exists():
        raise RuntimeError(f"SAM2_REPO does not exist: {repo}")


def load_image_rgb(path: Path) -> np.ndarray:
    # 遥感红外图常是单通道。SAM2 image predictor 需要 RGB，因此这里做 min-max 归一化后复制三通道。
    raw = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    min_v = float(raw.min())
    max_v = float(raw.max())
    denom = max(1e-6, max_v - min_v)
    norm = ((raw.astype(np.float32) - min_v) / denom * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


class SAM2ModelAdapter:
    """Unified adapter for local SAM2 repo, config, checkpoint, and inference APIs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo = config.sam2_repo
        self.model = None
        self.image_predictor = None
        self.video_predictor = None
        self.auto_mask_generator = None
        self.capabilities = ModelCapabilities(
            supports_auto_mask=True,
            supports_video_propagation=True,
            supports_transfer=True,
            supports_adapt=True,
            supports_distill_teacher=True,
            supports_quant_export=True,
        )

    def ensure_loaded(self) -> None:
        # SAM2 权重和 predictor 延迟加载，避免 dry-run、配置检查和分析阶段占用显存。
        if self.model is not None:
            return
        check_sam2_runtime(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        build_sam2 = self._resolve_symbol("sam2.build_sam", "build_sam2")
        image_predictor_cls = self._resolve_symbol("sam2.sam2_image_predictor", "SAM2ImagePredictor")
        if not GlobalHydra.instance().is_initialized():
            initialize_config_module(config_module="sam2_configs", version_base=None)
        ckpt_path = self._resolve_checkpoint_path()
        model_kwargs = {"device": self.config.runtime.device}
        try:
            self.model = build_sam2(self.config.model.cfg, str(ckpt_path), **model_kwargs)
        except TypeError:
            self.model = build_sam2(self.config.model.cfg, str(ckpt_path), mode="eval", **model_kwargs)
        self.model.eval()
        self.image_predictor = image_predictor_cls(self.model)
        self.video_predictor = self._build_video_predictor()
        self.auto_mask_generator = self._build_auto_mask_generator()

    def _resolve_symbol(self, module_name: str, symbol_name: str):
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)

    def _build_video_predictor(self):
        # 不同 SAM2 checkout 暴露的视频 API 名称略有差异，这里按常见入口做兼容探测。
        ckpt_path = self._resolve_checkpoint_path()
        candidates = [
            ("sam2.build_sam", "build_sam2_video_predictor"),
            ("sam2.sam2_video_predictor", "SAM2VideoPredictor"),
        ]
        for module_name, symbol_name in candidates:
            try:
                symbol = self._resolve_symbol(module_name, symbol_name)
                if symbol_name.startswith("build_"):
                    return symbol(self.config.model.cfg, str(ckpt_path), device=self.config.runtime.device)
                return symbol(self.model)
            except Exception:
                continue
        return None

    def _resolve_checkpoint_path(self) -> Path:
        # 允许 YAML 写绝对路径、相对 SAM2 repo 的路径，或相对 benchmark 项目的路径。
        ckpt = Path(self.config.model.ckpt)
        if ckpt.is_absolute():
            return ckpt
        repo_candidate = self.repo / ckpt
        if repo_candidate.exists():
            return repo_candidate
        root_candidate = self.config.root / ckpt
        if root_candidate.exists():
            return root_candidate
        return ckpt

    def _build_auto_mask_generator(self):
        # 自动掩码类在不同版本 SAM2 中模块名不同；找不到时 no-prompt 模式会显式报错。
        candidates = [
            ("sam2.automatic_mask_generator", "SAM2AutomaticMaskGenerator"),
            ("sam2.sam2_automatic_mask_generator", "SAM2AutomaticMaskGenerator"),
        ]
        for module_name, symbol_name in candidates:
            try:
                cls = self._resolve_symbol(module_name, symbol_name)
                return cls(self.model, points_per_batch=int(self.config.runtime.auto_mask_points_per_batch))
            except Exception:
                continue
        return None

    def predict_image(
        self,
        image_rgb: np.ndarray,
        *,
        box: Optional[Iterable[float]] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> dict[str, Any]:
        # 单图 prompted SAM2 推理。所有 box/point 坐标都已由上层按原图像素坐标生成。
        self.ensure_loaded()
        self.image_predictor.set_image(image_rgb)
        masks, scores, logits = self.image_predictor.predict(
            box=None if box is None else np.array(box, dtype=np.float32),
            point_coords=points,
            point_labels=point_labels,
            multimask_output=multimask_output,
            return_logits=True,
        )
        return {
            "masks": np.asarray(masks, dtype=np.float32),
            "scores": np.asarray(scores, dtype=np.float32),
            "logits": np.asarray(logits, dtype=np.float32),
        }

    def predict_images(
        self,
        image_rgbs: list[np.ndarray],
        *,
        boxes: list[Optional[Iterable[float]]] | None = None,
        points: list[Optional[np.ndarray]] | None = None,
        point_labels: list[Optional[np.ndarray]] | None = None,
        multimask_output: bool = False,
    ) -> list[dict[str, Any]]:
        # 优先使用 SAM2 的 batch API；本地 checkout 不支持时回退为逐图预测。
        if not image_rgbs:
            return []
        self.ensure_loaded()
        if len(image_rgbs) == 1 or not hasattr(self.image_predictor, "set_image_batch") or not hasattr(self.image_predictor, "predict_batch"):
            return [
                self.predict_image(
                    image_rgbs[idx],
                    box=None if boxes is None else boxes[idx],
                    points=None if points is None else points[idx],
                    point_labels=None if point_labels is None else point_labels[idx],
                    multimask_output=multimask_output,
                )
                for idx in range(len(image_rgbs))
            ]
        box_batch = None if boxes is None else [None if box is None else np.array(box, dtype=np.float32) for box in boxes]
        point_batch = None if points is None else points
        label_batch = None if point_labels is None else point_labels
        self.image_predictor.set_image_batch(image_rgbs)
        masks, scores, logits = self.image_predictor.predict_batch(
            point_coords_batch=point_batch,
            point_labels_batch=label_batch,
            box_batch=box_batch,
            multimask_output=multimask_output,
            return_logits=True,
        )
        return [
            {
                "masks": np.asarray(masks[idx], dtype=np.float32),
                "scores": np.asarray(scores[idx], dtype=np.float32),
                "logits": np.asarray(logits[idx], dtype=np.float32),
            }
            for idx in range(len(image_rgbs))
        ]

    def predict_auto_masks(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        # no-prompt 模式返回的是多个候选 instance，后续由 image-level evaluator 统一匹配 GT。
        self.ensure_loaded()
        if self.auto_mask_generator is None:
            raise RuntimeError("Automatic mask generation is not available in the local SAM2 checkout.")
        return list(self.auto_mask_generator.generate(image_rgb))

    def predict_video_sequence(self, samples: list[Sample], prompt_policy: PromptPolicy) -> dict[str, np.ndarray]:
        # SAM2 video predictor 读取目录中的帧，因此这里临时把同一 track 的帧写成连续 png。
        self.ensure_loaded()
        if self.video_predictor is None:
            raise RuntimeError("Video propagation is not available in the local SAM2 checkout.")
        if not samples:
            return {}

        sequence_ids = {sample.sequence_id for sample in samples}
        track_ids = {sample.track_id for sample in samples}
        if len(sequence_ids) != 1:
            raise RuntimeError(f"Video propagation expects one sequence per call, got {sorted(sequence_ids)!r}.")
        if len(track_ids) != 1 or None in track_ids:
            raise RuntimeError("Video propagation expects one non-empty track_id per call.")

        seen_frame_ids: set[str] = set()
        for sample in samples:
            if sample.frame_id in seen_frame_ids:
                raise RuntimeError(
                    f"Video propagation expects one sample per frame within a track, got duplicate frame_id={sample.frame_id!r}."
                )
            seen_frame_ids.add(sample.frame_id)

        with tempfile.TemporaryDirectory(prefix="irsam2_video_") as temp_dir:
            temp_root = Path(temp_dir)
            for idx, sample in enumerate(samples):
                with Image.open(sample.image_path) as image:
                    image.save(temp_root / f"{idx:06d}.png")

            init_state = getattr(self.video_predictor, "init_state", None)
            add_prompt = getattr(self.video_predictor, "add_new_points_or_box", None)
            propagate = getattr(self.video_predictor, "propagate_in_video", None)
            if init_state is None or add_prompt is None or propagate is None:
                raise RuntimeError("The local SAM2 video predictor does not expose the expected API.")

            state = init_state(video_path=str(temp_root))
            first_sample = samples[0]
            # 初始 prompt 只加在第一帧；refresh_interval 可按策略在后续帧重新注入 prompt。
            point_coords = None
            point_labels = None
            box = None
            if prompt_policy.prompt_type == PromptType.BOX and first_sample.bbox_loose is not None:
                box = np.array(first_sample.bbox_loose, dtype=np.float32)
            elif prompt_policy.prompt_type == PromptType.POINT and first_sample.point_prompt is not None:
                point_coords = np.array([first_sample.point_prompt], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
            elif prompt_policy.prompt_type == PromptType.BOX_POINT:
                box = np.array(first_sample.bbox_loose, dtype=np.float32) if first_sample.bbox_loose is not None else None
                if first_sample.point_prompt is not None:
                    point_coords = np.array([first_sample.point_prompt], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)

            add_prompt(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
                box=box,
            )

            predictions: dict[str, np.ndarray] = {}
            refresh_interval = prompt_policy.refresh_interval
            for frame_idx, object_ids, mask_logits in propagate(state):
                del object_ids
                if hasattr(mask_logits, "detach"):
                    logits = mask_logits.detach().cpu().numpy()
                else:
                    logits = np.asarray(mask_logits)
                if logits.ndim >= 3:
                    logits = logits[0]
                mask = (logits > 0).astype(np.float32)
                predictions[samples[frame_idx].sample_id] = mask
                if refresh_interval and frame_idx > 0 and frame_idx % refresh_interval == 0 and frame_idx < len(samples):
                    sample = samples[frame_idx]
                    refresh_box = np.array(sample.bbox_loose, dtype=np.float32) if sample.bbox_loose is not None else None
                    refresh_points = None
                    refresh_labels = None
                    if sample.point_prompt is not None:
                        refresh_points = np.array([sample.point_prompt], dtype=np.float32)
                        refresh_labels = np.array([1], dtype=np.int32)
                    add_prompt(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=1,
                        points=refresh_points,
                        labels=refresh_labels,
                        box=refresh_box,
                    )
            return predictions
