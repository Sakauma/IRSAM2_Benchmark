from __future__ import annotations

import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import Sample, build_box_prior, clamp_box, load_ir_image
from .models import PromptConditionedMaskAdapter, SegFormerWrapper, TinyPIDNetS


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * target).sum() + 1.0
    denominator = probs.sum() + target.sum() + 1.0
    return 1.0 - numerator / denominator


def _balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.sum()
    if float(pos.detach().cpu()) <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, target)
    neg = target.numel() - pos
    pos_weight = torch.clamp(neg / (pos + 1.0), min=1.0, max=20.0).to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _box_projection_loss(logits: torch.Tensor, box_prior: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    row_target = box_prior.amax(dim=-1)
    col_target = box_prior.amax(dim=-2)
    row_pred = probs.amax(dim=-1).clamp(1e-4, 1.0 - 1e-4)
    col_pred = probs.amax(dim=-2).clamp(1e-4, 1.0 - 1e-4)
    row_loss = F.binary_cross_entropy(row_pred, row_target)
    col_loss = F.binary_cross_entropy(col_pred, col_target)
    return row_loss + col_loss


def _box_outside_loss(logits: torch.Tensor, box_prior: torch.Tensor) -> torch.Tensor:
    outside = 1.0 - box_prior
    zeros = torch.zeros_like(logits)
    loss_map = F.binary_cross_entropy_with_logits(logits, zeros, reduction="none")
    return _masked_mean(loss_map, outside)


def _sample_training_loss(
    logits: torch.Tensor,
    target: Optional[torch.Tensor],
    box_prior: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    if target is not None:
        loss = config.lambda_bce * _balanced_bce_with_logits(logits, target)
        return loss + config.lambda_dice * _dice_loss(logits, target)
    loss = config.lambda_box_projection * _box_projection_loss(logits, box_prior)
    return loss + config.lambda_box_outside * _box_outside_loss(logits, box_prior)


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 1.0
    intersection = float(np.logical_and(a, b).sum())
    return intersection / union


class SAM2Teacher:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if str(config.sam2_repo) not in sys.path:
            sys.path.insert(0, str(config.sam2_repo))
        try:
            from hydra import initialize_config_module
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "Failed to import SAM2 dependencies. Ensure SAM2_REPO points to a valid local SAM2 checkout."
            ) from exc

        if not GlobalHydra.instance().is_initialized():
            initialize_config_module(config_module="sam2_configs", version_base=None)

        try:
            self.model = build_sam2(config.sam2_cfg, str(config.sam2_ckpt), device=config.device)
        except TypeError:
            self.model = build_sam2(config.sam2_cfg, str(config.sam2_ckpt), device=config.device, mode="eval")
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)

    def predict(self, image_rgb: np.ndarray, bbox: Sequence[float]):
        with torch.no_grad():
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(
                box=np.array(bbox, dtype=np.float32),
                multimask_output=False,
                return_logits=True,
            )
        return masks[0].astype(np.float32), float(scores[0]), logits[0].astype(np.float32)


class BaseMethod:
    is_trainable = False

    def build_optimizer(self, config: ExperimentConfig):
        raise NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        raise NotImplementedError

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self):
        return None

    def load_state_dict(self, state) -> None:
        return None

    def set_train(self) -> None:
        return None

    def set_eval(self) -> None:
        return None


class ZeroShotSAM2BoxPromptIR(BaseMethod):
    def __init__(self, teacher: SAM2Teacher):
        self.teacher = teacher

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        pred_mask, _, _ = self.teacher.predict(item["image_rgb"], item["bbox"])
        return pred_mask


class BBoxRectMaskBaseline(BaseMethod):
    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        height, width = item["image_rgb"].shape[:2]
        return build_box_prior(item["bbox"], height, width)


class TrainableMethod(BaseMethod):
    is_trainable = True

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def parameters(self):
        raise NotImplementedError

    def build_optimizer(self, config: ExperimentConfig):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


class CleanBoxPEFTSAM2Adapter(TrainableMethod):
    learning_rate: float
    weight_decay: float

    def __init__(self, teacher: SAM2Teacher, config: ExperimentConfig, mode: str = "clean"):
        super().__init__(config)
        self.teacher = teacher
        self.mode = mode
        self.adapter = PromptConditionedMaskAdapter().to(config.device)
        self.learning_rate = config.lr_teacher_adapter
        self.weight_decay = config.weight_decay
        self.rng = random.Random(0)

    def parameters(self):
        return [p for p in self.adapter.parameters() if p.requires_grad]

    def state_dict(self):
        return self.adapter.state_dict()

    def load_state_dict(self, state) -> None:
        self.adapter.load_state_dict(state)

    def set_train(self) -> None:
        self.adapter.train()

    def set_eval(self) -> None:
        self.adapter.eval()

    def sample_prompt_box(self, bbox: Sequence[float], image_shape, rng: random.Random):
        return clamp_box(bbox, image_shape[1], image_shape[0])

    def _forward_single(self, image_rgb: np.ndarray, prompt_box: Sequence[float], target_shape) -> torch.Tensor:
        _, _, lowres_logits = self.teacher.predict(image_rgb, prompt_box)
        teacher_logits = torch.from_numpy(lowres_logits[None, None]).float().to(self.config.device)
        teacher_logits = F.interpolate(teacher_logits, size=target_shape, mode="bilinear", align_corners=False)
        box_prior = torch.from_numpy(build_box_prior(prompt_box, target_shape[0], target_shape[1])[None, None]).float().to(self.config.device)
        logits = self.adapter(teacher_logits, box_prior)
        if logits.shape[-2:] != tuple(target_shape):
            logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        losses: List[torch.Tensor] = []
        sample_weights = batch["sample_weights"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)

        for idx, image_rgb in enumerate(batch["image_rgb"]):
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            clean_box = batch["bboxes"][idx].cpu().numpy()
            prompt_box = self.sample_prompt_box(clean_box, image_rgb.shape[:2], self.rng)
            target_shape = target.shape[-2:] if target is not None else image_rgb.shape[:2]
            logits = self._forward_single(image_rgb, prompt_box, target_shape)
            box_prior = torch.from_numpy(build_box_prior(clean_box, target_shape[0], target_shape[1])[None, None]).float().to(config.device)
            loss = _sample_training_loss(logits, target, box_prior, config)
            losses.append(loss * sample_weights[idx])

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        prompt_box = clamp_box(item["bbox"], item["image_rgb"].shape[1], item["image_rgb"].shape[0])
        logits = self._forward_single(item["image_rgb"], prompt_box, item["image_rgb"].shape[:2])
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


class NoisyBoxPromptRobustSAM2Adapter(CleanBoxPEFTSAM2Adapter):
    def __init__(self, teacher: SAM2Teacher, config: ExperimentConfig, mode: str = "full"):
        super().__init__(teacher, config, mode=mode)

    def sample_prompt_box(self, bbox: Sequence[float], image_shape, rng: random.Random):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        height, width = image_shape
        bw = x2 - x1
        bh = y2 - y1
        if self.mode == "clean_only":
            return clamp_box([x1, y1, x2, y2], width, height)
        if self.mode == "jitter_only":
            dx = bw * self.config.prompt_jitter_scale * (rng.random() * 2 - 1)
            dy = bh * self.config.prompt_jitter_scale * (rng.random() * 2 - 1)
            return clamp_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy], width, height)
        if rng.random() < 0.5:
            dx = bw * self.config.prompt_jitter_scale * (rng.random() * 2 - 1)
            dy = bh * self.config.prompt_jitter_scale * (rng.random() * 2 - 1)
            return clamp_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy], width, height)
        return clamp_box(
            [
                x1 + bw * self.config.prompt_offset_scale,
                y1 + bh * self.config.prompt_offset_scale,
                x2 - bw * self.config.prompt_truncation_ratio,
                y2 - bh * self.config.prompt_truncation_ratio,
            ],
            width,
            height,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        losses: List[torch.Tensor] = []
        sample_weights = batch["sample_weights"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)

        for idx, image_rgb in enumerate(batch["image_rgb"]):
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            clean_box = batch["bboxes"][idx].cpu().numpy()
            noisy_box = self.sample_prompt_box(clean_box, image_rgb.shape[:2], self.rng)
            target_shape = target.shape[-2:] if target is not None else image_rgb.shape[:2]
            noisy_logits = self._forward_single(image_rgb, noisy_box, target_shape)
            box_prior = torch.from_numpy(build_box_prior(clean_box, target_shape[0], target_shape[1])[None, None]).float().to(config.device)
            loss = _sample_training_loss(noisy_logits, target, box_prior, config)
            if self.mode != "clean_only":
                clean_logits = self._forward_single(image_rgb, clean_box, target_shape)
                consistency = F.l1_loss(torch.sigmoid(clean_logits), torch.sigmoid(noisy_logits))
                loss = loss + config.lambda_consistency * consistency
            losses.append(loss * sample_weights[idx])

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()
        return float(loss.detach().cpu())


class QualityFilteredPseudoMaskSelfTrainingSAM2(NoisyBoxPromptRobustSAM2Adapter):
    def __init__(self, teacher: SAM2Teacher, config: ExperimentConfig, quality_filter: bool):
        super().__init__(teacher, config, mode="full")
        self.quality_filter = quality_filter

    def score_pseudo_mask_quality(
        self,
        sample: Sample,
        pseudo_mask: np.ndarray,
        pseudo_prob: np.ndarray,
        teacher_mask: np.ndarray,
        teacher_score: float,
    ) -> float:
        bbox = sample.bbox
        box_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        mask_area = float(np.clip(pseudo_mask, 0.0, 1.0).sum())
        compactness = min(1.0, mask_area / box_area)
        agreement = _binary_iou(pseudo_mask, teacher_mask)
        if mask_area > 0.0:
            confidence = float(pseudo_prob[pseudo_mask > 0.5].mean())
        else:
            confidence = float(np.clip(pseudo_prob.max(), 0.0, 1.0))
        return 0.35 * confidence + 0.35 * agreement + 0.15 * compactness + 0.15 * teacher_score

    def generate_pseudo_samples(self, unlabeled_samples: Sequence[Sample]) -> List[Sample]:
        pseudo_samples: List[Sample] = []
        for sample in unlabeled_samples:
            image_rgb = load_ir_image(sample.image_path)
            pseudo_prob = self.predict({"image_rgb": image_rgb, "bbox": sample.bbox})
            pseudo_mask = (pseudo_prob > 0.5).astype(np.float32)
            teacher_mask, teacher_score, _ = self.teacher.predict(image_rgb, sample.bbox)
            quality = self.score_pseudo_mask_quality(sample, pseudo_mask, pseudo_prob, teacher_mask, teacher_score)
            if self.quality_filter:
                if quality < self.config.pseudo_quality_threshold or teacher_score < self.config.pseudo_score_threshold:
                    continue
            if float(pseudo_mask.sum()) <= 0.0:
                continue
            pseudo_samples.append(
                sample.with_mask(
                    mask=pseudo_mask,
                    supervision_source="pseudo_mask",
                    sample_weight=max(0.25, quality),
                    pseudo_score=teacher_score,
                    pseudo_quality=quality,
                )
            )
        return pseudo_samples


class DirectSupervisedIRSegFormerB0(TrainableMethod):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = SegFormerWrapper().to(config.device)
        self.learning_rate = config.lr_segformer
        self.weight_decay = config.weight_decay

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state)

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        images = batch["images"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)
        sample_weights = batch["sample_weights"].to(config.device)
        bboxes = batch["bboxes"].to(config.device)
        logits = self.model(images)
        losses: List[torch.Tensor] = []
        for idx in range(logits.shape[0]):
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            box_prior = torch.from_numpy(
                build_box_prior(
                    bboxes[idx].detach().cpu().numpy(),
                    logits.shape[-2],
                    logits.shape[-1],
                )[None, None]
            ).float().to(config.device)
            sample_loss = _sample_training_loss(logits[idx : idx + 1], target, box_prior, config)
            losses.append(sample_loss * sample_weights[idx])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        image = torch.from_numpy(item["image_rgb"]).permute(2, 0, 1).float()[None].to(self.config.device) / 255.0
        with torch.no_grad():
            logits = self.model(image)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


class DirectSupervisedIRPIDNetS(TrainableMethod):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = TinyPIDNetS().to(config.device)
        self.learning_rate = config.lr_pidnet
        self.weight_decay = config.weight_decay

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state)

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        images = batch["images"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)
        sample_weights = batch["sample_weights"].to(config.device)
        bboxes = batch["bboxes"].to(config.device)
        logits = self.model(images)
        losses: List[torch.Tensor] = []
        for idx in range(logits.shape[0]):
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            box_prior = torch.from_numpy(
                build_box_prior(
                    bboxes[idx].detach().cpu().numpy(),
                    logits.shape[-2],
                    logits.shape[-1],
                )[None, None]
            ).float().to(config.device)
            sample_loss = _sample_training_loss(logits[idx : idx + 1], target, box_prior, config)
            losses.append(sample_loss * sample_weights[idx])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        image = torch.from_numpy(item["image_rgb"]).permute(2, 0, 1).float()[None].to(self.config.device) / 255.0
        with torch.no_grad():
            logits = self.model(image)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


def build_method_registry(teacher: SAM2Teacher, config: ExperimentConfig):
    return {
        "BBoxRectMaskBaseline": lambda: BBoxRectMaskBaseline(),
        "ZeroShotSAM2BoxPromptIR": lambda: ZeroShotSAM2BoxPromptIR(teacher),
        "CleanBoxPEFTSAM2Adapter": lambda: CleanBoxPEFTSAM2Adapter(teacher, config),
        "NoisyBoxPromptRobustSAM2Adapter": lambda: NoisyBoxPromptRobustSAM2Adapter(teacher, config, mode="full"),
        "CleanPromptOnlyWithinPromptRobustAdapter": lambda: NoisyBoxPromptRobustSAM2Adapter(teacher, config, mode="clean_only"),
        "JitterOnlyPromptRobustSAM2Adapter": lambda: NoisyBoxPromptRobustSAM2Adapter(teacher, config, mode="jitter_only"),
        "QualityFilteredPseudoMaskSelfTrainingSAM2": lambda: QualityFilteredPseudoMaskSelfTrainingSAM2(teacher, config, quality_filter=True),
        "PseudoMaskSelfTrainingWithoutIRQualityFilter": lambda: QualityFilteredPseudoMaskSelfTrainingSAM2(teacher, config, quality_filter=False),
        "DirectSupervisedIRSegFormerB0": lambda: DirectSupervisedIRSegFormerB0(config),
        "DirectSupervisedIRPIDNetS": lambda: DirectSupervisedIRPIDNetS(config),
    }
