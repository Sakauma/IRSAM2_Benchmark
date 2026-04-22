"""方法实现层。

这里集中定义 benchmark 当前支持的全部方法，包括：
1. 几何矩形框 baseline
2. 零样本 SAM2
3. 基于 SAM2 的 adapter / robust / pseudo 变体
4. direct-train 控制组

训练与推理协议都在这一层落地。
"""

from __future__ import annotations

import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .config import ExperimentConfig
from .data import Sample, build_box_prior, clamp_box, load_ir_image
from .models import PromptConditionedMaskAdapter, SegFormerWrapper, TinyPIDNetS


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对二值分割 logits 计算 Dice loss。"""
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * target).sum() + 1.0
    denominator = probs.sum() + target.sum() + 1.0
    return 1.0 - numerator / denominator


def _balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """带正负样本重平衡的 BCE。

    红外小目标通常前景像素极少，如果直接用普通 BCE，模型很容易学成全背景。
    """
    pos = target.sum()
    if float(pos.detach().cpu()) <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, target)
    neg = target.numel() - pos
    pos_weight = torch.clamp(neg / (pos + 1.0), min=1.0, max=20.0).to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """带权均值，用于只在 box 外区域统计抑制损失。"""
    return (value * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _box_projection_loss(logits: torch.Tensor, box_prior: torch.Tensor) -> torch.Tensor:
    """box-only 协议下的投影损失。

    思路是约束预测在行投影和列投影上至少覆盖 box 指定的目标范围，
    而不是粗暴把整个矩形内部都当成正样本。
    """
    probs = torch.sigmoid(logits)
    row_target = box_prior.amax(dim=-1)
    col_target = box_prior.amax(dim=-2)
    row_pred = probs.amax(dim=-1).clamp(1e-4, 1.0 - 1e-4)
    col_pred = probs.amax(dim=-2).clamp(1e-4, 1.0 - 1e-4)
    row_loss = F.binary_cross_entropy(row_pred, row_target)
    col_loss = F.binary_cross_entropy(col_pred, col_target)
    return row_loss + col_loss


def _box_outside_loss(logits: torch.Tensor, box_prior: torch.Tensor) -> torch.Tensor:
    """box-only 协议下的外部抑制损失。

    约束模型不要在 box 外大面积激活，从而给弱监督一个最基本的形状边界。
    """
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
    """根据当前样本是否有 mask，自动选择训练损失。

    - 有 mask：BCE + Dice
    - 无 mask：box projection + outside suppression
    """
    if target is not None:
        loss = config.lambda_bce * _balanced_bce_with_logits(logits, target)
        return loss + config.lambda_dice * _dice_loss(logits, target)
    loss = config.lambda_box_projection * _box_projection_loss(logits, box_prior)
    return loss + config.lambda_box_outside * _box_outside_loss(logits, box_prior)


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """二值 mask 的 IoU，用于伪标签质量评分。"""
    a = mask_a > 0.5
    b = mask_b > 0.5
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 1.0
    intersection = float(np.logical_and(a, b).sum())
    return intersection / union


def _unwrap_module(module):
    """兼容普通 module 与 DDP 包装 module 的取底层逻辑。"""
    return module.module if hasattr(module, "module") else module


class SAM2Teacher:
    """SAM2 teacher 包装器。

    这个类负责：
    1. 动态把本地 SAM2 仓库加入 `sys.path`
    2. 初始化 Hydra / SAM2 模型
    3. 提供统一的 `predict(image, bbox)` 接口
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if str(config.sam2_repo) not in sys.path:
            # 使用本地 checkout，而不是要求 benchmark 工程把 SAM2 作为子包内置。
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
            # SAM2 的配置系统基于 Hydra，这里只在首次初始化时注册一次。
            initialize_config_module(config_module="sam2_configs", version_base=None)

        try:
            self.model = build_sam2(config.sam2_cfg, str(config.sam2_ckpt), device=config.device)
        except TypeError:
            # 兼容不同版本 SAM2 构建接口。
            self.model = build_sam2(config.sam2_cfg, str(config.sam2_ckpt), device=config.device, mode="eval")
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)

    def predict(self, image_rgb: np.ndarray, bbox: Sequence[float]):
        """对单张图和单个 box prompt 做一次 SAM2 推理。"""
        with torch.no_grad():
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(
                box=np.array(bbox, dtype=np.float32),
                multimask_output=False,
                return_logits=True,
            )
        return masks[0].astype(np.float32), float(scores[0]), logits[0].astype(np.float32)


class BaseMethod:
    """所有 benchmark 方法的统一接口。"""

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

    def configure_distributed(self, config: ExperimentConfig) -> None:
        """允许方法在训练前做 DDP 包装；非训练方法默认不需要。"""
        return None


class ZeroShotSAM2BoxPromptIR(BaseMethod):
    """零样本 SAM2 基线。"""

    def __init__(self, teacher: SAM2Teacher):
        self.teacher = teacher

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        pred_mask, _, _ = self.teacher.predict(item["image_rgb"], item["bbox"])
        return pred_mask


class BBoxRectMaskBaseline(BaseMethod):
    """最简单的矩形框 baseline：直接把 canonical bbox 当成预测 mask。"""

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        height, width = item["image_rgb"].shape[:2]
        return build_box_prior(item["bbox"], height, width)


class TrainableMethod(BaseMethod):
    """所有可训练方法的公共基类。"""

    is_trainable = True

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def parameters(self):
        raise NotImplementedError

    def build_optimizer(self, config: ExperimentConfig):
        # 当前统一采用 AdamW，具体学习率由子类给出。
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """在多卡模式下把模块包成 DDP。"""
        if not self.config.distributed:
            return module
        if isinstance(module, DistributedDataParallel):
            return module
        return DistributedDataParallel(
            module,
            device_ids=[self.config.local_rank] if self.config.device.type == "cuda" else None,
            output_device=self.config.local_rank if self.config.device.type == "cuda" else None,
            find_unused_parameters=False,
        )


class CleanBoxPEFTSAM2Adapter(TrainableMethod):
    learning_rate: float
    weight_decay: float

    def __init__(self, teacher: SAM2Teacher, config: ExperimentConfig, mode: str = "clean"):
        super().__init__(config)
        self.teacher = teacher
        self.mode = mode
        # adapter 学的是 teacher logits 的残差修正，而不是重训整个分割器。
        self.adapter = PromptConditionedMaskAdapter().to(config.device)
        self.learning_rate = config.lr_teacher_adapter
        self.weight_decay = config.weight_decay
        self.rng = random.Random(0)

    def parameters(self):
        return [p for p in self.adapter.parameters() if p.requires_grad]

    def state_dict(self):
        return _unwrap_module(self.adapter).state_dict()

    def load_state_dict(self, state) -> None:
        _unwrap_module(self.adapter).load_state_dict(state)

    def set_train(self) -> None:
        self.adapter.train()

    def set_eval(self) -> None:
        self.adapter.eval()

    def configure_distributed(self, config: ExperimentConfig) -> None:
        self.adapter = self._wrap_module(_unwrap_module(self.adapter))

    def sample_prompt_box(self, bbox: Sequence[float], image_shape, rng: random.Random):
        # clean 版本直接使用 canonical bbox，不引入额外扰动。
        return clamp_box(bbox, image_shape[1], image_shape[0])

    def _forward_single(self, image_rgb: np.ndarray, prompt_box: Sequence[float], target_shape) -> torch.Tensor:
        # 先用 SAM2 得到 low-res logits，再交给 adapter 做带 box 先验的细化。
        _, _, lowres_logits = self.teacher.predict(image_rgb, prompt_box)
        teacher_logits = torch.from_numpy(lowres_logits[None, None]).float().to(self.config.device)
        teacher_logits = F.interpolate(teacher_logits, size=target_shape, mode="bilinear", align_corners=False)
        box_prior = torch.from_numpy(build_box_prior(prompt_box, target_shape[0], target_shape[1])[None, None]).float().to(self.config.device)
        logits = self.adapter(teacher_logits, box_prior)
        if logits.shape[-2:] != tuple(target_shape):
            logits = F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        """逐样本训练 teacher adapter。

        为什么这里不用整 batch 一次过？
        因为 SAM2 teacher 推理仍以单图单框接口为主，逐样本更直接也更稳。
        """
        losses: List[torch.Tensor] = []
        sample_weights = batch["sample_weights"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)

        for idx, image_rgb in enumerate(batch["image_rgb"]):
            # box_only 协议下 target 会是 None，此时自动退化到 box-only loss。
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            clean_box = batch["bboxes"][idx].cpu().numpy()
            prompt_box = self.sample_prompt_box(clean_box, image_rgb.shape[:2], self.rng)
            target_shape = target.shape[-2:] if target is not None else image_rgb.shape[:2]
            logits = self._forward_single(image_rgb, prompt_box, target_shape)
            box_prior = torch.from_numpy(build_box_prior(clean_box, target_shape[0], target_shape[1])[None, None]).float().to(config.device)
            loss = _sample_training_loss(logits, target, box_prior, config)
            losses.append(loss * sample_weights[idx])

        # 当前 batch_size 往往很小，所以直接对逐样本 loss 求均值即可。
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, item: Dict[str, np.ndarray]) -> np.ndarray:
        # 测试期固定使用 canonical bbox prompt，确保协议一致。
        prompt_box = clamp_box(item["bbox"], item["image_rgb"].shape[1], item["image_rgb"].shape[0])
        logits = self._forward_single(item["image_rgb"], prompt_box, item["image_rgb"].shape[:2])
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


class NoisyBoxPromptRobustSAM2Adapter(CleanBoxPEFTSAM2Adapter):
    """加入 prompt 扰动和一致性约束的鲁棒版本。"""

    def __init__(self, teacher: SAM2Teacher, config: ExperimentConfig, mode: str = "full"):
        super().__init__(teacher, config, mode=mode)

    def sample_prompt_box(self, bbox: Sequence[float], image_shape, rng: random.Random):
        """根据模式生成 prompt 扰动。

        - clean_only: 不做任何扰动
        - jitter_only: 只做平移抖动
        - full: jitter 与 offset/truncation 混合
        """
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
        """训练时同时看 noisy prompt 与 clean prompt。

        full / jitter 模式下还会增加 clean-noisy 之间的一致性约束。
        """
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
                # 一致性项鼓励模型对 prompt 扰动更稳，而不是只记住单一框位置。
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
    """基于 adapter 的伪标签自训练方法。"""

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
        """对伪标签质量打分。

        当前综合考虑：
        1. 伪标签内部平均置信度
        2. 与 teacher mask 的一致性
        3. 与 box 的紧致程度
        4. teacher 自身打分
        """
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
        """在未标注池上生成伪标签样本。"""
        pseudo_samples: List[Sample] = []
        for sample in unlabeled_samples:
            image_rgb = load_ir_image(sample.image_path)
            pseudo_prob = self.predict({"image_rgb": image_rgb, "bbox": sample.bbox})
            pseudo_mask = (pseudo_prob > 0.5).astype(np.float32)
            teacher_mask, teacher_score, _ = self.teacher.predict(image_rgb, sample.bbox)
            quality = self.score_pseudo_mask_quality(sample, pseudo_mask, pseudo_prob, teacher_mask, teacher_score)
            if self.quality_filter:
                # 开启过滤时，必须同时过质量阈值与 teacher 分数阈值。
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
    """直接训练的 SegFormer 控制组。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = SegFormerWrapper().to(config.device)
        self.learning_rate = config.lr_segformer
        self.weight_decay = config.weight_decay

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self):
        return _unwrap_module(self.model).state_dict()

    def load_state_dict(self, state) -> None:
        _unwrap_module(self.model).load_state_dict(state)

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    def configure_distributed(self, config: ExperimentConfig) -> None:
        self.model = self._wrap_module(_unwrap_module(self.model))

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        """direct-train 路径也兼容 mask 与 box-only 两种损失。"""
        images = batch["images"].to(config.device)
        masks = batch["masks"].to(config.device)
        has_masks = batch["has_masks"].to(config.device)
        sample_weights = batch["sample_weights"].to(config.device)
        bboxes = batch["bboxes"].to(config.device)
        logits = self.model(images)
        losses: List[torch.Tensor] = []
        for idx in range(logits.shape[0]):
            target = masks[idx : idx + 1] if bool(has_masks[idx].item()) else None
            # 即使是 direct baseline，也统一复用 box-only 弱监督损失。
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
    """直接训练的 PIDNet-S 控制组。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = TinyPIDNetS().to(config.device)
        self.learning_rate = config.lr_pidnet
        self.weight_decay = config.weight_decay

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self):
        return _unwrap_module(self.model).state_dict()

    def load_state_dict(self, state) -> None:
        _unwrap_module(self.model).load_state_dict(state)

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    def configure_distributed(self, config: ExperimentConfig) -> None:
        self.model = self._wrap_module(_unwrap_module(self.model))

    def training_step(self, batch: Dict[str, torch.Tensor], optimizer, config: ExperimentConfig) -> float:
        """逻辑与 SegFormer 控制组一致，只是骨干网络不同。"""
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
    """返回方法名到构造函数的底层映射。

    上层 `method_registry.py` 会在这个映射基础上补充 benchmark 语义信息。
    """
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
