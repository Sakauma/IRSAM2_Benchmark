from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..data.prompt_synthesis import clamp_box_xyxy


LEARNED_IR_AUTO_PROMPT_PROTOCOL = "learned_ir_auto_prompt_v1"


@dataclass(frozen=True)
class AutoPromptModelConfig:
    architecture: str = "small"
    input_channels: int = 3
    hidden_channels: int = 16
    depth: int = 4
    fpn_channels: int = 64
    dropout: float = 0.0
    min_box_side: float = 2.0
    negative_ring_offset: float = 4.0
    use_local_contrast: bool = True
    use_top_hat: bool = True


@dataclass(frozen=True)
class LearnedAutoPrompt:
    point: list[float]
    box: list[float]
    points: list[list[float]]
    point_labels: list[int]
    metadata: dict[str, Any]
    objectness: np.ndarray


def _normalize_map(score: np.ndarray) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    denom = max(1e-6, max_v - min_v)
    return (arr - min_v) / denom


def _box_mean(gray: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    padded = np.pad(gray.astype(np.float32), ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return total / float(size * size)


def _white_tophat(gray: np.ndarray, size: int) -> np.ndarray:
    kernel = max(3, int(size))
    if kernel % 2 == 0:
        kernel += 1
    image = Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8))
    opened = image.filter(ImageFilter.MinFilter(kernel)).filter(ImageFilter.MaxFilter(kernel))
    opened_arr = np.asarray(opened, dtype=np.float32) / 255.0
    return np.maximum(gray - opened_arr, 0.0)


def load_ir_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    return _normalize_map(arr)


def ir_prior_stack(gray: np.ndarray, *, use_local_contrast: bool = True, use_top_hat: bool = True) -> np.ndarray:
    arr = _normalize_map(gray)
    small_mean = _box_mean(arr, radius=2)
    large_mean = _box_mean(arr, radius=8)
    local_contrast = _normalize_map(np.maximum(small_mean - large_mean, 0.0)) if use_local_contrast else np.zeros_like(arr, dtype=np.float32)
    top_hat = (
        _normalize_map(np.maximum.reduce([_white_tophat(arr, size=3), _white_tophat(arr, size=5), _white_tophat(arr, size=9)]))
        if use_top_hat
        else np.zeros_like(arr, dtype=np.float32)
    )
    return np.stack([arr, local_contrast, top_hat], axis=0).astype(np.float32)


def ir_prior_stack_from_path(path: Path, *, use_local_contrast: bool = True, use_top_hat: bool = True) -> np.ndarray:
    return ir_prior_stack(load_ir_gray(path), use_local_contrast=use_local_contrast, use_top_hat=use_top_hat)


def _negative_ring_points(box: list[float], width: int, height: int, offset: float) -> list[list[float]]:
    x1, y1, x2, y2 = [float(value) for value in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    candidates = [
        [cx, y1 - offset],
        [cx, y2 + offset],
        [x1 - offset, cy],
        [x2 + offset, cy],
    ]
    return [
        [float(min(max(0.0, x), max(0.0, width - 1.0))), float(min(max(0.0, y), max(0.0, height - 1.0)))]
        for x, y in candidates
    ]


def _to_numpy_first(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def _border_distance(x: int, y: int, width: int, height: int) -> int:
    left = max(0, int(x))
    top = max(0, int(y))
    right = max(0, int(width) - 1 - int(x))
    bottom = max(0, int(height) - 1 - int(y))
    return int(min(left, top, right, bottom))


def _suppress_border(score: np.ndarray, border_px: int) -> np.ndarray:
    work = np.asarray(score, dtype=np.float32).copy()
    border = max(0, int(border_px))
    if border <= 0:
        return work
    h, w = work.shape
    if h <= 2 * border or w <= 2 * border:
        return work
    work[:border, :] = -1.0
    work[-border:, :] = -1.0
    work[:, :border] = -1.0
    work[:, -border:] = -1.0
    return work


def _topk_candidates(
    score: np.ndarray,
    *,
    top_k: int,
    threshold: float,
    nms_radius: int,
    border_suppression_px: int = 0,
) -> list[tuple[int, int, float]]:
    work = np.asarray(score, dtype=np.float32)
    h, w = work.shape
    suppressed = _suppress_border(work, border_suppression_px)
    candidates: list[tuple[int, int, float]] = []
    for _ in range(max(1, min(int(top_k), int(work.size)))):
        flat_idx = int(np.argmax(suppressed))
        value = float(suppressed.reshape(-1)[flat_idx])
        if candidates and value < threshold:
            break
        y, x = divmod(int(flat_idx), w)
        if 0 <= y < h and 0 <= x < w:
            candidates.append((x, y, value))
            radius = max(0, int(nms_radius))
            y1 = max(0, y - radius)
            y2 = min(h, y + radius + 1)
            x1 = max(0, x - radius)
            x2 = min(w, x + radius + 1)
            suppressed[y1:y2, x1:x2] = -1.0
    if not candidates and work.size:
        flat_idx = int(np.argmax(work))
        y, x = divmod(flat_idx, w)
        candidates.append((x, y, float(work.reshape(-1)[flat_idx])))
    return candidates


def decode_auto_prompt(
    *,
    objectness_logits: Any,
    box_size: Any,
    image_width: int,
    image_height: int,
    confidence_logit: Any | None = None,
    min_box_side: float = 2.0,
    negative_ring: bool = False,
    negative_ring_offset: float = 4.0,
    top_k: int = 1,
    point_budget: int = 1,
    response_threshold: float = 0.0,
    nms_radius: int = 4,
    border_suppression_px: int = 0,
) -> LearnedAutoPrompt:
    logits = _to_numpy_first(objectness_logits)
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Expected objectness logits with shape HxW or 1xHxW, got {logits.shape}.")

    sizes = _to_numpy_first(box_size)
    if sizes.ndim == 3 and sizes.shape[0] == 2:
        size_channels = sizes
    elif sizes.ndim == 3 and sizes.shape[-1] == 2:
        size_channels = np.moveaxis(sizes, -1, 0)
    else:
        raise ValueError(f"Expected box_size with shape 2xHxW or HxWx2, got {sizes.shape}.")

    objectness = 1.0 / (1.0 + np.exp(-logits))
    candidates = _topk_candidates(
        objectness,
        top_k=max(1, int(top_k)),
        threshold=float(response_threshold),
        nms_radius=int(nms_radius),
        border_suppression_px=int(border_suppression_px),
    )
    x, y, primary_score = candidates[0]
    box_w = max(float(min_box_side), float(size_channels[0, y, x]))
    box_h = max(float(min_box_side), float(size_channels[1, y, x]))
    cx = float(x)
    cy = float(y)
    box = clamp_box_xyxy(
        [cx - 0.5 * box_w, cy - 0.5 * box_h, cx + 0.5 * box_w + 1.0, cy + 0.5 * box_h + 1.0],
        width=image_width,
        height=image_height,
    )
    point = [cx, cy]
    positive_candidates = candidates[: max(1, int(point_budget))]
    points = [[float(px), float(py)] for px, py, _ in positive_candidates]
    labels = [1] * len(points)
    negative_points: list[list[float]] = []
    if negative_ring:
        negative_points = _negative_ring_points(box, image_width, image_height, negative_ring_offset)
        points.extend(negative_points)
        labels.extend([0] * len(negative_points))

    confidence = float(primary_score)
    if confidence_logit is not None:
        raw_conf = _to_numpy_first(confidence_logit)
        confidence = float(1.0 / (1.0 + np.exp(-float(raw_conf.reshape(-1)[0]))))
    metadata = {
        "source": "learned_auto_prompt",
        "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
        "point": point,
        "box": box,
        "points": points,
        "point_labels": labels,
        "candidate_score": float(primary_score),
        "candidate_points": [[float(px), float(py), float(score)] for px, py, score in candidates],
        "candidate_rank": 0,
        "candidate_count": len(candidates),
        "candidate_top_k": int(top_k),
        "candidate_nms_radius": int(nms_radius),
        "border_suppression_px": int(border_suppression_px),
        "primary_border_distance_px": _border_distance(int(x), int(y), int(image_width), int(image_height)),
        "positive_point_count": len(positive_candidates),
        "fallback": bool(float(primary_score) < float(response_threshold)),
        "response_threshold": float(response_threshold),
        "prompt_confidence": confidence,
        "negative_point_count": len(negative_points),
        "box_width": float(box[2] - box[0]),
        "box_height": float(box[3] - box[1]),
    }
    return LearnedAutoPrompt(point=point, box=box, points=points, point_labels=labels, metadata=metadata, objectness=objectness.astype(np.float32))


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except Exception as exc:
        raise RuntimeError("learned auto prompt requires PyTorch.") from exc
    return torch, nn, F


def build_ir_prompt_net(config: AutoPromptModelConfig | dict[str, Any] | None = None):
    torch, nn, F = _require_torch()
    cfg = config if isinstance(config, AutoPromptModelConfig) else AutoPromptModelConfig(**(config or {}))

    class IRPromptNetSmall(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(cfg.hidden_channels)
            self.config = cfg
            self.model_name = "IRPromptNetSmall"
            self.features = nn.Sequential(
                nn.Conv2d(int(cfg.input_channels), hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.objectness_head = nn.Conv2d(hidden, 1, kernel_size=1)
            self.box_size_head = nn.Conv2d(hidden, 2, kernel_size=1)
            self.confidence_head = nn.Linear(hidden, 1)

        def forward(self, x):
            feat = self.features(x)
            pooled = feat.mean(dim=(2, 3))
            return {
                "objectness_logits": self.objectness_head(feat),
                "box_size": F.softplus(self.box_size_head(feat)) + float(cfg.min_box_side),
                "confidence_logits": self.confidence_head(pooled),
            }

    class ResidualDepthwiseBlock(nn.Module):
        def __init__(self, channels: int, *, expansion: int = 8) -> None:
            super().__init__()
            expanded = int(channels) * int(expansion)
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GroupNorm(1, channels),
                nn.GELU(),
                nn.Conv2d(channels, expanded, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(expanded, channels, kernel_size=1),
                nn.GroupNorm(1, channels),
            )

        def forward(self, x):
            return x + self.block(x)

    class MultiScaleContext(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False),
                        nn.GroupNorm(1, channels),
                        nn.GELU(),
                        nn.Conv2d(channels, channels, kernel_size=1),
                        nn.GELU(),
                    )
                    for dilation in (1, 2, 4)
                ]
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(channels * len(self.branches), channels, kernel_size=1),
                nn.GroupNorm(1, channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))

    class SEAttention(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            reduced = max(4, channels // 4)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced),
                nn.ReLU(inplace=True),
                nn.Linear(reduced, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            weight = self.fc(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
            return x * weight[:, :, None, None]

    class IRPromptNetV2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(cfg.hidden_channels)
            self.config = cfg
            self.model_name = "IRPromptNetV2"
            self.stem = nn.Sequential(
                nn.Conv2d(int(cfg.input_channels), hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
            )
            self.residual = nn.Sequential(*(ResidualDepthwiseBlock(hidden) for _ in range(4)))
            self.context = MultiScaleContext(hidden)
            self.attention = SEAttention(hidden)
            self.objectness_head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
            self.box_size_head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 2, kernel_size=1),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            feat = self.stem(x)
            feat = self.residual(feat)
            feat = feat + self.context(feat)
            feat = self.attention(feat)
            pooled = feat.mean(dim=(2, 3))
            return {
                "objectness_logits": self.objectness_head(feat),
                "box_size": F.softplus(self.box_size_head(feat)) + float(cfg.min_box_side),
                "confidence_logits": self.confidence_head(pooled),
            }

    class ConvNormAct(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.block(x)

    class IRPromptNetV3FPN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(cfg.hidden_channels)
            fpn_channels = int(cfg.fpn_channels)
            depth = max(2, int(cfg.depth))
            dropout = max(0.0, float(cfg.dropout))
            self.config = cfg
            self.model_name = "IRPromptNetV3FPN"
            self.stem = ConvNormAct(int(cfg.input_channels), hidden)
            self.stage1 = nn.Sequential(*(ResidualDepthwiseBlock(hidden, expansion=4) for _ in range(max(1, depth // 2))))
            self.down2 = ConvNormAct(hidden, hidden * 2, stride=2)
            self.stage2 = nn.Sequential(*(ResidualDepthwiseBlock(hidden * 2, expansion=4) for _ in range(max(1, depth // 2))))
            self.down4 = ConvNormAct(hidden * 2, hidden * 4, stride=2)
            self.stage3 = nn.Sequential(*(ResidualDepthwiseBlock(hidden * 4, expansion=4) for _ in range(max(1, depth // 3))))
            self.lateral1 = nn.Conv2d(hidden, fpn_channels, kernel_size=1)
            self.lateral2 = nn.Conv2d(hidden * 2, fpn_channels, kernel_size=1)
            self.lateral3 = nn.Conv2d(hidden * 4, fpn_channels, kernel_size=1)
            self.fuse = nn.Sequential(
                ConvNormAct(fpn_channels, fpn_channels),
                ResidualDepthwiseBlock(fpn_channels, expansion=4),
                nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.context = MultiScaleContext(fpn_channels)
            self.attention = SEAttention(fpn_channels)
            self.objectness_head = nn.Sequential(
                ConvNormAct(fpn_channels, fpn_channels),
                nn.Conv2d(fpn_channels, 1, kernel_size=1),
            )
            self.box_size_head = nn.Sequential(
                ConvNormAct(fpn_channels, fpn_channels),
                nn.Conv2d(fpn_channels, 2, kernel_size=1),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(fpn_channels, fpn_channels),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(fpn_channels, 1),
            )

        def forward(self, x):
            feat1 = self.stage1(self.stem(x))
            feat2 = self.stage2(self.down2(feat1))
            feat3 = self.stage3(self.down4(feat2))
            target_size = feat1.shape[-2:]
            pyramid = self.lateral1(feat1)
            pyramid = pyramid + F.interpolate(self.lateral2(feat2), size=target_size, mode="bilinear", align_corners=False)
            pyramid = pyramid + F.interpolate(self.lateral3(feat3), size=target_size, mode="bilinear", align_corners=False)
            feat = self.fuse(pyramid)
            feat = feat + self.context(feat)
            feat = self.attention(feat)
            pooled = feat.mean(dim=(2, 3))
            return {
                "objectness_logits": self.objectness_head(feat),
                "box_size": F.softplus(self.box_size_head(feat)) + float(cfg.min_box_side),
                "confidence_logits": self.confidence_head(pooled),
            }

    architecture = str(cfg.architecture).strip().lower()
    if architecture in {"v2", "ir_prompt_v2", "promptnet_v2"}:
        return IRPromptNetV2()
    if architecture in {"v3", "ir_prompt_v3", "ir_prompt_v3_fpn", "promptnet_v3_fpn"}:
        return IRPromptNetV3FPN()
    return IRPromptNetSmall()


def count_auto_prompt_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def save_auto_prompt_checkpoint(
    path: Path,
    model: Any,
    *,
    config: AutoPromptModelConfig | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    torch, _, _ = _require_torch()
    if config is None and isinstance(getattr(model, "config", None), AutoPromptModelConfig):
        cfg = model.config
    else:
        cfg = config if isinstance(config, AutoPromptModelConfig) else AutoPromptModelConfig(**(config or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("architecture", cfg.architecture)
    metadata_payload.setdefault("parameter_count", count_auto_prompt_parameters(model))
    torch.save(
        {
            "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
            "model": str(getattr(model, "model_name", "IRPromptNetSmall")),
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "metadata": metadata_payload,
        },
        path,
    )


def load_auto_prompt_model(checkpoint_path: Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    torch, _, _ = _require_torch()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint.get("config", {})
    cfg = AutoPromptModelConfig(**raw_config)
    model = build_ir_prompt_net(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, {"protocol": checkpoint.get("protocol", LEARNED_IR_AUTO_PROMPT_PROTOCOL), "config": asdict(cfg), "metadata": checkpoint.get("metadata", {})}


def predict_learned_auto_prompt_from_path(
    *,
    model: Any,
    image_path: Path,
    device: str = "cpu",
    negative_ring: bool = False,
    min_box_side: float = 2.0,
    negative_ring_offset: float = 4.0,
    top_k: int = 1,
    point_budget: int = 1,
    response_threshold: float = 0.0,
    nms_radius: int = 4,
    border_suppression_px: int = 0,
    use_local_contrast: bool = True,
    use_top_hat: bool = True,
) -> LearnedAutoPrompt:
    torch, _, _ = _require_torch()
    prior = ir_prior_stack_from_path(image_path, use_local_contrast=use_local_contrast, use_top_hat=use_top_hat)
    with torch.no_grad():
        x = torch.from_numpy(prior[None]).to(device=device, dtype=torch.float32)
        outputs = model(x)
    return decode_auto_prompt(
        objectness_logits=outputs["objectness_logits"],
        box_size=outputs["box_size"],
        confidence_logit=outputs.get("confidence_logits"),
        image_width=int(prior.shape[2]),
        image_height=int(prior.shape[1]),
        min_box_side=min_box_side,
        negative_ring=negative_ring,
        negative_ring_offset=negative_ring_offset,
        top_k=top_k,
        point_budget=point_budget,
        response_threshold=response_threshold,
        nms_radius=nms_radius,
        border_suppression_px=border_suppression_px,
    )
