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
    input_channels: int = 3
    hidden_channels: int = 16
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


def _topk_candidates(score: np.ndarray, *, top_k: int, threshold: float, nms_radius: int) -> list[tuple[int, int, float]]:
    work = np.asarray(score, dtype=np.float32)
    h, w = work.shape
    suppressed = work.copy()
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
    candidates = _topk_candidates(objectness, top_k=max(1, int(top_k)), threshold=float(response_threshold), nms_radius=int(nms_radius))
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
        "candidate_rank": 0,
        "candidate_count": len(candidates),
        "candidate_top_k": int(top_k),
        "candidate_nms_radius": int(nms_radius),
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

    return IRPromptNetSmall()


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
    torch.save(
        {
            "protocol": LEARNED_IR_AUTO_PROMPT_PROTOCOL,
            "model": "IRPromptNetSmall",
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
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
    )
