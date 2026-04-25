from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import PriorModule, normalize_map


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _avg_pool(image: Tensor, kernel_size: int) -> Tensor:
    kernel = _odd_kernel(kernel_size)
    return F.avg_pool2d(image, kernel_size=kernel, stride=1, padding=kernel // 2)


def _max_pool(image: Tensor, kernel_size: int) -> Tensor:
    kernel = _odd_kernel(kernel_size)
    return F.max_pool2d(image, kernel_size=kernel, stride=1, padding=kernel // 2)


def _min_pool(image: Tensor, kernel_size: int) -> Tensor:
    return -_max_pool(-image, kernel_size)


def local_contrast(image: Tensor, scales: Iterable[int]) -> Tensor:
    responses = []
    for scale in scales:
        background = _avg_pool(image, scale)
        responses.append((image - background).relu())
    return torch.stack(responses, dim=0).amax(dim=0)


def multi_scale_top_hat(image: Tensor, scales: Iterable[int]) -> Tensor:
    responses = []
    for scale in scales:
        opened = _max_pool(_min_pool(image, scale), scale)
        responses.append((image - opened).relu())
    return torch.stack(responses, dim=0).amax(dim=0)


def snr_like(image: Tensor, scales: Iterable[int]) -> Tensor:
    responses = []
    for scale in scales:
        mean = _avg_pool(image, scale)
        mean_sq = _avg_pool(image * image, scale)
        std = (mean_sq - mean * mean).clamp_min(0.0).sqrt()
        responses.append(((image - mean) / std.clamp_min(1e-6)).relu())
    return torch.stack(responses, dim=0).amax(dim=0)


class PriorFusion(PriorModule):
    def __init__(
        self,
        enabled: list[str] | None = None,
        scales: list[int] | None = None,
        weights: Dict[str, float] | None = None,
        **_: object,
    ):
        super().__init__()
        self.enabled = enabled or ["local_contrast", "multi_scale_top_hat", "snr_like"]
        self.scales = scales or [7, 15, 31]
        self.weights = weights or {"local_contrast": 0.4, "multi_scale_top_hat": 0.4, "snr_like": 0.2}

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        outputs: Dict[str, Tensor] = {}
        if "local_contrast" in self.enabled:
            outputs["local_contrast"] = normalize_map(local_contrast(image, self.scales))
        if "multi_scale_top_hat" in self.enabled:
            outputs["multi_scale_top_hat"] = normalize_map(multi_scale_top_hat(image, self.scales))
        if "snr_like" in self.enabled:
            outputs["snr_like"] = normalize_map(snr_like(image, self.scales))
        if not outputs:
            outputs["identity"] = normalize_map(image)

        fused = torch.zeros_like(next(iter(outputs.values())))
        total_weight = 0.0
        for name, response in outputs.items():
            weight = float(self.weights.get(name, 1.0))
            fused = fused + response * weight
            total_weight += weight
        outputs["fused"] = normalize_map(fused / max(total_weight, 1e-6))
        return outputs

