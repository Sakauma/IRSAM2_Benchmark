from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class PriorModule(nn.Module):
    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError


class IdentityPrior(PriorModule):
    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        return {"fused": image}


def normalize_map(value: Tensor) -> Tensor:
    dims = tuple(range(2, value.ndim))
    min_v = value.amin(dim=dims, keepdim=True)
    max_v = value.amax(dim=dims, keepdim=True)
    return (value - min_v) / (max_v - min_v).clamp_min(1e-6)

