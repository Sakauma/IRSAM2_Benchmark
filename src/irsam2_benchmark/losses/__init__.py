from __future__ import annotations

import torch
from torch import Tensor, nn

from ..registry import Registry


class NoLoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor | None = None) -> Tensor:
        del prediction, target
        return torch.tensor(0.0)


LossFactory: Registry[nn.Module] = Registry("loss")
LossFactory.register("none")(NoLoss)
LossFactory.register("identity")(NoLoss)

__all__ = ["LossFactory", "NoLoss"]

