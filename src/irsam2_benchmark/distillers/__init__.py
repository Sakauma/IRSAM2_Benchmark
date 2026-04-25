from __future__ import annotations

import torch
from torch import Tensor, nn

from ..registry import Registry


class NoDistiller(nn.Module):
    def forward(self, student: Tensor, teacher: Tensor | None = None) -> Tensor:
        del student, teacher
        return torch.tensor(0.0)


DistillerFactory: Registry[nn.Module] = Registry("distiller")
DistillerFactory.register("none")(NoDistiller)
DistillerFactory.register("identity")(NoDistiller)

__all__ = ["DistillerFactory", "NoDistiller"]

