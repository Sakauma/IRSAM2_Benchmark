from __future__ import annotations

from torch import Tensor, nn

from ..registry import Registry


class IdentityAdapter(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return features


AdapterFactory: Registry[nn.Module] = Registry("adapter")
AdapterFactory.register("identity")(IdentityAdapter)

__all__ = ["AdapterFactory", "IdentityAdapter"]

