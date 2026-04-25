from __future__ import annotations

from torch import Tensor, nn

from ..registry import Registry


class NoQuantizer(nn.Module):
    def forward(self, model_or_tensor):
        return model_or_tensor


QuantizerFactory: Registry[nn.Module] = Registry("quantizer")
QuantizerFactory.register("none")(NoQuantizer)
QuantizerFactory.register("identity")(NoQuantizer)

__all__ = ["NoQuantizer", "QuantizerFactory"]

