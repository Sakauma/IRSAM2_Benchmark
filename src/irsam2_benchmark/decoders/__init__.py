from __future__ import annotations

from torch import Tensor, nn

from ..registry import Registry


class SAM2MaskDecoderPlaceholder(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return features


DecoderFactory: Registry[nn.Module] = Registry("decoder")
DecoderFactory.register("sam2_mask_decoder")(SAM2MaskDecoderPlaceholder)
DecoderFactory.register("identity")(SAM2MaskDecoderPlaceholder)

__all__ = ["DecoderFactory", "SAM2MaskDecoderPlaceholder"]

