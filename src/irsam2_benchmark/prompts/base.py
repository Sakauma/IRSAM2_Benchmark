from __future__ import annotations

from typing import Dict

from torch import Tensor, nn


class PromptGenerator(nn.Module):
    def forward(self, image: Tensor, prior_maps: Dict[str, Tensor]) -> Dict[str, object]:
        raise NotImplementedError


class IdentityPrompt(PromptGenerator):
    def forward(self, image: Tensor, prior_maps: Dict[str, Tensor]) -> Dict[str, object]:
        del image, prior_maps
        return {}

