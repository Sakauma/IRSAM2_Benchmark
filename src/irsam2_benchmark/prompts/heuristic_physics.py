from __future__ import annotations

from typing import Dict

import numpy as np
from torch import Tensor

from ..data.prompt_synthesis import connected_components, expand_box_xyxy, mask_to_point_prompt, mask_to_tight_box
from .base import PromptGenerator


class HeuristicPhysicsPrompt(PromptGenerator):
    def __init__(
        self,
        type: str = "box+point",
        percentile: float = 99.5,
        top_k: int = 1,
        min_component_area: int = 2,
        box_pad_ratio: float = 0.25,
        **_: object,
    ):
        super().__init__()
        self.prompt_type = type
        self.percentile = float(percentile)
        self.top_k = max(1, int(top_k))
        self.min_component_area = max(1, int(min_component_area))
        self.box_pad_ratio = float(box_pad_ratio)

    def forward(self, image: Tensor, prior_maps: Dict[str, Tensor]) -> Dict[str, object]:
        del image
        fused = prior_maps.get("fused")
        if fused is None:
            raise RuntimeError("HeuristicPhysicsPrompt requires a 'fused' prior map.")
        heatmap = fused.detach().float().cpu().numpy()
        while heatmap.ndim > 2:
            heatmap = heatmap[0]
        height, width = heatmap.shape
        threshold = float(np.percentile(heatmap, self.percentile))
        candidate = (heatmap >= threshold).astype(np.float32)
        components = [
            component
            for component in connected_components(candidate)
            if int((component > 0.5).sum()) >= self.min_component_area
        ]

        if not components:
            y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            fallback_box = expand_box_xyxy(
                [x - 1, y - 1, x + 2, y + 2],
                width=width,
                height=height,
                pad_ratio=self.box_pad_ratio,
                max_side_multiplier=None,
            )
            return {
                "type": self.prompt_type,
                "box": fallback_box if "box" in self.prompt_type else None,
                "point": [float(x), float(y)] if "point" in self.prompt_type else None,
                "heatmap": heatmap,
                "score": float(heatmap[y, x]),
            }

        ranked = sorted(components, key=lambda item: float((item * heatmap).sum()), reverse=True)
        merged = np.zeros_like(heatmap, dtype=np.float32)
        for component in ranked[: self.top_k]:
            merged = np.maximum(merged, component.astype(np.float32))
        tight = mask_to_tight_box(merged)
        box = expand_box_xyxy(
            tight,
            width=width,
            height=height,
            pad_ratio=self.box_pad_ratio,
            max_side_multiplier=None,
        )
        point = mask_to_point_prompt(merged)
        return {
            "type": self.prompt_type,
            "box": box if "box" in self.prompt_type else None,
            "point": point if "point" in self.prompt_type else None,
            "heatmap": heatmap,
            "score": float((merged * heatmap).sum() / max(float(merged.sum()), 1.0)),
        }
