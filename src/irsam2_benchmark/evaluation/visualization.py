"""Benchmark visualization helpers.
Author: Egor Izmaylov

These helpers save lightweight PNG overlays so every baseline run leaves behind
human-readable qualitative results in addition to JSON metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw

from ..core.interfaces import InferenceMode
from ..data.sample import Sample
from ..models.sam2_adapter import load_image_rgb


def _to_pil(image_rgb: np.ndarray) -> Image.Image:
    clipped = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    base = image_rgb.astype(np.float32).copy()
    binary = (np.asarray(mask, dtype=np.float32) > 0.5)[..., None]
    tint = np.zeros_like(base)
    tint[..., 0] = float(color[0])
    tint[..., 1] = float(color[1])
    tint[..., 2] = float(color[2])
    blended = np.where(binary > 0, base * (1.0 - alpha) + tint * alpha, base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _draw_box(image: Image.Image, box: list[float] | None, color: tuple[int, int, int], width: int = 3) -> None:
    if box is None:
        return
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_point(image: Image.Image, point: list[float] | None, color: tuple[int, int, int], radius: int = 5) -> None:
    if point is None:
        return
    draw = ImageDraw.Draw(image)
    x, y = [float(v) for v in point]
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, fill=color)


def _stack_panels(panels: Iterable[Image.Image]) -> Image.Image:
    panel_list = list(panels)
    if not panel_list:
        raise ValueError("At least one panel is required.")
    width = sum(panel.width for panel in panel_list)
    height = max(panel.height for panel in panel_list)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    cursor = 0
    for panel in panel_list:
        canvas.paste(panel, (cursor, 0))
        cursor += panel.width
    return canvas


def _union_instances(instances: List[Dict[str, Any]], height: int, width: int) -> np.ndarray:
    union: np.ndarray = np.zeros((height, width), dtype=np.float32)
    for item in instances:
        union = np.maximum(union, np.asarray(item["mask"], dtype=np.float32))
    return union


def save_visualizations(
    *,
    output_dir: Path,
    visual_records: List[Dict[str, Any]],
    inference_mode: InferenceMode,
    method_name: str,
    seed: int,
) -> List[Path]:
    if not visual_records:
        return []

    visual_dir = output_dir / "visuals" / method_name / f"seed_{seed}"
    visual_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for idx, record in enumerate(visual_records):
        sample: Sample = record["sample"]
        image_rgb = load_image_rgb(sample.image_path)
        base_panel = _to_pil(image_rgb)

        if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
            gt_instances = record.get("gt_instances", [])
            pred_instances = record.get("pred_instances", [])
            gt_union = _union_instances(gt_instances, sample.height, sample.width) if gt_instances else np.zeros((sample.height, sample.width), dtype=np.float32)
            pred_union = _union_instances(pred_instances, sample.height, sample.width) if pred_instances else np.zeros((sample.height, sample.width), dtype=np.float32)
            gt_panel = _to_pil(_overlay_mask(image_rgb, gt_union, color=(0, 255, 0)))
            pred_panel = _to_pil(_overlay_mask(image_rgb, pred_union, color=(255, 64, 64)))
            composite = _stack_panels([base_panel, gt_panel, pred_panel])
        else:
            pred_mask = np.asarray(record["pred_mask"], dtype=np.float32)
            gt_mask = np.asarray(record["gt_mask"], dtype=np.float32)
            gt_panel = _to_pil(_overlay_mask(image_rgb, gt_mask, color=(0, 255, 0)))
            pred_panel = _to_pil(_overlay_mask(image_rgb, pred_mask, color=(255, 64, 64)))
            _draw_box(pred_panel, sample.bbox_loose or sample.bbox_tight, color=(64, 160, 255))
            _draw_point(pred_panel, sample.point_prompt, color=(255, 255, 0))
            composite = _stack_panels([base_panel, gt_panel, pred_panel])

        safe_id = sample.sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        path = visual_dir / f"{idx:03d}_{safe_id}.png"
        composite.save(path)
        saved_paths.append(path)
    return saved_paths
