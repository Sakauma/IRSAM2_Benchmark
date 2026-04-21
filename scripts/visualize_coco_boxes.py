#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageColor, ImageDraw


BOX_COLORS = [
    "#ff595e",
    "#1982c4",
    "#8ac926",
    "#ffca3a",
    "#6a4c93",
    "#ff924c",
    "#00b4d8",
    "#b5179e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize COCO annotations with tight and loose bounding boxes."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="COCO-style dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for rendered outputs.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on the number of images to render. Default: 0 (render all).",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=None,
        help="Optional explicit annotation JSON path. Defaults to the first instances_*.json under annotations_coco/.",
    )
    return parser.parse_args()


def resolve_annotation_json(src_root: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    annotation_dir = src_root / "annotations_coco"
    candidates = sorted(annotation_dir.glob("instances_*_train2017.json"))
    if not candidates:
        candidates = sorted(annotation_dir.glob("*.json"))
    if not candidates:
        raise RuntimeError(f"No annotation JSON found under {annotation_dir}.")
    return candidates[0]


def color_for_category(category_id: int) -> Tuple[int, int, int]:
    return ImageColor.getrgb(BOX_COLORS[category_id % len(BOX_COLORS)])


def segmentation_to_xy(segmentation: Sequence[float]) -> List[Tuple[float, float]]:
    return [(float(segmentation[idx]), float(segmentation[idx + 1])) for idx in range(0, len(segmentation), 2)]


def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def draw_annotation_panel(
    image_rgb: Image.Image,
    annotations: List[Dict],
    category_names: Dict[int, str],
    box_key: str,
    title: str,
) -> Image.Image:
    canvas = image_rgb.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_draw = ImageDraw.Draw(canvas)
    for ann in annotations:
        category_id = int(ann["category_id"])
        color = color_for_category(category_id)
        fill = (*color, 55)
        outline = (*color, 230)
        for polygon in ann.get("segmentation", []):
            if len(polygon) >= 6:
                draw.polygon(segmentation_to_xy(polygon), fill=fill, outline=outline)
        box_xywh = ann.get(box_key) or ann.get("bbox")
        if box_xywh is None:
            continue
        x1, y1, x2, y2 = xywh_to_xyxy(box_xywh)
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
        label = category_names.get(category_id, str(category_id))
        label_draw.text((x1 + 2, max(0.0, y1 - 12.0)), label, fill=outline)
    merged = Image.alpha_composite(canvas, overlay).convert("RGB")
    title_band = Image.new("RGB", (merged.width, 24), color=(18, 18, 18))
    title_draw = ImageDraw.Draw(title_band)
    title_draw.text((8, 5), title, fill=(240, 240, 240))
    stacked = Image.new("RGB", (merged.width, merged.height + title_band.height), color=(0, 0, 0))
    stacked.paste(title_band, (0, 0))
    stacked.paste(merged, (0, title_band.height))
    return stacked


def main() -> None:
    args = parse_args()
    annotation_json = resolve_annotation_json(args.src_root, args.annotation_json)
    data = json.loads(annotation_json.read_text(encoding="utf8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    categories = {int(item["id"]): item.get("name", str(item["id"])) for item in data.get("categories", [])}

    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    tight_dir = args.output_root / "tight"
    loose_dir = args.output_root / "loose"
    compare_dir = args.output_root / "comparison"
    tight_dir.mkdir(parents=True, exist_ok=True)
    loose_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for image_id in sorted(anns_by_image):
        if args.max_images > 0 and rendered >= args.max_images:
            break
        image_info = images.get(image_id)
        if image_info is None:
            continue
        image_path = args.src_root / "image" / image_info["file_name"]
        if not image_path.exists():
            continue
        annotations = anns_by_image[image_id]
        image_rgb = Image.open(image_path).convert("RGB")
        tight_panel = draw_annotation_panel(
            image_rgb=image_rgb,
            annotations=annotations,
            category_names=categories,
            box_key="bbox_tight",
            title="tight bbox",
        )
        loose_panel = draw_annotation_panel(
            image_rgb=image_rgb,
            annotations=annotations,
            category_names=categories,
            box_key="bbox_loose",
            title="loose bbox",
        )
        comparison = Image.new("RGB", (tight_panel.width * 2, max(tight_panel.height, loose_panel.height)), color=(0, 0, 0))
        comparison.paste(tight_panel, (0, 0))
        comparison.paste(loose_panel, (tight_panel.width, 0))

        file_stem = Path(image_info["file_name"]).stem
        tight_panel.save(tight_dir / f"{file_stem}.jpg", quality=95)
        loose_panel.save(loose_dir / f"{file_stem}.jpg", quality=95)
        comparison.save(compare_dir / f"{file_stem}.jpg", quality=95)
        rendered += 1

    print(f"annotation_json={annotation_json}")
    print(f"rendered_images={rendered}")
    print(f"tight_dir={tight_dir}")
    print(f"loose_dir={loose_dir}")
    print(f"comparison_dir={compare_dir}")


if __name__ == "__main__":
    main()
