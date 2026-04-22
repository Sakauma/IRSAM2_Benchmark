#!/usr/bin/env python
"""把原始 MultiModal 导出为 COCO 风格数据集。

导出结果同时保留：
1. bbox_tight
2. bbox_loose
3. canonical bbox（当前直接写成 bbox_loose）
4. 原始 segmentation
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    """解析导出脚本参数。"""
    parser = argparse.ArgumentParser(
        description="Export MultiModal annotations into a COCO-style dataset aligned with RBGT-Tiny layout."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Path to the raw MultiModal dataset root.")
    parser.add_argument("--dst-root", type=Path, required=True, help="Path to the exported COCO-style dataset root.")
    parser.add_argument(
        "--image-mode",
        choices=["copy", "skip"],
        default="copy",
        help="Whether to copy images into dst-root/image. Default: copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing annotation JSON and image files.",
    )
    parser.add_argument(
        "--bbox-pad-ratio",
        type=float,
        default=0.15,
        help="Relative padding ratio applied to tight boxes when generating loose boxes. Default: 0.15.",
    )
    parser.add_argument(
        "--bbox-min-pad",
        type=float,
        default=2.0,
        help="Minimum pixel padding applied on each side when generating loose boxes. Default: 2.",
    )
    parser.add_argument(
        "--bbox-min-side",
        type=float,
        default=12.0,
        help="Minimum loose-box side length in pixels after expansion. Default: 12.",
    )
    return parser.parse_args()


def polygon_bbox(points: Sequence[float]) -> List[float]:
    """根据 polygon 计算 tight bbox（xywh）。"""
    xs = [float(points[idx]) for idx in range(0, len(points), 2)]
    ys = [float(points[idx + 1]) for idx in range(0, len(points), 2)]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def polygon_area(points: Sequence[float]) -> float:
    """用鞋带公式估计 polygon 面积。"""
    coords = [(float(points[idx]), float(points[idx + 1])) for idx in range(0, len(points), 2)]
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(coords)):
        x1, y1 = coords[idx]
        x2, y2 = coords[(idx + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def resolve_image_path(img_dir: Path, stem: str) -> Optional[Path]:
    """按常见后缀寻找与标注同 stem 的图像文件。"""
    for suffix in (".bmp", ".png", ".jpg", ".jpeg"):
        candidate = img_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def clip_xyxy(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def build_loose_box(
    tight_box_xywh: Sequence[float],
    width: int,
    height: int,
    pad_ratio: float,
    min_pad: float,
    min_side: float,
) -> List[float]:
    """由 tight bbox 生成 benchmark 协议使用的 loose bbox。"""
    tight_x1, tight_y1, tight_x2, tight_y2 = xywh_to_xyxy(tight_box_xywh)
    tight_w = tight_x2 - tight_x1
    tight_h = tight_y2 - tight_y1
    pad_x = max(min_pad, tight_w * pad_ratio)
    pad_y = max(min_pad, tight_h * pad_ratio)

    loose_x1 = tight_x1 - pad_x
    loose_y1 = tight_y1 - pad_y
    loose_x2 = tight_x2 + pad_x
    loose_y2 = tight_y2 + pad_y

    loose_w = loose_x2 - loose_x1
    loose_h = loose_y2 - loose_y1
    if loose_w < min_side:
        half_extra = 0.5 * (min_side - loose_w)
        loose_x1 -= half_extra
        loose_x2 += half_extra
    if loose_h < min_side:
        half_extra = 0.5 * (min_side - loose_h)
        loose_y1 -= half_extra
        loose_y2 += half_extra

    return xyxy_to_xywh(clip_xyxy([loose_x1, loose_y1, loose_x2, loose_y2], width, height))


def load_multimodal_records(
    src_root: Path,
    bbox_pad_ratio: float,
    bbox_min_pad: float,
    bbox_min_side: float,
) -> Tuple[List[Dict], Dict[str, int]]:
    """把原始 MultiModal 样本整理成 COCO 记录。"""
    img_dir = src_root / "img"
    label_dir = src_root / "label"
    if not img_dir.exists() or not label_dir.exists():
        raise RuntimeError(f"Expected MultiModal layout with img/ and label/ under {src_root}")

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories: Dict[str, int] = {}
    image_id_by_stem: Dict[str, int] = {}
    next_image_id = 1
    next_ann_id = 1

    for label_path in sorted(label_dir.glob("*.json")):
        stem = label_path.stem
        image_path = resolve_image_path(img_dir, stem)
        if image_path is None:
            continue

        if stem not in image_id_by_stem:
            with Image.open(image_path) as image:
                width, height = image.size
            image_id_by_stem[stem] = next_image_id
            images.append(
                {
                    "id": next_image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )
            next_image_id += 1

        image_id = image_id_by_stem[stem]
        data = json.loads(label_path.read_text(encoding="utf8"))
        instances = data.get("detection", {}).get("instances", [])
        for inst in instances:
            category_name = inst.get("category", "unknown")
            if category_name not in categories:
                categories[category_name] = len(categories) + 1

            masks = inst.get("mask", [])
            polygon = masks[0] if masks and len(masks[0]) >= 6 else None

            if polygon is not None:
                # 有 polygon 时优先用 polygon 生成 tight bbox，并保留 segmentation。
                bbox_tight = polygon_bbox(polygon)
                area = polygon_area(polygon)
                segmentation = [polygon]
            else:
                raw_bbox = inst.get("bbox")
                if raw_bbox is None or len(raw_bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in raw_bbox]
                bbox_tight = [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
                area = bbox_tight[2] * bbox_tight[3]
                segmentation = []
            bbox_loose = build_loose_box(
                bbox_tight,
                width=width,
                height=height,
                pad_ratio=bbox_pad_ratio,
                min_pad=bbox_min_pad,
                min_side=bbox_min_side,
            )

            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": categories[category_name],
                    # 当前 benchmark 约定 canonical bbox 就是 loose box。
                    "bbox": bbox_loose,
                    "bbox_tight": bbox_tight,
                    "bbox_loose": bbox_loose,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            next_ann_id += 1

    return (
        [
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": cat_id, "name": cat_name} for cat_name, cat_id in sorted(categories.items(), key=lambda item: item[1])],
            }
        ],
        image_id_by_stem,
    )


def export_images(src_root: Path, dst_root: Path, stems: Sequence[str], overwrite: bool) -> None:
    """可选地把原始图像复制到导出目录。"""
    src_img_dir = src_root / "img"
    dst_img_dir = dst_root / "image"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        image_path = resolve_image_path(src_img_dir, stem)
        if image_path is None:
            continue
        destination = dst_img_dir / image_path.name
        if destination.exists() and not overwrite:
            continue
        shutil.copy2(image_path, destination)


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    datasets, image_id_by_stem = load_multimodal_records(
        args.src_root,
        bbox_pad_ratio=args.bbox_pad_ratio,
        bbox_min_pad=args.bbox_min_pad,
        bbox_min_side=args.bbox_min_side,
    )
    dataset = datasets[0]

    annotations_dir = args.dst_root / "annotations_coco"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    output_json = annotations_dir / "instances_multimodal_train2017.json"
    if output_json.exists() and not args.overwrite:
        raise RuntimeError(f"{output_json} already exists. Pass --overwrite to replace it.")

    if args.image_mode == "copy":
        export_images(args.src_root, args.dst_root, sorted(image_id_by_stem), overwrite=args.overwrite)

    payload = {
        "info": {
            "description": "MultiModal exported to COCO-style annotations",
            "version": "1.1",
            "bbox_policy": {
                "canonical_bbox": "bbox_loose",
                "bbox_pad_ratio": args.bbox_pad_ratio,
                "bbox_min_pad": args.bbox_min_pad,
                "bbox_min_side": args.bbox_min_side,
            },
        },
        "licenses": [],
        "images": dataset["images"],
        "annotations": dataset["annotations"],
        "categories": dataset["categories"],
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"exported_images={len(dataset['images'])}")
    print(f"exported_annotations={len(dataset['annotations'])}")
    print(f"exported_categories={len(dataset['categories'])}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
