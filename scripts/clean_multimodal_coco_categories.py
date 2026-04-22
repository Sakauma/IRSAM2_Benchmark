#!/usr/bin/env python
"""清洗 MultiModalCOCO 的类别体系。

目标是把原始较脏的类别名映射到 RBGT-Tiny 风格的目标集合，
为后续跨数据集 benchmark 对齐协议。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TARGET_CATEGORIES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
TARGET_CATEGORY_IDS = {name: idx + 1 for idx, name in enumerate(TARGET_CATEGORIES)}


def parse_args() -> argparse.Namespace:
    """解析类别清洗脚本参数。"""
    parser = argparse.ArgumentParser(
        description="Clean MultiModal COCO categories and map them onto the RBGT-Tiny target taxonomy."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Path to the exported MultiModalCOCO dataset root.")
    parser.add_argument("--dst-root", type=Path, required=True, help="Path to the cleaned COCO dataset root.")
    parser.add_argument(
        "--image-mode",
        choices=["copy", "skip"],
        default="copy",
        help="Whether to copy referenced images into dst-root/image. Default: copy.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    return parser.parse_args()


def tokenize(name: str) -> List[str]:
    """把类别名拆成小写 token，便于做规则映射。"""
    return re.findall(r"[a-z]+", name.lower())


def map_category(name: str) -> Optional[str]:
    """把原始类别名映射到目标类别。

    返回 None 表示该类别被过滤掉。
    """
    tokens = tokenize(name)
    token_set = set(tokens)

    if "drone" in token_set:
        return "drone"
    if "airplane" in token_set or "plane" in token_set:
        return "plane"
    if "ship" in token_set or "boat" in token_set or "boats" in token_set or "vessel" in token_set:
        return "ship"
    if "bus" in token_set or "buses" in token_set:
        return "bus"
    if "cyclist" in token_set or "cyclists" in token_set:
        return "cyclist"
    if "person" in token_set or "people" in token_set or "pedestrian" in token_set or "pedestrians" in token_set:
        if "no" in token_set:
            return None
        return "pedestrian"
    if "car" in token_set or "cars" in token_set:
        return "car"
    return None


def load_coco(path: Path) -> Dict:
    """读取目录下唯一的 COCO 标注文件。"""
    ann_files = sorted((path / "annotations_coco").glob("*.json"))
    if not ann_files:
        raise RuntimeError(f"No COCO annotations found under {path / 'annotations_coco'}")
    if len(ann_files) > 1:
        raise RuntimeError(f"Expected exactly one annotation json under {path / 'annotations_coco'}, found {len(ann_files)}")
    return json.loads(ann_files[0].read_text(encoding="utf8"))


def copy_images(src_root: Path, dst_root: Path, file_names: List[str], overwrite: bool) -> None:
    """把清洗后仍被引用的图像复制到目标目录。"""
    src_img_root = src_root / "image"
    dst_img_root = dst_root / "image"
    dst_img_root.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        src = src_img_root / file_name
        dst = dst_img_root / file_name
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    src_root = args.src_root
    dst_root = args.dst_root
    annotations_dir = dst_root / "annotations_coco"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    output_json = annotations_dir / "instances_multimodal_clean_train2017.json"
    report_json = annotations_dir / "category_cleaning_report.json"
    if (output_json.exists() or report_json.exists()) and not args.overwrite:
        raise RuntimeError(f"Output already exists under {annotations_dir}. Pass --overwrite to replace it.")

    data = load_coco(src_root)
    image_by_id = {image["id"]: image for image in data.get("images", [])}
    raw_name_by_id = {category["id"]: category["name"] for category in data.get("categories", [])}

    kept_annotations = []
    kept_image_ids = set()
    mapped_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()
    raw_to_clean: Dict[str, str] = {}

    next_ann_id = 1
    for ann in data.get("annotations", []):
        raw_name = raw_name_by_id.get(ann["category_id"], "unknown")
        clean_name = map_category(raw_name)
        if clean_name is None:
            # 无法可靠映射到目标体系的类别直接丢弃。
            dropped_counter[raw_name] += 1
            continue
        raw_to_clean[raw_name] = clean_name
        mapped_counter[clean_name] += 1
        kept_image_ids.add(ann["image_id"])

        cleaned_ann = dict(ann)
        cleaned_ann["id"] = next_ann_id
        cleaned_ann["category_id"] = TARGET_CATEGORY_IDS[clean_name]
        kept_annotations.append(cleaned_ann)
        next_ann_id += 1

    kept_images = [image_by_id[image_id] for image_id in sorted(kept_image_ids)]
    kept_file_names = [image["file_name"] for image in kept_images]

    if args.image_mode == "copy":
        copy_images(src_root, dst_root, kept_file_names, overwrite=args.overwrite)

    cleaned_payload = {
        "info": {"description": "Cleaned MultiModal COCO aligned to RBGT-Tiny taxonomy", "version": "1.0"},
        "licenses": data.get("licenses", []),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": [{"id": TARGET_CATEGORY_IDS[name], "name": name} for name in TARGET_CATEGORIES],
    }
    output_json.write_text(json.dumps(cleaned_payload, indent=2, ensure_ascii=False), encoding="utf8")

    report = {
        "source_dataset": str(src_root),
        "output_dataset": str(dst_root),
        "kept_images": len(kept_images),
        "kept_annotations": len(kept_annotations),
        "target_categories": TARGET_CATEGORIES,
        "mapped_annotation_counts": dict(mapped_counter),
        "mapped_raw_categories": dict(sorted(raw_to_clean.items())),
        "top_dropped_categories": dropped_counter.most_common(100),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"kept_images={len(kept_images)}")
    print(f"kept_annotations={len(kept_annotations)}")
    print(f"mapped_annotation_counts={dict(mapped_counter)}")
    print(f"output_json={output_json}")
    print(f"report_json={report_json}")


if __name__ == "__main__":
    main()
