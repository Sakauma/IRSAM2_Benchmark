#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IR_TOKENS = {"01", "ir", "infrared", "thermal"}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True)
class RbgtBoxRecord:
    image_path: Path
    image_rel: str
    xml_path: Path
    sequence_id: str
    width: int
    height: int
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    ann_id: str


@dataclass
class SplitStats:
    image_count: int = 0
    annotation_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    area_buckets: dict[str, int] = field(default_factory=dict)


@dataclass
class ExportStats:
    root: str
    annotations_dir: str
    images_dir: str
    seed: int
    small_target_filter: bool
    xml_files_read: int = 0
    objects_seen: int = 0
    annotations_kept: int = 0
    images_kept: int = 0
    skipped_non_ir: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_box: int = 0
    skipped_small_target_filter: int = 0
    splits: dict[str, SplitStats] = field(default_factory=dict)


def _xml_text(root: ET.Element, path: str, default: str = "") -> str:
    node = root.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _xml_float(root: ET.Element, path: str, default: float = 0.0) -> float:
    text = _xml_text(root, path)
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _path_parts(value: object) -> set[str]:
    return {part.lower() for part in Path(str(value)).as_posix().split("/") if part}


def _is_ir_path(value: object) -> bool:
    return bool(_path_parts(value) & IR_TOKENS)


def _resolve_image_root(root: Path, images_dir: str) -> Path:
    configured = root / images_dir
    if configured.exists():
        return configured
    for candidate_name in ("image", "images"):
        candidate = root / candidate_name
        if candidate.exists():
            return candidate
    return configured


def _existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_image_path(image_root: Path, ann_dir: Path, xml_path: Path, xml_root: ET.Element) -> Path | None:
    for tag in ("resolved_image_path", "path", "filename"):
        value = _xml_text(xml_root, tag)
        if not value:
            continue
        relative = Path(value)
        if relative.is_absolute() and relative.exists():
            return relative
        candidates = [image_root / relative, image_root.parent / relative]
        if len(relative.parts) > 1:
            candidates.extend(image_root / relative.parent / branch / relative.name for branch in IR_TOKENS)
        try:
            xml_relative = xml_path.relative_to(ann_dir)
            candidates.append(image_root / xml_relative.with_name(relative.name))
            if relative.suffix:
                candidates.append(image_root / xml_relative.with_suffix(relative.suffix))
        except ValueError:
            pass
        found = _existing_path(*candidates)
        if found is not None:
            return found
    try:
        base = image_root / xml_path.relative_to(ann_dir).with_suffix("")
    except ValueError:
        base = image_root / xml_path.with_suffix("").name
    for suffix in IMAGE_EXTENSIONS:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _image_size(xml_root: ET.Element, image_path: Path) -> tuple[int, int]:
    width_text = _xml_text(xml_root, "size/width")
    height_text = _xml_text(xml_root, "size/height")
    if width_text and height_text:
        try:
            return int(float(width_text)), int(float(height_text))
        except ValueError:
            pass
    with Image.open(image_path) as image:
        return image.size


def _clip_box(obj: ET.Element, width: int, height: int) -> tuple[float, float, float, float] | None:
    bbox = obj.find("bndbox")
    if bbox is None:
        return None
    x1 = max(0.0, min(float(width), _xml_float(bbox, "xmin")))
    y1 = max(0.0, min(float(height), _xml_float(bbox, "ymin")))
    x2 = max(0.0, min(float(width), _xml_float(bbox, "xmax")))
    y2 = max(0.0, min(float(height), _xml_float(bbox, "ymax")))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _area_bucket(area: float) -> str:
    if area < 16:
        return "tiny_0_15"
    if area < 64:
        return "tiny_16_63"
    if area < 256:
        return "small_64_255"
    if area < 1024:
        return "small_256_1023"
    return "mid_1024_plus"


def _passes_small_target_filter(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    max_area_ratio: float,
    max_box_side: float,
    min_box_side: float,
) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    if min(box_w, box_h) < float(min_box_side):
        return False
    if max(box_w, box_h) > float(max_box_side):
        return False
    return (box_w * box_h) / max(1.0, float(width * height)) <= float(max_area_ratio)


def _sequence_id(image_rel: str) -> str:
    parts = Path(image_rel).parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return Path(image_rel).stem


def _collect_records(
    *,
    root: Path,
    annotations_dir: str,
    images_dir: str,
    small_target_filter: bool,
    max_area_ratio: float,
    max_box_side: float,
    min_box_side: float,
    stats: ExportStats,
) -> list[RbgtBoxRecord]:
    ann_dir = root / annotations_dir
    image_root = _resolve_image_root(root, images_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing RBGT-Tiny annotation directory: {ann_dir}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing RBGT-Tiny image directory: {image_root}")
    records: list[RbgtBoxRecord] = []
    for xml_path in sorted(ann_dir.rglob("*.xml")):
        stats.xml_files_read += 1
        xml_root = ET.parse(xml_path).getroot()
        image_path = _resolve_image_path(image_root, ann_dir, xml_path, xml_root)
        if image_path is None or not image_path.exists():
            stats.skipped_missing_image += 1
            continue
        try:
            image_rel = image_path.relative_to(image_root).as_posix()
        except ValueError:
            image_rel = image_path.name
        if not _is_ir_path(image_rel) and not _is_ir_path(xml_path.relative_to(ann_dir).as_posix()):
            stats.skipped_non_ir += 1
            continue
        width, height = _image_size(xml_root, image_path)
        for obj_idx, obj in enumerate(xml_root.findall("object")):
            stats.objects_seen += 1
            bbox = _clip_box(obj, width, height)
            if bbox is None:
                stats.skipped_invalid_box += 1
                continue
            if small_target_filter and not _passes_small_target_filter(
                bbox_xyxy=bbox,
                width=width,
                height=height,
                max_area_ratio=max_area_ratio,
                max_box_side=max_box_side,
                min_box_side=min_box_side,
            ):
                stats.skipped_small_target_filter += 1
                continue
            x1, y1, x2, y2 = bbox
            area = float(_xml_float(obj, "area", (x2 - x1) * (y2 - y1)))
            ann_id = _xml_text(obj, "coco_annotation_id", str(obj_idx))
            records.append(
                RbgtBoxRecord(
                    image_path=image_path,
                    image_rel=image_rel,
                    xml_path=xml_path,
                    sequence_id=_sequence_id(image_rel),
                    width=width,
                    height=height,
                    category=_xml_text(obj, "name", "unknown") or "unknown",
                    bbox_xyxy=bbox,
                    area=area,
                    ann_id=ann_id,
                )
            )
            stats.annotations_kept += 1
    stats.images_kept = len({record.image_rel for record in records})
    return records


def _split_sequences(records: list[RbgtBoxRecord], *, seed: int) -> dict[str, list[str]]:
    sequences = sorted({record.sequence_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(sequences)
    total = len(sequences)
    train_end = int(round(total * SPLIT_RATIOS["train"]))
    val_end = train_end + int(round(total * SPLIT_RATIOS["val"]))
    return {
        "train": sequences[:train_end],
        "val": sequences[train_end:val_end],
        "test": sequences[val_end:],
    }


def _write_coco_split(*, root: Path, split_name: str, records: list[RbgtBoxRecord]) -> SplitStats:
    output_dir = root / f"annotations_coco_ir_box_m9_{split_name}"
    output_path = output_dir / f"instances_rbgt_tiny_ir_box_{split_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_id_by_rel: dict[str, int] = {}
    category_id_by_name: dict[str, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    stats = SplitStats()
    for record in records:
        if record.image_rel not in image_id_by_rel:
            image_id_by_rel[record.image_rel] = len(image_id_by_rel) + 1
            images.append(
                {
                    "id": image_id_by_rel[record.image_rel],
                    "file_name": record.image_rel,
                    "width": record.width,
                    "height": record.height,
                    "sequence_id": record.sequence_id,
                }
            )
        if record.category not in category_id_by_name:
            category_id_by_name[record.category] = len(category_id_by_name) + 1
        x1, y1, x2, y2 = record.bbox_xyxy
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id_by_rel[record.image_rel],
                "category_id": category_id_by_name[record.category],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": record.area,
                "iscrowd": 0,
                "source_xml": record.xml_path.relative_to(root).as_posix(),
                "source_annotation_id": record.ann_id,
            }
        )
        stats.category_counts[record.category] = stats.category_counts.get(record.category, 0) + 1
        bucket = _area_bucket(record.area)
        stats.area_buckets[bucket] = stats.area_buckets.get(bucket, 0) + 1
    stats.image_count = len(images)
    stats.annotation_count = len(annotations)
    payload = {
        "info": {
            "description": f"RBGT-Tiny IR box split for SAM2-IR-QD M9 ({split_name}).",
            "split": split_name,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": value, "name": key} for key, value in sorted(category_id_by_name.items(), key=lambda item: item[1])],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def export_rbgt_tiny_box_coco(
    *,
    root: Path,
    annotations_dir: str,
    images_dir: str,
    seed: int,
    small_target_filter: bool,
    max_area_ratio: float,
    max_box_side: float,
    min_box_side: float,
    overwrite: bool,
) -> ExportStats:
    root = root.resolve()
    stats = ExportStats(
        root=str(root),
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        seed=int(seed),
        small_target_filter=bool(small_target_filter),
    )
    for split_name in SPLIT_RATIOS:
        output_path = root / f"annotations_coco_ir_box_m9_{split_name}" / f"instances_rbgt_tiny_ir_box_{split_name}.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    records = _collect_records(
        root=root,
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        small_target_filter=small_target_filter,
        max_area_ratio=max_area_ratio,
        max_box_side=max_box_side,
        min_box_side=min_box_side,
        stats=stats,
    )
    splits = _split_sequences(records, seed=seed)
    by_sequence: dict[str, list[RbgtBoxRecord]] = {}
    for record in records:
        by_sequence.setdefault(record.sequence_id, []).append(record)
    for split_name, sequences in splits.items():
        split_records: list[RbgtBoxRecord] = []
        for sequence in sequences:
            split_records.extend(by_sequence.get(sequence, []))
        stats.splits[split_name] = _write_coco_split(root=root, split_name=split_name, records=split_records)
    summary_path = root / "annotations_coco_ir_box_m9_summary.json"
    summary_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RBGT-Tiny IR VOC boxes to fixed COCO splits for M9.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations-dir", default="annotations_voc")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--split", action="store_true", help="Kept for symmetry with other exporters; M9 always writes train/val/test splits.")
    parser.add_argument("--small-target-filter", action="store_true")
    parser.add_argument("--max-area-ratio", type=float, default=0.02)
    parser.add_argument("--max-box-side", type=float, default=128.0)
    parser.add_argument("--min-box-side", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = export_rbgt_tiny_box_coco(
        root=args.root,
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        seed=args.seed,
        small_target_filter=args.small_target_filter,
        max_area_ratio=args.max_area_ratio,
        max_box_side=args.max_box_side,
        min_box_side=args.min_box_side,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
