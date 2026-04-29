#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IR_TOKENS = {"01", "ir", "infrared", "thermal"}


@dataclass
class ConversionSummary:
    root: str
    annotations_dir: str
    images_dir: str
    output_dir: str
    dry_run: bool
    annotation_files_read: int = 0
    images_seen: int = 0
    ir_images_seen: int = 0
    images_written: int = 0
    objects_written: int = 0
    skipped_visible_images: int = 0
    skipped_missing_images: int = 0
    skipped_annotations_without_bbox: int = 0
    skipped_existing_xml: int = 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _path_parts(value: object) -> set[str]:
    return {part.lower() for part in Path(str(value)).as_posix().split("/") if part}


def _is_ir_annotation_file(path: Path) -> bool:
    return bool(set(path.stem.lower().split("_")) & IR_TOKENS)


def _is_ir_file_name(file_name: object) -> bool:
    return bool(_path_parts(file_name) & IR_TOKENS)


def _annotation_files(annotations_dir: Path) -> list[Path]:
    ir_named = sorted(annotations_dir.glob("instances_01*.json"))
    if ir_named:
        return ir_named
    branch_named = [
        path
        for path in sorted(annotations_dir.glob("*.json"))
        if bool(set(path.stem.lower().split("_")) & IR_TOKENS)
    ]
    return branch_named or sorted(annotations_dir.glob("*.json"))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _resolve_image_path(image_root: Path, file_name: object, *, ann_path: Path) -> Path:
    relative = Path(str(file_name))
    direct = image_root / relative
    if direct.exists():
        return direct
    if len(relative.parts) > 1:
        for branch_name in ("01", "ir", "infrared", "thermal"):
            branched = image_root / relative.parent / branch_name / relative.name
            if branched.exists():
                return branched
    if _is_ir_annotation_file(ann_path):
        ir_candidates = []
        fallback_candidates = []
        for candidate in image_root.rglob(relative.name):
            if not candidate.is_file():
                continue
            fallback_candidates.append(candidate)
            if _is_ir_file_name(candidate.relative_to(image_root).as_posix()):
                ir_candidates.append(candidate)
        if ir_candidates:
            return sorted(ir_candidates)[0]
        if fallback_candidates:
            return sorted(fallback_candidates)[0]
    return direct


def _image_relative_path(image_root: Path, image_path: Path, fallback_file_name: object) -> Path:
    try:
        return image_path.relative_to(image_root)
    except ValueError:
        return Path(str(fallback_file_name))


def _is_number_list(value: object) -> bool:
    return isinstance(value, list) and len(value) >= 2 and all(isinstance(item, (int, float)) for item in value)


def _segmentation_points(segmentation: object) -> list[float]:
    if _is_number_list(segmentation):
        return [float(item) for item in segmentation]  # type: ignore[arg-type]
    if not isinstance(segmentation, list):
        return []
    points: list[float] = []
    for item in segmentation:
        points.extend(_segmentation_points(item))
    return points


def _bbox_from_annotation(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = [float(value) for value in bbox]
        if w > 0.0 and h > 0.0:
            return x, y, x + w, y + h
    points = _segmentation_points(annotation.get("segmentation"))
    if len(points) < 6:
        return None
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0.0, min(float(width), xmin))
    ymin = max(0.0, min(float(height), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _add_text(parent: ET.Element, tag: str, value: object) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = "" if value is None else str(value)
    return child


def _add_json(parent: ET.Element, tag: str, value: object) -> ET.Element:
    return _add_text(parent, tag, _json_text(value))


def _build_xml(
    *,
    image_info: dict[str, Any],
    image_rel_path: Path,
    image_path: Path,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
) -> ET.ElementTree:
    annotation = ET.Element("annotation")
    _add_text(annotation, "folder", image_rel_path.parent.as_posix())
    _add_text(annotation, "filename", image_rel_path.name)
    _add_text(annotation, "path", image_rel_path.as_posix())
    source = ET.SubElement(annotation, "source")
    _add_text(source, "database", "RBGT-Tiny")
    _add_text(source, "annotation", "COCO")
    size = ET.SubElement(annotation, "size")
    _add_text(size, "width", width)
    _add_text(size, "height", height)
    _add_text(size, "depth", 1)
    _add_text(annotation, "segmented", int(any(obj["annotation"].get("segmentation") for obj in objects)))
    _add_text(annotation, "coco_image_id", image_info.get("id", ""))
    _add_json(annotation, "coco_image_json", image_info)
    _add_text(annotation, "resolved_image_path", image_path.as_posix())

    for obj in objects:
        annotation_record = obj["annotation"]
        object_el = ET.SubElement(annotation, "object")
        _add_text(object_el, "name", obj["category"])
        _add_text(object_el, "pose", "Unspecified")
        _add_text(object_el, "truncated", 0)
        _add_text(object_el, "difficult", 0)
        bbox = ET.SubElement(object_el, "bndbox")
        xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
        _add_text(bbox, "xmin", _fmt_number(xmin))
        _add_text(bbox, "ymin", _fmt_number(ymin))
        _add_text(bbox, "xmax", _fmt_number(xmax))
        _add_text(bbox, "ymax", _fmt_number(ymax))
        _add_text(object_el, "coco_annotation_id", annotation_record.get("id", ""))
        _add_text(object_el, "category_id", annotation_record.get("category_id", ""))
        _add_text(object_el, "area", annotation_record.get("area", (xmax - xmin) * (ymax - ymin)))
        _add_text(object_el, "iscrowd", annotation_record.get("iscrowd", 0))
        track_id = _track_id(annotation_record)
        if track_id is not None:
            _add_text(object_el, "track_id", track_id)
        _add_json(object_el, "coco_segmentation_json", annotation_record.get("segmentation"))
        _add_json(object_el, "coco_annotation_json", annotation_record)

    ET.indent(annotation, space="  ")
    return ET.ElementTree(annotation)


def _track_id(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("track_id", "trackId", "instance_id", "instanceId", "object_id", "objectId"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    for nested_key in ("attributes", "metadata"):
        found = _track_id(record.get(nested_key))
        if found is not None:
            return found
    return None


def convert_dataset(
    *,
    root: Path,
    annotations_dir: str = "annotations_coco",
    images_dir: str = "image",
    output_dir: str = "annotations_voc",
    overwrite: bool = False,
    dry_run: bool = False,
) -> ConversionSummary:
    root = root.resolve()
    ann_root = root / annotations_dir
    image_root = root / images_dir
    out_root = root / output_dir
    summary = ConversionSummary(
        root=str(root),
        annotations_dir=str(ann_root),
        images_dir=str(image_root),
        output_dir=str(out_root),
        dry_run=dry_run,
    )

    grouped: dict[str, dict[str, Any]] = {}
    for ann_path in _annotation_files(ann_root):
        summary.annotation_files_read += 1
        data = _read_json(ann_path)
        images = {item["id"]: item for item in data.get("images", [])}
        categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
        ann_is_ir_specific = _is_ir_annotation_file(ann_path)
        for annotation in data.get("annotations", []):
            image_info = images.get(annotation.get("image_id"))
            if image_info is None:
                continue
            summary.images_seen += 1
            file_name = image_info.get("file_name", "")
            image_path = _resolve_image_path(image_root, file_name, ann_path=ann_path)
            image_rel = _image_relative_path(image_root, image_path, file_name)
            if not ann_is_ir_specific and not _is_ir_file_name(file_name) and not _is_ir_file_name(image_rel.as_posix()):
                summary.skipped_visible_images += 1
                continue
            summary.ir_images_seen += 1
            if not image_path.exists():
                summary.skipped_missing_images += 1
                continue
            width = int(image_info.get("width") or _image_size(image_path)[0])
            height = int(image_info.get("height") or _image_size(image_path)[1])
            bbox = _bbox_from_annotation(annotation)
            if bbox is None:
                summary.skipped_annotations_without_bbox += 1
                continue
            bbox = _clip_bbox(bbox, width=width, height=height)
            if bbox is None:
                summary.skipped_annotations_without_bbox += 1
                continue
            key = image_rel.as_posix()
            if key not in grouped:
                grouped[key] = {
                    "image_info": image_info,
                    "image_rel_path": image_rel,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "objects": [],
                }
            grouped[key]["objects"].append(
                {
                    "annotation": annotation,
                    "category": categories.get(annotation.get("category_id"), "unknown"),
                    "bbox_xyxy": bbox,
                }
            )

    for item in grouped.values():
        xml_path = out_root / item["image_rel_path"].with_suffix(".xml")
        if xml_path.exists() and not overwrite:
            summary.skipped_existing_xml += 1
            continue
        summary.images_written += 1
        summary.objects_written += len(item["objects"])
        if dry_run:
            continue
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        tree = _build_xml(**item)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RBGT-Tiny COCO annotations to per-image VOC-style XML for the IR branch.")
    parser.add_argument("--root", required=True, type=Path, help="RBGT-Tiny dataset root.")
    parser.add_argument("--annotations-dir", default="annotations_coco")
    parser.add_argument("--images-dir", default="image")
    parser.add_argument("--output-dir", default="annotations_voc")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = convert_dataset(
        root=args.root,
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
