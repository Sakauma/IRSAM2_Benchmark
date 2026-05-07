#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass
class ExportSummary:
    root: str
    images_dir: str
    labels_dir: str
    output_path: str
    summary_path: str
    max_area_px: float
    min_area_px: float
    label_files_read: int = 0
    images_read: int = 0
    instances_seen: int = 0
    annotations_kept: int = 0
    images_kept: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_polygon: int = 0
    skipped_too_small: int = 0
    skipped_too_large: int = 0


@dataclass
class SplitSummary:
    name: str
    output_dir: str
    output_path: str
    image_count: int
    annotation_count: int
    area_bins: dict[str, int]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _image_index_by_stem(image_root: Path, extensions: Iterable[str]) -> dict[str, Path]:
    suffixes = {item.lower() for item in extensions}
    index: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        index.setdefault(path.stem, path)
        index.setdefault(path.stem.lower(), path)
    return index


def _valid_polygon(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2 != 0:
        return None
    try:
        points = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return points


def _instance_polygons(instance: dict[str, Any], *, multi_polygon: str) -> list[list[float]]:
    masks = instance.get("mask", [])
    if not isinstance(masks, list):
        return []
    polygons = [_valid_polygon(item) for item in masks]
    valid = [item for item in polygons if item is not None]
    if multi_polygon == "first":
        return valid[:1]
    return valid


def _rasterized_area(points: list[float], *, width: int, height: int) -> float:
    mask = Image.new("L", (width, height), 0)
    xy = list(zip(points[0::2], points[1::2]))
    ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return float(np.asarray(mask, dtype=np.uint8).sum())


def _bbox_from_polygons(polygons: list[list[float]], *, width: int, height: int) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for points in polygons:
        xs.extend(points[0::2])
        ys.extend(points[1::2])
    if not xs or not ys:
        return None
    xmin = max(0.0, min(float(width), min(xs)))
    ymin = max(0.0, min(float(height), min(ys)))
    xmax = max(0.0, min(float(width), max(xs)))
    ymax = max(0.0, min(float(height), max(ys)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def export_multimodal_small_target_coco(
    *,
    root: Path,
    images_dir: str = "img",
    labels_dir: str = "label",
    output_dir: str = "annotations_coco_small_target",
    output_file: str = "instances_multimodal_small_target.json",
    max_area_px: float = 1024.0,
    min_area_px: float = 1.0,
    multi_polygon: str = "first",
    overwrite: bool = False,
) -> ExportSummary:
    root = root.resolve()
    image_root = root / images_dir
    label_root = root / labels_dir
    out_dir = root / output_dir
    output_path = out_dir / output_file
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing MultiModal image directory: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Missing MultiModal label directory: {label_root}")

    summary = ExportSummary(
        root=str(root),
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_path=str(output_path),
        summary_path=str(summary_path),
        max_area_px=float(max_area_px),
        min_area_px=float(min_area_px),
    )
    image_index = _image_index_by_stem(image_root, DEFAULT_IMAGE_EXTENSIONS)
    image_id_by_rel: dict[str, int] = {}
    categories: dict[str, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for label_path in sorted(label_root.rglob("*.json")):
        summary.label_files_read += 1
        image_path = image_index.get(label_path.stem) or image_index.get(label_path.stem.lower())
        if image_path is None:
            summary.skipped_missing_image += 1
            continue
        summary.images_read += 1
        with Image.open(image_path) as image:
            width, height = image.size
        relative_image = image_path.relative_to(image_root).as_posix()
        data = _read_json(label_path)
        instances = data.get("detection", {}).get("instances", [])
        if not isinstance(instances, list):
            continue
        kept_for_image = 0
        for inst_idx, instance in enumerate(instances):
            if not isinstance(instance, dict):
                continue
            summary.instances_seen += 1
            polygons = _instance_polygons(instance, multi_polygon=multi_polygon)
            if not polygons:
                summary.skipped_invalid_polygon += 1
                continue
            areas = [_rasterized_area(points, width=width, height=height) for points in polygons]
            area = float(sum(areas))
            if area < float(min_area_px):
                summary.skipped_too_small += 1
                continue
            if area > float(max_area_px):
                summary.skipped_too_large += 1
                continue
            bbox = _bbox_from_polygons(polygons, width=width, height=height)
            if bbox is None:
                summary.skipped_invalid_polygon += 1
                continue
            if relative_image not in image_id_by_rel:
                image_id = len(image_id_by_rel) + 1
                image_id_by_rel[relative_image] = image_id
                images.append({"id": image_id, "file_name": relative_image, "width": width, "height": height})
            category_name = str(instance.get("category") or "unknown")
            if category_name not in categories:
                categories[category_name] = len(categories) + 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id_by_rel[relative_image],
                    "category_id": categories[category_name],
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "source_label": label_path.relative_to(label_root).as_posix(),
                    "source_instance_index": inst_idx,
                }
            )
            annotation_id += 1
            kept_for_image += 1
            summary.annotations_kept += 1
        if kept_for_image > 0:
            summary.images_kept += 1

    payload = {
        "info": {
            "description": "MultiModal small-target subset exported from polygon JSON labels.",
            "max_area_px": float(max_area_px),
            "min_area_px": float(min_area_px),
            "multi_polygon": multi_polygon,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": value, "name": key} for key, value in sorted(categories.items(), key=lambda item: item[1])],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _area_bin(area: float) -> str:
    if area < 16:
        return "tiny_0_15"
    if area < 64:
        return "tiny_16_63"
    if area < 256:
        return "small_64_255"
    if area < 1024:
        return "small_256_1023"
    return "large_ge_1024"


def _split_image_ids(
    image_ids: list[int],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, set[int]]:
    if not image_ids:
        return {"train": set(), "val": set(), "test": set()}
    rng = random.Random(int(seed))
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_count = int(round(n * float(train_ratio)))
    val_count = int(round(n * float(val_ratio)))
    if n >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        if train_count + val_count >= n:
            train_count = max(1, n - 2)
            val_count = 1
    else:
        train_count = max(1, min(n, train_count))
        val_count = 0
    test_count = max(0, n - train_count - val_count)
    if n >= 3 and test_count <= 0:
        train_count = max(1, train_count - 1)
        test_count = n - train_count - val_count
    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _write_split_payload(
    *,
    root: Path,
    base_output_dir: str,
    base_output_file: str,
    payload: dict[str, Any],
    split_name: str,
    image_ids: set[int],
    overwrite: bool,
) -> SplitSummary:
    out_dir = root / f"{base_output_dir}_{split_name}"
    output_path = out_dir / base_output_file.replace(".json", f"_{split_name}.json")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    images = [item for item in payload.get("images", []) if int(item["id"]) in image_ids]
    annotations = [item for item in payload.get("annotations", []) if int(item["image_id"]) in image_ids]
    split_payload = {
        **payload,
        "info": {**payload.get("info", {}), "split": split_name},
        "images": images,
        "annotations": annotations,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    area_bins: dict[str, int] = {}
    for annotation in annotations:
        key = _area_bin(float(annotation.get("area", 0.0)))
        area_bins[key] = area_bins.get(key, 0) + 1
    summary = SplitSummary(
        name=split_name,
        output_dir=str(out_dir),
        output_path=str(output_path),
        image_count=len(images),
        annotation_count=len(annotations),
        area_bins=area_bins,
    )
    output_path.with_suffix(".summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_multimodal_small_target_coco_splits(
    *,
    root: Path,
    images_dir: str = "img",
    labels_dir: str = "label",
    output_dir: str = "annotations_coco_small_target",
    output_file: str = "instances_multimodal_small_target.json",
    max_area_px: float = 1024.0,
    min_area_px: float = 1.0,
    multi_polygon: str = "first",
    split_seed: int = 20260508,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    overwrite: bool = False,
) -> dict[str, Any]:
    full_summary = export_multimodal_small_target_coco(
        root=root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        output_file=output_file,
        max_area_px=max_area_px,
        min_area_px=min_area_px,
        multi_polygon=multi_polygon,
        overwrite=overwrite,
    )
    full_payload = json.loads(Path(full_summary.output_path).read_text(encoding="utf-8"))
    image_ids = [int(item["id"]) for item in full_payload.get("images", [])]
    split_ids = _split_image_ids(image_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed)
    split_summaries = [
        _write_split_payload(
            root=root.resolve(),
            base_output_dir=output_dir,
            base_output_file=output_file,
            payload=full_payload,
            split_name=split_name,
            image_ids=ids,
            overwrite=overwrite,
        )
        for split_name, ids in split_ids.items()
    ]
    summary = {
        "full": asdict(full_summary),
        "split_seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": max(0.0, 1.0 - float(train_ratio) - float(val_ratio)),
        "splits": [asdict(item) for item in split_summaries],
    }
    split_summary_path = root.resolve() / output_dir / "split_summary.json"
    summary["split_summary_path"] = str(split_summary_path)
    split_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a COCO-format MultiModal small-target subset from raw polygon JSON labels.")
    parser.add_argument("--root", type=Path, required=True, help="MultiModal dataset root containing img/ and label/.")
    parser.add_argument("--images-dir", default="img")
    parser.add_argument("--labels-dir", default="label")
    parser.add_argument("--output-dir", default="annotations_coco_small_target")
    parser.add_argument("--output-file", default="instances_multimodal_small_target.json")
    parser.add_argument("--max-area-px", type=float, default=1024.0)
    parser.add_argument("--min-area-px", type=float, default=1.0)
    parser.add_argument("--multi-polygon", choices=("first", "all"), default="first")
    parser.add_argument("--split", action="store_true", help="Also export deterministic train/val/test COCO directories.")
    parser.add_argument("--split-seed", type=int, default=20260508)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.split:
        summary = export_multimodal_small_target_coco_splits(
            root=args.root,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            output_file=args.output_file,
            max_area_px=args.max_area_px,
            min_area_px=args.min_area_px,
            multi_polygon=args.multi_polygon,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = export_multimodal_small_target_coco(
        root=args.root,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        output_file=args.output_file,
        max_area_px=args.max_area_px,
        min_area_px=args.min_area_px,
        multi_polygon=args.multi_polygon,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
