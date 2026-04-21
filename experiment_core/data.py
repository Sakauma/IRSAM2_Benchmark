from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .config import ExperimentConfig


def _limit_reached(limit: int, count: int) -> bool:
    return limit > 0 and count >= limit


def _image_limit_reached(limit: int, seen_images: Sequence[str]) -> bool:
    return limit > 0 and len(seen_images) >= limit


def sample_image_key(frame_id: str) -> str:
    if "__inst_" in frame_id:
        return frame_id.split("__inst_")[0]
    if "__ann_" in frame_id:
        return frame_id.split("__ann_")[0]
    return frame_id


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    bbox: List[float]
    bbox_tight: Optional[List[float]]
    bbox_loose: Optional[List[float]]
    category_name: str
    polygon_points: Optional[List[float]]
    device_source: str
    frame_id: str
    annotation_protocol_flag: str
    supervision_source: str = "gt_mask"
    mask_array: Optional[np.ndarray] = None
    sample_weight: float = 1.0
    pseudo_score: Optional[float] = None
    pseudo_quality: Optional[float] = None

    def has_mask(self) -> bool:
        return self.mask_array is not None or bool(self.polygon_points)

    def with_mask(
        self,
        mask: np.ndarray,
        supervision_source: str,
        sample_weight: float,
        pseudo_score: Optional[float] = None,
        pseudo_quality: Optional[float] = None,
    ) -> "Sample":
        return replace(
            self,
            polygon_points=None,
            mask_array=mask.astype(np.float32),
            supervision_source=supervision_source,
            sample_weight=sample_weight,
            pseudo_score=pseudo_score,
            pseudo_quality=pseudo_quality,
        )


def load_ir_image(path: Path) -> np.ndarray:
    raw = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    min_v = float(raw.min())
    max_v = float(raw.max())
    norm = ((raw.astype(np.float32) - min_v) / max(1e-6, max_v - min_v) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def polygon_to_mask(points: Sequence[float], h: int, w: int) -> np.ndarray:
    canvas = Image.new("L", (w, h), 0)
    xy = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
    ImageDraw.Draw(canvas).polygon(xy, outline=1, fill=1)
    return np.array(canvas, dtype=np.float32)


def polygon_bbox(points: Sequence[float]) -> List[float]:
    xs = [float(points[idx]) for idx in range(0, len(points), 2)]
    ys = [float(points[idx + 1]) for idx in range(0, len(points), 2)]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return [x1, y1, x2, y2]


def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def clamp_box(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(x1 + 1.0, min(float(w), x2))
    y2 = max(y1 + 1.0, min(float(h), y2))
    return [x1, y1, x2, y2]


def build_loose_box_xyxy(
    tight_box_xyxy: Sequence[float],
    width: int,
    height: int,
    pad_ratio: float,
    min_pad: float,
    min_side: float,
) -> List[float]:
    tight_x1, tight_y1, tight_x2, tight_y2 = [float(v) for v in tight_box_xyxy]
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

    return clamp_box([loose_x1, loose_y1, loose_x2, loose_y2], width, height)


def build_box_prior(box: Sequence[float], h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(box, w, h)]
    prior = np.zeros((h, w), dtype=np.float32)
    prior[y1:y2, x1:x2] = 1.0
    return prior


def parse_device_source(label_path: Path) -> str:
    return label_path.stem.split("_")[0].split("-")[0]


def load_mask_for_sample(sample: Sample, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if sample.mask_array is not None:
        return sample.mask_array.astype(np.float32)
    if sample.polygon_points:
        return polygon_to_mask(sample.polygon_points, image_shape[0], image_shape[1])
    return None


def load_samples(config: ExperimentConfig) -> List[Sample]:
    if config.dataset_name == "MultiModal":
        return load_multimodal_samples(config)
    if (config.data_root / "annotations_coco").exists() and (config.data_root / "image").exists():
        return load_coco_samples(config)
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")


def load_multimodal_samples(config: ExperimentConfig) -> List[Sample]:
    samples: List[Sample] = []
    seen_images = set()
    assert config.label_dir is not None and config.img_dir is not None
    for label_path in sorted(config.label_dir.glob("*.json")):
        stem = label_path.stem
        if _image_limit_reached(config.max_images, seen_images) and stem not in seen_images:
            break
        image_path = config.img_dir / f"{stem}.bmp"
        if not image_path.exists():
            image_path = config.img_dir / f"{stem}.png"
        if not image_path.exists():
            continue
        seen_images.add(stem)
        with Image.open(image_path) as image:
            width, height = image.size
        data = json.loads(label_path.read_text(encoding="utf8"))
        instances = data.get("detection", {}).get("instances", [])
        for inst_idx, inst in enumerate(instances):
            masks = inst.get("mask", [])
            polygon = masks[0] if masks and len(masks[0]) >= 6 else None
            bbox = inst.get("bbox")
            if polygon is not None:
                bbox_tight = polygon_bbox(polygon)
            elif bbox is not None and len(bbox) == 4:
                bbox_tight = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            else:
                continue
            bbox_tight = clamp_box(bbox_tight, width, height)
            bbox_loose = build_loose_box_xyxy(
                bbox_tight,
                width=width,
                height=height,
                pad_ratio=config.bbox_pad_ratio,
                min_pad=config.bbox_min_pad,
                min_side=config.bbox_min_side,
            )
            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=label_path,
                    bbox=bbox_loose,
                    bbox_tight=bbox_tight,
                    bbox_loose=bbox_loose,
                    category_name=inst.get("category", "unknown"),
                    polygon_points=polygon,
                    device_source=parse_device_source(label_path),
                    frame_id=f"{label_path.stem}__inst_{inst_idx}",
                    annotation_protocol_flag="polygon_mask" if polygon else "raw_bbox_only",
                )
            )
        if _limit_reached(config.max_samples, len(samples)):
            break
    if len(samples) < 8:
        raise RuntimeError("Not enough samples found to run the experiment.")
    return samples


def _coco_segmentation_to_mask(segmentation, h: int, w: int) -> Optional[np.ndarray]:
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        canvas = np.zeros((h, w), dtype=np.float32)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            canvas = np.maximum(canvas, polygon_to_mask(polygon, h, w))
        return canvas if canvas.any() else None
    return None


def load_coco_samples(
    config: ExperimentConfig,
    annotation_patterns: Optional[Sequence[str]] = None,
    file_name_filter: Optional[Callable[[str], bool]] = None,
) -> List[Sample]:
    annotation_dir = config.data_root / "annotations_coco"
    image_root = config.data_root / "image"
    samples: List[Sample] = []
    seen_images = set()
    annotation_files: List[Path] = []
    patterns = list(annotation_patterns or ["instances_*_train2017.json"])
    for pattern in patterns:
        annotation_files.extend(sorted(annotation_dir.glob(pattern)))
    seen = set()
    annotation_files = [path for path in annotation_files if not (path in seen or seen.add(path))]
    if not annotation_files:
        annotation_files = sorted(annotation_dir.glob("*.json"))
    for ann_path in annotation_files:
        data = json.loads(ann_path.read_text(encoding="utf8"))
        images = {item["id"]: item for item in data.get("images", [])}
        categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
        for ann in data.get("annotations", []):
            image_info = images.get(ann.get("image_id"))
            if image_info is None:
                continue
            file_name = image_info["file_name"]
            if file_name_filter is not None and not file_name_filter(file_name):
                continue
            if _image_limit_reached(config.max_images, seen_images) and file_name not in seen_images:
                continue
            bbox_loose_xywh = ann.get("bbox_loose")
            bbox_tight_xywh = ann.get("bbox_tight")
            bbox_xywh = bbox_loose_xywh or ann.get("bbox") or bbox_tight_xywh
            if bbox_xywh is None or len(bbox_xywh) != 4:
                continue
            x, y, bw, bh = [float(v) for v in bbox_xywh]
            image_path = image_root / file_name
            if not image_path.exists():
                continue
            bbox_tight_xyxy = None
            if bbox_tight_xywh is not None and len(bbox_tight_xywh) == 4:
                tx, ty, tw, th = [float(v) for v in bbox_tight_xywh]
                bbox_tight_xyxy = [tx, ty, tx + tw, ty + th]
            bbox_loose_xyxy = None
            if bbox_loose_xywh is not None and len(bbox_loose_xywh) == 4:
                lx, ly, lw, lh = [float(v) for v in bbox_loose_xywh]
                bbox_loose_xyxy = [lx, ly, lx + lw, ly + lh]
            mask_array = _coco_segmentation_to_mask(
                ann.get("segmentation"),
                int(image_info["height"]),
                int(image_info["width"]),
            )
            category_name = categories.get(ann.get("category_id"), str(ann.get("category_id", "unknown")))
            frame_id = f"{Path(file_name).as_posix()}__ann_{ann.get('id')}"
            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=ann_path,
                    bbox=[x, y, x + bw, y + bh],
                    bbox_tight=bbox_tight_xyxy,
                    bbox_loose=bbox_loose_xyxy,
                    category_name=category_name,
                    polygon_points=None,
                    device_source=file_name.split("/")[0],
                    frame_id=frame_id,
                    annotation_protocol_flag="coco_segmentation" if mask_array is not None else "bbox_only",
                    mask_array=mask_array,
                )
            )
            seen_images.add(file_name)
            if _limit_reached(config.max_samples, len(samples)):
                break
        if _limit_reached(config.max_samples, len(samples)):
            break
    if len(samples) < 8:
        raise RuntimeError(f"Not enough COCO-format samples found under {config.data_root} to run the experiment.")
    return samples


def deterministic_source_split(
    samples: Sequence[Sample],
    budget: float,
    eval_limit: int,
) -> Tuple[List[Sample], List[Sample], List[Sample], List[Sample]]:
    groups = {}
    for sample in samples:
        groups.setdefault(sample.device_source, []).append(sample)

    group_sizes = [len(items) for items in groups.values()]
    if group_sizes and (max(group_sizes) <= 2 or len(groups) >= max(4, len(samples) // 2)):
        groups = _fallback_bucket_groups(samples)

    ordered_sources = sorted(groups)
    train_count = max(1, math.ceil(len(ordered_sources) * 0.5))
    train_sources = ordered_sources[:train_count]
    remaining_sources = ordered_sources[train_count:]
    if remaining_sources:
        val_count = max(1, len(remaining_sources) // 2)
        val_sources = remaining_sources[:val_count]
        test_sources = remaining_sources[val_count:]
    else:
        val_sources = []
        test_sources = []

    train_pool = [sample for source in train_sources for sample in groups[source]]
    val = [sample for source in val_sources for sample in groups[source]]
    test = [sample for source in test_sources for sample in groups[source]]
    if eval_limit > 0:
        val = val[:eval_limit]
        test = test[:eval_limit]

    if len(val) < 2:
        val = train_pool[:2]
    if len(test) < 2:
        test = train_pool[2:4]

    labeled, unlabeled = _split_train_pool_by_image(train_pool, budget)
    return labeled, val, test, unlabeled


def _fallback_bucket_groups(samples: Sequence[Sample]):
    image_groups = {}
    for sample in samples:
        image_id = sample_image_key(sample.frame_id)
        image_groups.setdefault(image_id, []).append(sample)
    ordered_images = sorted(image_groups)
    bucket_count = min(4, max(3, len(ordered_images)))
    grouped = {f"bucket_{idx}": [] for idx in range(bucket_count)}
    for idx, image_id in enumerate(ordered_images):
        grouped[f"bucket_{idx % bucket_count}"].extend(image_groups[image_id])
    return grouped


def _split_train_pool_by_image(train_pool: Sequence[Sample], budget: float):
    images = {}
    for sample in train_pool:
        image_id = sample_image_key(sample.frame_id)
        images.setdefault(image_id, []).append(sample)
    ordered_image_ids = sorted(images)
    keep_images = max(1, int(len(ordered_image_ids) * budget))
    if len(ordered_image_ids) > 1:
        keep_images = min(keep_images, len(ordered_image_ids) - 1)
    labeled_ids = set(ordered_image_ids[:keep_images])
    labeled = [sample for image_id in ordered_image_ids if image_id in labeled_ids for sample in images[image_id]]
    unlabeled = [sample for image_id in ordered_image_ids if image_id not in labeled_ids for sample in images[image_id]]
    return labeled, unlabeled


class InfraredDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], require_mask: bool, allow_gt_masks: bool = True):
        self.samples = list(samples)
        self.require_mask = require_mask
        self.allow_gt_masks = allow_gt_masks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_rgb = load_ir_image(sample.image_path)
        mask = load_mask_for_sample(sample, image_rgb.shape[:2])
        if not self.allow_gt_masks and sample.supervision_source == "gt_mask":
            mask = None
        if self.require_mask and mask is None:
            raise ValueError(f"Sample {sample.frame_id} does not provide a training mask.")
        return {
            "image_rgb": image_rgb,
            "mask": None if mask is None else mask.astype(np.float32),
            "has_mask": mask is not None,
            "bbox": np.array(clamp_box(sample.bbox, image_rgb.shape[1], image_rgb.shape[0]), dtype=np.float32),
            "bbox_tight": None
            if sample.bbox_tight is None
            else np.array(clamp_box(sample.bbox_tight, image_rgb.shape[1], image_rgb.shape[0]), dtype=np.float32),
            "bbox_loose": None
            if sample.bbox_loose is None
            else np.array(clamp_box(sample.bbox_loose, image_rgb.shape[1], image_rgb.shape[0]), dtype=np.float32),
            "sample_id": sample.frame_id,
            "category_name": sample.category_name,
            "device_source": sample.device_source,
            "sample_weight": float(sample.sample_weight),
            "supervision_source": sample.supervision_source,
            "annotation_protocol_flag": sample.annotation_protocol_flag,
        }


def collate_fn(batch):
    images = torch.stack(
        [torch.from_numpy(item["image_rgb"]).permute(2, 0, 1).float() / 255.0 for item in batch],
        dim=0,
    )
    masks = torch.stack(
        [
            torch.from_numpy(item["mask"]).float()
            if item["mask"] is not None
            else torch.zeros(item["image_rgb"].shape[:2], dtype=torch.float32)
            for item in batch
        ],
        dim=0,
    )[:, None]
    bboxes = torch.stack([torch.from_numpy(item["bbox"]).float() for item in batch], dim=0)
    weights = torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32)
    has_masks = torch.tensor([item["has_mask"] for item in batch], dtype=torch.bool)
    return {
        "images": images,
        "masks": masks,
        "has_masks": has_masks,
        "bboxes": bboxes,
        "bbox_tight_list": [item["bbox_tight"] for item in batch],
        "bbox_loose_list": [item["bbox_loose"] for item in batch],
        "image_rgb": [item["image_rgb"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "category_names": [item["category_name"] for item in batch],
        "device_sources": [item["device_source"] for item in batch],
        "sample_weights": weights,
        "supervision_sources": [item["supervision_source"] for item in batch],
        "annotation_protocol_flags": [item["annotation_protocol_flag"] for item in batch],
    }
