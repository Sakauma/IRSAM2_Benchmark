from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..config import AppConfig
from .prompt_synthesis import connected_components, expand_box_xyxy, mask_to_point_prompt, mask_to_tight_box
from .sample import Sample


@dataclass(frozen=True)
class DatasetManifest:
    adapter_name: str
    dataset_id: str
    root: str
    sample_count: int
    image_count: int
    sequence_count: int
    category_count: int
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedDataset:
    manifest: DatasetManifest
    samples: List[Sample]


class DatasetAdapter:
    adapter_name = "base"
    notes = ""

    def can_handle(self, config: AppConfig) -> bool:
        raise NotImplementedError

    def load(self, config: AppConfig) -> LoadedDataset:
        samples = self.load_samples(config)
        image_ids = {sample.frame_id for sample in samples}
        sequence_ids = {sample.sequence_id for sample in samples}
        categories = {sample.category for sample in samples}
        return LoadedDataset(
            manifest=DatasetManifest(
                adapter_name=self.adapter_name,
                dataset_id=config.dataset.dataset_id,
                root=str(_dataset_root(config)),
                sample_count=len(samples),
                image_count=len(image_ids),
                sequence_count=len(sequence_ids),
                category_count=len(categories),
                notes=self.notes,
            ),
            samples=samples,
        )

    def load_samples(self, config: AppConfig) -> List[Sample]:
        raise NotImplementedError


def _dataset_root(config: AppConfig) -> Path:
    return config.dataset_root


def _limit_reached(limit: int, count: int) -> bool:
    return limit > 0 and count >= limit


def _mask_to_numpy(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path))


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    xy = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
    ImageDraw.Draw(canvas).polygon(xy, outline=1, fill=1)
    return np.array(canvas, dtype=np.float32)


def _sorted_files(root: Path, extensions: Sequence[str]) -> List[Path]:
    lower = {ext.lower() for ext in extensions}
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in lower])


def _relative_sequence_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.parent != Path("."):
        return relative.parent.as_posix()
    return "__single_sequence__"


def _infer_device_source(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.parent != Path("."):
        return relative.parts[0]
    stem = path.stem
    for splitter in ("_", "-", "."):
        if splitter in stem:
            return stem.split(splitter)[0]
    return "unknown"


def _infer_frame_index(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else 0


def _build_sample_from_mask(
    *,
    image_path: Path,
    frame_id: str,
    sequence_id: str,
    frame_index: int,
    temporal_key: str,
    category: str,
    device_source: str,
    annotation_protocol_flag: str,
    mask_array: np.ndarray,
    width: int,
    height: int,
) -> Sample:
    tight = mask_to_tight_box(mask_array)
    loose = expand_box_xyxy(tight, width=width, height=height)
    point = mask_to_point_prompt(mask_array)
    target_scale = _target_scale_from_area(float((mask_array > 0.5).sum()))
    return Sample(
        image_path=image_path,
        sample_id=f"{frame_id}::{category}::{annotation_protocol_flag}",
        frame_id=frame_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
        temporal_key=temporal_key,
        width=width,
        height=height,
        category=category,
        target_scale=target_scale,
        device_source=device_source,
        annotation_protocol_flag=annotation_protocol_flag,
        supervision_type="mask",
        bbox_tight=tight,
        bbox_loose=loose,
        point_prompt=point,
        mask_array=mask_array.astype(np.float32),
    )


class MultiModalAdapter(DatasetAdapter):
    adapter_name = "multimodal_raw"
    notes = "Reads raw MultiModal img/ + label/ JSON files and synthesizes prompts from polygons."

    def can_handle(self, config: AppConfig) -> bool:
        root = _dataset_root(config)
        return config.dataset.adapter == self.adapter_name or ((root / "img").exists() and (root / "label").exists())

    def load_samples(self, config: AppConfig) -> List[Sample]:
        root = _dataset_root(config)
        img_dir = root / (config.dataset.images_dir or "img")
        label_dir = root / "label"
        samples: List[Sample] = []
        seen_images: set[str] = set()
        for label_path in sorted(label_dir.glob("*.json")):
            stem = label_path.stem
            if _limit_reached(config.runtime.max_images, len(seen_images)) and stem not in seen_images:
                break
            image_path = img_dir / f"{stem}.bmp"
            if not image_path.exists():
                image_path = img_dir / f"{stem}.png"
            if not image_path.exists():
                continue
            seen_images.add(stem)
            width, height = _image_size(image_path)
            data = json.loads(label_path.read_text(encoding="utf-8"))
            instances = data.get("detection", {}).get("instances", [])
            sequence_id = _relative_sequence_id(image_path, img_dir)
            frame_index = _infer_frame_index(image_path)
            device_source = _infer_device_source(label_path, label_dir)
            for inst_idx, inst in enumerate(instances):
                masks = inst.get("mask", [])
                polygon = masks[0] if masks and len(masks[0]) >= 6 else None
                if polygon is None:
                    continue
                mask_array = _polygon_to_mask(polygon, height=height, width=width)
                frame_id = f"{stem}__inst_{inst_idx}"
                samples.append(
                    _build_sample_from_mask(
                        image_path=image_path,
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        category=inst.get("category", "unknown"),
                        device_source=device_source,
                        annotation_protocol_flag="polygon_mask",
                        mask_array=mask_array,
                        width=width,
                        height=height,
                    )
                )
                if _limit_reached(config.runtime.max_samples, len(samples)):
                    return samples
        return samples


class CocoLikeAdapter(DatasetAdapter):
    adapter_name = "coco_like"
    notes = "Reads COCO-style annotations and keeps mask-aware prompts when segmentation exists."

    def can_handle(self, config: AppConfig) -> bool:
        root = _dataset_root(config)
        return config.dataset.adapter == self.adapter_name or ((root / "annotations_coco").exists() and (root / "image").exists())

    def load_samples(self, config: AppConfig) -> List[Sample]:
        root = _dataset_root(config)
        ann_dir = root / (config.dataset.annotations_dir or "annotations_coco")
        image_root = root / (config.dataset.images_dir or "image")
        ann_files = sorted(ann_dir.glob("*.json"))
        samples: List[Sample] = []
        seen_images: set[str] = set()
        for ann_path in ann_files:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            images = {item["id"]: item for item in data.get("images", [])}
            categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
            for ann in data.get("annotations", []):
                image_info = images.get(ann.get("image_id"))
                if image_info is None:
                    continue
                file_name = image_info["file_name"]
                image_path = image_root / file_name
                if not image_path.exists():
                    continue
                if _limit_reached(config.runtime.max_images, len(seen_images)) and file_name not in seen_images:
                    return samples
                seen_images.add(file_name)
                width = int(image_info.get("width", _image_size(image_path)[0]))
                height = int(image_info.get("height", _image_size(image_path)[1]))
                sequence_id = _relative_sequence_id(image_path, image_root)
                frame_index = _infer_frame_index(image_path)
                device_source = _infer_device_source(image_path, image_root)
                segmentation = ann.get("segmentation")
                mask_array = _decode_coco_segmentation(segmentation, height=height, width=width)
                frame_id = f"{Path(file_name).as_posix()}__ann_{ann.get('id')}"
                category = categories.get(ann.get("category_id"), "unknown")
                if mask_array is not None:
                    samples.append(
                        _build_sample_from_mask(
                            image_path=image_path,
                            frame_id=frame_id,
                            sequence_id=sequence_id,
                            frame_index=frame_index,
                            temporal_key=frame_id,
                            category=category,
                            device_source=device_source,
                            annotation_protocol_flag="coco_segmentation",
                            mask_array=mask_array,
                            width=width,
                            height=height,
                        )
                    )
                    if _limit_reached(config.runtime.max_samples, len(samples)):
                        return samples
                    continue
                bbox_xywh = ann.get("bbox")
                if not bbox_xywh or len(bbox_xywh) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox_xywh]
                tight = [x, y, x + w, y + h]
                loose = expand_box_xyxy(tight, width=width, height=height)
                point = [x + w / 2.0, y + h / 2.0]
                samples.append(
                    Sample(
                        image_path=image_path,
                        sample_id=frame_id,
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        width=width,
                        height=height,
                        category=category,
                        target_scale=_target_scale_from_area(w * h),
                        device_source=device_source,
                        annotation_protocol_flag="coco_bbox_only",
                        supervision_type="bbox",
                        bbox_tight=tight,
                        bbox_loose=loose,
                        point_prompt=point,
                    )
                )
                if _limit_reached(config.runtime.max_samples, len(samples)):
                    return samples
        return samples


class RBGTTinyIRAdapter(CocoLikeAdapter):
    adapter_name = "rbgt_tiny_ir_only"
    notes = "Reads RBGT-Tiny grayscale branch from COCO-style annotations."

    def can_handle(self, config: AppConfig) -> bool:
        return config.dataset.adapter == self.adapter_name or config.dataset.dataset_id == "RBGT-Tiny"

    def load_samples(self, config: AppConfig) -> List[Sample]:
        samples = super().load_samples(config)
        return [sample for sample in samples if "/01/" in sample.image_path.as_posix() or "\\01\\" in str(sample.image_path)]


class GenericImageMaskAdapter(DatasetAdapter):
    adapter_name = "generic_image_mask"
    notes = "Reads arbitrary images/ + masks/ datasets without requiring VOC/COCO conversion."

    def can_handle(self, config: AppConfig) -> bool:
        root = _dataset_root(config)
        images_dir = root / (config.dataset.images_dir or "images")
        masks_dir = root / (config.dataset.masks_dir or "masks")
        return config.dataset.adapter == self.adapter_name or (images_dir.exists() and masks_dir.exists())

    def load_samples(self, config: AppConfig) -> List[Sample]:
        root = _dataset_root(config)
        images_dir = root / (config.dataset.images_dir or "images")
        masks_dir = root / (config.dataset.masks_dir or "masks")
        images = _sorted_files(images_dir, config.dataset.image_extensions)
        mask_index = {
            path.relative_to(masks_dir).with_suffix("").as_posix(): path
            for path in _sorted_files(masks_dir, config.dataset.mask_extensions)
        }
        samples: List[Sample] = []
        seen_images: set[str] = set()
        for image_path in images:
            rel_stem = image_path.relative_to(images_dir).with_suffix("").as_posix()
            if _limit_reached(config.runtime.max_images, len(seen_images)) and rel_stem not in seen_images:
                break
            mask_path = mask_index.get(rel_stem)
            if mask_path is None:
                continue
            seen_images.add(rel_stem)
            mask_image = _mask_to_numpy(mask_path)
            width, height = _image_size(image_path)
            sequence_id = _relative_sequence_id(image_path, images_dir)
            frame_index = _infer_frame_index(image_path)
            frame_id = rel_stem
            temporal_key = rel_stem
            device_source = _infer_device_source(image_path, images_dir)
            for sample in _samples_from_generic_mask(
                image_path=image_path,
                frame_id=frame_id,
                sequence_id=sequence_id,
                frame_index=frame_index,
                temporal_key=temporal_key,
                device_source=device_source,
                mask=mask_image,
                width=width,
                height=height,
                mask_mode=config.dataset.mask_mode,
                class_map=config.dataset.class_map,
            ):
                samples.append(sample)
                if _limit_reached(config.runtime.max_samples, len(samples)):
                    return samples
        return samples


def _samples_from_generic_mask(
    *,
    image_path: Path,
    frame_id: str,
    sequence_id: str,
    frame_index: int,
    temporal_key: str,
    device_source: str,
    mask: np.ndarray,
    width: int,
    height: int,
    mask_mode: str,
    class_map: Dict[str, str],
) -> List[Sample]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.int64)
    positive_values = [int(v) for v in np.unique(mask) if int(v) > 0]
    if not positive_values:
        return []

    resolved_mode = mask_mode
    if resolved_mode == "auto":
        resolved_mode = "binary" if len(positive_values) == 1 else "instance_id"

    samples: List[Sample] = []
    if resolved_mode == "binary":
        instance_mask = (mask > 0).astype(np.float32)
        samples.append(
            _build_sample_from_mask(
                image_path=image_path,
                frame_id=frame_id,
                sequence_id=sequence_id,
                frame_index=frame_index,
                temporal_key=temporal_key,
                category="foreground",
                device_source=device_source,
                annotation_protocol_flag="generic_binary_mask",
                mask_array=instance_mask,
                width=width,
                height=height,
            )
        )
        return samples

    if resolved_mode == "instance_id":
        for value in positive_values:
            instance_mask = (mask == value).astype(np.float32)
            samples.append(
                _build_sample_from_mask(
                    image_path=image_path,
                    frame_id=f"{frame_id}__inst_{value}",
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    temporal_key=temporal_key,
                    category=class_map.get(str(value), "instance"),
                    device_source=device_source,
                    annotation_protocol_flag="generic_instance_mask",
                    mask_array=instance_mask,
                    width=width,
                    height=height,
                )
            )
        return samples

    if resolved_mode == "class_index":
        for value in positive_values:
            class_mask = (mask == value).astype(np.float32)
            for component_idx, component in enumerate(connected_components(class_mask)):
                samples.append(
                    _build_sample_from_mask(
                        image_path=image_path,
                        frame_id=f"{frame_id}__class_{value}__cc_{component_idx}",
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=temporal_key,
                        category=class_map.get(str(value), str(value)),
                        device_source=device_source,
                        annotation_protocol_flag="generic_class_mask",
                        mask_array=component,
                        width=width,
                        height=height,
                    )
                )
        return samples

    raise ValueError(f"Unsupported generic mask mode: {resolved_mode}")


def _decode_coco_segmentation(segmentation: object, height: int, width: int) -> Optional[np.ndarray]:
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        canvas = np.zeros((height, width), dtype=np.float32)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            canvas = np.maximum(canvas, _polygon_to_mask(polygon, height, width))
        return canvas if canvas.any() else None
    return None


def _target_scale_from_area(area: float) -> str:
    if area < float(32 * 32):
        return "small"
    if area < float(96 * 96):
        return "medium"
    return "large"


def build_dataset_adapter(config: AppConfig) -> DatasetAdapter:
    adapters: List[DatasetAdapter] = [
        MultiModalAdapter(),
        RBGTTinyIRAdapter(),
        CocoLikeAdapter(),
        GenericImageMaskAdapter(),
    ]
    for adapter in adapters:
        if adapter.can_handle(config):
            return adapter
    raise RuntimeError(f"No dataset adapter matched dataset_id={config.dataset.dataset_id!r} under {_dataset_root(config)}")
