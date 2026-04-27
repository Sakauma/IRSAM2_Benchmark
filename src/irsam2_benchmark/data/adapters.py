from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import DatasetAdapterProtocol
from .masks import MASK_SOURCE_KEY, polygon_to_mask
from .prompt_synthesis import connected_components, expand_box_xyxy, mask_derived_prompt_metadata, mask_to_point_prompt, mask_to_tight_box
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


def _mask_height_width(mask: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}.")
    return int(arr.shape[0]), int(arr.shape[1])


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
    return polygon_to_mask(points, height=height, width=width)


def _sorted_files(root: Path, extensions: Sequence[str]) -> List[Path]:
    lower = {ext.lower() for ext in extensions}
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in lower])


def _image_index_by_stem(root: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    # MultiModal 原始数据的 label 文件没有记录图片扩展名。
    # 这里先按允许的扩展名建立 stem -> image_path 索引，避免漏掉 .jpg/.jpeg 等图片。
    index: Dict[str, Path] = {}
    for path in _sorted_files(root, extensions):
        index.setdefault(path.stem, path)
        index.setdefault(path.stem.lower(), path)
    return index


def _generic_mask_index_keys(path: Path, root: Path) -> List[str]:
    rel_stem = path.relative_to(root).with_suffix("").as_posix()
    keys = [rel_stem]
    normalized = re.sub(r"_pixels\d+$", "", rel_stem)
    if normalized != rel_stem:
        keys.append(normalized)
    return keys


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
    sample_id: str,
    frame_id: str,
    sequence_id: str,
    frame_index: int,
    temporal_key: str,
    track_id: Optional[str],
    category: str,
    device_source: str,
    annotation_protocol_flag: str,
    mask_array: np.ndarray,
    width: int,
    height: int,
    store_mask_array: bool = True,
    mask_source: Optional[Dict[str, object]] = None,
) -> Sample:
    # 所有 mask-supervised 数据集都走同一套 prompt 派生规则：
    # 由 GT mask 生成 tight box、loose box、foreground centroid point 和 target_scale。
    tight = mask_to_tight_box(mask_array)
    loose = expand_box_xyxy(tight, width=width, height=height)
    point = mask_to_point_prompt(mask_array)
    target_scale = _target_scale_from_area(float((mask_array > 0.5).sum()))
    metadata: Dict[str, object] = {"prompt_generation": mask_derived_prompt_metadata()}
    if mask_source is not None:
        metadata[MASK_SOURCE_KEY] = mask_source
    return Sample(
        image_path=image_path,
        sample_id=f"{sample_id}::{category}::{annotation_protocol_flag}",
        frame_id=frame_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
        temporal_key=temporal_key,
        track_id=track_id,
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
        # MultiModal 可以只保存 polygon source，在评估阶段按需解码，避免一次性常驻数 GB GT mask。
        mask_array=mask_array.astype(np.float32) if store_mask_array else None,
        metadata=metadata,
    )


def _resolve_explicit_track_id(record: object) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    for key in ("track_id", "trackId", "instance_id", "instanceId", "object_id", "objectId"):
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    for nested_key in ("attributes", "metadata"):
        nested_track_id = _resolve_explicit_track_id(record.get(nested_key))
        if nested_track_id is not None:
            return nested_track_id
    return None


class MultiModalAdapter(DatasetAdapter):
    adapter_name = "multimodal_raw"
    notes = "Reads raw MultiModal img/ + label/ JSON files and synthesizes prompts from polygons."

    def can_handle(self, config: AppConfig) -> bool:
        root = _dataset_root(config)
        return config.dataset.adapter == self.adapter_name or ((root / "img").exists() and (root / "label").exists())

    def load_samples(self, config: AppConfig) -> List[Sample]:
        # 原始 MultiModal 目录结构是 img/ + label/*.json。
        # 每个 label 文件可以包含多个 instance；benchmark 的 eval unit 是 instance。
        root = _dataset_root(config)
        img_dir = root / (config.dataset.images_dir or "img")
        label_dir = root / "label"
        samples: List[Sample] = []
        seen_images: set[str] = set()
        image_index = _image_index_by_stem(img_dir, config.dataset.image_extensions)
        for label_path in sorted(label_dir.glob("*.json")):
            stem = label_path.stem
            if _limit_reached(config.runtime.max_images, len(seen_images)) and stem not in seen_images:
                break
            image_path = image_index.get(stem) or image_index.get(stem.lower())
            if image_path is None:
                continue
            seen_images.add(stem)
            width, height = _image_size(image_path)
            data = _read_json(label_path)
            instances = data.get("detection", {}).get("instances", [])
            sequence_id = _relative_sequence_id(image_path, img_dir)
            frame_index = _infer_frame_index(image_path)
            device_source = _infer_device_source(label_path, label_dir)
            frame_id = stem
            for inst_idx, inst in enumerate(instances):
                masks = inst.get("mask", [])
                # 当前协议使用第一个 polygon 作为实例 GT。若后续需要多 polygon 合并，
                # 应在这里显式 rasterize 所有 polygon，而不是静默改变评估口径。
                polygon = masks[0] if masks and len(masks[0]) >= 6 else None
                if polygon is None:
                    continue
                mask_array = _polygon_to_mask(polygon, height=height, width=width)
                sample_id = f"{frame_id}__inst_{inst_idx}"
                mask_source = {
                    "type": "polygon",
                    "points": [float(value) for value in polygon],
                    "height": height,
                    "width": width,
                }
                samples.append(
                    _build_sample_from_mask(
                        image_path=image_path,
                        sample_id=sample_id,
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        track_id=_resolve_explicit_track_id(inst),
                        category=inst.get("category", "unknown"),
                        device_source=device_source,
                        annotation_protocol_flag="polygon_mask",
                        mask_array=mask_array,
                        width=width,
                        height=height,
                        store_mask_array=False,
                        mask_source=mask_source,
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
            data = _read_json(ann_path)
            images = {item["id"]: item for item in data.get("images", [])}
            categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
            for ann_idx, ann in enumerate(data.get("annotations", [])):
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
                frame_id = Path(file_name).as_posix()
                segmentation = ann.get("segmentation")
                mask_array = _decode_coco_segmentation(segmentation, height=height, width=width)
                sample_id = f"{frame_id}__ann_{ann.get('id', ann_idx)}"
                category = categories.get(ann.get("category_id"), "unknown")
                if mask_array is not None:
                    samples.append(
                        _build_sample_from_mask(
                            image_path=image_path,
                            sample_id=sample_id,
                            frame_id=frame_id,
                            sequence_id=sequence_id,
                            frame_index=frame_index,
                            temporal_key=frame_id,
                            track_id=_resolve_explicit_track_id(ann),
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
                        sample_id=f"{sample_id}::{category}::coco_bbox_only",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        track_id=_resolve_explicit_track_id(ann),
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
        root = _dataset_root(config)
        ann_dir = root / (config.dataset.annotations_dir or "annotations_coco")
        image_root = root / (config.dataset.images_dir or "image")
        ann_files = sorted(ann_dir.glob("instances_01*.json"))
        samples: List[Sample] = []
        seen_images: set[str] = set()
        for ann_path in ann_files:
            data = _read_json(ann_path)
            images = {item["id"]: item for item in data.get("images", [])}
            categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
            for ann_idx, ann in enumerate(data.get("annotations", [])):
                image_info = images.get(ann.get("image_id"))
                if image_info is None:
                    continue
                file_name = image_info["file_name"]
                if "/01/" not in Path(file_name).as_posix():
                    continue
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
                frame_id = Path(file_name).as_posix()
                segmentation = ann.get("segmentation")
                mask_array = _decode_coco_segmentation(segmentation, height=height, width=width)
                sample_id = f"{frame_id}__ann_{ann.get('id', ann_idx)}"
                category = categories.get(ann.get("category_id"), "unknown")
                if mask_array is not None:
                    samples.append(
                        _build_sample_from_mask(
                            image_path=image_path,
                            sample_id=sample_id,
                            frame_id=frame_id,
                            sequence_id=sequence_id,
                            frame_index=frame_index,
                            temporal_key=frame_id,
                            track_id=_resolve_explicit_track_id(ann),
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
                        sample_id=f"{sample_id}::{category}::coco_bbox_only",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        track_id=_resolve_explicit_track_id(ann),
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


class GenericImageMaskAdapter(DatasetAdapter):
    adapter_name = "generic_image_mask"
    notes = "Reads arbitrary images/ + masks/ datasets without requiring VOC/COCO conversion."

    def can_handle(self, config: AppConfig) -> bool:
        root = _dataset_root(config)
        images_dir = root / (config.dataset.images_dir or "images")
        masks_dir = root / (config.dataset.masks_dir or "masks")
        return config.dataset.adapter == self.adapter_name or (images_dir.exists() and masks_dir.exists())

    def load_samples(self, config: AppConfig) -> List[Sample]:
        base_notes = self.__class__.notes
        skipped_size_mismatches: List[str] = []

        def finish(samples: List[Sample]) -> List[Sample]:
            if skipped_size_mismatches:
                preview = ", ".join(skipped_size_mismatches[:5])
                suffix = "..." if len(skipped_size_mismatches) > 5 else ""
                self.notes = f"{base_notes} Skipped {len(skipped_size_mismatches)} image(s) with image/mask size mismatch: {preview}{suffix}."
            else:
                self.notes = base_notes
            return samples

        root = _dataset_root(config)
        images_dir = root / (config.dataset.images_dir or "images")
        masks_dir = root / (config.dataset.masks_dir or "masks")
        images = _sorted_files(images_dir, config.dataset.image_extensions)
        mask_index: Dict[str, Path] = {}
        for path in _sorted_files(masks_dir, config.dataset.mask_extensions):
            for key in _generic_mask_index_keys(path, masks_dir):
                mask_index.setdefault(key, path)
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
            mask_height, mask_width = _mask_height_width(mask_image)
            if (mask_width, mask_height) != (width, height):
                message = (
                    "Skipping image/mask size mismatch. "
                    f"dataset_id={config.dataset.dataset_id!r}, "
                    f"image={image_path}, image_size={(width, height)}, "
                    f"mask={mask_path}, mask_size={(mask_width, mask_height)}."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                skipped_size_mismatches.append(image_path.relative_to(images_dir).as_posix())
                continue
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
                    return finish(samples)
        return finish(samples)


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
        binary_rule = "positive_values_gt_0"
        if 255 in positive_values and len(positive_values) > 1:
            instance_mask = (mask == 255).astype(np.float32)
            binary_rule = "value_255_when_mixed_positive_values"
        else:
            instance_mask = (mask > 0).astype(np.float32)
        if float(instance_mask.sum()) <= 0.0:
            return []
        sample = _build_sample_from_mask(
            image_path=image_path,
            sample_id=frame_id,
            frame_id=frame_id,
            sequence_id=sequence_id,
            frame_index=frame_index,
            temporal_key=temporal_key,
            track_id=None,
            category="foreground",
            device_source=device_source,
            annotation_protocol_flag="generic_binary_mask",
            mask_array=instance_mask,
            width=width,
            height=height,
        )
        sample.metadata["mask_decode"] = {
            "mask_mode": "binary",
            "binary_rule": binary_rule,
            "positive_values": positive_values,
        }
        samples.append(
            sample
        )
        return samples

    if resolved_mode == "instance_id":
        for value in positive_values:
            instance_mask = (mask == value).astype(np.float32)
            samples.append(
                _build_sample_from_mask(
                    image_path=image_path,
                    sample_id=f"{frame_id}__inst_{value}",
                    frame_id=frame_id,
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    temporal_key=temporal_key,
                    track_id=str(value),
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
                        sample_id=f"{frame_id}__class_{value}__cc_{component_idx}",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=temporal_key,
                        track_id=None,
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
            if polygon and isinstance(polygon[0], list):
                for nested_polygon in polygon:
                    if len(nested_polygon) >= 6:
                        canvas = np.maximum(canvas, _polygon_to_mask(nested_polygon, height, width))
                continue
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


def build_dataset_adapter(config: AppConfig) -> DatasetAdapterProtocol:
    adapters: List[DatasetAdapter] = [
        MultiModalAdapter(),
        RBGTTinyIRAdapter(),
        CocoLikeAdapter(),
        GenericImageMaskAdapter(),
    ]
    requested_adapter = str(config.dataset.adapter or "").strip()
    if requested_adapter and requested_adapter != "auto":
        for adapter in adapters:
            if adapter.adapter_name == requested_adapter:
                return adapter
        available = ", ".join(adapter.adapter_name for adapter in adapters)
        raise RuntimeError(
            f"Unknown dataset adapter {requested_adapter!r} for dataset_id={config.dataset.dataset_id!r}. "
            f"Available adapters: {available}"
        )
    for adapter in adapters:
        if adapter.can_handle(config):
            return adapter
    raise RuntimeError(f"No dataset adapter matched dataset_id={config.dataset.dataset_id!r} under {_dataset_root(config)}")
