from __future__ import annotations

import json
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from PIL import Image

from ..config import AppConfig
from ..core.interfaces import DatasetAdapterProtocol
from .masks import (
    MASK_SOURCE_KEY,
    coco_rle_is_decodable,
    coco_segmentation_to_mask,
    coco_segmentation_to_polygons,
    is_coco_rle_segmentation,
    polygon_to_mask,
)
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

    def iter_samples(self, config: AppConfig, *, shard_id: int = 0, num_shards: int = 1) -> Iterable[Sample]:
        for index, sample in enumerate(self.load_samples(config)):
            if _shard_matches(index, shard_id=shard_id, num_shards=num_shards):
                yield sample


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _polygon_to_mask(points: Sequence[float], height: int, width: int) -> np.ndarray:
    return polygon_to_mask(points, height=height, width=width)


def _sorted_files(root: Path, extensions: Sequence[str]) -> List[Path]:
    lower = {ext.lower() for ext in extensions}
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in lower])


def _shard_matches(index: int, *, shard_id: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return index % max(1, int(num_shards)) == int(shard_id)


def _iter_files_depth_first(root: Path, *, suffix: str, shard_id: int = 0, num_shards: int = 1) -> Iterator[Path]:
    stack = [root]
    suffix = suffix.lower()
    file_index = 0
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        directories: list[Path] = []
        for entry in entries:
            if entry.is_dir():
                directories.append(entry)
            elif entry.is_file() and entry.suffix.lower() == suffix:
                if _shard_matches(file_index, shard_id=shard_id, num_shards=num_shards):
                    yield entry
                file_index += 1
        stack.extend(reversed(directories))


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


def _path_parts(value: object) -> tuple[str, ...]:
    return tuple(part for part in Path(str(value)).as_posix().split("/") if part)


def _xml_text(parent: ET.Element, path: str, default: str = "") -> str:
    element = parent.find(path)
    if element is None or element.text is None:
        return default
    return element.text.strip()


def _xml_float(parent: ET.Element, path: str, default: float = 0.0) -> float:
    text = _xml_text(parent, path)
    if text == "":
        return default
    return float(text)


def _xml_json(parent: ET.Element, path: str) -> object:
    text = _xml_text(parent, path)
    if not text:
        return None
    return json.loads(text)


def _mask_mode_requests_segmentation(mask_mode: str) -> bool:
    return mask_mode.strip().lower() in {"segmentation", "segmentations", "mask", "masks"}


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
    notes = "Reads RBGT-Tiny grayscale branch from COCO-style or per-image VOC-style annotations."

    def can_handle(self, config: AppConfig) -> bool:
        return config.dataset.adapter == self.adapter_name or config.dataset.dataset_id == "RBGT-Tiny"

    def _annotation_files(self, ann_dir: Path) -> List[Path]:
        ir_named = sorted(ann_dir.glob("instances_01*.json"))
        if ir_named:
            return ir_named
        branch_named = [
            path
            for path in sorted(ann_dir.glob("*.json"))
            if any(token in path.stem.lower().split("_") for token in ("01", "ir", "infrared", "thermal"))
        ]
        return branch_named or sorted(ann_dir.glob("*.json"))

    def _annotation_file_is_ir_specific(self, ann_path: Path) -> bool:
        stem_parts = ann_path.stem.lower().split("_")
        return any(token in stem_parts for token in ("01", "ir", "infrared", "thermal"))

    def _image_file_is_ir_branch(self, file_name: object) -> bool:
        parts = {part.lower() for part in _path_parts(file_name)}
        return bool(parts & {"01", "ir", "infrared", "thermal"})

    def _resolve_image_root(self, root: Path, images_dir: Optional[str]) -> Path:
        configured = root / (images_dir or "image")
        if configured.exists():
            return configured
        for candidate_name in ("image", "images"):
            candidate = root / candidate_name
            if candidate.exists():
                return candidate
        return configured

    def _image_root_is_ir_only_layout(self, image_root: Path) -> bool:
        image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        non_ir_branch_tokens = {"00", "rgb", "visible", "vis", "color"}
        seen_image = False
        try:
            for path in image_root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in image_suffixes:
                    continue
                seen_image = True
                parent_parts = {part.lower() for part in path.relative_to(image_root).parts[:-1]}
                if parent_parts & non_ir_branch_tokens:
                    return False
        except OSError:
            return False
        return seen_image

    def _resolve_image_path(self, image_root: Path, file_name: object, ann_path: Path, *, image_root_is_ir_only_layout: bool) -> Path:
        relative = Path(str(file_name))
        direct = image_root / relative
        if direct.exists():
            return direct
        if len(relative.parts) > 1:
            for branch_name in ("01", "ir", "infrared", "thermal"):
                branched = image_root / relative.parent / branch_name / relative.name
                if branched.exists():
                    return branched
        if self._annotation_file_is_ir_specific(ann_path) or image_root_is_ir_only_layout:
            for candidate in image_root.rglob(relative.name):
                if candidate.is_file():
                    return candidate
        return direct

    def _use_voc_annotations(self, ann_dir: Path) -> bool:
        if "voc" in ann_dir.name.lower():
            return True
        try:
            return next(ann_dir.rglob("*.xml"), None) is not None
        except OSError:
            return False

    def _existing_path(self, *candidates: Path) -> Optional[Path]:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _voc_image_path(self, image_root: Path, ann_dir: Path, xml_root: ET.Element, ann_path: Path) -> Path:
        path_text = _xml_text(xml_root, "path")
        if path_text:
            path = Path(path_text)
            if path.is_absolute():
                return path
            found = self._existing_path(image_root / path, image_root.parent / path)
            if found is not None:
                return found
        resolved_text = _xml_text(xml_root, "resolved_image_path")
        if resolved_text:
            resolved = Path(resolved_text)
            if resolved.is_absolute():
                return resolved
            found = self._existing_path(image_root / resolved, image_root.parent / resolved)
            if found is not None:
                return found
        filename = _xml_text(xml_root, "filename")
        if filename:
            relative = Path(filename)
            candidates = [image_root / relative]
            if len(relative.parts) > 1:
                candidates.extend(image_root / relative.parent / branch_name / relative.name for branch_name in ("01", "ir", "infrared", "thermal"))
            try:
                xml_relative = ann_path.relative_to(ann_dir)
                candidates.append(image_root / xml_relative.with_name(relative.name))
                if relative.suffix:
                    candidates.append(image_root / xml_relative.with_suffix(relative.suffix))
            except ValueError:
                pass
            found = self._existing_path(*candidates)
            if found is not None:
                return found
            return candidates[0]
        try:
            return image_root / ann_path.relative_to(ann_dir).with_suffix("")
        except ValueError:
            return image_root / ann_path.with_suffix("").name

    def _voc_image_size(self, xml_root: ET.Element, image_path: Path) -> tuple[int, int]:
        width_text = _xml_text(xml_root, "size/width")
        height_text = _xml_text(xml_root, "size/height")
        if width_text and height_text:
            try:
                return int(float(width_text)), int(float(height_text))
            except ValueError:
                pass
        return _image_size(image_path)

    def _voc_frame_id(self, image_path: Path, image_root: Path, xml_root: ET.Element) -> str:
        path_text = _xml_text(xml_root, "path")
        if path_text and not Path(path_text).is_absolute():
            return Path(path_text).as_posix()
        try:
            return image_path.relative_to(image_root).as_posix()
        except ValueError:
            filename = _xml_text(xml_root, "filename")
            return filename or image_path.name

    def _voc_bbox(self, obj: ET.Element, width: int, height: int) -> Optional[List[float]]:
        bbox = obj.find("bndbox")
        if bbox is None:
            return None
        xmin = max(0.0, min(float(width), _xml_float(bbox, "xmin")))
        ymin = max(0.0, min(float(height), _xml_float(bbox, "ymin")))
        xmax = max(0.0, min(float(width), _xml_float(bbox, "xmax")))
        ymax = max(0.0, min(float(height), _xml_float(bbox, "ymax")))
        if xmax <= xmin or ymax <= ymin:
            return None
        return [xmin, ymin, xmax, ymax]

    def _voc_object_track_id(self, obj: ET.Element) -> Optional[str]:
        track_id = _xml_text(obj, "track_id")
        if track_id:
            return track_id
        annotation_json = _xml_json(obj, "coco_annotation_json")
        return _resolve_explicit_track_id(annotation_json)

    def _voc_segmentation_source(self, obj: ET.Element, width: int, height: int) -> Optional[Dict[str, object]]:
        segmentation = _xml_json(obj, "coco_segmentation_json")
        if segmentation is None:
            annotation_json = _xml_json(obj, "coco_annotation_json")
            if isinstance(annotation_json, dict):
                segmentation = annotation_json.get("segmentation")
        if coco_segmentation_to_polygons(segmentation):
            return {
                "type": "coco_polygon",
                "segmentation": segmentation,
                "height": height,
                "width": width,
            }
        if is_coco_rle_segmentation(segmentation) and coco_rle_is_decodable(segmentation):
            return {
                "type": "coco_rle",
                "segmentation": segmentation,
                "height": height,
                "width": width,
            }
        return None

    def _iter_voc_annotation_files(self, ann_dir: Path, *, shard_id: int = 0, num_shards: int = 1) -> Iterator[Path]:
        yield from _iter_files_depth_first(ann_dir, suffix=".xml", shard_id=shard_id, num_shards=num_shards)

    def _iter_voc_samples(
        self,
        config: AppConfig,
        ann_dir: Path,
        image_root: Path,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> Iterator[Sample]:
        sample_count = 0
        missing_image_count = 0
        first_xml_reported = False
        first_image_reported = False
        seen_images: set[str] = set()
        use_segmentation = _mask_mode_requests_segmentation(config.dataset.mask_mode)
        for ann_path in self._iter_voc_annotation_files(ann_dir, shard_id=shard_id, num_shards=num_shards):
            if not first_xml_reported:
                print(f"[train-load] rbgt_voc_first_xml path={ann_path}", file=sys.stderr, flush=True)
                first_xml_reported = True
            xml_root = ET.parse(ann_path).getroot()
            image_path = self._voc_image_path(image_root, ann_dir, xml_root, ann_path)
            if not image_path.exists():
                missing_image_count += 1
                continue
            if not first_image_reported:
                print(f"[train-load] rbgt_voc_first_image path={image_path}", file=sys.stderr, flush=True)
                first_image_reported = True
            frame_id = self._voc_frame_id(image_path, image_root, xml_root)
            if not self._image_file_is_ir_branch(frame_id) and not self._annotation_file_is_ir_specific(ann_path):
                continue
            if _limit_reached(config.runtime.max_images, len(seen_images)) and frame_id not in seen_images:
                return
            seen_images.add(frame_id)
            width, height = self._voc_image_size(xml_root, image_path)
            sequence_id = _relative_sequence_id(image_path, image_root)
            frame_index = _infer_frame_index(image_path)
            device_source = _infer_device_source(image_path, image_root)
            for obj_idx, obj in enumerate(xml_root.findall("object")):
                tight = self._voc_bbox(obj, width, height)
                if tight is None:
                    continue
                category = _xml_text(obj, "name", "unknown")
                ann_id = _xml_text(obj, "coco_annotation_id", str(obj_idx))
                area = _xml_float(obj, "area", (tight[2] - tight[0]) * (tight[3] - tight[1]))
                loose = expand_box_xyxy(tight, width=width, height=height)
                point = [(tight[0] + tight[2]) / 2.0, (tight[1] + tight[3]) / 2.0]
                track_id = self._voc_object_track_id(obj)
                sample_id = f"{frame_id}__ann_{ann_id}"
                metadata: Dict[str, object] = {"coco_annotation_id": ann_id}
                segmentation_source = self._voc_segmentation_source(obj, width, height) if use_segmentation else None
                if segmentation_source is not None:
                    metadata[MASK_SOURCE_KEY] = segmentation_source
                    sample = Sample(
                        image_path=image_path,
                        sample_id=f"{sample_id}::{category}::voc_coco_segmentation",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        track_id=track_id,
                        width=width,
                        height=height,
                        category=category,
                        target_scale=_target_scale_from_area(area),
                        device_source=device_source,
                        annotation_protocol_flag="voc_coco_segmentation",
                        supervision_type="mask",
                        bbox_tight=tight,
                        bbox_loose=loose,
                        point_prompt=point,
                        metadata=metadata,
                    )
                else:
                    sample = Sample(
                        image_path=image_path,
                        sample_id=f"{sample_id}::{category}::voc_bbox_only",
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        temporal_key=frame_id,
                        track_id=track_id,
                        width=width,
                        height=height,
                        category=category,
                        target_scale=_target_scale_from_area(area),
                        device_source=device_source,
                        annotation_protocol_flag="voc_bbox_only",
                        supervision_type="bbox",
                        bbox_tight=tight,
                        bbox_loose=loose,
                        point_prompt=point,
                        metadata=metadata,
                    )
                yield sample
                sample_count += 1
                if _limit_reached(config.runtime.max_samples, sample_count):
                    return
        if missing_image_count > 0:
            print(f"[train-load] rbgt_voc_missing_images count={missing_image_count}", file=sys.stderr, flush=True)

    def _load_voc_samples(self, config: AppConfig, ann_dir: Path, image_root: Path) -> List[Sample]:
        return list(self._iter_voc_samples(config, ann_dir, image_root))

    def load_samples(self, config: AppConfig) -> List[Sample]:
        root = _dataset_root(config)
        ann_dir = root / (config.dataset.annotations_dir or "annotations_coco")
        image_root = self._resolve_image_root(root, config.dataset.images_dir)
        if self._use_voc_annotations(ann_dir):
            return self._load_voc_samples(config, ann_dir, image_root)
        ann_files = self._annotation_files(ann_dir)
        image_root_is_ir_only_layout = self._image_root_is_ir_only_layout(image_root)
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
                ann_is_ir_specific = self._annotation_file_is_ir_specific(ann_path)
                if not ann_is_ir_specific and not image_root_is_ir_only_layout and not self._image_file_is_ir_branch(file_name):
                    continue
                image_path = self._resolve_image_path(image_root, file_name, ann_path, image_root_is_ir_only_layout=image_root_is_ir_only_layout)
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

    def iter_samples(self, config: AppConfig, *, shard_id: int = 0, num_shards: int = 1) -> Iterable[Sample]:
        root = _dataset_root(config)
        ann_dir = root / (config.dataset.annotations_dir or "annotations_coco")
        image_root = self._resolve_image_root(root, config.dataset.images_dir)
        if self._use_voc_annotations(ann_dir):
            yield from self._iter_voc_samples(config, ann_dir, image_root, shard_id=shard_id, num_shards=num_shards)
            return
        for index, sample in enumerate(self.load_samples(config)):
            if _shard_matches(index, shard_id=shard_id, num_shards=num_shards):
                yield sample


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
    return coco_segmentation_to_mask(segmentation, height=height, width=width)


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
