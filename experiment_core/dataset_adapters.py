from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import ExperimentConfig
from .data import Sample, load_coco_samples, load_multimodal_samples, sample_image_key


@dataclass(frozen=True)
class DatasetManifest:
    adapter_name: str
    dataset_name: str
    data_root: str
    sample_count: int
    image_count: int
    category_count: int
    device_source_count: int
    annotation_protocols: List[str]
    canonical_bbox: str
    notes: str


@dataclass(frozen=True)
class LoadedDataset:
    manifest: DatasetManifest
    samples: List[Sample]


class DatasetAdapter:
    adapter_name = "base"
    notes = ""

    def can_handle(self, config: ExperimentConfig) -> bool:
        raise NotImplementedError

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        raise NotImplementedError

    def canonical_bbox(self, config: ExperimentConfig) -> str:
        return "bbox_loose"

    def describe(self, config: ExperimentConfig, samples: Sequence[Sample]) -> DatasetManifest:
        image_ids = {sample_image_key(sample.frame_id) for sample in samples}
        categories = {sample.category_name for sample in samples}
        device_sources = {sample.device_source for sample in samples}
        annotation_protocols = sorted({sample.annotation_protocol_flag for sample in samples})
        return DatasetManifest(
            adapter_name=self.adapter_name,
            dataset_name=config.dataset_name,
            data_root=str(config.data_root),
            sample_count=len(samples),
            image_count=len(image_ids),
            category_count=len(categories),
            device_source_count=len(device_sources),
            annotation_protocols=annotation_protocols,
            canonical_bbox=self.canonical_bbox(config),
            notes=self.notes,
        )

    def load(self, config: ExperimentConfig) -> LoadedDataset:
        samples = self.load_samples(config)
        return LoadedDataset(manifest=self.describe(config, samples), samples=samples)


class MultiModalAdapter(DatasetAdapter):
    adapter_name = "multimodal_raw"
    notes = "Loads raw MultiModal img/ + label/ and derives canonical loose boxes on the fly."

    def can_handle(self, config: ExperimentConfig) -> bool:
        return config.dataset_name == "MultiModal"

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        return load_multimodal_samples(config)


class CocoLikeAdapter(DatasetAdapter):
    adapter_name = "coco_like"
    notes = "Loads COCO-style annotations under annotations_coco/ and image/."

    def can_handle(self, config: ExperimentConfig) -> bool:
        return (config.data_root / "annotations_coco").exists() and (config.data_root / "image").exists()

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        return load_coco_samples(config)


class RBGTTinyIRAdapter(DatasetAdapter):
    adapter_name = "rbgt_tiny_ir_only"
    notes = "Loads RBGT-Tiny COCO annotations from the grayscale 01 branch only."

    def can_handle(self, config: ExperimentConfig) -> bool:
        return config.dataset_name == "RBGT-Tiny"

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        return load_coco_samples(
            config,
            annotation_patterns=["instances_01_train2017.json", "*_01_train2017.json"],
            file_name_filter=lambda name: "/01/" in name.replace("\\", "/"),
        )


def build_dataset_adapter(config: ExperimentConfig) -> DatasetAdapter:
    adapters = [MultiModalAdapter(), RBGTTinyIRAdapter(), CocoLikeAdapter()]
    for adapter in adapters:
        if adapter.can_handle(config):
            return adapter
    raise RuntimeError(
        f"No dataset adapter matched dataset_name={config.dataset_name!r} at {config.data_root}."
    )
