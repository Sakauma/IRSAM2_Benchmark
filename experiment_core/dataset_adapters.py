"""数据集适配器层。

benchmark 平台的关键目标之一是“可切换数据集但不改主流程”，
因此这里把不同数据源都统一映射成 `Sample` 列表和 `DatasetManifest`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import ExperimentConfig
from .data import Sample, load_coco_samples, load_multimodal_samples, sample_image_key


@dataclass(frozen=True)
class DatasetManifest:
    """用于写入 summary 的数据集摘要信息。"""
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
    """适配器的统一返回值：样本本体 + 数据集摘要。"""
    manifest: DatasetManifest
    samples: List[Sample]


class DatasetAdapter:
    """数据集适配器抽象基类。"""

    adapter_name = "base"
    notes = ""

    def can_handle(self, config: ExperimentConfig) -> bool:
        raise NotImplementedError

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        raise NotImplementedError

    def canonical_bbox(self, config: ExperimentConfig) -> str:
        # 统一声明当前数据集默认采用哪种 bbox 作为 canonical prompt。
        return "bbox_loose"

    def describe(self, config: ExperimentConfig, samples: Sequence[Sample]) -> DatasetManifest:
        # 这里统计的是 benchmark 报告需要的元信息，而不是训练期逻辑。
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
        # 统一先拿样本，再生成 manifest，保持 adapter 输出结构一致。
        samples = self.load_samples(config)
        return LoadedDataset(manifest=self.describe(config, samples), samples=samples)


class MultiModalAdapter(DatasetAdapter):
    adapter_name = "multimodal_raw"
    notes = "Loads raw MultiModal img/ + label/ and derives canonical loose boxes on the fly."

    def can_handle(self, config: ExperimentConfig) -> bool:
        return config.dataset_name == "MultiModal"

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        # 原始 MultiModal 使用 img/ + label/ 自定义结构。
        return load_multimodal_samples(config)


class CocoLikeAdapter(DatasetAdapter):
    adapter_name = "coco_like"
    notes = "Loads COCO-style annotations under annotations_coco/ and image/."

    def can_handle(self, config: ExperimentConfig) -> bool:
        # 只要目录结构满足 COCO-like 约定，就按通用 COCO 读取。
        return (config.data_root / "annotations_coco").exists() and (config.data_root / "image").exists()

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        return load_coco_samples(config)


class RBGTTinyIRAdapter(DatasetAdapter):
    adapter_name = "rbgt_tiny_ir_only"
    notes = "Loads RBGT-Tiny COCO annotations from the grayscale 01 branch only."

    def can_handle(self, config: ExperimentConfig) -> bool:
        return config.dataset_name == "RBGT-Tiny"

    def load_samples(self, config: ExperimentConfig) -> List[Sample]:
        # RBGT-Tiny 当前 benchmark 只读取灰度 01 分支，避免多模态输入混入。
        return load_coco_samples(
            config,
            annotation_patterns=["instances_01_train2017.json", "*_01_train2017.json"],
            file_name_filter=lambda name: "/01/" in name.replace("\\", "/"),
        )


def build_dataset_adapter(config: ExperimentConfig) -> DatasetAdapter:
    """按顺序尝试适配器，返回第一个能处理当前数据集的实现。"""
    adapters = [MultiModalAdapter(), RBGTTinyIRAdapter(), CocoLikeAdapter()]
    for adapter in adapters:
        if adapter.can_handle(config):
            return adapter
    raise RuntimeError(
        f"No dataset adapter matched dataset_name={config.dataset_name!r} at {config.data_root}."
    )
