from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .core.fingerprints import sha256_file
from .core.interfaces import InferenceMode, PromptPolicy, PromptSource, PromptType, Track

import yaml


def _read_structured_file(path: Path) -> Dict[str, Any]:
    """读取 JSON/YAML 配置文件，统一返回普通 dict。"""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config file format: {path}")


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else fallback


@dataclass
class ModelConfig:
    model_id: str
    cfg: str
    ckpt: str
    repo: Optional[str] = None
    family: str = "sam2"


@dataclass
class DatasetConfig:
    # image_extensions 同时被通用 mask adapter 和 MultiModal adapter 使用。
    # MultiModal 原始数据中可能混有 bmp/png/jpg，不能在 adapter 里写死扩展名。
    dataset_id: str
    adapter: str
    root: str
    modality: str = "ir"
    images_dir: Optional[str] = None
    masks_dir: Optional[str] = None
    annotations_dir: Optional[str] = None
    mask_mode: str = "auto"
    class_map: Dict[str, str] = field(default_factory=dict)
    image_extensions: list[str] = field(default_factory=lambda: [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    mask_extensions: list[str] = field(default_factory=lambda: [".png", ".bmp", ".tif", ".tiff"])
    sequence_strategy: str = "parent_dir_or_stem"
    device_source_strategy: str = "parent_dir_or_unknown"


@dataclass
class RuntimeConfig:
    # max_samples/max_images 为 0 表示不截断；smoke/micro 配置会覆盖它们。
    artifact_root: str
    reference_results_root: str
    output_name: str
    device: str = "cuda"
    num_workers: int = 0
    smoke_test: bool = False
    max_samples: int = 0
    max_images: int = 0
    save_visuals: bool = True
    visual_limit: int = 24
    update_reference_results: bool = True
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    image_batch_size: int = 1
    reuse_image_embedding: bool = True
    auto_mask_points_per_batch: int = 64
    batch_oom_fallback: bool = True
    max_failure_rate: float = 0.05
    show_progress: bool = True
    progress_backend: str = "auto"
    progress_position: int = 0
    progress_update_interval_s: float = 1.0


@dataclass
class EvaluationConfig:
    # inference_mode 最终会转换成 InferenceMode 枚举，值必须和 core.interfaces 保持一致。
    benchmark_version: str
    track: str
    protocol: str
    inference_mode: str
    split_version: str = "split_v1"
    prompt_policy_version: str = "prompt_policy_v1"
    metric_schema_version: str = "metric_schema_v1"
    reference_result_version: str = "reference_results_v1"
    prompt_policy: PromptPolicy = field(
        default_factory=lambda: PromptPolicy(
            name="default_box_gt",
            prompt_type=PromptType.BOX,
            prompt_source=PromptSource.GT,
            prompt_budget=1,
            notes="Default image benchmark prompt policy.",
        )
    )


@dataclass
class AppConfig:
    root: Path
    config_path: Path
    model: ModelConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    method: Dict[str, Any] = field(default_factory=dict)
    fingerprints: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_root(self) -> Path:
        # DATASET_ROOT 是单次运行的最高优先级覆盖，方便在服务器脚本里临时切换数据目录。
        explicit = os.environ.get("DATASET_ROOT")
        if explicit:
            return Path(explicit)
        # 先按 config 所在项目根目录解析；如果 generated config 在 artifact 目录中，
        # 再回退到 root.parent。这兼容手写 configs/ 和 runner 生成的 config。
        candidate = self.root / self.dataset.root
        if candidate.exists():
            return candidate
        return (self.root.parent / self.dataset.root).resolve()

    @property
    def artifact_root(self) -> Path:
        # ARTIFACT_ROOT 可把同一份生成配置重定向到新的输出目录，便于复跑。
        return _env_path("ARTIFACT_ROOT", self.root / self.runtime.artifact_root)

    @property
    def reference_results_root(self) -> Path:
        return self.root / self.runtime.reference_results_root

    @property
    def output_dir(self) -> Path:
        return self.artifact_root / self.runtime.output_name

    @property
    def sam2_repo(self) -> Path:
        # SAM2_REPO 环境变量优先级最高；否则使用 config.model.repo 或项目同级 sam2。
        fallback = Path(self.model.repo) if self.model.repo else self.root.parent / "sam2"
        return _env_path("SAM2_REPO", fallback)

    @property
    def inference_mode(self) -> InferenceMode:
        return InferenceMode(self.evaluation.inference_mode)

    @property
    def track(self) -> Track:
        return Track(self.evaluation.track)


def _build_prompt_policy(raw: Dict[str, Any]) -> PromptPolicy:
    return PromptPolicy(
        name=raw["name"],
        prompt_type=PromptType(raw["prompt_type"]),
        prompt_source=PromptSource(raw["prompt_source"]),
        prompt_budget=int(raw.get("prompt_budget", 1)),
        refresh_interval=raw.get("refresh_interval"),
        multi_mask=bool(raw.get("multi_mask", False)),
        notes=raw.get("notes", ""),
    )


def load_app_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    raw = _read_structured_file(path)
    # 约定：仓库内 configs/*.yaml 的项目根是 configs 的上一级；
    # generated config 的项目根则是该 config 所在目录，绝对路径会在生成阶段写入。
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    evaluation = raw.get("evaluation", {})
    fingerprints = dict(raw.get("fingerprints", {}))
    fingerprints["config_file_sha256"] = sha256_file(path)
    return AppConfig(
        root=root,
        config_path=path,
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        evaluation=EvaluationConfig(
            benchmark_version=evaluation["benchmark_version"],
            track=evaluation["track"],
            protocol=evaluation["protocol"],
            inference_mode=evaluation["inference_mode"],
            split_version=evaluation.get("split_version", "split_v1"),
            prompt_policy_version=evaluation.get("prompt_policy_version", "prompt_policy_v1"),
            metric_schema_version=evaluation.get("metric_schema_version", "metric_schema_v1"),
            reference_result_version=evaluation.get("reference_result_version", "reference_results_v1"),
            prompt_policy=_build_prompt_policy(evaluation["prompt_policy"]),
        ),
        method=raw.get("method", {}),
        fingerprints=fingerprints,
    )
