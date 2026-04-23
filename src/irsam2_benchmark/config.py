"""配置解析与路径解析逻辑。

Author: Egor Izmaylov

这个模块承担三个核心职责：
1. 把 YAML/JSON 配置文件解析成类型化 dataclass；
2. 冻结 benchmark 治理字段，例如 split/prompt policy/schema version；
3. 把机器相关路径统一收敛到属性访问中，避免业务代码散落路径拼接逻辑。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .core.interfaces import InferenceMode, PromptPolicy, PromptSource, PromptType, Track

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _read_structured_file(path: Path) -> Dict[str, Any]:
    """读取 YAML/JSON 配置文件并返回字典。

    这里故意只支持结构化文本格式，避免把 Python 脚本当配置执行。
    """
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config file format: {path}")


def _env_path(name: str, fallback: Path) -> Path:
    """优先读取环境变量路径，否则退回默认路径。"""
    value = os.environ.get(name)
    return Path(value) if value else fallback


@dataclass
class ModelConfig:
    """模型接入配置。

    这里描述的是“如何找到一个 SAM2 模型”，而不是训练超参数。
    """

    model_id: str
    cfg: str
    ckpt: str
    repo: Optional[str] = None
    family: str = "sam2"


@dataclass
class DatasetConfig:
    """数据集接入配置。

    adapter 决定如何解释目录结构，其余字段用于细化该 adapter 的行为。
    """

    dataset_id: str
    adapter: str
    root: str
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
    """运行时配置。

    这些字段主要描述输出目录、设备、数据裁剪上限和 seed 集合。
    """

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
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class EvaluationConfig:
    """评测协议配置。

    这部分信息会被写入 benchmark spec，属于论文级可复现实验治理字段。
    """

    benchmark_version: str
    track: str
    protocol: str
    inference_mode: str
    split_version: str = "split_v1"
    prompt_policy_version: str = "prompt_policy_v1"
    metric_schema_version: str = "metric_schema_v1"
    reference_result_version: str = "reference_results_v1"
    temporal_eval_policy: str = "sequence_aware"
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
class StageConfig:
    """全链路 stage 配置容器。"""

    transfer: Dict[str, Any] = field(default_factory=dict)
    adapt: Dict[str, Any] = field(default_factory=dict)
    distill: Dict[str, Any] = field(default_factory=dict)
    quantize: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """平台总配置对象。"""

    root: Path
    config_path: Path
    model: ModelConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    stages: StageConfig
    ablations: Dict[str, Any]

    @property
    def dataset_root(self) -> Path:
        """解析数据集根目录。

        优先级：
        1. `DATASET_ROOT` 环境变量；
        2. 相对项目根目录；
        3. 相对项目父目录。
        """
        explicit = os.environ.get("DATASET_ROOT")
        if explicit:
            return Path(explicit)
        candidate = self.root / self.dataset.root
        if candidate.exists():
            return candidate
        return (self.root.parent / self.dataset.root).resolve()

    @property
    def artifact_root(self) -> Path:
        """统一解析产物根目录。"""
        return _env_path("ARTIFACT_ROOT", self.root / self.runtime.artifact_root)

    @property
    def reference_results_root(self) -> Path:
        """返回 reference snapshot 根目录。"""
        return self.root / self.runtime.reference_results_root

    @property
    def output_dir(self) -> Path:
        """当前运行的输出目录。"""
        return self.artifact_root / self.runtime.output_name

    @property
    def sam2_repo(self) -> Path:
        """解析本地 SAM2 仓库路径。"""
        fallback = Path(self.model.repo) if self.model.repo else self.root.parent / "sam2"
        return _env_path("SAM2_REPO", fallback)

    @property
    def inference_mode(self) -> InferenceMode:
        """把字符串配置提升成强类型枚举。"""
        return InferenceMode(self.evaluation.inference_mode)

    @property
    def track(self) -> Track:
        """把字符串赛道名提升成强类型枚举。"""
        return Track(self.evaluation.track)


def _build_prompt_policy(raw: Dict[str, Any]) -> PromptPolicy:
    """从原始字典构建 PromptPolicy。

    这里单独抽一层，方便后续扩展配置兼容逻辑。
    """
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
    """从配置文件加载项目总配置。"""
    path = Path(config_path).resolve()
    raw = _read_structured_file(path)

    # 如果配置文件放在 configs/ 目录下，则项目根目录取 configs 的父目录。
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    evaluation = raw.get("evaluation", {})
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
            temporal_eval_policy=evaluation.get("temporal_eval_policy", "sequence_aware"),
            prompt_policy=_build_prompt_policy(evaluation["prompt_policy"]),
        ),
        stages=StageConfig(**raw.get("stages", {})),
        ablations=raw.get("ablations", {}),
    )
