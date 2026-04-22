"""对外导出的 benchmark 核心接口。

外层脚本和入口文件只需要从这里拿配置加载器、数据集适配器和实验调度器，
不需要感知内部模块拆分细节。
"""

from .config import ExperimentConfig, load_config
from .dataset_adapters import DatasetAdapter, DatasetManifest, LoadedDataset, build_dataset_adapter
from .runner import run_experiment

__all__ = [
    "DatasetAdapter",
    "DatasetManifest",
    "ExperimentConfig",
    "LoadedDataset",
    "build_dataset_adapter",
    "load_config",
    "run_experiment",
]
