"""运行期配置解析。

这里集中管理 benchmark 的全部运行参数，包括：
1. 数据集与 SAM2 路径解析
2. 单卡 / 多卡运行信息探测
3. benchmark 阶段、监督协议、训练超参数默认值

后续如果迁移到新服务器，优先修改环境变量，而不是改这里的默认逻辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .distributed import detect_distributed

SUPPORTED_CONDITIONS = [
    # 主 benchmark 当前已经实现并允许直接运行的方法条件。
    "BBoxRectMaskBaseline",
    "ZeroShotSAM2BoxPromptIR",
    "CleanBoxPEFTSAM2Adapter",
    "NoisyBoxPromptRobustSAM2Adapter",
    "CleanPromptOnlyWithinPromptRobustAdapter",
    "JitterOnlyPromptRobustSAM2Adapter",
    "QualityFilteredPseudoMaskSelfTrainingSAM2",
    "PseudoMaskSelfTrainingWithoutIRQualityFilter",
    "DirectSupervisedIRSegFormerB0",
    "DirectSupervisedIRPIDNetS",
]

SUPPORTED_SUPERVISION_PROTOCOLS = {
    # 完整 mask 监督，用于上界或对照实验。
    "mask_supervised",
    # 仅保留 box 作为训练监督，mask 只用于评估或伪标签。
    "box_only",
}

DEFERRED_CONDITIONS = [
    # 这些条件在计划里存在，但当前 benchmark v1 暂不正式启用。
    "FinalMaskDistilledIRStudent",
    "CorrectionTrajectoryDistilledIRStudent",
    "FinalMaskDistilledINT8SegFormerB0",
    "CorrectionTrajectoryDistilledINT8SegFormerB0",
    "CorrectionTrajectoryDistilledINT8PIDNetS",
    "FinalStateOnlyDistillationWithinTrajectoryStudent",
    "UnorderedTrajectoryDistillationStudent",
    "TrajectoryStudentWithoutAuxiliaryTrajectoryHead",
    "INT8PostTrainingOnlyWithoutQuantizationAwareRecovery",
    "MixedProtocolTrainingWithoutProtocolConsistentSubset",
    "MetadataBlindEvaluationWithoutDeviceStratification",
]


@dataclass
class ExperimentConfig:
    """统一配置对象。

    这个 dataclass 会贯穿整个 benchmark 生命周期：
    数据加载、方法构建、训练、评估、报告生成都只从这里取配置。
    """

    root: Path
    dataset_root: Path
    dataset_name: str
    data_root: Path
    img_dir: Path | None
    label_dir: Path | None
    sam2_repo: Path
    sam2_ckpt: Path
    sam2_cfg: str
    output_dir: Path
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    smoke_test: bool
    experiment_phase: str
    supervision_protocol: str
    seeds: List[int]
    supervision_budgets: List[float]
    max_samples: int
    max_images: int
    train_epochs: int
    pseudo_finetune_epochs: int
    batch_size: int
    eval_limit: int
    num_workers: int
    prompt_jitter_scale: float
    prompt_offset_scale: float
    prompt_truncation_ratio: float
    bbox_pad_ratio: float
    bbox_min_pad: float
    bbox_min_side: float
    lambda_bce: float
    lambda_dice: float
    lambda_consistency: float
    lambda_box_projection: float
    lambda_box_outside: float
    pseudo_quality_threshold: float
    pseudo_score_threshold: float
    lr_teacher_adapter: float
    lr_segformer: float
    lr_pidnet: float
    weight_decay: float
    max_grad_norm: float
    active_conditions: List[str]
    deferred_conditions: List[str]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    """按顺序返回第一个存在的路径，用于实现多候选路径回退。"""
    for path in paths:
        if path.exists():
            return path
    return None


def _parse_csv_env(name: str) -> List[str]:
    """解析逗号分隔环境变量，例如 `42,123,456`。"""
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv_env(name: str) -> List[int]:
    return [int(part) for part in _parse_csv_env(name)]


def _parse_float_csv_env(name: str) -> List[float]:
    return [float(part) for part in _parse_csv_env(name)]


def _int_env(name: str, default: int) -> int:
    """读取整型环境变量，不存在时回退到默认值。"""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    """读取浮点型环境变量，不存在时回退到默认值。"""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _resolve_active_conditions(phase: str) -> List[str]:
    """根据实验阶段选择默认要跑的方法条件。

    如果用户显式设置了 `EXPERIMENT_CONDITIONS`，则以用户指定为准。
    """
    explicit = _parse_csv_env("EXPERIMENT_CONDITIONS")
    if explicit:
        return explicit
    if phase == "benchmark_v1":
        return [
            "BBoxRectMaskBaseline",
            "ZeroShotSAM2BoxPromptIR",
            "CleanBoxPEFTSAM2Adapter",
            "NoisyBoxPromptRobustSAM2Adapter",
            "QualityFilteredPseudoMaskSelfTrainingSAM2",
            "DirectSupervisedIRSegFormerB0",
            "DirectSupervisedIRPIDNetS",
        ]
    if phase == "teacher_only":
        return [
            "BBoxRectMaskBaseline",
            "ZeroShotSAM2BoxPromptIR",
            "CleanBoxPEFTSAM2Adapter",
            "NoisyBoxPromptRobustSAM2Adapter",
            "DirectSupervisedIRSegFormerB0",
            "DirectSupervisedIRPIDNetS",
        ]
    return list(SUPPORTED_CONDITIONS)


def _resolve_dataset_root(root: Path) -> Path:
    """解析数据集根目录。

    优先级：
    1. 显式环境变量
    2. 仓库附近常见目录
    3. 容器或当前工作目录下的 `dataset/`
    """
    explicit = os.environ.get("DATASET_ROOT") or os.environ.get("DATA_ROOT")
    if explicit:
        return Path(explicit)
    candidate = _first_existing(
        [
            root / "dataset",
            root.parent / "dataset",
            Path("/dataset"),
            Path.cwd() / "dataset",
        ]
    )
    if candidate is not None:
        return candidate
    raise RuntimeError(
        "Dataset root is not configured. Set DATASET_ROOT to the directory containing datasets such as "
        "'MultiModal' or 'RBGT-Tiny'."
    )


def _resolve_sam2_repo(root: Path) -> Path:
    """解析本地 SAM2 仓库路径。"""
    explicit = os.environ.get("SAM2_REPO")
    if explicit:
        return Path(explicit)
    candidate = _first_existing(
        [
            root / "third_party" / "sam2",
            root / "external" / "sam2",
            root.parent / "sam2",
            Path.cwd() / "sam2",
        ]
    )
    if candidate is not None:
        return candidate
    raise RuntimeError(
        "SAM2 repository is not configured. Set SAM2_REPO to a local SAM2 checkout path."
    )


def _resolve_sam2_ckpt(sam2_repo: Path) -> Path:
    """解析 SAM2 checkpoint 路径。"""
    explicit = os.environ.get("SAM2_CKPT")
    if explicit:
        return Path(explicit)
    candidate = _first_existing(
        [
            sam2_repo / "checkpoints" / "sam2.1_hiera_base_plus.pt",
            sam2_repo / "checkpoints" / "sam2.1_hiera_large.pt",
        ]
    )
    if candidate is not None:
        return candidate
    raise RuntimeError(
        "SAM2 checkpoint is not configured. Set SAM2_CKPT to a valid checkpoint file."
    )


def _validate_config_paths(config: ExperimentConfig) -> None:
    """在真正启动实验前做显式路径校验。

    这里的目的不是“自动修复”，而是尽早给出可定位的报错。
    """
    if not config.data_root.exists():
        raise RuntimeError(
            f"Dataset path does not exist: {config.data_root}. "
            f"Check DATASET_ROOT and DATASET_NAME."
        )
    if config.dataset_name == "MultiModal":
        if config.img_dir is None or config.label_dir is None:
            raise RuntimeError("MultiModal expects both img_dir and label_dir to be configured.")
        if not config.img_dir.exists() or not config.label_dir.exists():
            raise RuntimeError(
                f"MultiModal dataset expects img/ and label/ under {config.data_root}."
            )
    if not config.sam2_repo.exists():
        raise RuntimeError(
            f"SAM2 repository path does not exist: {config.sam2_repo}. Check SAM2_REPO."
        )
    if not config.sam2_ckpt.exists():
        raise RuntimeError(
            f"SAM2 checkpoint path does not exist: {config.sam2_ckpt}. Check SAM2_CKPT."
        )


def load_config() -> ExperimentConfig:
    """从环境变量和默认规则构造完整配置对象。"""
    root = Path(__file__).resolve().parents[1]
    # 先把所有路径依赖解析干净，后续模块不再关心路径发现逻辑。
    dataset_root = _resolve_dataset_root(root)
    dataset_name = os.environ.get("DATASET_NAME", "MultiModalCOCOClean")
    data_root = dataset_root / dataset_name
    sam2_repo = _resolve_sam2_repo(root)
    sam2_ckpt = _resolve_sam2_ckpt(sam2_repo)
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    experiment_phase = os.environ.get("EXPERIMENT_PHASE", "benchmark_v1")
    supervision_protocol = os.environ.get("SUPERVISION_PROTOCOL", "box_only").strip() or "box_only"
    if supervision_protocol not in SUPPORTED_SUPERVISION_PROTOCOLS:
        raise RuntimeError(
            "Unsupported SUPERVISION_PROTOCOL. Expected one of "
            f"{sorted(SUPPORTED_SUPERVISION_PROTOCOLS)}, got {supervision_protocol!r}."
        )
    distributed_cfg = detect_distributed()
    # 多卡模式下，每个进程绑定自己的 local_rank；单卡时统一用默认 cuda。
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{distributed_cfg.local_rank}" if distributed_cfg.enabled else "cuda")
    else:
        device = torch.device("cpu")

    seeds = _parse_int_csv_env("EXPERIMENT_SEEDS") or ([42] if smoke_test else [42, 123, 456])
    budgets = _parse_float_csv_env("SUPERVISION_BUDGETS") or ([0.1] if smoke_test else [0.1, 0.2, 0.5])

    config = ExperimentConfig(
        root=root,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        data_root=data_root,
        img_dir=(data_root / "img") if dataset_name == "MultiModal" else None,
        label_dir=(data_root / "label") if dataset_name == "MultiModal" else None,
        sam2_repo=sam2_repo,
        sam2_ckpt=sam2_ckpt,
        sam2_cfg=os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        # 输出目录允许完全由外部覆盖，便于在服务器上把结果写到大容量盘。
        output_dir=Path(os.environ.get("OUTPUT_DIR", str(root / "benchmark_runs" / "outputs"))),
        device=device,
        distributed=distributed_cfg.enabled,
        rank=distributed_cfg.rank,
        local_rank=distributed_cfg.local_rank,
        world_size=distributed_cfg.world_size,
        smoke_test=smoke_test,
        experiment_phase=experiment_phase,
        supervision_protocol=supervision_protocol,
        seeds=seeds,
        supervision_budgets=budgets,
        max_samples=_int_env("MAX_SAMPLES", 12 if smoke_test else 64),
        max_images=_int_env("MAX_IMAGES", 0),
        train_epochs=_int_env("TRAIN_EPOCHS", 1 if smoke_test else 6),
        pseudo_finetune_epochs=_int_env("PSEUDO_FINETUNE_EPOCHS", 1 if smoke_test else 4),
        batch_size=_int_env("BATCH_SIZE", 1),
        eval_limit=_int_env("EVAL_LIMIT", 4 if smoke_test else 12),
        num_workers=_int_env("NUM_WORKERS", 0),
        prompt_jitter_scale=0.10,
        prompt_offset_scale=0.15,
        prompt_truncation_ratio=0.10,
        bbox_pad_ratio=_float_env("BBOX_PAD_RATIO", 0.15),
        bbox_min_pad=_float_env("BBOX_MIN_PAD", 2.0),
        bbox_min_side=_float_env("BBOX_MIN_SIDE", 12.0),
        lambda_bce=1.0,
        lambda_dice=1.0,
        lambda_consistency=0.2,
        lambda_box_projection=_float_env("LAMBDA_BOX_PROJECTION", 1.0),
        lambda_box_outside=_float_env("LAMBDA_BOX_OUTSIDE", 1.0),
        pseudo_quality_threshold=_float_env("PSEUDO_QUALITY_THRESHOLD", 0.70),
        pseudo_score_threshold=_float_env("PSEUDO_SCORE_THRESHOLD", 0.70),
        lr_teacher_adapter=_float_env("LR_TEACHER_ADAPTER", 1e-4),
        lr_segformer=_float_env("LR_SEGFORMER", 6e-4),
        lr_pidnet=_float_env("LR_PIDNET", 1e-3),
        weight_decay=1e-4,
        max_grad_norm=1.0,
        # active/deferred 会直接出现在 summary 中，便于追踪这次到底跑了什么。
        active_conditions=_resolve_active_conditions(experiment_phase),
        deferred_conditions=list(DEFERRED_CONDITIONS),
    )
    _validate_config_paths(config)
    return config
