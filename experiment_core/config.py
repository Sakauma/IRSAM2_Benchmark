from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


SUPPORTED_CONDITIONS = [
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
    "mask_supervised",
    "box_only",
}

DEFERRED_CONDITIONS = [
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
    for path in paths:
        if path.exists():
            return path
    return None


def _parse_csv_env(name: str) -> List[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv_env(name: str) -> List[int]:
    return [int(part) for part in _parse_csv_env(name)]


def _parse_float_csv_env(name: str) -> List[float]:
    return [float(part) for part in _parse_csv_env(name)]


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _resolve_active_conditions(phase: str) -> List[str]:
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
    root = Path(__file__).resolve().parents[1]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        output_dir=Path(os.environ.get("OUTPUT_DIR", str(root / "benchmark_runs" / "outputs"))),
        device=device,
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
        active_conditions=_resolve_active_conditions(experiment_phase),
        deferred_conditions=list(DEFERRED_CONDITIONS),
    )
    _validate_config_paths(config)
    return config
