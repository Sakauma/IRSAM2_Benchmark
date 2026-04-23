from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..config import AppConfig
from ..core.interfaces import ArtifactRecord, PipelineStage


@dataclass
class StageResult:
    record: ArtifactRecord


def _write_stage_stub(config: AppConfig, stage: PipelineStage, payload: Dict[str, Any]) -> StageResult:
    stage_dir = config.output_dir / stage.value
    stage_dir.mkdir(parents=True, exist_ok=True)
    record = ArtifactRecord(
        stage=stage.value,
        artifact_dir=str(stage_dir),
        artifact_name=f"{stage.value}_artifact",
        metadata=payload,
    )
    return StageResult(record=record)


def run_transfer_stage(config: AppConfig) -> StageResult:
    return _write_stage_stub(
        config,
        PipelineStage.TRANSFER,
        {
            "model_id": config.model.model_id,
            "cfg": config.model.cfg,
            "ckpt": config.model.ckpt,
            "note": "Transfer stage initializes the SAM2 teacher artifact.",
        },
    )


def run_adapt_stage(config: AppConfig) -> StageResult:
    return _write_stage_stub(
        config,
        PipelineStage.ADAPT,
        {
            "note": "Reference adaptation stage scaffold. Replace with a concrete trainer artifact when available.",
            "stages": config.stages.adapt,
            "ablations": config.ablations,
        },
    )


def run_distill_stage(config: AppConfig) -> StageResult:
    return _write_stage_stub(
        config,
        PipelineStage.DISTILL,
        {
            "note": "Reference distillation stage scaffold. Replace with a trained student artifact.",
            "stages": config.stages.distill,
        },
    )


def run_quantize_stage(config: AppConfig) -> StageResult:
    return _write_stage_stub(
        config,
        PipelineStage.QUANTIZE,
        {
            "note": "Reference quantization stage scaffold. Replace with PTQ/QAT export artifacts.",
            "stages": config.stages.quantize,
        },
    )
