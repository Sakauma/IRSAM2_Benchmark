"""Stage scaffold 实现。

Author: Egor Izmaylov

当前版本里，transfer / adapt / distill / quantize 还没有全部变成最终论文方法。
但 benchmark 需要先冻结 artifact 接口，因此先以 scaffold 的形式把 stage 输出结构稳定下来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import AppConfig
from ..core.interfaces import ArtifactRecord, PipelineStage


@dataclass
class StageResult:
    """单个 stage 的标准返回对象。"""

    record: ArtifactRecord


def _write_stage_stub(config: AppConfig, stage: PipelineStage, payload: Dict[str, Any]) -> StageResult:
    """写出当前 stage 的占位 artifact。"""
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
    """transfer stage：初始化 teacher artifact。"""
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
    """adapt stage scaffold。"""
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
    """distill stage scaffold。"""
    return _write_stage_stub(
        config,
        PipelineStage.DISTILL,
        {
            "note": "Reference distillation stage scaffold. Replace with a trained student artifact.",
            "stages": config.stages.distill,
        },
    )


def run_quantize_stage(config: AppConfig) -> StageResult:
    """quantize stage scaffold。"""
    return _write_stage_stub(
        config,
        PipelineStage.QUANTIZE,
        {
            "note": "Reference quantization stage scaffold. Replace with PTQ/QAT export artifacts.",
            "stages": config.stages.quantize,
        },
    )
