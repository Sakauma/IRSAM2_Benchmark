"""方法注册表。

这个模块把“方法实现”和“benchmark 叙事层信息”绑定在一起：
每个方法不仅有 factory，还会标注自己属于哪个 layer / family / description。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .config import ExperimentConfig
from .methods import BaseMethod, SAM2Teacher, build_method_registry as build_method_factories


@dataclass(frozen=True)
class MethodSpec:
    """单个方法条件的静态描述。"""
    name: str
    layer: str
    family: str
    description: str
    factory: Callable[[], BaseMethod]


class MethodRegistry:
    """方法名到方法规格的统一索引。"""

    def __init__(self, specs: List[MethodSpec]):
        self._specs: Dict[str, MethodSpec] = {spec.name: spec for spec in specs}

    def names(self) -> List[str]:
        return list(self._specs)

    def build(self, name: str) -> BaseMethod:
        # 统一从 factory 构造方法实例，runner 不需要关心具体类名。
        if name not in self._specs:
            raise KeyError(f"Unknown method condition: {name}")
        return self._specs[name].factory()

    def has(self, name: str) -> bool:
        return name in self._specs

    def manifest(self, selected: List[str] | None = None) -> List[Dict[str, str]]:
        """导出方法清单，写入 summary。"""
        names = selected or self.names()
        manifest: List[Dict[str, str]] = []
        for name in names:
            spec = self._specs.get(name)
            if spec is None:
                continue
            manifest.append(
                {
                    "name": spec.name,
                    "layer": spec.layer,
                    "family": spec.family,
                    "description": spec.description,
                }
            )
        return manifest


def build_method_registry(teacher: SAM2Teacher, config: ExperimentConfig) -> MethodRegistry:
    """把当前 benchmark v1 已实现的所有方法整理进注册表。"""
    factories = build_method_factories(teacher, config)
    specs = [
        MethodSpec(
            name="BBoxRectMaskBaseline",
            layer="layer_0_geometry_baseline",
            family="baseline",
            description="Returns the canonical box as a rectangular mask baseline.",
            factory=factories["BBoxRectMaskBaseline"],
        ),
        MethodSpec(
            name="ZeroShotSAM2BoxPromptIR",
            layer="layer_1_zero_shot",
            family="sam2",
            description="Runs zero-shot SAM2 with the canonical bbox prompt.",
            factory=factories["ZeroShotSAM2BoxPromptIR"],
        ),
        MethodSpec(
            name="CleanBoxPEFTSAM2Adapter",
            layer="layer_2_adaptation",
            family="sam2",
            description="Learns a clean prompt-conditioned residual refinement on top of SAM2 logits.",
            factory=factories["CleanBoxPEFTSAM2Adapter"],
        ),
        MethodSpec(
            name="NoisyBoxPromptRobustSAM2Adapter",
            layer="layer_2_adaptation",
            family="sam2",
            description="Learns a prompt-robust SAM2 refinement with noisy box perturbations.",
            factory=factories["NoisyBoxPromptRobustSAM2Adapter"],
        ),
        MethodSpec(
            name="CleanPromptOnlyWithinPromptRobustAdapter",
            layer="layer_2_adaptation",
            family="ablation",
            description="Ablation that removes noisy prompt augmentation from the robust adapter.",
            factory=factories["CleanPromptOnlyWithinPromptRobustAdapter"],
        ),
        MethodSpec(
            name="JitterOnlyPromptRobustSAM2Adapter",
            layer="layer_2_adaptation",
            family="ablation",
            description="Ablation that keeps jitter but removes prompt truncation/offset noise.",
            factory=factories["JitterOnlyPromptRobustSAM2Adapter"],
        ),
        MethodSpec(
            name="QualityFilteredPseudoMaskSelfTrainingSAM2",
            layer="layer_3_pseudo",
            family="sam2",
            description="Adds pseudo-label self-training with IR-specific quality filtering.",
            factory=factories["QualityFilteredPseudoMaskSelfTrainingSAM2"],
        ),
        MethodSpec(
            name="PseudoMaskSelfTrainingWithoutIRQualityFilter",
            layer="layer_3_pseudo",
            family="ablation",
            description="Pseudo-label self-training without the IR quality filter.",
            factory=factories["PseudoMaskSelfTrainingWithoutIRQualityFilter"],
        ),
        MethodSpec(
            name="DirectSupervisedIRSegFormerB0",
            layer="control_baseline",
            family="direct_train_control",
            description="Directly trained SegFormer control baseline under the current supervision protocol.",
            factory=factories["DirectSupervisedIRSegFormerB0"],
        ),
        MethodSpec(
            name="DirectSupervisedIRPIDNetS",
            layer="control_baseline",
            family="direct_train_control",
            description="Directly trained PIDNet-S control baseline under the current supervision protocol.",
            factory=factories["DirectSupervisedIRPIDNetS"],
        ),
    ]
    return MethodRegistry(specs)
