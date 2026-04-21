# Benchmark Overview

## 1. 目标
这个 benchmark 要回答的不是“某个方法单次能不能跑通”，而是：

1. 在统一 `box_only` 协议下，几何 baseline 有多强。
2. SAM2 zero-shot 在红外数据上能达到什么水平。
3. SAM2 adaptation 是否稳定优于 zero-shot。
4. pseudo-label 扩展是否真的带来净收益。

## 2. 平台接口
当前工程已经按三层接口组织。

### Dataset Adapter
- 文件：[dataset_adapters.py](D:/workspace/sam_ir/ir_sam2_bench/experiment_core/dataset_adapters.py)
- 作用：统一把不同数据集转换成同一批 `Sample` 结构。
- 当前实现：
  - `MultiModalAdapter`
  - `RBGTTinyIRAdapter`
  - `CocoLikeAdapter`

### Method Registry
- 文件：[method_registry.py](D:/workspace/sam_ir/ir_sam2_bench/experiment_core/method_registry.py)
- 作用：统一注册 benchmark 条件，并给每个条件附上层级、家族和说明。
- 当前层级：
  - `layer_0_geometry_baseline`
  - `layer_1_zero_shot`
  - `layer_2_adaptation`
  - `layer_3_pseudo`
  - `control_baseline`

### Evaluation / Reporting
- 文件：[evaluation.py](D:/workspace/sam_ir/ir_sam2_bench/experiment_core/evaluation.py)
- 文件：[reporting.py](D:/workspace/sam_ir/ir_sam2_bench/experiment_core/reporting.py)
- 作用：把评估计算和结果落盘从 runner 中拆开，保证不同数据集、不同方法共用同一套 report schema。

## 3. 数据协议
- canonical box：`bbox_loose`
- `MultiModal` 原始数据会在线生成：
  - `bbox_tight`
  - `bbox_loose`
- `RBGT-Tiny` 默认只读灰度 `01` 分支，避免彩色 `00` 分支污染红外 benchmark。

## 4. 条件集合
- Geometry baseline：
  - `BBoxRectMaskBaseline`
- SAM2 zero-shot：
  - `ZeroShotSAM2BoxPromptIR`
- SAM2 adaptation：
  - `CleanBoxPEFTSAM2Adapter`
  - `NoisyBoxPromptRobustSAM2Adapter`
  - `CleanPromptOnlyWithinPromptRobustAdapter`
  - `JitterOnlyPromptRobustSAM2Adapter`
- SAM2 pseudo：
  - `QualityFilteredPseudoMaskSelfTrainingSAM2`
  - `PseudoMaskSelfTrainingWithoutIRQualityFilter`
- Control：
  - `DirectSupervisedIRSegFormerB0`
  - `DirectSupervisedIRPIDNetS`

## 5. 输出产物
- `results.json`
  - 条件级聚合结果。
- `summary.json`
  - 当前运行的 dataset manifest、method manifest、协议信息和主结果摘要。
- `eval_reports/*.json`
  - 每个 condition / seed / budget 的详细评估报告。
- `visualizations/`
  - 数据标注可视化结果。
