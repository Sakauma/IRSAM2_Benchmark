# 实现说明

本文档记录了全新 `IRSAM2_Benchmark` 工程的第一轮实现情况。

## 本轮已实现

- 在 `IRSAM2_Benchmark/` 下创建了新的独立工程结构
- 增加了基于 package 的 CLI 与配置加载器
- 增加了以下数据集 adapter：
  - 原始 `MultiModal`
  - COCO-like 数据集
  - `RBGT-Tiny` 红外单分支视图
  - 通用 `images/ + masks/` 数据集
- 为 mask-only 数据集增加了确定性的 prompt synthesis：
  - tight box
  - adaptive loose box
  - point prompt
- 增加了一等公民化的 SAM2 baseline：
  - `BBoxRectMaskBaseline`
  - `ZeroShotSAM2`
  - `NoPromptAutoMaskSAM2`
  - `ZeroShotSAM2VideoPropagation`
- 增加了 benchmark 治理字段：
  - `benchmark_version`
  - `split_version`
  - `prompt_policy_version`
  - `metric_schema_version`
  - `reference_result_version`
- 增加了分组评估报告与参考结果快照生成

## 已完成验证

- WSL 下通过了 Python 语法验证
- `sam_hq2` 环境下单元测试已通过
- 在 `MultiModalCOCOClean` 上完成了 `bbox_rect` 的端到端 baseline 运行
- 已完成真实 `SAM2` smoke 验证：
  - `sam2_zero_shot`
  - `sam2_zero_shot_point`
  - `sam2_no_prompt_auto_mask`

## 当前边界

平台骨架、结果 schema、数据层和 baseline 层已经可运行。

完整 pipeline 中的以下 stage：

- `adapt`
- `distill`
- `quantize`

虽然已经具备稳定的 stage 接口和 artifact scaffold，但目前还不是最终论文方法。
