# SAM2-IR-QD M9 RBGT-Tiny Box 预训练计划

日期：2026-05-08

## 当前结论

M8 说明 MultiModal 小目标混训不能稳定提升自动 prompt proposal。它在 MultiModal-small-test 上只带来很小提升，同时使 public IR3 的 PromptHitRate、TargetRecallIoU25 和 FalseAlarmPixelsPerMP 变差。

已有 oracle 结果说明 SAM2 decoder 在红外小目标上仍然有能力。只要 box 或 box+point prompt 准确，SAM2 可以得到较高的分割质量。因此当前瓶颈不是 mask decoder，而是自动 prompt proposal。

RBGT-Tiny 比 MultiModal 更适合作为 M9 的弱监督数据源。它规模大，标注形式是 box，正好对应自动 prompt proposal 的 objectness、point 和 box 学习目标。

## 新论文主线

M9 成功后，论文主线应调整为：

> 大规模 box 弱监督驱动的 SAM2 红外小目标自动提示迁移框架。

更具体地说：

> SAM2 在红外小目标分割中的主要瓶颈不是 mask decoder，而是自动 prompt proposal。我们利用 RBGT-Tiny 的大规模 box 标注进行弱监督 prompt proposal 预训练，再通过少量 mask-supervised 红外小目标数据校准，并结合 false-alarm-aware reranking，实现无需人工 prompt 的 SAM2 红外小目标分割。

## 贡献结构

1. Prompt-centric benchmark。
   - 统一评估 SAM2 类方法在红外小目标上的 prompted segmentation 能力。
   - 报告传统 mIoU/Dice，也报告 PromptHitRate、PromptTopKHitRate、TargetRecallIoU25、PromptBoxCoverage 和 FalseAlarmPixelsPerMP。

2. Box-supervised infrared prompt discovery。
   - 使用 RBGT-Tiny box-only 标注训练红外自动 prompt proposal。
   - 不把 RBGT-Tiny 混入 mask mIoU 主表。
   - 用 bbox-only prompt 指标验证 proposal 是否学到可迁移的目标定位能力。

3. Two-stage transfer。
   - 先做 RBGT-Tiny box pretraining。
   - 再用 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 做 public IR mask fine-tuning。
   - 最终只在 public mask benchmark 上判断主方法是否成立。

4. False-alarm-aware reranking。
   - 保留 M3/M4/M6 中有效的 SAM2 mask-feedback reranking。
   - 用它控制红外 clutter 导致的假阳性。

## M9 实验问题

1. RBGT-Tiny box pretraining 是否提升 public IR3 的 PromptHitRate。
2. RBGT-Tiny box pretraining 是否提升 TargetRecallIoU25 和 mIoU。
3. 大容量 PromptNetV3-FPN 是否优于 PromptNetV2。
4. staged pretraining 是否优于 public+RBGT 简单混训。
5. false-alarm-aware reranking 是否仍然是 M9 主方法中不可缺少的模块。

## 已加入的代码入口

- `scripts/export_rbgt_tiny_box_coco.py`
  - 将 RBGT-Tiny VOC box 标注导出为固定 COCO train/val/test split。
  - 默认按 sequence split，避免相邻帧泄漏。
  - 支持 `--small-target-filter`。

- `scripts/run_m9_rbgt_auto_prompt.py`
  - M9 专用入口。
  - 当前复用现有 auto-prompt runner，因此保留已有 8 卡并行训练、8 卡评估、断点跳过和 manifest 机制。

- `scripts/select_auto_prompt_checkpoint.py`
  - 根据 checkpoint history 或外部 metrics CSV 选择 M9 checkpoint。
  - 支持 prompt-centric score。

- `configs/server_auto_prompt_4090x8_m9_rbgt_pretrain.example.yaml`
  - M9 RBGT box pretraining 第一阶段配置。
  - 使用 `ir_prompt_v3_fpn`。
  - 训练 GPU 使用 0-7。
  - 评估 GPU 使用 0-7。

## 服务器命令

先导出 RBGT split：

```bash
cd /project/IDIP/MAJ/code/IRSAM2_Benchmark

PYTHONPATH=src python scripts/export_rbgt_tiny_box_coco.py \
  --root /project/IDIP/Dataset/RBGT-Tiny \
  --split \
  --small-target-filter \
  --overwrite
```

smoke：

```bash
PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_rbgt_pretrain.example.yaml \
  --smoke-test \
  --stop-on-error \
  --preflight-mode fast
```

完整 RBGT pretraining：

```bash
PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_rbgt_pretrain.example.yaml \
  --stop-on-error \
  --preflight-mode fast
```

## 判定标准

M9 可以进入论文主线，需要同时满足：

- public IR3 macro mIoU 高于 M6 `sam2_ir_fa_rerank`。
- public IR3 PromptHitRate 高于 M6。
- public IR3 TargetRecallIoU25 高于 M6。
- FalseAlarmPixelsPerMP 不高于 M6 的 1.2 倍。
- PromptNetV3-FPN 明显优于 PromptNetV2。
- RBGT pretraining 明显优于 public-only 大容量模型。

如果只提升 RBGT bbox-only 指标，但 public IR3 不提升，则 M9 只能作为弱监督迁移失败分析。

## 当前工程状态

当前提交实现了 M9 的第一阶段：RBGT split、PromptNetV3-FPN、初始化 checkpoint 支持、M9 pretrain 配置和叙事文档。

完整 two-stage 自动编排仍需在下一步增强：

1. 自动运行 RBGT pretrain。
2. 自动把 selected RBGT checkpoint 注入 public IR fine-tune。
3. 自动运行 M9-A 到 M9-G 的完整消融矩阵。
4. 自动生成 M9 与 M6/M8/第三方模型的对比表。
