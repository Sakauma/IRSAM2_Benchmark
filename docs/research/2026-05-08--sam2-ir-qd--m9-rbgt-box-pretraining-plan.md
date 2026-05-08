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
  - 负责 RBGT split 导出、RBGT box pretrain、public IR fine-tune、checkpoint 选择、public IR eval 和 analysis。
  - 支持 `--stage export/pretrain/finetune/select/eval/analysis/all`，可以断点续跑单个阶段。
  - 训练和评估都使用 `m9.gpus` 中的 GPU 列表。

- `scripts/select_auto_prompt_checkpoint.py`
  - 根据 checkpoint history、外部 metrics CSV 或 prompt-validation 数据集选择 M9 checkpoint。
  - mask 数据集使用 PromptHitRate、TargetRecallIoU25、PromptTopKHitRate、PromptBoxCoverage 和 FalseAlarmPixelsPerMP 综合打分。
  - box-only 数据集使用 PromptPointInBBox、PromptTopKInBBox 和 PromptBoxBBoxIoU 打分。

- `configs/server_auto_prompt_4090x8_m9_full.example.yaml`
  - M9 完整实验配置。
  - 包含 M9-A 到 M9-G 的默认变体定义所需字段。
  - RBGT-Tiny 标注路径使用 `annotations_voc`，导出的 COCO split 写在 RBGT-Tiny 同目录下。
  - RBGT pretrain 和 mixed 阶段默认使用 disk light cache，并显示 cache 构建进度。
  - `artifact_subdir` 为 `sam2_ir_qd_m9_full_v1`。

- `configs/server_auto_prompt_4090x8_m9_rbgt_pretrain.example.yaml`
  - M9 RBGT box pretraining 第一阶段旧配置。
  - 使用 `ir_prompt_v3_fpn`。
  - 仅保留为单阶段调试入口；正式实验优先使用 full 配置。

## 服务器命令

推荐先跑 smoke：

```bash
cd /project/IDIP/MAJ/code/IRSAM2_Benchmark

PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_full.example.yaml \
  --stage all \
  --smoke-test \
  --variants M9-E \
  --seeds 42 \
  --stop-on-error \
  --preflight-mode fast
```

完整 M9：

```bash
PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_full.example.yaml \
  --stage all \
  --stop-on-error \
  --preflight-mode fast
```

只重跑 eval：

```bash
PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_full.example.yaml \
  --stage eval \
  --stop-on-error \
  --preflight-mode fast
```

只重做 checkpoint 选择：

```bash
PYTHONPATH=src python scripts/run_m9_rbgt_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m9_full.example.yaml \
  --stage select \
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

当前工程已经实现 M9 完整编排：

- `M9-A`：PromptNetV2 public-only。
- `M9-B`：PromptNetV2 RBGT-pretrain-only。
- `M9-C`：PromptNetV2 RBGT-pretrain -> public fine-tune。
- `M9-D`：PromptNetV3-FPN public-only。
- `M9-E`：PromptNetV3-FPN RBGT-pretrain -> public fine-tune，作为主方法候选。
- `M9-F`：PromptNetV3-FPN public+RBGT mixed。
- `M9-G`：PromptNetV3-FPN staged no hard-negative。

runner 已支持：

- RBGT split 已存在时自动复用，不会重复导出。
- RBGT light cache 使用 shard 化磁盘缓存，首次构建显示进度条，后续运行显示 cache hit 并复用。
- 多进程同时启动同一份 RBGT cache 时使用构建锁，避免重复构建。
- RBGT pretrain checkpoint 自动注入 staged fine-tune。
- pretrain checkpoint 只作为 `M9-B` 评估对象；`M9-C/M9-E/M9-G` 只评估 final fine-tune checkpoint。
- `--stage select/eval/analysis` 可以从已有训练目录重建任务上下文。
- 每次运行写入 `m9_manifest_latest.json`、`m9_variant_summary.csv` 和 `m9_success_gate.json`。
