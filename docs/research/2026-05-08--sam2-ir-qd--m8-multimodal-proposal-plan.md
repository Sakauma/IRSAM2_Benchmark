# SAM2-IR-QD M8 MultiModal 小目标 Proposal 实验计划

日期：2026-05-08

## 目的

M5/M6 的结果说明当前瓶颈不是 SAM2 mask decoder，而是自动 proposal 对红外小目标的命中率不足。M8 的目标是用筛选后的 MultiModal 小目标样本训练 proposal 网络，同时保持 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三个公开数据集的性能不明显下降。

M8 不把 MultiModal 直接混入 benchmark 内部做动态过滤。它先在数据目录旁边导出确定性的 COCO train/val/test 标注，然后 benchmark 只读取这些固定 JSON。

## 实验问题

1. MultiModal 小目标训练能否提升 MultiModal-small-test 的自动 prompt 命中率。
2. 领域重采样能否避免 MultiModal 大量样本压过三个公开数据集。
3. 面积重采样和面积加权损失能否改善 tiny target 的 recall。
4. M8 是否能在公开 IR3 上保持 M6 的主要优势，同时缩小和监督式红外小目标模型的差距。

## 已实现代码

- `scripts/export_multimodal_small_target_coco.py`
  - 增加 `--split`，固定 seed 导出 train/val/test COCO 标注。
  - 默认 split 为 70% train、10% val、20% test。
  - 输出每个 split 的样本数、图像数、目标数和面积桶统计。
- `src/irsam2_benchmark/data/adapters.py`
  - COCO adapter 支持 `annotations_file`，可读取指定 split JSON。
  - 样本 metadata 增加 `gt_area_pixels` 和 `area_bucket`。
- `src/irsam2_benchmark/training/auto_prompt.py`
  - 支持 explicit validation dataset。
  - 支持 `domain_sampling_weights`。
  - 支持 `area_sampling_weights`。
  - 支持 `area_loss_weights`。
  - dense/light cache 均记录 sample weight 和 area bucket。
- `src/irsam2_benchmark/benchmark/auto_prompt_runner.py`
  - 训练配置生成支持 `validation_datasets`。
  - preflight summary 覆盖 train、eval、gpu cache、light cache、validation 配置。
- `configs/server_auto_prompt_4090x8_m8_multimodal_proposal.example.yaml`
  - M8 主实验配置。
  - 训练使用 GPU 5、6、7。
  - eval 使用 GPU 0 到 7。
  - 三个 seed：42、123、456。

## 服务器运行命令

先导出 MultiModal 小目标 split：

```bash
cd /project/IDIP/MAJ/code/IRSAM2_Benchmark

PYTHONPATH=src python scripts/export_multimodal_small_target_coco.py \
  --root /project/IDIP/MAJ/dataset/data/MultiModal \
  --split \
  --overwrite
```

smoke 测试：

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m8_multimodal_proposal.example.yaml \
  --smoke-test \
  --stop-on-error \
  --preflight-mode fast
```

完整实验：

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x8_m8_multimodal_proposal.example.yaml \
  --stop-on-error \
  --preflight-mode fast
```

建议使用 tmux：

```bash
tmux new -s irsam2_m8
```

## 预期输出

主 artifact：

```bash
artifacts/sam2_ir_qd_m8_multimodal_proposal_v1/
```

关键文件：

```bash
artifacts/sam2_ir_qd_m8_multimodal_proposal_v1/train/
artifacts/sam2_ir_qd_m8_multimodal_proposal_v1/runs/
artifacts/sam2_ir_qd_m8_multimodal_proposal_v1/analysis/auto_prompt_m8_multimodal_proposal/large/checkpoint_sweep_summary.csv
```

MultiModal split 标注：

```bash
/project/IDIP/MAJ/dataset/data/MultiModal/annotations_coco_small_target_train/instances_multimodal_small_target_train.json
/project/IDIP/MAJ/dataset/data/MultiModal/annotations_coco_small_target_val/instances_multimodal_small_target_val.json
/project/IDIP/MAJ/dataset/data/MultiModal/annotations_coco_small_target_test/instances_multimodal_small_target_test.json
```

## 初始判定标准

M8 值得进入论文主线，需要同时满足：

- MultiModal-small-test `PromptHitRate >= 0.12`。
- MultiModal-small-test `PromptBoxCoverage >= 0.08`。
- MultiModal-small-test `TargetRecallIoU25` 相比 M6 明显提升。
- 公开 IR3 macro mIoU 相比 M6 `sam2_ir_fa_rerank` 下降不超过 0.03。
- FApx/MP 不出现数量级恶化。

如果 MultiModal 指标提升但公开 IR3 明显下降，M8 只能作为域扩展失败分析或后续域适配动机，不能作为主方法结论。

## 对比矩阵更新

M8 完成 analysis 后，把它加入第三方模型对比矩阵：

```bash
python scripts/build_comparison_evaluation_matrix.py \
  --ours-analysis M6=artifacts/sam2_ir_qd_m6_promptnet_v2_v1/analysis/checkpoint_sweep_summary.csv \
  --ours-analysis M8=artifacts/sam2_ir_qd_m8_multimodal_proposal_v1/analysis/auto_prompt_m8_multimodal_proposal/large/checkpoint_sweep_summary.csv
```

如果 M8 的 analysis CSV 路径和上面不同，先用下面命令查找：

```bash
find artifacts/sam2_ir_qd_m8_multimodal_proposal_v1 -name checkpoint_sweep_summary.csv
```

## 消融顺序

先跑当前 M8-D 主配置。若 M8-D 有正向结果，再做以下消融：

1. M8-A：只加入 MultiModal-small-train，不启用 domain/area 权重。
2. M8-B：启用 domain sampling，不启用 area sampling/loss。
3. M8-C：启用 domain sampling 和 area sampling，不启用 area loss。
4. M8-D：启用 domain sampling、area sampling、area loss。

如果 M8-D 不提升 MultiModal-small-test 的 prompt 命中率，则不要继续扩大训练量，应回到 proposal 网络结构和目标函数。

## 论文方向影响

M8 的正向结果可以支撑论文主张：

红外小目标 SAM2 迁移的核心难点是自动 prompt proposal，而不是单纯 mask refinement。通过小目标筛选、领域平衡采样和面积敏感训练，可以让 SAM2 在无人工 prompt 场景下获得更可用的红外目标候选。

M8 的负向结果也有价值：

如果筛选后的 MultiModal 仍不能提升 public IR3 或 MultiModal-small-test，说明 box-only 或 polygon-derived weak supervision 对 SAM2 自动 prompt 的迁移不足，下一步应转向 teacher-guided pseudo prompt、特征级蒸馏或更强 proposal backbone。
