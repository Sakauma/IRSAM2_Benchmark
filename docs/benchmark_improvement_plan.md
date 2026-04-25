# Benchmark 改进计划

本文档记录当前 `IRSAM2_Benchmark` 平台在测试 SAM2 红外小目标分割 baseline 时的可用边界、已知风险和改进动作。

评估日期：2026-04-25

## 实施状态

已完成第一轮 P0/P1 改进：

- Track A prompted segmentation 与 Track B no-prompt automatic mask 已拆分，默认不再做跨 track paired test。
- box-only 样本不再输出 mask-only 指标。
- 分析表中缺失指标输出为 `null` / 空 CSV 单元格，并保留 `<metric>_count`。
- 已增加 `BoundaryF1Exact` 与 `BoundaryF1Tol1`，论文配置优先使用 `BoundaryF1Tol1`。
- 普通 paper runner 已持久化展开后的 run config。
- 正式 baseline run 已输出 `run_metadata.json`。

## 结论摘要

当前平台可以用于测试 SAM2 prompted baseline 在 mask-supervised 红外数据集上的效果。

推荐优先使用以下数据集：

- `NUAA-SIRST`
- `NUDT-SIRST`
- `IRSTD-1K`
- `MultiModal`

推荐优先使用以下方法：

- `sam2_box_oracle`
- `sam2_tight_box_oracle`
- `sam2_point_oracle`
- `sam2_box_point_oracle`

推荐主指标：

- `mIoU`
- `Dice`
- `TargetRecallIoU25`
- `FalseAlarmPixelsPerMP`

当前平台不建议直接把全部矩阵结果作为论文最终结论。主要原因是部分 track 的评估语义不同，部分分析表会把缺失指标显示成 `0.0`，普通 paper runner 的展开配置没有持久保存。

## 对 SAM2 Baseline 测试的影响

### 影响较小的问题

这些问题不会改变一次 prompted SAM2 baseline 的原始推理结果，但会影响复现、归档或论文写作：

- `run_paper_experiments.py` 会把展开后的单次运行配置写入临时目录，运行结束后配置文件消失。
- `benchmark_spec.json` 中的 `config_path` 可能指向已经不存在的临时 YAML。
- `.gitignore` 忽略了一批仍被 Git 跟踪的 legacy 配置和脚本，会影响维护者用 `rg --files` 检索文件。

### 影响解释的问题

这些问题会影响结果表和结论解释：

- no-prompt automatic mask 是图像级评估，box/point/box+point 是实例级 prompted segmentation，不能直接混成同一个主表解释。
- `BoundaryF1` 当前没有容忍半径，对红外小目标的 1 像素偏移很敏感，只适合作为辅助指标。
- 分析表中缺失指标可能被汇总成 `0.0`，这会把“没有这个指标”和“真实得分为 0”混淆。

### 需要禁止误用的问题

这些问题可能直接产生误导性 benchmark 结果：

- `RBGT-Tiny` 在当前平台中主要是 box-only 补充数据集，不应进入主 mask `mIoU/Dice` 表。
- box-only 样本如果走普通 segmentation evaluator，可能用全零 GT mask 计算 mask 指标。
- Track A prompted segmentation 与 Track B no-prompt automatic mask 的 paired statistical test 语义不一致，默认不应作为同一配对比较。

## 风险清单

| 优先级 | 问题 | 影响 | 建议动作 |
| --- | --- | --- | --- |
| P0 | 普通 paper runner 不持久化展开后的运行配置 | 影响复现实验和审稿追溯 | 将生成的 YAML 写入 artifact 目录，并在 `benchmark_spec.json` 记录持久路径 |
| P0 | 缺失指标在分析表中显示为 `0.0` | 可能误读为真实零分 | 缺失指标输出 `null` 或 `NA`，并在表格中记录 count |
| P0 | box-only 数据可能产生 mask 指标 | 可能产生无意义 `mIoU/Dice` | 对 `supervision_type=bbox` 禁止输出 mask metrics，或单独走 bbox/prompt 指标 |
| P0 | Track A 与 Track B 混合配对统计 | 统计结论不成立 | 拆分 image-level 与 instance-level 分析，默认禁止跨 track paired test |
| P1 | `BoundaryF1` 没有容忍半径 | 小目标边界指标过于苛刻 | 增加 tolerance-based BoundaryF1，并保留当前 exact 版本作诊断 |
| P1 | 矩阵中未知 method 被静默跳过 | 配置拼写错误不容易发现 | 对未知 dataset/method fail-fast |
| P1 | 5090 runner 与普通 paper runner 的 artifact 元数据不一致 | 归档格式不统一 | 统一生成 `run_metadata.json`，记录命令、配置、Git、checkpoint 和环境 |
| P2 | `.gitignore` 忽略已跟踪 legacy 文件 | 日常检索和维护容易漏文件 | 清理 ignore 规则，或正式迁移 legacy 文件到 archive |

## 改进方案

### 1. 持久化运行配置

目标：每个 benchmark run 都能从 artifact 中恢复完整配置。

建议修改：

- 在 `scripts/run_paper_experiments.py` 中增加 `generated/run_configs/` 输出目录。
- 每个 run 的 YAML 写入 artifact root 下的持久路径。
- `benchmark_spec.json` 记录持久配置路径和矩阵来源。
- dry-run 时打印持久配置路径，而不是临时路径。

验收标准：

- 运行结束后，每个 output dir 能找到对应 generated config。
- `benchmark_spec.json` 中的 `config_path` 指向存在的文件。
- 删除临时目录后，仍能复跑同一 run。

### 2. 修正缺失指标语义

目标：区分“指标不存在”和“指标真实为 0”。

建议修改：

- 分析表对缺失指标输出 `null` 或空值。
- 每个指标保留 `<metric>_count`。
- 当 `<metric>_count == 0` 时，不输出 `<metric>_mean=0.0`。
- Markdown 报告中标注该方法不适用的指标。

验收标准：

- no-prompt automatic mask 的主表不会出现误导性的 `mIoU_mean=0.0`。
- 缺失指标在 JSON/CSV/Markdown 中语义一致。

### 3. 约束 box-only 数据的评估路径

目标：避免 `RBGT-Tiny` 等 box-only 数据被误当 mask-supervised 数据集。

建议修改：

- 在 evaluator 入口检查 `Sample.supervision_type`。
- 当样本为 `bbox` 且没有 `mask_array` 时，不计算 `mIoU`、`Dice`、`BoundaryF1`、小目标 mask recall。
- 为 box-only 数据输出 `BBoxIoU`、prompt coverage 相关指标和 artifact 记录。
- 分析配置默认排除 box-only 数据的 mask table。

验收标准：

- `RBGT-Tiny` 不进入主 mask `mIoU/Dice` 表。
- box-only run 的结果文件不会产生全零 GT mask 的 segmentation 指标。

### 4. 拆分 Track A 与 Track B 分析

目标：保证不同 track 的比较语义清楚。

建议修改：

- Track A：实例级 prompted segmentation 表。
- Track B：图像级 automatic mask instance matching 表。
- 跨 track 只允许写作“辅助对照”或“不同任务设置下的趋势”，默认不做 paired statistical test。
- 如果要比较 Track A 和 Track B，需要新增 image-level 聚合协议，先把 Track A 也聚合到图像级。

验收标准：

- `sam2_no_prompt_auto_mask` 不再和 `sam2_box_oracle` 默认进入同一个 paired test。
- 分析报告中明确标注 Track A 和 Track B 的评估单位。

### 5. 增加 tolerance-based BoundaryF1

目标：让边界指标更符合小目标分割评估习惯。

建议修改：

- 保留当前 exact boundary F1，命名为 `BoundaryF1Exact`。
- 新增 `BoundaryF1Tol1` 或 `BoundaryF1Tol2`。
- 指标卡片说明 tolerance 半径和适用场景。

验收标准：

- exact 与 tolerance 版本同时出现在 eval rows。
- 论文主表优先使用 tolerance 版本或仅把 exact 版本作为诊断指标。

### 6. 统一 run metadata

目标：所有正式运行都有足够的审计信息。

建议记录：

- 展开后的完整 config。
- 命令行参数。
- Git commit、branch、dirty status。
- Python、PyTorch、CUDA、GPU 信息。
- SAM2 repo 路径和 commit。
- checkpoint 路径、大小、mtime、sha256。
- 数据集 root 和样本计数。

验收标准：

- 每个 output dir 都有 `run_metadata.json`。
- 5090 runner 与普通 paper runner 的 metadata 字段一致。

## 推荐的当前使用方式

在上述问题修复前，建议按以下规则使用平台测试 SAM2 baseline：

1. 主表只使用 mask-supervised 红外数据集。
2. 主表只比较 `sam2_box_oracle`、`sam2_tight_box_oracle`、`sam2_point_oracle`、`sam2_box_point_oracle`。
3. 主结论优先看 `mIoU`、`Dice`、`TargetRecallIoU25`、`FalseAlarmPixelsPerMP`。
4. `BoundaryF1` 只作为辅助诊断。
5. `RBGT-Tiny` 单独作为 box-only 补充实验。
6. `sam2_no_prompt_auto_mask` 单独作为 Track B automatic mask 对照，不混入 prompted baseline 主表。
7. 写论文前，保留每次运行的 generated config 和 manifest。

## 建议执行顺序

1. 修复配置持久化和 run metadata。
2. 修复缺失指标输出语义。
3. 增加 box-only supervision guard。
4. 拆分 Track A 和 Track B 分析表。
5. 增加 tolerance-based BoundaryF1。
6. 清理 `.gitignore` 中对已跟踪文件的忽略规则。

## 最小验证清单

每次修复后至少运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.example.yaml --group p0_auto_prompt --dry-run
python scripts/run_5090_full_benchmark.py --paths configs/local_paths.example.yaml --suites mask --checkpoints tiny --modes box --dry-run
```

如果本机具备 SAM2 repo、checkpoint 和数据集，再运行一个真实 smoke run：

```bash
python scripts/run_5090_full_benchmark.py --paths configs/local_paths.yaml --suites mask --checkpoints tiny --modes box --smoke-test --no-analysis
```
