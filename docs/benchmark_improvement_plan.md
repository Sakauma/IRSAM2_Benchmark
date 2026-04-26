# Benchmark 改进计划

本文档记录 `IRSAM2_Benchmark` 平台在测试 SAM2 红外小目标分割 baseline 时的当前状态、已完成改进和剩余工程事项。

评估日期：2026-04-26

## 当前结论

当前平台可以用于测试 SAM2 prompted baseline 在 mask-supervised 红外数据集上的效果。
一次 run 只有在产出非空 `eval_reports/rows.json` 和非空 `summary.mean` 时，才应被视为有效完成。

推荐主实验数据集：

- `NUAA-SIRST`
- `NUDT-SIRST`
- `IRSTD-1K`
- `MultiModal`

推荐主表方法：

- `sam2_box_oracle`
- `sam2_tight_box_oracle`
- `sam2_point_oracle`
- `sam2_box_point_oracle`

推荐主指标：

- `mIoU`
- `Dice`
- `BoundaryF1Tol1`
- `TargetRecallIoU25`
- `FalseAlarmPixelsPerMP`

`RBGT-Tiny` 仍应作为 box-only 弱标注补充实验，不应进入主 mask 指标表。`sam2_no_prompt_auto_mask` 是 Track B 图像级 automatic mask baseline，应和 Track A prompted segmentation 分开解释。

## 已完成改进

第一轮 P0/P1 改进已经完成：

- Track A prompted segmentation 与 Track B no-prompt automatic mask 已拆分，默认不做跨 track paired test。
- box-only 样本不再输出 mask-only 指标。
- 分析表中缺失指标输出为 `null` 或空 CSV 单元格，并保留 `<metric>_count`。
- 已增加 `BoundaryF1Exact` 与 `BoundaryF1Tol1`，论文配置优先使用 `BoundaryF1Tol1`。
- evaluator 会对预测 mask 做尺寸对齐，并记录 resize 元数据。
- prompted SAM2 支持 batch 推理，CUDA OOM 时会拆 batch fallback。
- 单图异常会写入 `eval_reports/error_log.jsonl`，不会直接中断整个 run。
- 如果所有样本都预测失败，baseline/evaluate 会返回非零退出码，避免空指标被记录为成功。
- 正式 baseline run 会输出 `run_metadata.json`。
- `run_paper_experiments.py` 会持久化展开后的 generated config，并在 generated config 中写入绝对 artifact、reference、dataset、SAM2 repo 和 checkpoint 路径。

## 本轮处理结果

| 优先级 | 问题 | 处理结果 |
| --- | --- | --- |
| P0 | 普通 paper runner 的 generated config 曾经保留相对 `runtime.artifact_root` | 已改为写入绝对路径，避免输出落到 `generated/run_configs/.../artifacts` |
| P0 | SAM2 加载失败时可能写出空指标但 run 返回成功 | 已改为零有效评估行时报错，5090 runner 不再把空 `rows.json` 或空 `summary.mean` 当作可 resume 的完成结果 |
| P1 | 改进文档中曾经保留旧风险描述 | 已改为当前状态和已完成事项 |
| P2 | `artifact_schema_spec.md` 有重复描述 | 已删除重复句子 |

## 推荐使用方式

在正式跑完整 benchmark 时，优先使用：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --dry-run
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --smoke-test
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml
```

在使用普通 paper runner 时，先 dry-run 并确认 generated config 和 output dir：

```bash
python scripts/run_paper_experiments.py \
  --matrix configs/paper_experiments_v1.yaml \
  --paths configs/local_paths.yaml \
  --group p0_all \
  --dry-run
```

正式论文主表建议只混合相同 eval unit 的方法：

- Track A：实例级 prompted segmentation。
- Track B：图像级 automatic mask instance matching。
- Box-only：只报告 bbox/prompt 相关指标。

## 验证清单

每次修改 runner、analysis 或 metric 逻辑后至少运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.example.yaml --group p0_auto_prompt --dry-run
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.example.yaml --suites mask --checkpoints tiny --modes box --dry-run
```

如果本机具备 SAM2 repo、checkpoint 和数据集，再运行一个真实 smoke run：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --suites mask --checkpoints tiny --modes box --smoke-test --no-analysis
```
