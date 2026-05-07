# 脚本索引

当前保留四个主要脚本入口。

## `run_5090_full_benchmark.py`

完整 benchmark runner。它读取 `configs/server_benchmark_full.local.yaml`，展开 suite/checkpoint/dataset/method 组合，生成单 run config，调用 `main.py run baseline`，并在需要时执行分析。

常用命令：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --dry-run
```

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --smoke-test \
  --stop-on-error
```

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml
```

筛选组合：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny \
  --modes box,point
```

## `analyze_paper_results.py`

分析已完成的 benchmark artifacts，不重新运行模型。

```bash
python scripts/analyze_paper_results.py \
  --analysis artifacts/paper_5090/generated/analysis_configs/mask/tiny.yaml
```

分析配置必须显式传入 `--analysis`。矩阵脚本会为每个需要分析的 suite/checkpoint 自动生成该文件。

## `build_comparison_evaluation_matrix.py`

汇总第三方模型导出的 mask 与当前 SAM2-IR-QD analysis CSV，生成论文方向判断用的对比评估表。默认读取 `artifacts/external_predictions/`、本地 `/home/sakauma/dataset` 数据集，以及 M5/M6 analysis 输出。

```bash
python scripts/build_comparison_evaluation_matrix.py
```

输出目录默认为 `artifacts/comparison_evaluation_matrix_latest/`，包含：

- `comparison-report.md`
- `manifest.json`
- `tables/external_public_dataset.csv`
- `tables/external_public_macro.csv`
- `tables/ours_public_dataset.csv`
- `tables/ours_public_macro.csv`
- `tables/comparison_public_macro.csv`

注意：第三方 mask 的 `WholeMaskIoU25Rate/50Rate` 是快速代理召回；论文最终目标级 `TargetRecallIoU25/50` 仍需用 benchmark evaluator 正式生成。

## `run_4090x4_auto_prompt.py`

SAM2-IR-QD M1 自动 prompt runner。它先训练 learned IR auto prompt，再用 4 张 4090 并行跑 E2 自动 prompt 评估。

```bash
python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.local.yaml \
  --dry-run
```

```bash
python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.local.yaml \
  --smoke-test \
  --stop-on-error
```

正式运行建议放在 tmux 中：

```bash
python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.local.yaml
```

默认会显示 auto-prompt 训练进度条和 E2 run 级进度条；并行 E2 子进程内部 sample/image 进度条会关闭，避免 4 卡输出互相覆盖。需要把子进程普通日志也转发到终端时，加 `--stream-logs`。

## 运行日志

需要把控制台输出保存为日志时：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  2>&1 | tee artifacts/paper_5090/full_run.log
```

## 设计约束

- 不再保留单独 quick/smoke 包装脚本。
- 不再保留旧版 paper runner。
- 不再保留拆分式双 YAML 入口。
- smoke 和 dry-run 都通过 `run_5090_full_benchmark.py` 参数实现。
