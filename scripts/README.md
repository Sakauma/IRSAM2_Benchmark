# 脚本索引

本目录提供面向本地 Linux、Windows PowerShell 和 AutoDL 风格环境的常用入口。

## 本地验证

- `run_smoke.sh`：Linux 下的 smoke baseline 包装脚本。
- `run_smoke.ps1`：Windows PowerShell 下的 smoke baseline 包装脚本。
- `run_baseline.sh`：通用 Linux baseline 包装脚本。
- `run_tests.sh`：Linux 单元测试包装脚本。

## Benchmark 自动化

- `run_official_baseline_matrix.sh`：官方 SAM2 baseline matrix 的 shell 入口。
- `run_official_baseline_matrix.py`：从完整 benchmark YAML 展开模型、数据集和模式组合的矩阵驱动脚本。
- `run_paper_experiments.py`：展开 IR-only 论文实验矩阵，支持 `--paths`、`--group` 与 `--dry-run`。
- `run_5090_full_benchmark.py`：面向单张 RTX 5090 的完整基准入口，展开 4 个 SAM2.1 checkpoint、4 种主 prompt policy、全部论文数据集，并额外运行 tight-box/loose-box 协议诊断。
- `run_5090_micro_benchmark.py`：和 5090 完整基准展开相同组合，但每个数据集只跑前 24 张图像，输出到 `paper_5090_micro/`。
- `analyze_paper_results.py`：分析已完成的论文实验 artifacts，生成表格、显著性检验、错误分桶和 Markdown 报告。

官方矩阵默认启用断点续跑：`MATRIX_RESUME=1`。如果某个组合已经包含完整的 `summary.json`、`results.json`、`eval_reports/rows.json`，并且 `run_metadata.json` 标记为 `completed`，脚本会跳过该组合并把它记为 `skipped`。如需强制重跑，设置：

```bash
MATRIX_RESUME=0 bash scripts/run_official_baseline_matrix.sh
```

官方矩阵脚本默认读取 `configs/server_benchmark_full.local.yaml`。如需指定其他完整配置：

```bash
BENCHMARK_CONFIG=configs/server_benchmark_full.local.yaml bash scripts/run_official_baseline_matrix.sh
```

每个组合会在自己的输出目录写入 `run_metadata.json`，记录命令、配置、数据集路径、checkpoint 信息、关键环境变量、Git commit、Python 版本和 GPU 信息。矩阵根目录还会生成：

- `matrix_summary.json` / `matrix_summary.csv`：成功或跳过组合的汇总。
- `matrix_failures.json` / `matrix_failures.csv`：失败组合清单。存在失败时脚本会继续跑剩余组合，但最终返回非零退出码。

单图级异常不会中断当前 run。预测、指标计算或可视化阶段的坏图会写入该 run 的 `eval_reports/error_log.jsonl`，后续可按 `sample_id`、`image_path`、`stage` 和 `error_message` 定位并单独重跑。

## AutoDL 辅助脚本

- `setup_autodl_server.sh`：初始化数据、输出和 checkpoint 路径。
- `run_autodl_smoke.sh`：面向 `MultiModal` 或 `RBGT-Tiny` 的 smoke 运行辅助脚本。

所有脚本都默认以仓库根目录作为工作目录锚点，并在需要时自动把 `src/` 注入到 `PYTHONPATH`。

## 5090 单卡完整基准

先复制完整配置模板，并在其中填写服务器路径、模型、数据集、方法、seed 和分析参数：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
```

然后检查展开的命令：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --dry-run
python scripts/run_5090_micro_benchmark.py --config configs/server_benchmark_full.local.yaml --dry-run
```

正式运行：

```bash
python scripts/run_5090_micro_benchmark.py --config configs/server_benchmark_full.local.yaml --stop-on-error
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml
```

详细说明见 `docs/server_5090_benchmark.md`。
