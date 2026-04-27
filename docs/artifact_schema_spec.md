# Artifact Schema 规范

每次单组 benchmark 运行都必须写出以下文件：

- `benchmark_spec.json`
- `run_metadata.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`

一个 baseline run 只有在 `eval_reports/rows.json` 非空且 `summary.json` 中的 `mean` 非空时，才应被矩阵 runner 视为完成。所有样本都失败时必须保留 `eval_reports/error_log.jsonl`，并让上层 runner 记录失败。

`artifact_manifest.json` 至少需要记录：

- stage 名称
- artifact 目录
- artifact 标识符
- stage 元数据

这是 benchmark 可复现性的锚点之一。

## Run Metadata

每个正式 benchmark run 都会写出：

- `run_metadata.json`：记录命令、展开后的配置、数据集路径、checkpoint 路径/大小/mtime/sha256、SAM2 repo、Git commit、Python、PyTorch、CUDA 和 GPU 信息。

## 完整矩阵附加产物

通过 `scripts/run_5090_full_benchmark.py` 运行完整矩阵时，矩阵根目录还会额外写出：

- `benchmark_manifest_latest.json`
- `run_manifest_latest.json`
- `run_manifest_latest.csv`
- `run_failures_latest.json`
- `run_failures_latest.csv`
- `analysis/checkpoint_sweep_summary.json`
- `analysis/checkpoint_sweep_summary.csv`

`run_manifest_latest.*` 汇总成功、失败、dry-run 和断点续跑跳过的组合。`run_failures_latest.*` 记录失败组合，包含 suite、checkpoint、数据集、方法、输出目录、命令、返回码和错误消息。
