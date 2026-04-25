# Artifact Schema 规范

每次单组 benchmark 运行都必须写出以下文件：

- `benchmark_spec.json`
- `run_metadata.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`

`artifact_manifest.json` 至少需要记录：

- stage 名称
- artifact 目录
- artifact 标识符
- stage 元数据

这是 benchmark 可复现性的锚点之一。

## Run Metadata

每个正式 benchmark run 都会写出：

- `run_metadata.json`：记录命令、展开后的配置、数据集路径、checkpoint 路径/大小/mtime/sha256、SAM2 repo、Git commit、Python、PyTorch、CUDA 和 GPU 信息。

## 官方矩阵附加产物

通过 `scripts/run_official_baseline_matrix.py` 运行官方矩阵时，矩阵根目录还会额外写出：

矩阵根目录会额外写出：

- `matrix_summary.json`
- `matrix_summary.csv`
- `matrix_failures.json`
- `matrix_failures.csv`

`matrix_summary.*` 汇总成功和断点续跑跳过的组合。`matrix_failures.*` 记录失败组合，包含数据集、模型、模式、输出目录、错误类型、返回码、错误消息、开始/结束时间和耗时。
