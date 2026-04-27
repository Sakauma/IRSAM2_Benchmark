# Artifact Schema 规范

每次单组 benchmark 运行都必须写出以下文件：

- `benchmark_spec.json`
- `run_metadata.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`

一个 baseline run 只有在 `eval_reports/rows.json` 非空且 `summary.json` 中的 `mean` 非空时，才应被矩阵 runner 视为完成。所有样本都失败时必须保留 `eval_reports/error_log.jsonl`，并让上层 runner 记录失败。

`summary.json` 还应记录 run health 字段：

- `expected_sample_count`
- `expected_eval_units`
- `expected_row_count`
- `row_count`
- `error_count`
- `missing_row_count`
- `failure_rate`
- `failure_rate_threshold`

当 `failure_rate` 大于 `failure_rate_threshold` 时，矩阵 runner 应把该 run 视为失败或不完整。

`summary.json` 还应记录 `runtime_resources`：

- `wall_time_s`
- `samples_per_s`
- `eval_units_per_s`
- `rows_per_s`
- `cuda_available`
- `cuda_peak_memory_bytes`

`artifact_manifest.json` 至少需要记录：

- stage 名称
- artifact 目录
- artifact 标识符
- stage 元数据

这是 benchmark 可复现性的锚点之一。

## Run Metadata

每个正式 benchmark run 都会写出：

- `run_metadata.json`：记录命令、展开后的配置、数据集路径、checkpoint 路径/大小/mtime/sha256、SAM2 repo、Git commit、Python、PyTorch、CUDA 和 GPU 信息。

`benchmark_spec.json` 和 `run_metadata.json` 必须包含 `fingerprints`。至少应包含当前 run config 的 `config_file_sha256`。由完整矩阵 runner 生成的配置还会包含源完整 YAML 的 `source_config_sha256`。

`run_metadata.json` 同时包含 `runtime_resources`，用于后续追溯 wall time、吞吐和 CUDA peak memory。

## Schema 验证

可以用 CLI 检查单个 run artifact：

```bash
python -m irsam2_benchmark.cli validate artifacts --run-dir artifacts/paper_5090/runs/mask/tiny/...
```

该命令会验证关键文件、JSON 可读性、非空结果行、health 字段、非有限数值和 `eval_unit` 协议一致性。

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
每个子进程还应写入 `logs/{suite}/{checkpoint}/{dataset}_{method}.log`；失败记录应包含 `log_path` 和日志尾部摘要。

`benchmark_manifest_latest.json` 和 `run_manifest_latest.*` 会记录生成 run config 的 `config_sha256`。分析配置记录会包含 `analysis_config_sha256` 和对应 matrix 的 `matrix_config_sha256`。
