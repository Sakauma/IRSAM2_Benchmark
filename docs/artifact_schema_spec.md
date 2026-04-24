# Artifact Schema 规范

每次运行都必须写出以下文件：

- `benchmark_spec.json`
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
