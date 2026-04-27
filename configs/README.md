# 配置索引

## 清理结论

- 当前没有删除 tracked YAML。它们仍被脚本、测试或文档引用。
- `server_benchmark_full.example.yaml` 是新服务器运行的推荐模板。
- `paper_experiments_v1.yaml`、`paper_analysis_v1.yaml` 和 `local_paths.example.yaml` 是旧版 paper runner 的兼容配置，暂时保留。
- `benchmark_*.yaml` 是旧版单 run/quick/smoke 配置，仍被 smoke 脚本和 README 示例引用，暂时保留。
- `server_5090_full_benchmark.yaml` 是旧版 5090 suite 配置，只给兼容模式使用。
- `configs/local_paths.yaml` 和 `configs/*.local.yaml` 是机器私有文件，被 `.gitignore` 排除，不应提交。

## 论文实验矩阵

- `server_benchmark_full.example.yaml`：服务器 benchmark 的完整配置模板，包含路径、模型、方法、数据集、seed、suite 和分析参数。复制为 `server_benchmark_full.local.yaml` 后运行 5090/full、micro 或 official matrix。
- `paper_experiments_v1.yaml`：IR-only 论文实验矩阵，配合 `scripts/run_paper_experiments.py` 使用。
- `local_paths.example.yaml`：机器本地路径配置模板，复制为 `local_paths.yaml` 后填写数据集、SAM2 和 artifact 路径。
- `paper_analysis_v1.yaml`：论文基准结果分析配置，配合 `scripts/analyze_paper_results.py` 使用。
- `server_5090_full_benchmark.yaml`：旧版单张 RTX 5090 suite 配置，仍可通过兼容参数使用；新服务器运行推荐用完整 YAML。

本目录按运行规模与数据集组织 benchmark 配置。

## MultiModalCOCOClean

- `benchmark_quick10.yaml`：最快的 10 图 sanity check
- `benchmark_smoke.yaml`：用于 adapter 和报告链路验证的小规模 smoke run
- `benchmark_v1.yaml`：默认完整 benchmark 配置

## RBGT-Tiny IR

- `benchmark_quick10_rbgt_tiny.yaml`：`RBGT-Tiny` 的 10 图快速检查
- `benchmark_smoke_rbgt_tiny.yaml`：`RBGT-Tiny` smoke run
- `benchmark_v1_rbgt_tiny.yaml`：完整 `RBGT-Tiny` benchmark 配置

## 使用建议

优先用 quick 配置做环境检查，用 smoke 配置做流程 sanity check，用 `v1` 配置执行正式 benchmark。
