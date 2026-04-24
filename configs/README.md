# 配置索引

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
