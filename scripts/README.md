# 脚本索引

本目录提供面向本地 Linux、Windows PowerShell 和 AutoDL 风格环境的便捷入口。

## 本地验证

- `run_smoke.sh`：Linux 下的 smoke baseline 包装脚本
- `run_smoke.ps1`：Windows PowerShell 下的 smoke baseline 包装脚本
- `run_baseline.sh`：通用 Linux baseline 包装脚本
- `run_tests.sh`：Linux 单元测试包装脚本

## Benchmark 自动化

- `run_official_baseline_matrix.sh`：官方 SAM2 baseline matrix 的 shell 入口
- `run_official_baseline_matrix.py`：展开模型、数据集和模式组合的矩阵驱动脚本

## AutoDL 辅助脚本

- `setup_autodl_server.sh`：初始化数据、输出和 checkpoint 路径
- `run_autodl_smoke.sh`：面向 `MultiModal` 或 `RBGT-Tiny` 的 smoke 运行辅助脚本

所有脚本都默认以仓库根目录作为工作目录锚点，并在需要时自动把 `src/` 注入到 `PYTHONPATH`。
