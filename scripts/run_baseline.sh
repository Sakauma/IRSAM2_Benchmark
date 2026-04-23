#!/usr/bin/env bash
# Author: Egor Izmaylov
#
# Linux 下的通用 baseline 启动脚本。
# 用法：
#   bash scripts/run_baseline.sh [config_path] [baseline_name]
# 例如：
#   bash scripts/run_baseline.sh configs/benchmark_smoke.yaml sam2_zero_shot

set -euo pipefail

CONFIG_PATH="${1:-configs/benchmark_v1.yaml}"
BASELINE_NAME="${2:-bbox_rect}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 以脚本所在目录的父目录作为项目根目录，避免从任意 cwd 调用时路径错乱。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# 统一把 src 注入导入路径，这样无需预先安装 package 也能直接运行。
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" main.py run baseline --config "${CONFIG_PATH}" --baseline "${BASELINE_NAME}"
