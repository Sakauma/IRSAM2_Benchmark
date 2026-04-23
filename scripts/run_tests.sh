#!/usr/bin/env bash
# Author: Egor Izmaylov
#
# Linux 下的单元测试脚本。
# 默认使用仓库根目录的 src 作为导入路径，直接运行 unittest。
# 如果当前环境里没有 numpy / Pillow / PyYAML 等依赖，测试会按正常方式失败，
# 这样可以帮助快速发现服务器环境是否准备完整。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m unittest discover -s tests -v
