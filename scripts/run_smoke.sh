#!/usr/bin/env bash
# Author: Egor Izmaylov
#
# Linux 下的 smoke baseline 启动脚本。
# 这个脚本默认使用最小 smoke 配置，适合服务器首轮验证：
# - 数据集 adapter 是否正常
# - prompt synthesis 是否正常
# - SAM2 路径接线是否正常
# - 结果 schema 与 reference snapshot 是否正常

set -euo pipefail

BASELINE_NAME="${1:-bbox_rect}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_baseline.sh" "configs/benchmark_smoke.yaml" "${BASELINE_NAME}"
