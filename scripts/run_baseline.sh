#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/benchmark_v1.yaml}"
BASELINE_NAME="${2:-bbox_rect}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" main.py run baseline --config "${CONFIG_PATH}" --baseline "${BASELINE_NAME}"
