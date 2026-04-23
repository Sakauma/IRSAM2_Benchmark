#!/usr/bin/env bash
# Author: Egor Izmaylov
#
# One-command entrypoint for the official SAM2 baseline matrix.
# Default matrix:
# - 4 official SAM2.1 checkpoints
# - 2 datasets: MultiModalCOCOClean, RBGT-Tiny
# - 4 modes: box / point / box+point / no-prompt auto-mask

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_official_baseline_matrix.py"
