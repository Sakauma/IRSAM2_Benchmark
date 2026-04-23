#!/usr/bin/env bash
# Author: Egor Izmaylov
#
# AutoDL 环境的一键 smoke 入口。
# 按数据集关键字切换到对应配置，并自动设置 DATASET_ROOT。

set -euo pipefail

DATASET_KEY="${1:-multimodal}"
BASELINE_NAME="${2:-bbox_rect}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.autodl_env.sh"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

case "${DATASET_KEY}" in
  multimodal)
    export DATASET_ROOT="${DATASET_ROOT:-${MULTIMODAL_DATASET_ROOT:-/root/autodl-tmp/datasets/MultiModalCOCOClean}}"
    CONFIG_PATH="configs/benchmark_smoke.yaml"
    ;;
  rbgt|rbgt-tiny|rbgt_tiny)
    export DATASET_ROOT="${DATASET_ROOT:-${RBGT_DATASET_ROOT:-/root/autodl-tmp/datasets/RBGT-Tiny}}"
    CONFIG_PATH="configs/benchmark_smoke_rbgt_tiny.yaml"
    ;;
  *)
    echo "[error] unsupported dataset key: ${DATASET_KEY}" >&2
    echo "[hint] use one of: multimodal, rbgt" >&2
    exit 1
    ;;
esac

if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
  export ARTIFACT_ROOT
fi

bash "${SCRIPT_DIR}/run_baseline.sh" "${CONFIG_PATH}" "${BASELINE_NAME}"
