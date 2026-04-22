#!/usr/bin/env bash
# 只运行 baseline 层，适合快速检查 benchmark 主表起点是否正常。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_common_env

# baseline 默认使用清洗后的 COCO 数据，以减少原始类别噪声。
export DATASET_NAME="${DATASET_NAME:-MultiModalCOCOClean}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/baselines_v1}"
export EXPERIMENT_CONDITIONS="${EXPERIMENT_CONDITIONS:-BBoxRectMaskBaseline,ZeroShotSAM2BoxPromptIR,DirectSupervisedIRSegFormerB0,DirectSupervisedIRPIDNetS}"

echo "[run_baselines] dataset=${DATASET_NAME} supervision=${SUPERVISION_PROTOCOL} output=${OUTPUT_DIR}"
run_experiment_main
