#!/usr/bin/env bash
# 只运行 zero-shot 与 adaptation 层，用于分析 SAM2 适配收益。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_common_env

export DATASET_NAME="${DATASET_NAME:-MultiModalCOCOClean}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/adaptation_v1}"
export EXPERIMENT_CONDITIONS="${EXPERIMENT_CONDITIONS:-ZeroShotSAM2BoxPromptIR,CleanBoxPEFTSAM2Adapter,NoisyBoxPromptRobustSAM2Adapter,CleanPromptOnlyWithinPromptRobustAdapter,JitterOnlyPromptRobustSAM2Adapter}"

echo "[run_adaptation] dataset=${DATASET_NAME} supervision=${SUPERVISION_PROTOCOL} output=${OUTPUT_DIR}"
run_experiment_main
