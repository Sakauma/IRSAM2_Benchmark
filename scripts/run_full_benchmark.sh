#!/usr/bin/env bash
# 运行 benchmark v1 的完整条件集合。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_common_env

# 完整 benchmark 默认从原始 MultiModal 起跑，便于直接复现当前主实验入口。
export DATASET_NAME="${DATASET_NAME:-MultiModal}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DATASET_NAME}_full_benchmark_v1}"
export EXPERIMENT_CONDITIONS="${EXPERIMENT_CONDITIONS:-BBoxRectMaskBaseline,ZeroShotSAM2BoxPromptIR,CleanBoxPEFTSAM2Adapter,NoisyBoxPromptRobustSAM2Adapter,CleanPromptOnlyWithinPromptRobustAdapter,JitterOnlyPromptRobustSAM2Adapter,QualityFilteredPseudoMaskSelfTrainingSAM2,PseudoMaskSelfTrainingWithoutIRQualityFilter,DirectSupervisedIRSegFormerB0,DirectSupervisedIRPIDNetS}"

# 打印当前运行摘要，方便日志回溯。
echo "[run_full_benchmark] dataset=${DATASET_NAME} supervision=${SUPERVISION_PROTOCOL} output=${OUTPUT_DIR}"
run_experiment_main
