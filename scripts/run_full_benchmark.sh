#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_common_env

export DATASET_NAME="${DATASET_NAME:-MultiModal}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${DATASET_NAME}_full_benchmark_v1}"
export EXPERIMENT_CONDITIONS="${EXPERIMENT_CONDITIONS:-BBoxRectMaskBaseline,ZeroShotSAM2BoxPromptIR,CleanBoxPEFTSAM2Adapter,NoisyBoxPromptRobustSAM2Adapter,CleanPromptOnlyWithinPromptRobustAdapter,JitterOnlyPromptRobustSAM2Adapter,QualityFilteredPseudoMaskSelfTrainingSAM2,PseudoMaskSelfTrainingWithoutIRQualityFilter,DirectSupervisedIRSegFormerB0,DirectSupervisedIRPIDNetS}"

echo "[run_full_benchmark] dataset=${DATASET_NAME} supervision=${SUPERVISION_PROTOCOL} output=${OUTPUT_DIR}"
run_experiment_main
