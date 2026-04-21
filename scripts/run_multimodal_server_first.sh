#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prepare_common_env

export DATASET_NAME="${DATASET_NAME:-MultiModal}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/multimodal_server_first_run_v1}"
export EXPERIMENT_SEEDS="${EXPERIMENT_SEEDS:-42,123}"
export SUPERVISION_BUDGETS="${SUPERVISION_BUDGETS:-0.1,0.2}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
export PSEUDO_FINETUNE_EPOCHS="${PSEUDO_FINETUNE_EPOCHS:-2}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
export MAX_IMAGES="${MAX_IMAGES:-0}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

echo "[run_multimodal_server_first] dataset=${DATASET_NAME} output=${OUTPUT_DIR}"
run_experiment_main
