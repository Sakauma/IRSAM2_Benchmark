#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

prepare_common_env() {
  require_env "DATASET_ROOT"
  require_env "SAM2_REPO"

  export PYTHON_BIN="${PYTHON_BIN:-python}"
  export TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/benchmark_runs}"
  export EXPERIMENT_SEEDS="${EXPERIMENT_SEEDS:-42,123,456}"
  export SUPERVISION_BUDGETS="${SUPERVISION_BUDGETS:-0.1,0.2,0.5}"
  export MAX_SAMPLES="${MAX_SAMPLES:-0}"
  export MAX_IMAGES="${MAX_IMAGES:-0}"
  export EVAL_LIMIT="${EVAL_LIMIT:-0}"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
  export PSEUDO_FINETUNE_EPOCHS="${PSEUDO_FINETUNE_EPOCHS:-4}"
  export BATCH_SIZE="${BATCH_SIZE:-1}"
  export NUM_WORKERS="${NUM_WORKERS:-0}"
  export SUPERVISION_PROTOCOL="${SUPERVISION_PROTOCOL:-box_only}"
  export EXPERIMENT_PHASE="${EXPERIMENT_PHASE:-benchmark_v1}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"

  mkdir -p "${OUTPUT_ROOT}"
}

run_experiment_main() {
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${EXPERIMENT_DIR}/main.py"
  else
    "${PYTHON_BIN}" "${EXPERIMENT_DIR}/main.py"
  fi
}
