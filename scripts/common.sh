#!/usr/bin/env bash
# benchmark 公共脚本函数。
# 所有 run_*.sh 都先 source 这个文件，以复用统一的环境准备与启动逻辑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

require_env() {
  # 显式检查必需环境变量，避免实验跑了一半才发现路径没配。
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

prepare_common_env() {
  # DATASET_ROOT 和 SAM2_REPO 是 benchmark 的最低外部依赖。
  require_env "DATASET_ROOT"
  require_env "SAM2_REPO"

  # 下面这些变量都允许被外部覆盖，脚本只提供平台级默认值。
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
  # 多卡时统一走 torchrun；单卡时直接调用 Python 入口。
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${EXPERIMENT_DIR}/main.py"
  else
    "${PYTHON_BIN}" "${EXPERIMENT_DIR}/main.py"
  fi
}
