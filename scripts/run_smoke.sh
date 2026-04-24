#!/usr/bin/env bash

set -euo pipefail

BASELINE_NAME="${1:-bbox_rect}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_baseline.sh" "configs/benchmark_smoke.yaml" "${BASELINE_NAME}"
