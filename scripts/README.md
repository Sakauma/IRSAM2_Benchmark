# Script Map

This directory contains convenience entrypoints for local Linux, Windows PowerShell, and AutoDL-style environments.

## Local Validation

- `run_smoke.sh`: smoke baseline wrapper on Linux
- `run_smoke.ps1`: smoke baseline wrapper on Windows PowerShell
- `run_baseline.sh`: generic Linux baseline wrapper
- `run_tests.sh`: Linux unit-test wrapper

## Benchmark Automation

- `run_official_baseline_matrix.sh`: shell entrypoint for the official SAM2 baseline matrix
- `run_official_baseline_matrix.py`: matrix driver that expands models, datasets, and modes

## AutoDL Helpers

- `setup_autodl_server.sh`: initializes dataset, output, and checkpoint paths
- `run_autodl_smoke.sh`: smoke-run helper for MultiModal or RBGT-Tiny on AutoDL-style servers

All scripts assume the repository root is the working directory anchor and inject `src/` into `PYTHONPATH` when needed.
