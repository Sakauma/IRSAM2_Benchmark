# Config Map

This directory groups benchmark configs by runtime scale and dataset.

## MultiModalCOCOClean

- `benchmark_quick10.yaml`: fastest 10-image sanity check
- `benchmark_smoke.yaml`: small smoke run for adapter and reporting validation
- `benchmark_v1.yaml`: full default benchmark configuration

## RBGT-Tiny IR

- `benchmark_quick10_rbgt_tiny.yaml`: 10-image RBGT-Tiny quick check
- `benchmark_smoke_rbgt_tiny.yaml`: RBGT-Tiny smoke run
- `benchmark_v1_rbgt_tiny.yaml`: full RBGT-Tiny benchmark configuration

## Usage Pattern

Use quick configs for environment validation, smoke configs for pipeline sanity checks, and `v1` configs for formal benchmark runs.
