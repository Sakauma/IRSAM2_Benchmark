# IRSAM2_Benchmark

`IRSAM2_Benchmark/` is a fresh benchmark platform for evaluating SAM2-based infrared segmentation pipelines. It is intentionally independent from [ir_sam2_bench](D:/workspace/sam_ir/ir_sam2_bench), which remains a read-only historical reference.

This v0.1 implementation focuses on four things:

- a clean package structure and CLI
- dataset adapters for raw MultiModal, COCO-like, RBGT-Tiny, and generic `images/ + masks/`
- first-class SAM2 baselines, including prompt-free and sequence-aware evaluation hooks
- frozen result schema, manifests, and paper-oriented reporting

## Current Scope

This codebase fully implements the benchmark platform skeleton, dataset ingestion, result schema, baseline registry, and evaluation stack. The baseline layer is functional. The full-chain training stages (`adapt`, `distill`, `quantize`) are implemented as pipeline stages with stable artifacts and interfaces, but they are still reference-stage scaffolds rather than finalized paper methods.

That split is deliberate:

- benchmark infrastructure must be stable before methods are swapped in
- baseline and evaluation behavior must be reproducible before claiming pipeline gains

## Directory Layout

- `src/irsam2_benchmark/`: runtime package
- `configs/`: YAML benchmark configs
- `docs/`: benchmark specifications
- `tests/`: unit tests
- `artifacts/`: default output root
- `reference_results/`: frozen per-baseline reference snapshots for regression checks

## CLI

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_box_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_no_prompt_auto_mask
python -m irsam2_benchmark.cli run evaluate --config configs/benchmark_v1.yaml
```

For fast validation, use the smoke config:

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

For a lighter 10-image quick check, use:

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_quick10.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_quick10_rbgt_tiny.yaml --baseline sam2_zero_shot
```

The smoke config is intentionally small enough to validate:

- dataset adapter loading
- prompt synthesis
- SAM2 checkpoint wiring
- report schema and grouped eval outputs
- reference snapshot generation under `reference_results/`

The quick-10 configs are intended for shorter sanity checks:

- `configs/benchmark_quick10.yaml`: `MultiModalCOCOClean`
- `configs/benchmark_quick10_rbgt_tiny.yaml`: `RBGT-Tiny` IR-only branch
- fixed `max_images: 10`
- fixed `seeds: [42]`
- `save_visuals: false`

When the dataset is stored outside the repo-relative default location, set `DATASET_ROOT` explicitly.
This is especially common for `configs/benchmark_quick10_rbgt_tiny.yaml`.

## Linux Scripts

For Linux servers, `scripts/` now includes shell entrypoints again:

```bash
bash scripts/run_smoke.sh
bash scripts/run_smoke.sh sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1.yaml sam2_zero_shot_point
bash scripts/run_tests.sh
```

RBGT-Tiny also has dedicated smoke and v1 configs now:

```bash
bash scripts/run_baseline.sh configs/benchmark_smoke_rbgt_tiny.yaml sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1_rbgt_tiny.yaml sam2_zero_shot
```

The official matrix launcher is available again:

```bash
bash scripts/run_official_baseline_matrix.sh
MATRIX_MODELS=tiny,small MATRIX_DATASETS=multimodal MATRIX_MODES=box,point bash scripts/run_official_baseline_matrix.sh
```

If the official SAM2.1 checkpoints are not stored under `${SAM2_REPO}/checkpoints`, set a dedicated checkpoint root:

```bash
export SAM2_CKPT_ROOT=/path/to/official_sam2_checkpoints
bash scripts/run_official_baseline_matrix.sh
```

For AutoDL-style servers:

```bash
bash scripts/setup_autodl_server.sh
source .autodl_env.sh
bash scripts/run_autodl_smoke.sh multimodal sam2_zero_shot
bash scripts/run_autodl_smoke.sh rbgt sam2_zero_shot
```

## Machine Paths

Machine-specific paths are configured through:

- `DATASET_ROOT`
- `SAM2_REPO`
- `ARTIFACT_ROOT`

The benchmark config keeps experimental meaning in YAML. Environment variables are used only for path resolution.
