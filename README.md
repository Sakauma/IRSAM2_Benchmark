# IRSAM2_Benchmark

`IRSAM2_Benchmark` is a fresh benchmark platform for evaluating SAM2-based infrared segmentation pipelines. It replaces the earlier monolithic repository layout with a package-based benchmark runtime and frozen reporting schema.

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

The smoke config is intentionally small enough to validate:

- dataset adapter loading
- prompt synthesis
- SAM2 checkpoint wiring
- report schema and grouped eval outputs
- reference snapshot generation under `reference_results/`

## Machine Paths

Machine-specific paths are configured through:

- `DATASET_ROOT`
- `SAM2_REPO`
- `ARTIFACT_ROOT`

The benchmark config keeps experimental meaning in YAML. Environment variables are used only for path resolution.
