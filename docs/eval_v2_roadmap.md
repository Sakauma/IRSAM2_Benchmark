# Eval v2 Roadmap

This note records planned evaluation improvements beyond the first benchmark-analysis implementation.

## Current Eval v2 Target

- Collect completed benchmark artifacts.
- Build paper-ready CSV/JSON tables.
- Run paired sample-level significance tests.
- Select failure cases and copy available visualizations.
- Write `analysis-report.md`, `stats-appendix.md`, and `figure-catalog.md`.

## Future Improvements

### Training Analysis

- Parse training logs and validation curves.
- Compare checkpoint selection policies.
- Add overfitting and convergence diagnostics.

### Stronger Statistical Analysis

- Add hierarchical analysis when multiple seeds or checkpoints exist.
- Add dataset-level random-effects summaries.
- Add bootstrap over images and over datasets separately.

### Publication Figures

- Generate publication-grade PDF figures from CSV tables.
- Add consistent colors, method ordering, metric direction labels, and caption metadata.
- Add final figure QA before paper writing.

### Deployment Analysis

- Add model size, FLOPs, throughput, memory, and edge latency summaries.
- Add PTQ/QAT comparison tables.
- Add speed-accuracy Pareto plots.

### Failure Taxonomy

- Add local contrast buckets.
- Add clutter and edge-target buckets.
- Add multi-target and near-target confusion buckets.
- Add false-positive type labels after qualitative review.

