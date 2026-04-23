# Artifact Schema Spec

Every run must write:

- `benchmark_spec.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`

`artifact_manifest.json` records:

- stage name
- artifact directory
- artifact identifier
- stage metadata

This is the benchmark's reproducibility anchor.
