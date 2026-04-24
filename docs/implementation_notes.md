# Implementation Notes

This note records the first implementation pass of the greenfield `IRSAM2_Benchmark` project.

## Implemented in this pass

- Created a new standalone project layout under `IRSAM2_Benchmark/`
- Added a package-based CLI and config loader
- Added dataset adapters for:
  - raw `MultiModal`
  - COCO-like datasets
  - `RBGT-Tiny` IR-only view
  - generic `images/ + masks/` datasets
- Added deterministic prompt synthesis for mask-only datasets:
  - tight box
  - loose box
  - point prompt
- Added first-class SAM2 baselines:
  - `BBoxRectMaskBaseline`
  - `ZeroShotSAM2`
  - `NoPromptAutoMaskSAM2`
  - `ZeroShotSAM2VideoPropagation`
- Added benchmark governance fields:
  - `benchmark_version`
  - `split_version`
  - `prompt_policy_version`
  - `metric_schema_version`
  - `reference_result_version`
- Added grouped evaluation reports and reference snapshot generation

## Validation completed

- Python syntax validation passed in WSL
- Unit tests passed in the `sam_hq2` environment
- End-to-end `bbox_rect` baseline run completed on `MultiModalCOCOClean`
- Real `SAM2` smoke validation completed for:
  - `sam2_zero_shot`
  - `sam2_zero_shot_point`
  - `sam2_no_prompt_auto_mask`

## Current boundary

The platform skeleton, result schema, dataset layer, and baseline layer are operational.

The full pipeline stages:

- `adapt`
- `distill`
- `quantize`

already have stable stage interfaces and artifact scaffolding, but they are not yet final paper methods.
