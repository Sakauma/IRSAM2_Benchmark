# Paper Experiment Matrix

This file records the IR-only experiments required by the paper.

Machine-readable source:

- `configs/paper_experiments_v1.yaml`

Launcher:

```bash
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_all --dry-run
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_all
```

## Global Rules

- All experiments are infrared-only.
- RGB images must not be loaded, logged, or used as comparison inputs.
- `RBGT-Tiny` is used only through the IR branch and box annotations.
- `MultiModal` uses `img/ + label/` annotations through `multimodal_raw`; JSON polygons are decoded into mask supervision.
- SAM2 receives a 3-channel image only by copying the normalized single-channel IR image.

## Dataset Roots

Copy `configs/local_paths.example.yaml` to `configs/local_paths.yaml`, then edit machine-specific paths there:

```yaml
sam2:
  repo: "/path/to/sam2"
  checkpoint_root: "/path/to/sam2/checkpoints"
artifacts:
  root: "artifacts"
datasets:
  nuaa_sirst: "/path/to/NUAA-SIRST"
  nudt_sirst: "/path/to/NUDT-SIRST"
  irstd_1k: "/path/to/IRSTD-1K"
  multimodal: "/path/to/MultiModal"
  rbgt_tiny_ir_box: "/path/to/RBGT-Tiny"
```

Environment variables remain a fallback for legacy scripts, but paper experiments should use the YAML path file.

## P0 Runnable Experiments

### T1 Oracle Prompt Baselines

Purpose:

- Measure SAM2 oracle-prompt upper bound on IR small-target mask datasets.
- Box and point prompts are deterministically derived from GT masks. They are diagnostic oracle protocols, not native dataset annotations.

Datasets:

- `nuaa_sirst`
- `nudt_sirst`
- `irstd_1k`
- `MultiModal`

Methods:

- `bbox_rect`
- `sam2_box_oracle`：mask-derived adaptive loose-box oracle
- `sam2_point_oracle`
- `sam2_box_point_oracle`

Command:

```bash
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_baselines
```

### T2 IR Auto Prompt

Purpose:

- Evaluate IR physics-prior automatic prompting as a Track A prompted segmentation method.

Datasets:

- `nuaa_sirst`
- `nudt_sirst`
- `irstd_1k`
- `MultiModal`

Methods:

- `sam2_physics_auto_prompt`

Command:

```bash
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_auto_prompt
```

### T7 No-Prompt Auto Mask

Purpose:

- Evaluate no-prompt SAM2 automatic masks as a separate Track B image-level baseline.

Methods:

- `sam2_no_prompt_auto_mask`

Note:

- Track B results are not paired with Track A prompted segmentation in the default statistical tests.

### T3 Prior And Prompt Ablation

Purpose:

- Identify which physical prior and prompt post-processing configuration contributes most.

Datasets:

- `nuaa_sirst`
- `nudt_sirst`
- `irstd_1k`
- `MultiModal`

Methods:

- `sam2_physics_auto_prompt`
- `physics_no_local_contrast`
- `physics_no_top_hat`
- `physics_no_snr`
- `physics_percentile_99_0`
- `physics_percentile_99_8`
- `physics_top_k_3`
- `physics_pad_0_10`
- `physics_pad_0_40`

Command:

```bash
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_ablation
```

### T6 Prompt Generation Protocol Ablation

Purpose:

- Test whether the adaptive loose-box expansion rule changes the SAM2 oracle-prompt conclusion.

Datasets:

- `nuaa_sirst`
- `nudt_sirst`
- `irstd_1k`
- `MultiModal`

Methods:

- `sam2_tight_box_oracle`
- `sam2_box_oracle`

Command:

```bash
python scripts/run_paper_experiments.py --matrix configs/paper_experiments_v1.yaml --paths configs/local_paths.yaml --group p0_prompt_protocol
```

## Planned After P0

### T4 Trainable Adapter Placeholder

Status:

- `planned_after_p0`

Planned modules:

- `adapter: pgsa_adapter`
- `decoder: ir_fpn_decoder`
- `loss: dice_bce_tiny_weighted`

### T5 Distillation And Quantization Placeholder

Status:

- `planned_after_training`

Planned modules:

- `distiller: feature_mask_distiller`
- `quantizer: ptq_int8`
- `quantizer: qat_int8`

## Core Metrics

- `mIoU`
- `Dice`
- `BoundaryF1Tol1`
- `TargetRecallIoU10`
- `TargetRecallIoU25`
- `TargetRecallIoU50`
- `FalseAlarmPixelsPerMP`
- `FalseAlarmComponents`
- `PromptHitRate`
- `PromptDistanceToCentroid`
- `PromptBoxCoverage`
- `LatencyMs`
