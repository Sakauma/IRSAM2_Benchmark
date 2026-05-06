---
type: results-report
date: 2026-05-06
experiment_line: sam2-ir-qd
round: 4
purpose: experiment-review-and-paper-direction
status: active
source_artifacts:
  - artifacts/analysis_remote_20260430/analysis-report.md
  - artifacts/notebook_prompted_benchmark/analysis/overall_result_review.md
  - artifacts/sam2_ir_qd_m2_prompt_proposal_v2_20260504/analysis/auto_prompt/large/analysis-report.md
  - artifacts/sam2_ir_qd_m3_prompt_rerank_v1/analysis/auto_prompt/large/analysis-report.md
  - artifacts/sam2_ir_qd_m3_rerank_ablation_v1/analysis/auto_prompt_rerank_ablation/large/analysis-report.md
  - artifacts/sam2_ir_qd_m3_rerank_ablation_v1/train/sam2_ir_qd_m3_rerank_ablation_v1/train_summary.json
---

# SAM2-IR-QD / Round 4 / Experiment Review and Paper Direction / 2026-05-06

## Executive Summary

The current evidence says the publishable problem is not simply "SAM2 cannot segment infrared small targets." The stronger claim is that SAM2 has usable prompt-conditioned response, but infrared small target deployment is limited by automatic prompt discovery and false-alarm control.

The most promising paper direction is:

> SAM2-IR-QD: false-alarm-aware prompt discovery and quantized distillation for infrared small target segmentation.

M3.1 provides the strongest current method evidence. On the three canonical mask datasets, the no-local-contrast reranker improves learned point prompting from `0.3674` to `0.3965` mIoU and from `0.5971` to `0.6449` TargetRecall@IoU25, while reducing false-alarm pixels from `9308.8` to `3352.9` px/MP.

M4 should now test whether this result is stable across training seeds. If M4 confirms the M3.1 trend, the main method contribution is defensible enough to support the next paper-stage experiments.

## Experiment Identity and Decision Context

This report freezes the research discussion after M2, M3, M3.1, and the implementation of the M4 seeded workflow.

The decision question is:

> Should the paper focus on infrared-domain SAM2 transfer, or on false-alarm-aware automatic prompt discovery with later quantized distillation?

The current answer is:

> Focus the paper on false-alarm-aware prompt discovery first, then use quantized distillation as the efficiency contribution.

This direction is supported because oracle-prompt SAM2 performs meaningfully better than automatic prompting, while learned automatic prompts improve target finding but can trigger severe false alarms.

## Completed Evidence

### Remote SAM2 Baseline Analysis

Artifact:

- `artifacts/analysis_remote_20260430/analysis-report.md`

Status:

- `82` completed remote runs.
- `4,094,151` evaluated rows.
- `308` completed paired tests in the combined significance table.

Key findings:

- Oracle-prompted SAM2 is useful but strongly prompt-mode dependent.
- Best unweighted mean mIoU in that analysis is `0.431` for `large` checkpoint with box prompt.
- Heuristic automatic prompting is weak before adaptation. Best average heuristic auto-prompt mIoU is `0.292` from auto point.
- Tight box protocol is strong, with average protocol mIoU `0.632`.
- RBGT-Tiny is useful for box-prompt pseudo-mask probing, not for direct mask mIoU reporting.
- MultiModal is an important diagnostic split because it exposes large domain variation and automatic prompt failures.

Decision impact:

- The teacher should not be framed as generic SAM2 fine-tuning.
- The paper should evaluate prompt discovery, prompt-response preservation, and false-alarm behavior.
- Quantization/distillation must report prompt-response fidelity, not only final mIoU.

### Notebook Prompted Baseline

Artifact:

- `artifacts/notebook_prompted_benchmark/analysis/overall_result_review.md`

Status:

- `48/48` runs completed.
- Seed `42`.
- Four SAM2 checkpoints, four datasets, and three prompted methods.

Key findings:

- Dataset-balanced prompted comparison:
  - `sam2_box_oracle`: mIoU `0.421`, Dice `0.549`, Recall@IoU25 `0.778`, false-alarm px/MP `18,264`.
  - `sam2_box_point_oracle`: mIoU `0.414`, Dice `0.535`, Recall@IoU25 `0.741`, false-alarm px/MP `20,656`.
  - `sam2_point_oracle`: mIoU `0.398`, Dice `0.505`, Recall@IoU25 `0.671`, false-alarm px/MP `108,989`.
- Box prompt is the strongest default prompted SAM2 baseline when false alarms are considered.
- Point-only prompts can raise mIoU on NUAA/NUDT but introduce much larger false-alarm area.
- MultiModal needs small-target and large-target splits.

Important caveat:

- That run evaluated only the `.bmp/.png` MultiModal subset because `.jpg` matching was not yet fixed.

Decision impact:

- Do not use point-only prompt as the paper's default upper-bound.
- Use box prompt as the conservative SAM2 prompted baseline.
- Report MultiModal by size split, not only aggregate score.

### M2 Learned Prompt Proposal

Artifact:

- `artifacts/sam2_ir_qd_m2_prompt_proposal_v2_20260504`

Status:

- `48/48` runs completed.
- `135,606` sample-level rows.

Weighted IR3 results:

| Method | mIoU | TR25 | FA px/MP | FA comp | PromptHit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sam2_heuristic_auto_box_point` | 0.3342 | 0.5890 | 543.3 | 0.32 | 0.7727 |
| `sam2_learned_auto_point` | 0.3674 | 0.5971 | 9308.8 | 7.35 | 0.7683 |
| `sam2_learned_auto_box_point` | 0.3062 | 0.5375 | 414.4 | 0.58 | 0.7683 |
| `sam2_learned_auto_box_point_neg` | 0.3033 | 0.5318 | 417.9 | 0.62 | 0.7683 |

Interpretation:

- Learned point prompting improves mIoU over heuristic box-point on IR3.
- Learned point prompting creates severe false alarms.
- Learned box and box+point suppress false alarms but lose target quality.

Decision impact:

- The learned proposal can find useful target points.
- Directly passing learned point prompts to SAM2 is not enough.
- False-alarm-aware reranking is the next necessary module.

### M3 Prompt Rerank

Artifact:

- `artifacts/sam2_ir_qd_m3_prompt_rerank_v1`

Status:

- `12/12` runs completed.
- `36,051` sample-level rows.

Weighted IR3 results:

| Method | mIoU | TR25 | FA px/MP | FA comp | PromptHit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sam2_learned_auto_point_rerank` | 0.3838 | 0.6197 | 3693.0 | 1.25 | 0.7934 |
| `sam2_learned_auto_box_point_calibrated` | 0.3151 | 0.5445 | 759.4 | 0.76 | 0.7934 |
| `sam2_learned_auto_box_point_calibrated_neg` | 0.3129 | 0.5336 | 643.6 | 1.17 | 0.7934 |

Interpretation:

- Mask-feedback reranking improves learned point prompting.
- Box calibration reduces false alarms but sacrifices mIoU and recall.

Decision impact:

- Reranking is a valid core method branch.
- Box calibration should be optional or constrained, not the default main method.

### M3.1 Rerank Ablation

Artifact:

- `artifacts/sam2_ir_qd_m3_rerank_ablation_v1`

Status:

- `32/32` runs completed.
- `96,136` sample-level rows.
- Training used `2754` samples from NUAA-SIRST, NUDT-SIRST, and IRSTD-1K.
- Training loss decreased from `6.2629` at epoch 1 to `4.2200` at epoch 20.

Weighted IR3 results:

| Method | mIoU | TR25 | FA px/MP | FA comp | PromptHit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sam2_learned_auto_point` | 0.3674 | 0.5971 | 9308.8 | 7.35 | 0.7683 |
| `sam2_learned_auto_point_rerank` | 0.3838 | 0.6197 | 3693.0 | 1.25 | 0.7934 |
| `sam2_learned_auto_point_rerank_no_local_contrast` | 0.3965 | 0.6449 | 3352.9 | 1.21 | 0.8235 |
| `sam2_learned_auto_point_rerank_mask_feedback_only` | 0.3874 | 0.6225 | 3540.0 | 1.13 | 0.7927 |
| `sam2_learned_auto_box_point_gated` | 0.3839 | 0.6188 | 1215.9 | 0.40 | 0.7934 |

Main interpretation:

- The strongest current method is the no-local-contrast reranker.
- Local contrast appears to overfit to infrared clutter in this setup.
- SAM2 mask feedback is the most useful reranking signal.
- Gated box strongly reduces false alarms but can suppress recall, especially outside IRSTD-1K.

Decision impact:

- M4 main method should be `sam2_ir_fa_rerank`, which maps to the no-local-contrast reranker.
- `sam2_ir_fa_rerank_feedback_only` should remain as an ablation.
- `sam2_ir_fa_rerank_gated_box_strict` should be reported as a false-alarm-control branch, not the default main method.

## Current M4 Design

Implemented code status:

- Commit in benchmark repo: `3ab5a3d feat: add m4 seeded fa rerank workflow`.
- Outer repo submodule update: `07d0a70 chore: update benchmark submodule for m4 workflow`.

Full M4 config:

- `configs/server_auto_prompt_4090x4.example.yaml`

Smoke M4 config:

- `configs/server_auto_prompt_4090x4_smoke.yaml`

Full M4 artifact root:

- `artifacts/sam2_ir_qd_m4_fa_rerank_seeded_v1`

M4 training:

- Train datasets: `NUAA-SIRST`, `NUDT-SIRST`, `IRSTD-1K`.
- Excludes RBGT-Tiny by default.
- Train seeds: `42`, `123`, `456`.
- GPU 0 trains each seed checkpoint.

M4 evaluation:

- Eval datasets: `NUAA-SIRST`, `NUDT-SIRST`, `IRSTD-1K`, `MultiModal`.
- Reference methods run once:
  - `sam2_no_prompt_auto_mask`
  - `sam2_heuristic_auto_box_point`
  - `sam2_box_oracle`
  - `sam2_box_point_oracle`
- Learned methods run once per train seed:
  - `sam2_learned_auto_point`
  - `sam2_learned_auto_point_rerank`
  - `sam2_ir_fa_rerank`
  - `sam2_ir_fa_rerank_feedback_only`
  - `sam2_ir_fa_rerank_gated_box_strict`

Expected full M4 run count:

- Reference: `4 datasets x 4 methods = 16`.
- Learned: `4 datasets x 5 methods x 3 train seeds = 60`.
- Total: `76` eval runs.

M4 analysis behavior:

- Analysis merges `E4_fa_rerank_reference`, `E4_fa_rerank_seed42`, `E4_fa_rerank_seed123`, and `E4_fa_rerank_seed456`.
- Paired tests use `TrainSeed + sample_id` when `TrainSeed` exists.

## Third-Party Comparison Status

External prediction masks exist under:

- `artifacts/external_predictions`

Completed mask groups include:

- Infrared small target models: `dnanet`, `bgm`, `drpcanet`, `mshnet`, `rpcanet_pp`, `sctransnet`, `serankdet`, `hdnet`, `uiu_net`, `pconv_mshnet_p43`, `pconv_yolov8n_p2_p43_boxmask`.
- SAM/SAM-like models: `sam_vit_b`, `hq_sam_vit_b`, `mobile_sam`, `fastsam_s`, `fastsam_x`, `edge_sam`, `efficient_sam_vitt`, `sam2_unet_cod`.

Important coverage difference:

- Many infrared trained models have `665` MultiModal image-level masks because they output one mask per image.
- Promptable SAM-like models have `9263` MultiModal instance-level masks.

Decision impact:

- Third-party masks must be imported into the unified evaluator before paper tables are final.
- MultiModal comparison must clearly separate image-level external predictors from instance-level prompted SAM predictors.

## What Is Supported Now

Stable conclusions:

- SAM2 is prompt-sensitive in infrared small target segmentation.
- Box prompts are the safest conservative oracle baseline.
- Learned point proposals improve target discovery but introduce false alarms.
- SAM2 mask-feedback reranking reduces false alarms while improving mIoU and recall.
- The no-local-contrast rerank variant is the strongest current method on IR3.
- Strict gated box is useful for false-alarm suppression but should be treated as a tradeoff branch.

Tentative conclusions:

- Local contrast may be harmful because it amplifies infrared clutter.
- Feedback-only reranking may be a strong simplified version.
- M4 may turn the method into a seed-stable contribution, but the evidence is not available until full M4 finishes.

Not yet supported:

- Training-seed stability.
- Quantized student quality.
- RBGT-Tiny weak-supervision gain.
- Final superiority over all third-party models.
- Full MultiModal conclusions without careful unit separation.

## Paper Direction

Working title:

> SAM2-IR-QD: False-Alarm-Aware Prompt Discovery and Quantized Distillation for Infrared Small Target Segmentation

Core thesis:

> Infrared small target segmentation does not only require adapting SAM2's mask decoder. It requires discovering prompts that hit true small targets while suppressing infrared clutter-induced false alarms. A false-alarm-aware prompt reranker can recover SAM2's useful prompt response and provide a teacher for efficient quantized students.

Proposed contribution structure:

1. Benchmark contribution.
   - Unified evaluation over infrared small target datasets and MultiModal.
   - SAM2 prompt protocol analysis.
   - Third-party model mask import and uniform evaluation.
   - Prompt-specific and false-alarm-specific metrics.

2. Method contribution.
   - Learned infrared prompt proposal.
   - SAM2 feedback-guided prompt reranking.
   - False-alarm-aware reranking, with `sam2_ir_fa_rerank` as main method.
   - Strict gated box as an explicit false-alarm-control branch.

3. Efficiency contribution.
   - Teacher-to-student distillation from the best M4 teacher.
   - Two student branches:
     - custom PicoSAM2-like student.
     - tiny SAM2-based student.
   - Quantization after or during distillation.
   - Report prompt-response fidelity, latency, memory, and false-alarm metrics.

## Next Experiments

### Priority 1: Run M4 Smoke

Command:

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4_smoke.yaml \
  --stop-on-error \
  --preflight-mode fast
```

Purpose:

- Verify multi-seed runner logic.
- Verify `TrainSeed` columns and paired statistics.
- Verify M4 methods load and evaluate.
- Check MultiModal adapter path before full run.

### Priority 2: Run Full M4

Command:

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.example.yaml \
  --stop-on-error \
  --preflight-mode fast
```

Purpose:

- Confirm whether M3.1's no-local-contrast gain is stable across seeds.
- Estimate seed variance for M4 methods.
- Decide whether `sam2_ir_fa_rerank` is the main method for the paper.

### Priority 3: Evaluate Third-Party Masks

Tasks:

- Import masks from `artifacts/external_predictions`.
- Normalize image-level and instance-level evaluation units.
- Generate final comparison tables.
- Separate infrared-specialized models, SAM-like models, and SAM2-IR-QD variants.

Purpose:

- Establish external baselines for a high-level journal submission.
- Avoid relying only on internal SAM2 ablations.

### Priority 4: M4 Error Analysis

Tasks:

- Compare prompt hit, prompt distance, false-alarm components, and hit-conditioned IoU.
- Inspect cases where `sam2_ir_fa_rerank` beats learned point.
- Inspect cases where strict gated box loses recall.
- Generate qualitative panels:
  - input image.
  - objectness heatmap.
  - selected prompt.
  - SAM2 feedback candidate masks.
  - final mask.

Purpose:

- Turn the method into a mechanism-backed claim, not only a metric improvement.

### Priority 5: Quantized Distillation

Teacher:

- Use the best M4 teacher if M4 confirms M3.1.

Student branches:

- Custom PicoSAM2-like student.
- Tiny SAM2 student.

Metrics:

- mIoU.
- Dice.
- TargetRecall@IoU25.
- FalseAlarmPixelsPerMP.
- FalseAlarmComponents.
- PromptHitRate.
- Prompt-response fidelity.
- Latency.
- Memory.

Purpose:

- Convert the method contribution into an end-to-end deployable framework.

## Risks and Mitigations

Risk: M4 no-local-contrast gain is not stable across seeds.

Mitigation:

- Fall back to feedback-only reranker if it is more stable.
- Present local-contrast removal as an ablation rather than the main novelty.

Risk: MultiModal remains weak for learned prompts.

Mitigation:

- Treat MultiModal as a domain-generalization stress test.
- Report small/large target split.
- Do not overclaim full MultiModal generalization.

Risk: Third-party comparisons are unfair because masks have different evaluation units.

Mitigation:

- Explicitly separate image-level masks and instance-level prompted masks.
- Report evaluation-unit metadata in tables.

Risk: Quantization hurts false-alarm control more than mIoU.

Mitigation:

- Use prompt-response fidelity and false-alarm metrics as primary efficiency-stage guardrails.

## Current Decision

Proceed with M4 full as the next decisive experiment.

If M4 confirms the M3.1 result, promote `sam2_ir_fa_rerank` to the paper's main method and start the quantized distillation branch.

If M4 does not confirm the result, keep the paper around the benchmark and prompt-response analysis, then redesign the reranker before starting distillation.

