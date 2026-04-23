# Metric Cards

## Core Image Metrics

- `mIoU`: mask IoU between prediction and GT
- `Dice`: Sørensen–Dice score
- `BoundaryF1`: boundary-aware F1
- `LatencyMs`: per-sample or per-frame inference latency

## Prompt / Protocol Audit Metrics

- `BBoxIoU`
- `TightBoxMaskIoU`
- `LooseBoxMaskIoU`
- `PredAreaRatio`
- `GTAreaRatio`

## Automatic Masking Metrics

- `num_pred_instances`
- `num_matched_instances`
- `instance_precision`
- `instance_recall`
- `instance_f1`
- `matched_instance_iou`

## Temporal Metrics

- `temporal_iou_mean`
- `temporal_boundary_f1`
- `mask_jitter_score`
- `propagation_decay`
- `track_recall`
- `track_precision`
- `identity_switch_count`
