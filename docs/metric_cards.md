# Metric Cards

## Core Image Metrics

- `mIoU`: mask IoU between prediction and GT
- `Dice`: Sørensen–Dice score
- `BoundaryF1`: boundary-aware F1
- `LatencyMs`: per-sample or per-frame inference latency

## Prompt / Protocol Audit Metrics

- `BBoxIoU`: IoU between the predicted mask tight box and the GT tight box
- `TightBoxMaskIoU`
- `LooseBoxMaskIoU`
- `PredAreaRatio`
- `GTAreaRatio`

## Automatic Masking Metrics

- image-level evaluation: one automatic-mask inference per image, then matched against all GT instances from that image
- `num_pred_instances`
- `num_matched_instances`
- `instance_precision`
- `instance_recall`
- `instance_f1`
- `matched_instance_iou`

## Temporal Metrics

- Track C rows are evaluated per `(sequence_id, track_id)` stream
- `temporal_iou_mean`
- `temporal_boundary_f1`
- `mask_jitter_score`
- `propagation_decay`
- `track_recall`
- `track_precision`
- `identity_switch_count`
