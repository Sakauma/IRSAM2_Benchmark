# 指标卡片

## 核心图像指标

- `mIoU`：预测 mask 与 GT mask 的 IoU
- `Dice`：Sørensen-Dice 分数
- `BoundaryF1`：边界感知 F1
- `LatencyMs`：单样本或单帧推理时延

## Prompt / 协议审计指标

- `BBoxIoU`：预测 mask 的 tight box 与 GT tight box 之间的 IoU
- `TightBoxMaskIoU`
- `LooseBoxMaskIoU`
- `PredAreaRatio`
- `GTAreaRatio`

## 自动分割指标

- 图像级评估：每张图只做一次 automatic-mask 推理，再与该图中的全部 GT 实例匹配
- `num_pred_instances`
- `num_matched_instances`
- `instance_precision`
- `instance_recall`
- `instance_f1`
- `matched_instance_iou`

## 时序指标

- Track C 按 `(sequence_id, track_id)` 流进行评估
- `temporal_iou_mean`
- `temporal_boundary_f1`
- `mask_jitter_score`
- `propagation_decay`
- `track_recall`
- `track_precision`
- `identity_switch_count`
