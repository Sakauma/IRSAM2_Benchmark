# 指标卡片

## 核心图像指标

- `mIoU`：预测 mask 与 GT mask 的 IoU
- `Dice`：Sørensen-Dice 分数
- `BoundaryF1`：兼容字段，当前等同于 exact boundary F1
- `BoundaryF1Exact`：无容忍半径的边界 F1，仅建议作诊断
- `BoundaryF1Tol1`：1 像素容忍半径的边界 F1，优先用于红外小目标论文表格
- `LatencyMs`：单样本或单帧推理时延

## Prompt / 协议审计指标

- `BBoxIoU`：预测 mask 的 tight box 与 GT tight box 之间的 IoU
- `TightBoxMaskIoU`
- `LooseBoxMaskIoU`
- `PredAreaRatio`
- `GTAreaRatio`
- `PromptHitRate`：自动 point prompt 是否落在 GT mask 内
- `PromptDistanceToCentroid`：自动 point prompt 与 GT mask 质心的像素距离
- `PromptBoxCoverage`：自动 box prompt 覆盖的 GT mask 像素比例
- `PromptBoxBBoxIoU`：box-only 数据中 prompt box 与 GT bbox 的 IoU
- `PromptPointInBBox`：box-only 数据中 point prompt 是否落在 GT bbox 内

## 红外小目标指标

- `TargetRecallIoU10`：以 connected component 为目标，IoU ≥ 0.10 的召回比例
- `TargetRecallIoU25`：以 connected component 为目标，IoU ≥ 0.25 的召回比例
- `TargetRecallIoU50`：以 connected component 为目标，IoU ≥ 0.50 的召回比例
- `FalseAlarmPixelsPerMP`：每百万像素中的误报像素数
- `FalseAlarmComponents`：与 GT 无交集的预测连通域数量
- `GTAreaPixels`：GT mask 面积
- `PredAreaPixels`：预测 mask 面积

## 自动分割指标

- 图像级评估：每张图只做一次 automatic-mask 推理，再与该图中的全部 GT 实例匹配
- `num_pred_instances`
- `num_matched_instances`
- `instance_precision`
- `instance_recall`
- `instance_f1`
- `matched_instance_iou`

## 大小目标分组

- `target_scale`：由数据 adapter 根据 GT mask 面积和标注类型写入，至少包含 `small`、`large` 或数据集原始值。
- 总体指标：所有样本共同汇总。
- 小目标指标：只汇总 `target_scale == "small"` 的样本。
- 大目标指标：只汇总非 small 的样本，主要用于解释 MultiModal 中建筑物等大目标。
