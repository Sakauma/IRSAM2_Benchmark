# 数据集卡片

## 必需字段

每个数据集 adapter 都必须产出统一的 `Sample` schema，至少包含：

- `image_path`
- `sample_id`
- `frame_id`
- `sequence_id`
- `frame_index`
- `temporal_key`
- `track_id`（可选，但 Track C / 视频传播场景下必需）
- `category`
- `device_source`
- `annotation_protocol_flag`
- `supervision_type`

字段语义：

- `frame_id`：图像级标识符，同一帧中的所有实例共享它
- `sample_id`：实例级唯一标识符
- `track_id`：当数据集支持跨帧传播评估时，显式表示同一目标的跨帧身份

## 已支持的数据集族

- `MultiModalAdapter`：原始 `img/ + label/`
- `CocoLikeAdapter`：`annotations_coco/ + image/`
- `RBGTTinyIRAdapter`：`RBGT-Tiny` 灰度红外分支
- `GenericImageMaskAdapter`：任意 `images/ + masks/`

## 通用 Image+Mask 策略

`GenericImageMaskAdapter` 必须支持：

- 二值前景 mask
- 类别索引 mask
- 实例 id mask

对于 `instance-id` mask，如果某个实例值在跨帧语义上是稳定的，那么该值可以直接复用为 `track_id`。

如果数据集中没有显式 prompt 标注，则必须使用确定性的 prompt synthesis。
