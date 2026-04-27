# 数据集卡片

## 必需字段

每个数据集 adapter 都必须产出统一的 `Sample` schema，至少包含：

- `image_path`
- `sample_id`
- `frame_id`
- `sequence_id`
- `frame_index`
- `temporal_key`
- `track_id`（可选；当前 baseline 不依赖它）
- `category`
- `device_source`
- `annotation_protocol_flag`
- `supervision_type`

字段语义：

- `frame_id`：图像级标识符，同一帧中的所有实例共享它
- `sample_id`：实例级唯一标识符
- `track_id`：如果数据集本身提供跨帧身份，可以保留该字段供后续分析使用

## 已支持的数据集族

- `MultiModalAdapter`：原始 `img/ + label/`
- `CocoLikeAdapter`：`annotations_coco/ + image/`
- `RBGTTinyIRAdapter`：`RBGT-Tiny` 灰度红外分支
- `GenericImageMaskAdapter`：任意 `images/ + masks/`

## 论文 IR-only 约束

论文实验只允许使用红外图像。

- `modality` 必须记录为 `ir`
- `SAM2` 所需的三通道输入只能由单通道红外图像复制得到
- 不允许读取真实 RGB 图像作为输入或对比
- `RBGT-Tiny` 只使用红外分支和 box 标注
- `MultiModal` 使用 `img/ + label/` 结构，通过 `MultiModalAdapter` 解码 JSON polygon 为 mask

## 通用 Image+Mask 策略

`GenericImageMaskAdapter` 必须支持：

- 二值前景 mask
- 类别索引 mask
- 实例 id mask

对于 `instance-id` mask，如果某个实例值在跨帧语义上是稳定的，那么该值可以直接复用为 `track_id`。

如果数据集中没有显式 prompt 标注，则必须使用确定性的 prompt synthesis。
