# Track 定义

## Track A：图像级 Prompted Segmentation

在 `box`、`point` 或 `box+point` 条件下进行提示式分割评估。

## Track B：无 Prompt Automatic Masking

不提供外部 prompt，直接做 automatic mask generation，并在图像级与该图中的全部 GT 实例进行匹配评估。

## Track C：序列 / 视频传播

使用带冻结 prompt policy 的 SAM2 时序传播能力进行 sequence-aware 推理。
该 track 要求显式 `track_id`，从而确保每条传播流都对应真实的跨帧目标。

## Track D：完整部署链路

同时在图像指标和时序指标下比较 `adapted teacher -> student -> quantized student`。
