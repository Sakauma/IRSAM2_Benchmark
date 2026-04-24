# Track Definitions

## Track A: Image Prompted Segmentation

Prompted segmentation with `box`, `point`, or `box+point`.

## Track B: No-Prompt Automatic Masking

Prompt-free automatic mask generation with image-level inference and instance matching against all GT objects in that image.

## Track C: Sequence / Video Propagation

Sequence-aware inference using SAM2 temporal propagation with frozen prompt policy.
This track requires explicit `track_id` values so each propagated stream represents one real target across frames.

## Track D: Full Pipeline Deployment

Compare `adapted teacher -> student -> quantized student` under both image and temporal metrics.
