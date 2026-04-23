# Dataset Cards

## Required Fields

Every dataset adapter must emit a unified `Sample` schema with:

- `image_path`
- `sample_id`
- `frame_id`
- `sequence_id`
- `frame_index`
- `temporal_key`
- `category`
- `device_source`
- `annotation_protocol_flag`
- `supervision_type`

## Supported Dataset Families

- `MultiModalAdapter`: raw `img/ + label/`
- `CocoLikeAdapter`: `annotations_coco/ + image/`
- `RBGTTinyIRAdapter`: RBGT-Tiny grayscale branch
- `GenericImageMaskAdapter`: arbitrary `images/ + masks/`

## Generic Image+Mask Policy

`GenericImageMaskAdapter` must support:

- binary foreground masks
- class-index masks
- instance-id masks

If prompts are not explicitly annotated, deterministic prompt synthesis is mandatory.
