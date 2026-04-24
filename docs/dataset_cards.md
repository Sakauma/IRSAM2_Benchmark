# Dataset Cards

## Required Fields

Every dataset adapter must emit a unified `Sample` schema with:

- `image_path`
- `sample_id`
- `frame_id`
- `sequence_id`
- `frame_index`
- `temporal_key`
- `track_id` (optional, required for Track C / video propagation)
- `category`
- `device_source`
- `annotation_protocol_flag`
- `supervision_type`

Field semantics:

- `frame_id`: image-level identifier shared by all instances from the same frame
- `sample_id`: instance-level unique identifier
- `track_id`: explicit cross-frame identity when the dataset can support sequence propagation evaluation

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

For `instance-id` masks, the instance value may be reused as `track_id` when that value is explicitly stable across frames.

If prompts are not explicitly annotated, deterministic prompt synthesis is mandatory.
