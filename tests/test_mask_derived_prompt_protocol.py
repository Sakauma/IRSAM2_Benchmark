import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.baselines.methods import PretrainedPromptedSAM2
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.adapters import _samples_from_generic_mask
from irsam2_benchmark.data.prompt_synthesis import (
    MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL,
    MASK_DERIVED_PROMPT_PROTOCOL,
    MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL,
    mask_derived_prompt_metadata,
)
from irsam2_benchmark.evaluation.runner import build_segmentation_row


class DummySAM2Adapter:
    def __init__(self):
        self.kwargs = None

    def predict_image(self, image_rgb, **kwargs):
        self.kwargs = kwargs
        return {"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)}


class MaskDerivedPromptProtocolTests(unittest.TestCase):
    def test_generic_mask_samples_record_prompt_generation_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:5, 3:6] = 255

            samples = _samples_from_generic_mask(
                image_path=image_path,
                frame_id="sample",
                sequence_id="seq",
                frame_index=0,
                temporal_key="sample",
                device_source="cam",
                mask=mask,
                width=8,
                height=8,
                mask_mode="binary",
                class_map={},
            )

            self.assertEqual(samples[0].metadata["prompt_generation"]["protocol"], "mask_derived_gt_prompt_rules_v1")
            self.assertEqual(samples[0].metadata["prompt_generation"]["loose_box_point_protocol"], MASK_DERIVED_PROMPT_PROTOCOL)
            self.assertEqual(samples[0].bbox_tight, [3.0, 2.0, 6.0, 5.0])

    def test_tight_and_loose_box_baselines_use_different_prompt_boxes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            sample = _samples_from_generic_mask(
                image_path=image_path,
                frame_id="sample",
                sequence_id="seq",
                frame_index=0,
                temporal_key="sample",
                device_source="cam",
                mask=np.pad(np.ones((2, 2), dtype=np.uint8), ((2, 4), (3, 3))),
                width=8,
                height=8,
                mask_mode="binary",
                class_map={},
            )[0]

            tight_adapter = DummySAM2Adapter()
            tight_method = PretrainedPromptedSAM2(tight_adapter, prompt_mode=InferenceMode.BOX, box_variant="tight")
            tight_pred = tight_method.predict_sample(sample)

            loose_adapter = DummySAM2Adapter()
            loose_method = PretrainedPromptedSAM2(loose_adapter, prompt_mode=InferenceMode.BOX, box_variant="loose")
            loose_pred = loose_method.predict_sample(sample)

            self.assertEqual(tight_adapter.kwargs["box"], sample.bbox_tight)
            self.assertEqual(loose_adapter.kwargs["box"], sample.bbox_loose)
            self.assertEqual(tight_pred["prompt"]["box_variant"], "tight")
            self.assertEqual(loose_pred["prompt"]["box_variant"], "loose")
            self.assertEqual(tight_pred["prompt"]["protocol"], MASK_DERIVED_TIGHT_BOX_CENTROID_POINT_PROTOCOL)
            self.assertEqual(loose_pred["prompt"]["protocol"], MASK_DERIVED_LOOSE_BOX_CENTROID_POINT_PROTOCOL)

    def test_eval_row_records_prompt_generation_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)
            sample = _samples_from_generic_mask(
                image_path=image_path,
                frame_id="sample",
                sequence_id="seq",
                frame_index=0,
                temporal_key="sample",
                device_source="cam",
                mask=np.ones((4, 4), dtype=np.uint8),
                width=4,
                height=4,
                mask_mode="binary",
                class_map={},
            )[0]
            prompt = {
                **mask_derived_prompt_metadata(),
                "protocol": MASK_DERIVED_PROMPT_PROTOCOL,
                "box_variant": "loose",
                "box": sample.bbox_loose,
            }
            row = build_segmentation_row(sample, sample.mask_array, sample.mask_array, elapsed_ms=1.0, prompt=prompt)

            self.assertEqual(row["PromptProtocol"], MASK_DERIVED_PROMPT_PROTOCOL)
            self.assertEqual(row["PromptSource"], "gt_mask")
            self.assertEqual(row["PromptBoxVariant"], "loose")
            self.assertEqual(row["PromptLooseBoxMinSide"], 0.0)
            self.assertEqual(row["PromptLooseBoxMaxSideMultiplier"], 2.0)


if __name__ == "__main__":
    unittest.main()
