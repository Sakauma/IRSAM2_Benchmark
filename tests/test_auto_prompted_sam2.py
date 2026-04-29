import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.baselines.methods import HeuristicAutoPromptedSAM2
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.sample import Sample


class DummySAM2Adapter:
    def __init__(self):
        self.kwargs = None

    def predict_image(self, image_rgb, **kwargs):
        self.kwargs = kwargs
        return {"masks": np.ones((1, 16, 16), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)}

    def predict_images(self, image_rgbs, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        return [{"masks": np.ones((1, 16, 16), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in image_rgbs]


def _sample(path: Path) -> Sample:
    return Sample(
        image_path=path,
        sample_id="sample",
        frame_id="sample",
        sequence_id="seq",
        frame_index=0,
        temporal_key="sample",
        width=16,
        height=16,
        category="target",
        target_scale="small",
        device_source="test",
        annotation_protocol_flag="mask",
        supervision_type="mask",
    )


class AutoPromptedSAM2Tests(unittest.TestCase):
    def test_auto_box_point_negative_prompt_passes_labels_to_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((16, 16), dtype=np.uint8)
            image[7:9, 7:9] = 255
            Image.fromarray(image).save(path)
            adapter = DummySAM2Adapter()
            method = HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX_POINT, use_negative_ring=True)

            pred = method.predict_sample(_sample(path))

            self.assertIn("box", adapter.kwargs)
            self.assertEqual(adapter.kwargs["points"].shape[1], 2)
            self.assertGreater(adapter.kwargs["points"].shape[0], 1)
            self.assertIn(0, adapter.kwargs["point_labels"].tolist())
            self.assertEqual(pred["prompt"]["source"], "synthesized")
            self.assertEqual(pred["prompt"]["negative_point_count"], 4)

    def test_auto_box_prompt_does_not_pass_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((16, 16), dtype=np.uint8)
            image[7:9, 7:9] = 255
            Image.fromarray(image).save(path)
            adapter = DummySAM2Adapter()
            method = HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX)

            method.predict_sample(_sample(path))

            self.assertIn("box", adapter.kwargs)
            self.assertNotIn("points", adapter.kwargs)
            self.assertNotIn("point_labels", adapter.kwargs)


if __name__ == "__main__":
    unittest.main()
