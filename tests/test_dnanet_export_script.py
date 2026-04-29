import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_dnanet_predictions.py"
SPEC = importlib.util.spec_from_file_location("export_dnanet_predictions", SCRIPT_PATH)
export_dnanet_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_dnanet_predictions)


class DNANetExportScriptTests(unittest.TestCase):
    def test_discovers_images_and_preserves_relative_frame_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "seq"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "frame.v1.bmp"
            Image.fromarray(np.zeros((3, 5), dtype=np.uint8), mode="L").save(image_path)

            images = export_dnanet_predictions.discover_images(root / "images", (".bmp",))

            self.assertEqual(images, [image_path])
            frame_id = export_dnanet_predictions.frame_id_from_image(image_path, root / "images")
            self.assertEqual(frame_id, "seq/frame.v1")
            self.assertEqual(
                export_dnanet_predictions.prediction_path_from_frame_id(root / "predictions", frame_id),
                root / "predictions" / "seq" / "frame.v1.png",
            )

    def test_preprocess_and_binary_mask_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.png"
            Image.fromarray(np.full((3, 5, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)

            tensor, original_hw = export_dnanet_predictions.preprocess_image(image_path, input_size=8)
            mask = export_dnanet_predictions.binary_mask_from_logits(
                torch.tensor([[[[-1.0, 0.2], [0.0, 2.0]]]]),
                threshold=0.0,
            )

            self.assertEqual(tuple(tensor.shape), (1, 3, 8, 8))
            self.assertEqual(original_hw, (3, 5))
            self.assertEqual(mask.tolist(), [[0, 255], [0, 255]])


if __name__ == "__main__":
    unittest.main()
