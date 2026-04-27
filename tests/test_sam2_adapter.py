import unittest
from pathlib import Path

import numpy as np

from irsam2_benchmark.core.interfaces import ModelCapabilities
from irsam2_benchmark.models.sam2_adapter import SAM2ModelAdapter


class StubImagePredictor:
    def __init__(self):
        self.image_batch = None
        self.box_batch = None

    def set_image_batch(self, image_batch):
        self.image_batch = image_batch

    def predict_batch(self, **kwargs):
        self.box_batch = kwargs.get("box_batch")
        masks = [np.full((1, 2, 2), idx + 1, dtype=np.float32) for idx in range(len(self.image_batch))]
        scores = [np.array([0.5 + idx], dtype=np.float32) for idx in range(len(self.image_batch))]
        logits = [np.full((1, 2, 2), idx, dtype=np.float32) for idx in range(len(self.image_batch))]
        return masks, scores, logits


class DummyImageBatchAdapter(SAM2ModelAdapter):
    def __init__(self):
        self.config = None
        self.repo = Path(".")
        self.model = object()
        self.image_predictor = StubImagePredictor()
        self.auto_mask_generator = None
        self.capabilities = ModelCapabilities()

    def ensure_loaded(self) -> None:
        return


class DummyRuntime:
    auto_mask_points_per_batch = 192


class DummyConfig:
    runtime = DummyRuntime()


class CapturingAutoMaskGenerator:
    def __init__(self, model, points_per_batch):
        self.model = model
        self.points_per_batch = points_per_batch


class DummyAutoMaskAdapter(SAM2ModelAdapter):
    def __init__(self):
        self.config = DummyConfig()
        self.repo = Path(".")
        self.model = object()
        self.image_predictor = None
        self.auto_mask_generator = None
        self.capabilities = ModelCapabilities()

    def _resolve_symbol(self, module_name: str, symbol_name: str):
        del module_name, symbol_name
        return CapturingAutoMaskGenerator


class SAM2AdapterImageBatchTests(unittest.TestCase):
    def test_predict_images_uses_batch_api_and_preserves_order(self):
        adapter = DummyImageBatchAdapter()
        images = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ]

        results = adapter.predict_images(images, boxes=[[0, 0, 2, 2], [1, 1, 3, 3]])

        self.assertEqual(len(results), 2)
        self.assertEqual(float(results[0]["masks"][0, 0, 0]), 1.0)
        self.assertEqual(float(results[1]["masks"][0, 0, 0]), 2.0)
        self.assertEqual(len(adapter.image_predictor.image_batch), 2)
        self.assertEqual(adapter.image_predictor.box_batch[0].tolist(), [0.0, 0.0, 2.0, 2.0])

    def test_auto_mask_generator_receives_points_per_batch(self):
        adapter = DummyAutoMaskAdapter()

        generator = adapter._build_auto_mask_generator()

        self.assertEqual(generator.points_per_batch, 192)
        self.assertIs(generator.model, adapter.model)


if __name__ == "__main__":
    unittest.main()
