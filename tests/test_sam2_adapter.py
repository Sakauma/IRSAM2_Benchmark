import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.core.interfaces import ModelCapabilities, PromptPolicy, PromptSource, PromptType
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.models.sam2_adapter import SAM2ModelAdapter


class StubVideoPredictor:
    def __init__(self):
        self.saved_frame_count = 0

    def init_state(self, video_path: str):
        self.saved_frame_count = len(list(Path(video_path).glob("*.png")))
        return {"video_path": video_path}

    def add_new_points_or_box(self, **kwargs):
        return kwargs

    def propagate_in_video(self, state):
        del state
        yield 0, [1], np.ones((1, 4, 4), dtype=np.float32)
        yield 1, [1], np.ones((1, 4, 4), dtype=np.float32)


class DummyVideoAdapter(SAM2ModelAdapter):
    def __init__(self):
        self.config = None
        self.repo = Path(".")
        self.model = object()
        self.image_predictor = None
        self.video_predictor = StubVideoPredictor()
        self.auto_mask_generator = None
        self.capabilities = ModelCapabilities()

    def ensure_loaded(self) -> None:
        return


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
        self.video_predictor = None
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
        self.video_predictor = None
        self.auto_mask_generator = None
        self.capabilities = ModelCapabilities()

    def _resolve_symbol(self, module_name: str, symbol_name: str):
        del module_name, symbol_name
        return CapturingAutoMaskGenerator


class SAM2AdapterVideoTests(unittest.TestCase):
    def _make_sample(self, image_path: Path, sample_id: str, frame_id: str, frame_index: int) -> Sample:
        return Sample(
            image_path=image_path,
            sample_id=sample_id,
            frame_id=frame_id,
            sequence_id="seq",
            frame_index=frame_index,
            temporal_key=frame_id,
            width=4,
            height=4,
            category="target",
            target_scale="small",
            device_source="cam",
            annotation_protocol_flag="mask",
            supervision_type="mask",
            track_id="track_a",
            bbox_tight=[0, 0, 2, 2],
            bbox_loose=[0, 0, 2, 2],
            point_prompt=[0.5, 0.5],
            mask_array=np.ones((4, 4), dtype=np.float32),
        )

    def test_predict_video_sequence_writes_one_frame_per_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "frame_0.png"
            image_b = root / "frame_1.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_a)
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_b)

            adapter = DummyVideoAdapter()
            policy = PromptPolicy(name="p", prompt_type=PromptType.BOX, prompt_source=PromptSource.GT, prompt_budget=1)
            predictions = adapter.predict_video_sequence(
                [
                    self._make_sample(image_a, "frame_0__track_a", "frame_0", 0),
                    self._make_sample(image_b, "frame_1__track_a", "frame_1", 1),
                ],
                policy,
            )

            self.assertEqual(adapter.video_predictor.saved_frame_count, 2)
            self.assertEqual(set(predictions), {"frame_0__track_a", "frame_1__track_a"})

    def test_predict_video_sequence_rejects_duplicate_frame_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "frame.png"
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_a)

            adapter = DummyVideoAdapter()
            policy = PromptPolicy(name="p", prompt_type=PromptType.BOX, prompt_source=PromptSource.GT, prompt_budget=1)
            with self.assertRaises(RuntimeError):
                adapter.predict_video_sequence(
                    [
                        self._make_sample(image_a, "frame_0__track_a", "frame_0", 0),
                        self._make_sample(image_a, "frame_0__track_a_dup", "frame_0", 1),
                    ],
                    policy,
                )


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
