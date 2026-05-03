import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.baselines.methods import LearnedAutoPromptedSAM2
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.models import (
    LEARNED_IR_AUTO_PROMPT_PROTOCOL,
    AutoPromptModelConfig,
    build_ir_prompt_net,
    decode_auto_prompt,
    ir_prior_stack_from_path,
    load_auto_prompt_model,
    save_auto_prompt_checkpoint,
)


class DummySAM2Adapter:
    def __init__(self):
        self.kwargs = None

    def predict_image(self, image_rgb, **kwargs):
        self.kwargs = kwargs
        return {"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)}

    def predict_images(self, image_rgbs, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        return [{"masks": np.ones((1, 8, 8), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in image_rgbs]


def _sample(path: Path) -> Sample:
    return Sample(
        image_path=path,
        sample_id="sample",
        frame_id="sample",
        sequence_id="seq",
        frame_index=0,
        temporal_key="sample",
        width=8,
        height=8,
        category="target",
        target_scale="small",
        device_source="test",
        annotation_protocol_flag="mask",
        supervision_type="mask",
    )


def _config(checkpoint_path: Path):
    config = type("Config", (), {})()
    config.method = {"prompt_checkpoint": str(checkpoint_path), "prompt_device": "cpu"}
    config.runtime = type("Runtime", (), {"device": "cpu"})()
    config.root = checkpoint_path.parent
    config.config_path = checkpoint_path.parent / "config.yaml"
    return config


class LearnedAutoPromptTests(unittest.TestCase):
    def test_ir_prior_stack_from_path_has_three_channels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((8, 8), dtype=np.uint8)
            image[3:5, 3:5] = 255
            Image.fromarray(image).save(path)

            prior = ir_prior_stack_from_path(path)

            self.assertEqual(prior.shape, (3, 8, 8))
            self.assertGreaterEqual(float(prior.min()), 0.0)
            self.assertLessEqual(float(prior.max()), 1.0)

    def test_decode_auto_prompt_uses_argmax_and_local_box_size(self):
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        logits[0, 2, 3] = 10.0
        box_size = np.zeros((2, 4, 5), dtype=np.float32)
        box_size[0, :, :] = 4.0
        box_size[1, :, :] = 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=5,
            image_height=4,
            negative_ring=True,
        )

        self.assertEqual(prompt.point, [3.0, 2.0])
        self.assertEqual(prompt.metadata["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)
        self.assertEqual(prompt.metadata["negative_point_count"], 4)
        self.assertEqual(len(prompt.points), 5)

    def test_decode_auto_prompt_supports_topk_nms_and_threshold_metadata(self):
        logits = np.zeros((1, 8, 8), dtype=np.float32)
        logits[0, 2, 2] = 8.0
        logits[0, 2, 3] = 7.0
        logits[0, 6, 6] = 6.0
        box_size = np.ones((2, 8, 8), dtype=np.float32) * 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=8,
            image_height=8,
            top_k=3,
            point_budget=2,
            nms_radius=2,
            response_threshold=0.1,
        )

        self.assertEqual(prompt.metadata["candidate_top_k"], 3)
        self.assertEqual(prompt.metadata["candidate_nms_radius"], 2)
        self.assertEqual(prompt.metadata["positive_point_count"], 2)
        self.assertEqual(prompt.points[0], [2.0, 2.0])
        self.assertEqual(prompt.points[1], [6.0, 6.0])
        self.assertEqual(prompt.metadata["candidate_points"][0][:2], [2.0, 2.0])

    def test_decode_auto_prompt_can_suppress_border_candidates(self):
        logits = np.zeros((1, 8, 8), dtype=np.float32)
        logits[0, 0, 7] = 10.0
        logits[0, 4, 4] = 8.0
        box_size = np.ones((2, 8, 8), dtype=np.float32) * 2.0

        prompt = decode_auto_prompt(
            objectness_logits=logits,
            box_size=box_size,
            image_width=8,
            image_height=8,
            border_suppression_px=1,
        )

        self.assertEqual(prompt.point, [4.0, 4.0])
        self.assertEqual(prompt.metadata["border_suppression_px"], 1)
        self.assertGreater(prompt.metadata["primary_border_distance_px"], 0)

    def test_checkpoint_roundtrip_loads_model_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            cfg = AutoPromptModelConfig(hidden_channels=4)
            model = build_ir_prompt_net(cfg)

            save_auto_prompt_checkpoint(checkpoint, model, config=cfg, metadata={"tag": "unit"})
            loaded, meta = load_auto_prompt_model(checkpoint, device="cpu")

            self.assertIsNotNone(loaded)
            self.assertEqual(meta["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)
            self.assertEqual(meta["metadata"]["tag"], "unit")

    def test_learned_auto_prompted_sam2_passes_prompt_to_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            save_auto_prompt_checkpoint(checkpoint, build_ir_prompt_net(AutoPromptModelConfig(hidden_channels=4)))
            adapter = DummySAM2Adapter()
            method = LearnedAutoPromptedSAM2(adapter, _config(checkpoint), prompt_mode=InferenceMode.BOX_POINT, use_negative_ring=True)

            pred = method.predict_sample(_sample(image_path))

            self.assertIn("box", adapter.kwargs)
            self.assertEqual(adapter.kwargs["points"].shape[1], 2)
            self.assertIn(0, adapter.kwargs["point_labels"].tolist())
            self.assertEqual(pred["prompt"]["source"], "learned_auto_prompt")
            self.assertEqual(pred["prompt"]["protocol"], LEARNED_IR_AUTO_PROMPT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
