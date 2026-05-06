import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from irsam2_benchmark.baselines.methods import HeuristicAutoPromptedSAM2, LearnedAutoPromptedSAM2
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.models import LearnedAutoPrompt


class DummySAM2Adapter:
    def __init__(self):
        self.kwargs = None
        self.image_batch_calls = 0
        self.prompt_batch_calls = 0

    def predict_image(self, image_rgb, **kwargs):
        self.kwargs = kwargs
        return {"masks": np.ones((1, 16, 16), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)}

    def predict_images(self, image_rgbs, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.image_batch_calls += 1
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        return [{"masks": np.ones((1, 16, 16), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in image_rgbs]

    def predict_prompts_for_image(self, image_rgb, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        self.prompt_batch_calls += 1
        self.kwargs = {"boxes": boxes, "points": points, "point_labels": point_labels, "multimask_output": multimask_output}
        prompt_count = len(boxes or points or point_labels or [])
        return [{"masks": np.ones((1, 16, 16), dtype=np.float32), "scores": np.array([1.0], dtype=np.float32)} for _ in range(prompt_count)]


class GroupedRerankAdapter:
    def __init__(self):
        self.set_image_calls = 0
        self.current_prompt_calls = 0
        self.prompt_predict_count = 0

    def set_image(self, image_rgb):
        self.set_image_calls += 1

    def predict_current_image_prompts(self, *, boxes=None, points=None, point_labels=None, multimask_output=False):
        del boxes, point_labels, multimask_output
        self.current_prompt_calls += 1
        prompt_count = len(points or [])
        self.prompt_predict_count += prompt_count
        return [
            {
                "masks": np.ones((1, 16, 16), dtype=np.float32),
                "scores": np.array([1.0], dtype=np.float32),
                "logits": np.ones((1, 16, 16), dtype=np.float32),
            }
            for _ in range(prompt_count)
        ]

    def predict_prompts_for_image(self, *args, **kwargs):
        raise AssertionError("Grouped learned rerank should reuse the current image embedding.")

    def predict_image(self, *args, **kwargs):
        raise AssertionError("Grouped learned rerank should not fall back to per-sample predict_image.")

    def profile_counters(self):
        return {"set_image": self.set_image_calls, "prompt_predict": self.prompt_predict_count}


def _sample(path: Path, sample_id: str = "sample") -> Sample:
    return Sample(
        image_path=path,
        sample_id=sample_id,
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


def _learned_config(root: Path, checkpoint: Path):
    return SimpleNamespace(
        root=root,
        config_path=root / "config.yaml",
        output_dir=root / "out",
        runtime=SimpleNamespace(device="cpu", output_name="test", profile_eval=True),
        dataset=SimpleNamespace(dataset_id="test"),
        method={
            "prompt_checkpoint": str(checkpoint),
            "prompt_device": "cpu",
            "prompt_top_k": 2,
            "prompt_reranker": {"use_mask_feedback": True, "max_feedback_candidates": 2, "use_frequency": False},
        },
    )


class DummyLearnedAutoPromptedSAM2(LearnedAutoPromptedSAM2):
    def __init__(self, adapter, config):
        super().__init__(adapter, config, prompt_mode=InferenceMode.POINT, rerank_candidates=True)
        self.auto_prompt_calls = 0

    def _auto_prompt_object(self, sample: Sample):
        self.auto_prompt_calls += 1
        objectness = np.zeros((16, 16), dtype=np.float32)
        objectness[8, 8] = 1.0
        return LearnedAutoPrompt(
            point=[8.0, 8.0],
            box=[6.0, 6.0, 10.0, 10.0],
            points=[[8.0, 8.0]],
            point_labels=[1],
            metadata={
                "candidate_score": 1.0,
                "candidate_points": [[8.0, 8.0, 1.0], [4.0, 4.0, 0.4]],
            },
            objectness=objectness,
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

    def test_auto_prompted_batch_reuses_single_image_embedding_for_same_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((16, 16), dtype=np.uint8)
            image[7:9, 7:9] = 255
            Image.fromarray(image).save(path)
            adapter = DummySAM2Adapter()
            method = HeuristicAutoPromptedSAM2(adapter, prompt_mode=InferenceMode.BOX)

            predictions = method.predict_samples([_sample(path, "sample_0"), _sample(path, "sample_1")])

            self.assertEqual(sorted(predictions), ["sample_0", "sample_1"])
            self.assertEqual(adapter.prompt_batch_calls, 1)
            self.assertEqual(adapter.image_batch_calls, 0)
            self.assertEqual(len(adapter.kwargs["boxes"]), 2)

    def test_learned_rerank_batch_reuses_image_embedding_for_same_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "sample.png"
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"placeholder")
            image = np.zeros((16, 16), dtype=np.uint8)
            image[7:9, 7:9] = 255
            Image.fromarray(image).save(path)
            adapter = GroupedRerankAdapter()
            method = DummyLearnedAutoPromptedSAM2(adapter, _learned_config(root, checkpoint))

            predictions = method.predict_samples([_sample(path, "sample_0"), _sample(path, "sample_1")])

            self.assertEqual(sorted(predictions), ["sample_0", "sample_1"])
            self.assertEqual(method.auto_prompt_calls, 1)
            self.assertEqual(adapter.set_image_calls, 1)
            self.assertEqual(adapter.current_prompt_calls, 1)
            self.assertEqual(adapter.prompt_predict_count, 2)
            metadata = predictions["sample_0"]["metadata"]
            self.assertEqual(metadata["EvalOptimizationPath"], "image_grouped_rerank_v1")
            self.assertEqual(metadata["EvalImageGroupSize"], 2.0)
            self.assertEqual(metadata["EvalSamSetImageCalls"], 1.0)


if __name__ == "__main__":
    unittest.main()
