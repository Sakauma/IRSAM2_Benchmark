import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.data.auto_prompt import (
    HEURISTIC_IR_AUTO_PROMPT_PROTOCOL,
    generate_heuristic_ir_auto_prompt,
    generate_heuristic_ir_auto_prompt_from_path,
)
from irsam2_benchmark.evaluation.prompt_metrics import prompt_metrics


class HeuristicAutoPromptTests(unittest.TestCase):
    def test_single_bright_target_returns_nearby_point_and_covering_box(self):
        image = np.zeros((32, 32), dtype=np.float32)
        image[14:18, 15:19] = 1.0

        prompt = generate_heuristic_ir_auto_prompt(image)

        self.assertEqual(prompt.metadata["protocol"], HEURISTIC_IR_AUTO_PROMPT_PROTOCOL)
        x, y = prompt.point
        self.assertLessEqual(abs(x - 16.5), 3.0)
        self.assertLessEqual(abs(y - 15.5), 3.0)
        x1, y1, x2, y2 = prompt.box
        self.assertLessEqual(x1, 15.0)
        self.assertLessEqual(y1, 14.0)
        self.assertGreaterEqual(x2, 19.0)
        self.assertGreaterEqual(y2, 18.0)

    def test_uniform_image_uses_deterministic_fallback(self):
        image = np.zeros((16, 16), dtype=np.float32)

        prompt = generate_heuristic_ir_auto_prompt(image)

        self.assertTrue(prompt.metadata["fallback"])
        self.assertEqual(prompt.point, [0.0, 0.0])
        self.assertEqual(prompt.metadata["candidate_score"], 0.0)

    def test_boundary_target_clips_box_and_negative_points(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[0:3, 0:3] = 1.0

        prompt = generate_heuristic_ir_auto_prompt(image, negative_ring=True)

        self.assertEqual(prompt.metadata["negative_point_count"], 4)
        for x, y in prompt.points:
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(x, 19.0)
            self.assertLessEqual(y, 19.0)
        for value in prompt.box:
            self.assertGreaterEqual(value, 0.0)

    def test_prompt_metrics_include_auto_prompt_fields(self):
        gt = np.zeros((16, 16), dtype=np.float32)
        gt[6:9, 6:9] = 1.0
        prompt = {
            "point": [7.0, 7.0],
            "box": [5.0, 5.0, 10.0, 10.0],
            "points": [[7.0, 7.0], [0.0, 0.0], [15.0, 15.0]],
            "point_labels": [1, 0, 0],
            "candidate_score": 0.75,
            "fallback": False,
        }

        metrics = prompt_metrics(prompt, gt)

        self.assertEqual(metrics["PromptHitRate"], 1.0)
        self.assertEqual(metrics["AutoPromptCandidateScore"], 0.75)
        self.assertEqual(metrics["AutoPromptNumPoints"], 3.0)
        self.assertEqual(metrics["AutoPromptNegativePointCount"], 2.0)
        self.assertEqual(metrics["NegativePromptInGtRate"], 0.0)
        self.assertEqual(metrics["AutoPromptFallback"], 0.0)

    def test_load_from_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            image = np.zeros((16, 16), dtype=np.uint8)
            image[8, 8] = 255
            Image.fromarray(image).save(path)

            prompt = generate_heuristic_ir_auto_prompt_from_path(path)

            self.assertEqual(prompt.metadata["protocol"], HEURISTIC_IR_AUTO_PROMPT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
