import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.models import LearnedAutoPrompt
from irsam2_benchmark.models.prompt_reranker import (
    PromptRerankerConfig,
    calibrate_box_from_results,
    make_scaled_boxes,
    prompt_reranker_config_from_dict,
    rank_prompt_candidates,
    score_mask_feedback,
)


def _write_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)).save(path)


class PromptRerankerTests(unittest.TestCase):
    def test_config_parser_supports_gated_box_fields_and_ignores_unknown_keys(self):
        config = prompt_reranker_config_from_dict(
            {
                "box_enable_margin": 0.05,
                "box_enable_min_score": 0.2,
                "unknown_note": "kept out of runtime config",
            }
        )

        self.assertEqual(config.box_enable_margin, 0.05)
        self.assertEqual(config.box_enable_min_score, 0.2)

    def test_rank_prompt_candidates_can_override_raw_objectness_with_ir_cues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image = np.zeros((32, 32), dtype=np.float32)
            image[23:26, 23:26] = 1.0
            _write_gray(image_path, image)
            prompt = LearnedAutoPrompt(
                point=[4.0, 4.0],
                box=[2.0, 2.0, 7.0, 7.0],
                points=[[4.0, 4.0]],
                point_labels=[1],
                metadata={
                    "candidate_score": 0.9,
                    "candidate_points": [[4.0, 4.0, 0.9], [24.0, 24.0, 0.45]],
                },
                objectness=np.zeros((32, 32), dtype=np.float32),
            )
            config = PromptRerankerConfig(
                use_frequency=False,
                prior_weight_objectness=0.1,
                prior_weight_local_contrast=0.45,
                prior_weight_top_hat=0.45,
                prior_weight_peak_sharpness=0.0,
            )

            result = rank_prompt_candidates(prompt, image_path, config)

            self.assertEqual(result.selected.point, [24.0, 24.0])
            self.assertEqual(result.selected.index, 1)
            self.assertGreater(result.selected.prior_score, result.candidates[-1].prior_score)

    def test_mask_feedback_penalizes_large_false_alarm_masks(self):
        gray = np.zeros((32, 32), dtype=np.float32)
        gray[15:17, 15:17] = 1.0
        compact = np.zeros_like(gray)
        compact[15:17, 15:17] = 1.0
        diffuse = np.ones_like(gray)
        config = PromptRerankerConfig(max_area_ratio=0.02)

        compact_score = score_mask_feedback(compact, sam_score=0.8, gray=gray, point=[16.0, 16.0], config=config)
        diffuse_score = score_mask_feedback(diffuse, sam_score=0.8, gray=gray, point=[16.0, 16.0], config=config)

        self.assertGreater(compact_score["feedback_score"], diffuse_score["feedback_score"])
        self.assertLess(diffuse_score["area_score"], compact_score["area_score"])

    def test_make_scaled_boxes_clamps_to_image_and_removes_duplicates(self):
        boxes = make_scaled_boxes(
            point=[1.0, 1.0],
            base_box=[0.0, 0.0, 3.0, 3.0],
            image_width=8,
            image_height=8,
            min_box_side=2.0,
            scales=[1.0, 1.0, 2.0],
        )

        self.assertEqual(len(boxes), 2)
        for _, box in boxes:
            self.assertGreaterEqual(min(box), 0.0)
            self.assertLessEqual(max(box), 8.0)

    def test_calibrate_box_selects_best_feedback_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image = np.zeros((16, 16), dtype=np.float32)
            image[7:9, 7:9] = 1.0
            _write_gray(image_path, image)
            small_mask = np.zeros((1, 16, 16), dtype=np.float32)
            small_mask[0, 7:9, 7:9] = 1.0
            large_mask = np.ones((1, 16, 16), dtype=np.float32)
            boxes = [(1.0, [6.0, 6.0, 10.0, 10.0]), (3.0, [0.0, 0.0, 16.0, 16.0])]

            result = calibrate_box_from_results(
                boxes=boxes,
                image_path=image_path,
                point=[8.0, 8.0],
                sam_results=[
                    {"masks": small_mask, "scores": np.array([0.8], dtype=np.float32)},
                    {"masks": large_mask, "scores": np.array([0.8], dtype=np.float32)},
                ],
                config=PromptRerankerConfig(max_area_ratio=0.05),
            )

            self.assertEqual(result.selected.scale, 1.0)
            self.assertEqual(result.selected.index, 0)


if __name__ == "__main__":
    unittest.main()
