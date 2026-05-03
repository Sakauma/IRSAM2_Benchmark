import unittest

import numpy as np

from irsam2_benchmark.evaluation.prompt_metrics import prompt_metrics
from irsam2_benchmark.evaluation.image_metrics import boundary_f1, boundary_f1_tolerance
from irsam2_benchmark.evaluation.small_target_metrics import small_target_metrics


class SmallTargetMetricTests(unittest.TestCase):
    def test_small_target_recall_and_false_alarm(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        pred = np.zeros((8, 8), dtype=np.float32)
        gt[2:4, 2:4] = 1.0
        pred[2:4, 2:4] = 1.0
        pred[7, 7] = 1.0
        metrics = small_target_metrics(pred, gt)
        self.assertEqual(metrics["TargetRecallIoU50"], 1.0)
        self.assertEqual(metrics["FalseAlarmComponents"], 1.0)
        self.assertEqual(metrics["GTAreaPixels"], 4.0)
        self.assertEqual(metrics["PredAreaPixels"], 5.0)

    def test_prompt_metrics(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[2:4, 2:4] = 1.0
        metrics = prompt_metrics({"point": [2.5, 2.5], "box": [1, 1, 5, 5], "candidate_points": [[0, 0, 0.9], [2.5, 2.5, 0.8]]}, gt)
        self.assertEqual(metrics["PromptHitRate"], 1.0)
        self.assertEqual(metrics["PromptBoxCoverage"], 1.0)
        self.assertLess(metrics["PromptDistanceToCentroid"], 1.0)
        self.assertEqual(metrics["PromptTopKHitRate"], 1.0)

    def test_prompt_metrics_detect_border_prompt(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[0, 7] = 1.0
        metrics = prompt_metrics({"point": [7.0, 0.0], "border_metric_px": 1}, gt)
        self.assertEqual(metrics["PromptHitRate"], 1.0)
        self.assertEqual(metrics["PromptBorderRate"], 1.0)

    def test_boundary_f1_tolerance_handles_one_pixel_shift(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        pred = np.zeros((8, 8), dtype=np.float32)
        gt[2:4, 2:4] = 1.0
        pred[2:4, 3:5] = 1.0
        self.assertLess(boundary_f1(pred, gt), boundary_f1_tolerance(pred, gt, radius=1))


if __name__ == "__main__":
    unittest.main()
