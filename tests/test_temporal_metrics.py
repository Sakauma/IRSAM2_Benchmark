import unittest

import numpy as np

from irsam2_benchmark.evaluation.temporal_metrics import compute_temporal_metrics


class TemporalMetricTests(unittest.TestCase):
    def test_temporal_metrics_shape(self):
        rows = [
            {"mIoU": 0.8, "BoundaryF1": 0.7, "PredAreaRatio": 0.1, "GTAreaRatio": 0.1, "pred_mask": np.ones((4, 4), dtype=np.float32)},
            {"mIoU": 0.6, "BoundaryF1": 0.5, "PredAreaRatio": 0.1, "GTAreaRatio": 0.1, "pred_mask": np.ones((4, 4), dtype=np.float32)},
        ]
        result = compute_temporal_metrics(rows)
        self.assertIn("temporal_iou_mean", result)
        self.assertIn("mask_jitter_score", result)


if __name__ == "__main__":
    unittest.main()
