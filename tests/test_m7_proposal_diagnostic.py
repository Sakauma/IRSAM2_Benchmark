import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.diagnostics.proposal_diagnostic import ProposalSweepItem, proposal_metric_row, summarize_rows


class M7ProposalDiagnosticTests(unittest.TestCase):
    def test_proposal_metric_row_reports_topk_hit_rank_and_box_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            sample = Sample(
                image_path=image_path,
                sample_id="sample",
                frame_id="frame",
                sequence_id="seq",
                frame_index=0,
                temporal_key="frame",
                width=8,
                height=8,
                category="target",
                target_scale="small",
                device_source="synthetic",
                annotation_protocol_flag="mask",
                supervision_type="mask",
            )
            gt = np.zeros((8, 8), dtype=np.float32)
            gt[3:5, 3:5] = 1.0
            prompt = {
                "point": [1.0, 1.0],
                "box": [2.0, 2.0, 6.0, 6.0],
                "candidate_score": 0.9,
                "candidate_points": [[1.0, 1.0, 0.9], [3.0, 3.0, 0.7]],
                "candidate_count": 2,
                "candidate_top_k": 2,
                "candidate_nms_radius": 4,
            }

            row = proposal_metric_row(
                sample=sample,
                prompt=prompt,
                gt_mask=gt,
                sweep=ProposalSweepItem(top_k=2, response_threshold=0.1, nms_radius=4),
                checkpoint_path=Path("checkpoint.pt"),
                device="cpu",
                inference_ms=12.5,
            )

        self.assertEqual(row["PromptHitRate"], 0.0)
        self.assertEqual(row["PromptTopKHitRate"], 1.0)
        self.assertEqual(row["PromptTopKFirstHitRank"], 1)
        self.assertAlmostEqual(row["PromptTopKFirstHitScore"], 0.7)
        self.assertEqual(row["PromptBoxCoverage"], 1.0)
        self.assertEqual(row["area_bin"], "tiny_0_15")

    def test_summarize_rows_groups_sweep_metrics(self):
        rows = [
            {"PromptTopKRequested": 5, "PromptResponseThreshold": 0.1, "PromptCandidateNmsRadius": 4, "PromptHitRate": 1.0},
            {"PromptTopKRequested": 5, "PromptResponseThreshold": 0.1, "PromptCandidateNmsRadius": 4, "PromptHitRate": 0.0},
            {"PromptTopKRequested": 10, "PromptResponseThreshold": 0.1, "PromptCandidateNmsRadius": 4, "PromptHitRate": 1.0},
        ]

        summary = summarize_rows(
            rows,
            group_keys=["PromptTopKRequested", "PromptResponseThreshold", "PromptCandidateNmsRadius"],
        )

        by_topk = {item["PromptTopKRequested"]: item for item in summary}
        self.assertEqual(by_topk[5]["sample_count"], 2)
        self.assertAlmostEqual(by_topk[5]["PromptHitRate_mean"], 0.5)
        self.assertEqual(by_topk[10]["sample_count"], 1)
        self.assertAlmostEqual(by_topk[10]["PromptHitRate_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
