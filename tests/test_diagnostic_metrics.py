import unittest

from irsam2_benchmark.analysis.diagnostics import (
    aor_rows,
    diagnostic_metric_rows,
    fab_tr_rows,
    hit_conditioned_iou_rows,
    pmcr_rows,
    prompt_aggregate_rows,
)


class DiagnosticMetricTests(unittest.TestCase):
    def test_aor_rows_compare_auto_prompt_against_no_prompt_and_oracle(self):
        rows = [
            {"dataset": "d", "method": "sam2_no_prompt_auto_mask", "mIoU": 0.2, "TargetRecallIoU25": 0.3},
            {"dataset": "d", "method": "sam2_pretrained_box_prompt", "mIoU": 0.8, "TargetRecallIoU25": 0.9},
            {"dataset": "d", "method": "sam2_heuristic_auto_box_point_prompt", "mIoU": 0.5, "TargetRecallIoU25": 0.6},
        ]

        table = aor_rows(rows, {"diagnostics": {"aor": {"metrics": ["mIoU", "TargetRecallIoU25"]}}})
        by_metric = {row["metric"]: row for row in table}

        self.assertEqual(set(by_metric), {"mIoU", "TargetRecallIoU25"})
        self.assertEqual(by_metric["mIoU"]["status"], "ok")
        self.assertAlmostEqual(by_metric["mIoU"]["AOR"], 0.5)
        self.assertAlmostEqual(by_metric["TargetRecallIoU25"]["AOR"], 0.5)

    def test_aor_rows_report_low_gap_and_missing_reference(self):
        low_gap_rows = [
            {"dataset": "d", "method": "sam2_no_prompt_auto_mask", "mIoU": 0.5},
            {"dataset": "d", "method": "sam2_pretrained_box_prompt", "mIoU": 0.52},
            {"dataset": "d", "method": "sam2_heuristic_auto_box_point_prompt", "mIoU": 0.51},
        ]
        missing_rows = [{"dataset": "d", "method": "sam2_heuristic_auto_box_point_prompt", "mIoU": 0.51}]

        low_gap = aor_rows(low_gap_rows, {"diagnostics": {"aor": {"metrics": ["mIoU"], "low_gap_threshold": 0.05}}})
        missing = aor_rows(missing_rows, {"diagnostics": {"aor": {"metrics": ["mIoU"]}}})

        self.assertEqual(low_gap[0]["status"], "low_gap")
        self.assertIsNone(low_gap[0]["AOR"])
        self.assertEqual(missing[0]["status"], "missing_reference")
        self.assertIsNone(missing[0]["AOR"])

    def test_pmcr_rows_measure_prompt_hit_conversion(self):
        rows = [
            {"dataset": "d", "method": "auto", "PromptHitRate": 1.0, "mIoU": 0.3},
            {"dataset": "d", "method": "auto", "PromptHitRate": 1.0, "mIoU": 0.7},
            {"dataset": "d", "method": "auto", "PromptHitRate": 0.0, "mIoU": 1.0},
        ]

        table = pmcr_rows(rows, {"diagnostics": {"pmcr": {"thresholds": [0.25, 0.5]}}})
        by_threshold = {row["threshold"]: row for row in table}

        self.assertEqual(by_threshold[0.25]["prompt_hit_count"], 2)
        self.assertAlmostEqual(by_threshold[0.25]["PMCR"], 1.0)
        self.assertAlmostEqual(by_threshold[0.5]["PMCR"], 0.5)

    def test_fab_tr_rows_measure_recall_under_false_alarm_budget(self):
        rows = [
            {"dataset": "d", "method": "m", "FalseAlarmPixelsPerMP": 500.0, "TargetRecallIoU25": 1.0},
            {"dataset": "d", "method": "m", "FalseAlarmPixelsPerMP": 2000.0, "TargetRecallIoU25": 0.0},
            {"dataset": "d", "method": "m", "FalseAlarmPixelsPerMP": 8000.0, "TargetRecallIoU25": 1.0},
        ]

        table = fab_tr_rows(rows, {"diagnostics": {"fab_tr": {"budgets": [1000.0, 5000.0]}}})
        by_budget = {row["false_alarm_budget_per_mp"]: row for row in table}

        self.assertEqual(by_budget[1000.0]["eligible_count"], 1)
        self.assertAlmostEqual(by_budget[1000.0]["FAB-TR"], 1.0)
        self.assertEqual(by_budget[5000.0]["eligible_count"], 2)
        self.assertAlmostEqual(by_budget[5000.0]["FAB-TR"], 0.5)

    def test_prompt_quality_rows_measure_border_topk_and_hit_conditioned_iou(self):
        rows = [
            {"dataset": "d", "method": "auto", "PromptHitRate": 1.0, "PromptBorderRate": 0.0, "PromptTopKHitRate": 1.0, "mIoU": 0.6},
            {"dataset": "d", "method": "auto", "PromptHitRate": 0.0, "PromptBorderRate": 1.0, "PromptTopKHitRate": 0.0, "mIoU": 0.1},
        ]

        aggregates = prompt_aggregate_rows(rows, {"diagnostics": {"prompt_quality": {}}})
        by_metric = {row["metric"]: row for row in aggregates}
        hit_conditioned = hit_conditioned_iou_rows(rows, {"diagnostics": {"prompt_quality": {}}})

        self.assertAlmostEqual(by_metric["PromptBorderRate"]["value"], 0.5)
        self.assertAlmostEqual(by_metric["PromptTopKHitRate"]["value"], 0.5)
        self.assertEqual(hit_conditioned[0]["hit_count"], 1)
        self.assertAlmostEqual(hit_conditioned[0]["HitConditionedIoU"], 0.6)

    def test_diagnostic_metric_rows_combines_all_sections(self):
        rows = [
            {"dataset": "d", "method": "sam2_no_prompt_auto_mask", "mIoU": 0.2, "PromptHitRate": 0.0, "FalseAlarmPixelsPerMP": 0.0, "TargetRecallIoU25": 0.0},
            {"dataset": "d", "method": "sam2_pretrained_box_prompt", "mIoU": 0.8, "PromptHitRate": 1.0, "FalseAlarmPixelsPerMP": 0.0, "TargetRecallIoU25": 1.0},
            {
                "dataset": "d",
                "method": "sam2_heuristic_auto_box_point_prompt",
                "mIoU": 0.5,
                "PromptHitRate": 1.0,
                "FalseAlarmPixelsPerMP": 0.0,
                "TargetRecallIoU25": 1.0,
            },
        ]

        table = diagnostic_metric_rows(rows, {"diagnostics": {"aor": {"metrics": ["mIoU"]}}})

        self.assertTrue(any(row["diagnostic"] == "AOR" for row in table))
        self.assertTrue(any(row["diagnostic"] == "PMCR" for row in table))
        self.assertTrue(any(row["diagnostic"] == "FAB-TR" for row in table))
        self.assertTrue(any(row["diagnostic"] == "HitConditionedIoU" for row in table))


if __name__ == "__main__":
    unittest.main()
