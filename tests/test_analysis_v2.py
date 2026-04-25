import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from irsam2_benchmark.analysis.collector import collect_runs
from irsam2_benchmark.analysis.runner import run_analysis
from irsam2_benchmark.analysis.stats import run_paired_tests
from irsam2_benchmark.analysis.tables import main_baseline_table


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root: Path, experiment: str, dataset: str, method: str, metric_shift: float) -> None:
    run_dir = root / experiment / dataset / method
    rows = [
        {
            "sample_id": "sample_a",
            "seed": 42,
            "mIoU": 0.2 + metric_shift,
            "Dice": 0.3 + metric_shift,
            "TargetRecallIoU25": 1.0 if metric_shift > 0 else 0.0,
            "FalseAlarmPixelsPerMP": 20.0 - metric_shift,
            "LatencyMs": 10.0 + metric_shift,
            "GTAreaPixels": 4.0,
            "PredAreaPixels": 4.0 + metric_shift,
            "target_scale": "small",
            "annotation_protocol_flag": "mask",
        },
        {
            "sample_id": "sample_b",
            "seed": 42,
            "mIoU": 0.4 + metric_shift,
            "Dice": 0.5 + metric_shift,
            "TargetRecallIoU25": 1.0,
            "FalseAlarmPixelsPerMP": 30.0 - metric_shift,
            "LatencyMs": 11.0 + metric_shift,
            "GTAreaPixels": 8.0,
            "PredAreaPixels": 8.0 + metric_shift,
            "target_scale": "small",
            "annotation_protocol_flag": "mask",
        },
    ]
    _write_json(run_dir / "benchmark_spec.json", {"benchmark_version": "test"})
    _write_json(run_dir / "summary.json", {"mean": {"mIoU": 0.3 + metric_shift}})
    _write_json(run_dir / "results.json", [{"seed": 42, "mIoU": 0.3 + metric_shift}])
    _write_json(run_dir / "eval_reports" / "rows.json", rows)


def _write_matrix(path: Path) -> None:
    payload = {
        "methods": {
            "bbox_rect": {"baseline": "bbox_rect"},
            "sam2_box_oracle": {"baseline": "sam2_zero_shot"},
        },
        "experiments": [
            {
                "experiment_id": "T1_oracle_prompt_baselines",
                "status": "planned",
                "datasets": ["dummy_dataset"],
                "methods": ["bbox_rect", "sam2_box_oracle"],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_analysis(path: Path, artifact_root: Path, matrix_path: Path, output_dir: Path) -> None:
    payload = {
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "matrix": str(matrix_path),
        "experiment_groups": ["T1_oracle_prompt_baselines"],
        "primary_metric": "mIoU",
        "metrics": ["mIoU", "Dice", "TargetRecallIoU25", "FalseAlarmPixelsPerMP", "LatencyMs"],
        "case_selection": {"top_k": 1, "primary_metric": "mIoU"},
        "statistics": {
            "n_bootstrap": 100,
            "ci": 0.95,
            "random_seed": 7,
            "low_power_threshold": 20,
            "comparisons": [{"name": "sam2_vs_bbox", "baseline": "bbox_rect", "candidate": "sam2_box_oracle"}],
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class AnalysisV2Tests(unittest.TestCase):
    def test_collect_tables_stats_and_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts" / "paper_v1"
            output_dir = artifact_root / "analysis"
            matrix_path = root / "matrix.yaml"
            analysis_path = root / "analysis.yaml"
            _write_matrix(matrix_path)
            _write_analysis(analysis_path, artifact_root, matrix_path, output_dir)
            _write_run(artifact_root, "T1_oracle_prompt_baselines", "dummy_dataset", "bbox_rect", 0.0)
            _write_run(artifact_root, "T1_oracle_prompt_baselines", "dummy_dataset", "sam2_box_oracle", 0.1)

            manifest = run_analysis(analysis_path)

            self.assertEqual(manifest["run_count"], 2)
            self.assertTrue((output_dir / "tables" / "main_baseline_table.csv").exists())
            self.assertTrue((output_dir / "tables" / "significance_tests.json").exists())
            self.assertTrue((output_dir / "analysis-report.md").exists())
            self.assertTrue((output_dir / "stats-appendix.md").exists())
            stats_rows = json.loads((output_dir / "tables" / "significance_tests.json").read_text(encoding="utf-8"))
            ok_rows = [row for row in stats_rows if row.get("status") == "ok"]
            self.assertTrue(ok_rows)
            self.assertIn("wilcoxon_p_holm", ok_rows[0])

    def test_collector_records_missing_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_path = root / "matrix.yaml"
            _write_matrix(matrix_path)
            matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
            runs, missing = collect_runs(root / "missing_artifacts", matrix, {"experiment_groups": ["T1_oracle_prompt_baselines"]})
            self.assertEqual(runs, [])
            self.assertEqual(len(missing), 2)

    def test_table_builder_summarizes_dataset_method(self):
        rows = [
            {"dataset": "d", "method": "m", "sample_id": "a", "seed": 1, "mIoU": 0.2},
            {"dataset": "d", "method": "m", "sample_id": "b", "seed": 1, "mIoU": 0.4},
        ]
        table = main_baseline_table(rows, ["mIoU"])
        self.assertEqual(table[0]["sample_count"], 2)
        self.assertAlmostEqual(table[0]["mIoU_mean"], 0.3)

    def test_paired_stats_skips_mismatched_samples(self):
        rows = [
            {"dataset": "d", "method": "a", "sample_id": "one", "seed": 1, "mIoU": 0.1},
            {"dataset": "d", "method": "b", "sample_id": "two", "seed": 1, "mIoU": 0.2},
        ]
        result = run_paired_tests(
            rows,
            {
                "metrics": ["mIoU"],
                "statistics": {"comparisons": [{"baseline": "a", "candidate": "b"}], "n_bootstrap": 10},
            },
        )
        self.assertEqual(result[0]["status"], "skipped_no_pairs")

    def test_script_dry_run(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_paper_results.py"
        spec = importlib.util.spec_from_file_location("analyze_paper_results_under_test", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_path = root / "matrix.yaml"
            analysis_path = root / "analysis.yaml"
            _write_matrix(matrix_path)
            _write_analysis(analysis_path, root / "artifacts", matrix_path, root / "analysis_out")
            self.assertEqual(module.main(["--analysis", str(analysis_path), "--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()

