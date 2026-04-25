import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml


def _load_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_5090_full_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_5090_full_benchmark_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_paths(path: Path, artifact_root: Path) -> None:
    payload = {
        "sam2": {"repo": "/server/sam2", "checkpoint_root": "/server/checkpoints"},
        "artifacts": {"root": str(artifact_root)},
        "datasets": {
            "nuaa_sirst": "/data/NUAA-SIRST",
            "nudt_sirst": "/data/NUDT-SIRST",
            "irstd_1k": "/data/IRSTD-1K",
            "multimodal": "/data/MultiModal",
            "rbgt_tiny_ir_box": "/data/RBGT-Tiny",
        },
        "execution": {"cuda_visible_devices": "0"},
        "runtime": {"seeds": [42]},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class Full5090BenchmarkTests(unittest.TestCase):
    def test_dry_run_generates_single_checkpoint_mask_subset(self):
        runner = _load_runner()
        suite_config = Path(__file__).resolve().parents[1] / "configs" / "server_5090_full_benchmark.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = root / "paths.yaml"
            _write_paths(paths, root / "artifacts")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    runner.main(
                        [
                            "--paths",
                            str(paths),
                            "--suite-config",
                            str(suite_config),
                            "--suites",
                            "mask",
                            "--checkpoints",
                            "tiny",
                            "--modes",
                            "box",
                            "--dry-run",
                            "--python-bin",
                            "python",
                        ]
                    ),
                    0,
            )
            text = output.getvalue()
            self.assertIn("[plan] runs=4", text)
            self.assertIn("sam2_box_oracle", text)
            self.assertIn("sam2.1_hiera_tiny", text)

            manifest_path = root / "artifacts" / "paper_5090" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_count"], 4)
            self.assertEqual(manifest["failed_count"], 0)
            first_config = Path(manifest["records"][0]["config_path"])
            config = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["repo"], "/server/sam2")
            self.assertEqual(config["model"]["ckpt"], "/server/checkpoints/sam2.1_hiera_tiny.pt")
            self.assertEqual(config["runtime"]["seeds"], [42])
            self.assertEqual(config["runtime"]["reference_results_root"], "reference_results")
            self.assertIn("/paper_5090/runs/mask/tiny", config["runtime"]["artifact_root"])

    def test_smoke_test_writes_to_separate_artifact_subdir(self):
        runner = _load_runner()
        suite_config = Path(__file__).resolve().parents[1] / "configs" / "server_5090_full_benchmark.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = root / "paths.yaml"
            _write_paths(paths, root / "artifacts")
            self.assertEqual(
                runner.main(
                    [
                        "--paths",
                        str(paths),
                        "--suite-config",
                        str(suite_config),
                        "--suites",
                        "mask",
                        "--checkpoints",
                        "tiny",
                        "--modes",
                        "box",
                        "--dry-run",
                        "--smoke-test",
                    ]
                ),
                0,
            )
            manifest_path = root / "artifacts" / "paper_5090_smoke" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_config = Path(manifest["records"][0]["config_path"])
            config = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            self.assertEqual(config["runtime"]["max_samples"], 10)
            self.assertEqual(config["runtime"]["max_images"], 10)

    def test_checkpoint_summary_reads_completed_runs(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "paper_5090" / "runs" / "mask" / "tiny" / "T" / "d" / "m"
            run_dir.mkdir(parents=True)
            (run_dir / "benchmark_spec.json").write_text("{}", encoding="utf-8")
            (run_dir / "results.json").write_text("[]", encoding="utf-8")
            (run_dir / "eval_reports").mkdir()
            (run_dir / "eval_reports" / "rows.json").write_text("[]", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps({"dataset_manifest": {"sample_count": 3}, "mean": {"mIoU": 0.5}, "std": {"mIoU": 0.1}}),
                encoding="utf-8",
            )
            outputs = runner._write_checkpoint_summary(
                root / "paper_5090",
                [
                    {
                        "status": "completed",
                        "suite": "mask",
                        "checkpoint": "tiny",
                        "model_id": "sam2.1_hiera_tiny",
                        "dataset": "d",
                        "method": "m",
                        "output_dir": str(run_dir),
                    }
                ],
            )
            summary = json.loads(Path(outputs["checkpoint_sweep_summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["sample_count"], 3)
            self.assertAlmostEqual(summary[0]["mIoU_mean"], 0.5)
            self.assertAlmostEqual(summary[0]["mIoU_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
