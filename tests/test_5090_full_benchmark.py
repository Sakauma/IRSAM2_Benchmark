import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import yaml


def _load_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_5090_full_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_5090_full_benchmark_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_full_config(path: Path, artifact_root: Path) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "server_benchmark_full.example.yaml"
    payload = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    sam2_repo = path.parent / "sam2"
    checkpoint_root = path.parent / "checkpoints"
    sam2_repo.mkdir()
    checkpoint_root.mkdir()
    for ckpt_name in (
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt",
    ):
        (checkpoint_root / ckpt_name).write_text("checkpoint", encoding="utf-8")
    dataset_roots = {
        "nuaa_sirst": path.parent / "datasets" / "NUAA-SIRST",
        "nudt_sirst": path.parent / "datasets" / "NUDT-SIRST",
        "irstd_1k": path.parent / "datasets" / "IRSTD-1K",
        "multimodal": path.parent / "datasets" / "MultiModal",
        "rbgt_tiny_ir_box": path.parent / "datasets" / "RBGT-Tiny",
    }
    for dataset_root in dataset_roots.values():
        dataset_root.mkdir(parents=True)
    payload["paths"] = {
        "sam2": {"repo": str(sam2_repo), "checkpoint_root": str(checkpoint_root)},
        "artifacts": {"root": str(artifact_root)},
        "reference_results": {"root": str(path.parent / "reference_results")},
        "datasets": {key: str(value) for key, value in dataset_roots.items()},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_rbgt_voc_safe_config(path: Path, artifact_root: Path) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "server_benchmark_4090x3_rbgt_voc_safe.example.yaml"
    payload = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    sam2_repo = path.parent / "sam2"
    checkpoint_root = path.parent / "checkpoints"
    rbgt_root = path.parent / "datasets" / "RBGT-Tiny"
    sam2_repo.mkdir()
    checkpoint_root.mkdir()
    rbgt_root.mkdir(parents=True)
    (checkpoint_root / "sam2.1_hiera_large.pt").write_text("checkpoint", encoding="utf-8")
    payload["paths"] = {
        "sam2": {"repo": str(sam2_repo), "checkpoint_root": str(checkpoint_root)},
        "artifacts": {"root": str(artifact_root)},
        "reference_results": {"root": str(path.parent / "reference_results")},
        "datasets": {"rbgt_tiny_ir_box": str(rbgt_root)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FakeRunProgress:
    def __init__(self):
        self.descriptions = []
        self.postfixes = []
        self.updates = 0
        self.closed = False
        self.desc = "runs"

    def set_description_str(self, text):
        self.desc = text
        self.descriptions.append(text)

    def set_postfix(self, **kwargs):
        self.postfixes.append(dict(kwargs))

    def update(self, n):
        self.updates += n

    def close(self):
        self.closed = True


class Full5090BenchmarkTests(unittest.TestCase):
    def test_run_subprocess_streams_output_and_writes_log(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.log"
            stderr = io.StringIO()
            command = [sys.executable, "-c", "import sys; print('stdout line'); sys.stderr.write('stderr line\\n')"]

            with redirect_stderr(stderr):
                result = runner._full_runner._run_subprocess(command, os.environ.copy(), log_path, stream_logs=True)

            self.assertEqual(result.returncode, 0)
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("stdout line", log_text)
            self.assertIn("stderr line", log_text)
            self.assertIn("stdout line", stderr.getvalue())
            self.assertIn("stderr line", stderr.getvalue())

    def test_dry_run_generates_single_checkpoint_mask_subset(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")
            output = io.StringIO()
            with patch.object(runner._full_runner, "_make_run_progress", return_value=None), redirect_stdout(output):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
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
            run_id = manifest["run_id"]
            self.assertEqual(manifest["run_count"], 4)
            self.assertEqual(manifest["failed_count"], 0)
            self.assertTrue((root / "artifacts" / "paper_5090" / f"benchmark_manifest_{run_id}.json").exists())
            run_manifest = json.loads((root / "artifacts" / "paper_5090" / f"run_manifest_{run_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["run_id"], run_id)
            first_config = Path(manifest["records"][0]["config_path"])
            config = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["repo"], str(root / "sam2"))
            self.assertEqual(config["model"]["ckpt"], str(root / "checkpoints" / "sam2.1_hiera_tiny.pt"))
            self.assertEqual(config["runtime"]["seeds"], [42])
            self.assertEqual(config["runtime"]["image_batch_size"], 1)
            self.assertEqual(config["runtime"]["auto_mask_points_per_batch"], 256)
            self.assertTrue(config["runtime"]["show_progress"])
            self.assertEqual(config["runtime"]["progress_backend"], "tqdm")
            self.assertEqual(config["runtime"]["progress_position"], 1)
            self.assertEqual(config["runtime"]["reference_results_root"], str(root / "reference_results"))
            self.assertIn("/paper_5090/runs/mask/tiny", config["runtime"]["artifact_root"])
            self.assertIn("/paper_5090/logs/mask/tiny", manifest["records"][0]["log_path"])
            self.assertIn("config_sha256", manifest["records"][0])
            self.assertEqual(config["fingerprints"]["source_config"], str(config_path.resolve()))
            self.assertEqual(config["fingerprints"]["source_config_sha256"], manifest["config_sha256"])
            self.assertEqual(manifest["config_mode"], "complete")
            self.assertEqual(manifest["config"], str(config_path.resolve()))

    def test_dry_run_advances_outer_run_progress(self):
        runner = _load_runner()
        progress = FakeRunProgress()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")

            with patch.object(runner._full_runner, "_make_run_progress", return_value=progress):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
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

        self.assertEqual(progress.updates, 4)
        self.assertTrue(progress.closed)
        self.assertTrue(any("nuaa_sirst" in item for item in progress.descriptions))
        self.assertEqual(progress.postfixes[-1]["dry_run"], 4)

    def test_no_progress_disables_generated_single_run_progress(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")
            self.assertEqual(
                runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--suites",
                        "mask",
                        "--checkpoints",
                        "tiny",
                        "--modes",
                        "box",
                        "--dry-run",
                        "--no-progress",
                        "--python-bin",
                        "python",
                    ]
                ),
                0,
            )

            manifest = json.loads((root / "artifacts" / "paper_5090" / "benchmark_manifest_latest.json").read_text(encoding="utf-8"))
            first_config = Path(manifest["records"][0]["config_path"])
            config = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            self.assertFalse(config["runtime"]["show_progress"])
            self.assertEqual(config["runtime"]["progress_backend"], "none")
            self.assertEqual(config["runtime"]["progress_position"], 1)

    def test_rbgt_voc_safe_example_generates_single_large_run(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_4090x3_rbgt_voc_safe.local.yaml"
            _write_rbgt_voc_safe_config(config_path, root / "artifacts")
            self.assertEqual(
                runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--dry-run",
                        "--no-analysis",
                        "--no-progress",
                        "--python-bin",
                        "python",
                    ]
                ),
                0,
            )

            manifest_path = root / "artifacts" / "paper_4090x3_rbgt_voc_safe" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_count"], 1)
            config = yaml.safe_load(Path(manifest["records"][0]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["model_id"], "sam2.1_hiera_large")
            self.assertEqual(config["dataset"]["images_dir"], "images")
            self.assertEqual(config["dataset"]["annotations_dir"], "annotations_voc")
            self.assertEqual(config["dataset"]["mask_mode"], "bbox")
            self.assertEqual(config["runtime"]["image_batch_size"], 1)
            self.assertTrue(config["runtime"]["reuse_image_embedding"])

    def test_smoke_test_writes_to_separate_artifact_subdir(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")
            self.assertEqual(
                runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--suites",
                        "mask",
                        "--checkpoints",
                        "tiny",
                        "--modes",
                            "box",
                            "--dry-run",
                            "--smoke-test",
                            "--no-progress",
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

    def test_suite_runtime_overrides_checkpoint_runtime(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            for model in payload["models"]:
                if model["alias"] == "large":
                    model["runtime"]["image_batch_size"] = 8
                    model["runtime"]["auto_mask_points_per_batch"] = 128
            payload["suites"]["rbgt_box"]["runtime"] = {
                "image_batch_size": 1,
                "auto_mask_points_per_batch": 64,
                "save_visuals": False,
                "visual_limit": 0,
            }
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

            self.assertEqual(
                runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--suites",
                        "rbgt_box",
                        "--checkpoints",
                        "large",
                        "--modes",
                        "box",
                        "--dry-run",
                        "--no-analysis",
                        "--no-progress",
                    ]
                ),
                0,
            )

            manifest_path = root / "artifacts" / "paper_5090" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = yaml.safe_load(Path(manifest["records"][0]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["runtime"]["image_batch_size"], 1)
            self.assertEqual(config["runtime"]["auto_mask_points_per_batch"], 64)
            self.assertFalse(config["runtime"]["save_visuals"])
            self.assertEqual(config["runtime"]["visual_limit"], 0)

    def test_checkpoint_summary_reads_completed_runs(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "paper_5090" / "runs" / "mask" / "tiny" / "T" / "d" / "m"
            run_dir.mkdir(parents=True)
            (run_dir / "benchmark_spec.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
            (run_dir / "results.json").write_text("[]", encoding="utf-8")
            (run_dir / "eval_reports").mkdir()
            (run_dir / "eval_reports" / "rows.json").write_text(json.dumps([{"sample_id": "s0", "mIoU": 0.5}]), encoding="utf-8")
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

    def test_run_is_complete_rejects_empty_metric_artifacts(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "benchmark_spec.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
            (run_dir / "results.json").write_text("[{\"seed\": 42}]", encoding="utf-8")
            (run_dir / "summary.json").write_text(json.dumps({"mean": {}, "std": {}}), encoding="utf-8")
            (run_dir / "eval_reports").mkdir()
            (run_dir / "eval_reports" / "rows.json").write_text("[]", encoding="utf-8")

            self.assertFalse(runner._run_is_complete(run_dir))

            (run_dir / "summary.json").write_text(json.dumps({"mean": {"mIoU": 0.5}, "std": {"mIoU": 0.0}}), encoding="utf-8")
            self.assertFalse(runner._run_is_complete(run_dir))

            (run_dir / "eval_reports" / "rows.json").write_text(json.dumps([{"sample_id": "s0", "mIoU": 0.5}]), encoding="utf-8")
            self.assertTrue(runner._run_is_complete(run_dir))

            (run_dir / "summary.json").write_text(
                json.dumps({"mean": {"mIoU": 0.5}, "std": {"mIoU": 0.0}, "failure_rate": 0.5, "failure_rate_threshold": 0.05}),
                encoding="utf-8",
            )
            self.assertFalse(runner._run_is_complete(run_dir))

    def test_config_validation_reports_missing_dataset_root(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["paths"]["datasets"]["nuaa_sirst"] = str(root / "missing" / "NUAA-SIRST")
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Dataset root does not exist for nuaa_sirst"):
                runner.main(["--config", str(config_path), "--suites", "mask", "--checkpoints", "tiny", "--modes", "box", "--dry-run"])

    def test_failed_run_records_log_path_and_tail(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")

            def fake_run_subprocess(command, env, log_path, *, stream_logs=False):
                del command, env, stream_logs
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("first line\nlast failure line\n", encoding="utf-8")
                return CompletedProcess(args=[], returncode=7)

            with patch.object(runner._full_runner, "_run_subprocess", side_effect=fake_run_subprocess):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
                            "--suites",
                            "mask",
                            "--checkpoints",
                            "tiny",
                            "--modes",
                            "box",
                            "--no-analysis",
                            "--stop-on-error",
                            "--no-progress",
                        ]
                    ),
                    1,
                )

            failures = json.loads((root / "artifacts" / "paper_5090" / "run_failures_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["log_path"].endswith("logs/mask/tiny/nuaa_sirst_sam2_box_oracle.log"))
            self.assertIn("last failure line", failures[0]["log_tail"])

    def test_successful_process_with_invalid_artifacts_is_failure(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_benchmark_full.local.yaml"
            _write_full_config(config_path, root / "artifacts")

            def fake_run_subprocess(command, env, log_path, *, stream_logs=False):
                del command, env, stream_logs
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("process returned zero\n", encoding="utf-8")
                return CompletedProcess(args=[], returncode=0)

            with patch.object(runner._full_runner, "_run_subprocess", side_effect=fake_run_subprocess):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
                            "--suites",
                            "mask",
                            "--checkpoints",
                            "tiny",
                            "--modes",
                            "box",
                            "--no-analysis",
                            "--stop-on-error",
                            "--no-progress",
                        ]
                    ),
                    1,
                )

            failures = json.loads((root / "artifacts" / "paper_5090" / "run_failures_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(failures[0]["status"], "failed_invalid_artifacts")
            self.assertEqual(failures[0]["returncode"], 0)
            self.assertTrue(any("Missing required artifact file" in error for error in failures[0]["validation_errors"]))


if __name__ == "__main__":
    unittest.main()
