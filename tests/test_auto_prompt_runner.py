import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from irsam2_benchmark.benchmark import auto_prompt_runner


class FakeProgress:
    def __init__(self):
        self.postfixes = []
        self.updates = 0

    def set_description_str(self, value):
        raise AssertionError(f"eval progress should keep a fixed description, got: {value}")

    def set_postfix(self, **kwargs):
        self.postfixes.append(kwargs)

    def update(self, value):
        self.updates += value


def _load_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_4090x4_auto_prompt.py"
    spec = importlib.util.spec_from_file_location("run_4090x4_auto_prompt_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_sample_dataset(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:5] = 255
    Image.fromarray(image).save(images / "sample.png")
    Image.fromarray(mask).save(masks / "sample.png")


def _write_auto_prompt_config(path: Path, artifact_root: Path, *, write_samples: bool = True) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "server_auto_prompt_4090x4.example.yaml"
    payload = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    sam2_repo = path.parent / "sam2"
    checkpoint_root = path.parent / "checkpoints"
    sam2_repo.mkdir()
    checkpoint_root.mkdir()
    (checkpoint_root / "sam2.1_hiera_large.pt").write_text("checkpoint", encoding="utf-8")
    dataset_roots = {
        "nuaa_sirst": path.parent / "datasets" / "NUAA-SIRST",
        "nudt_sirst": path.parent / "datasets" / "NUDT-SIRST",
        "irstd_1k": path.parent / "datasets" / "IRSTD-1K",
        "multimodal": path.parent / "datasets" / "MultiModal",
        "rbgt_tiny_ir_box": path.parent / "datasets" / "RBGT-Tiny",
    }
    for dataset_root in dataset_roots.values():
        dataset_root.mkdir(parents=True)
    if write_samples:
        _write_sample_dataset(dataset_roots["nuaa_sirst"])
    payload["auto_prompt"]["train_datasets"] = ["nuaa_sirst"]
    payload["suites"]["auto_prompt"]["datasets"] = ["nuaa_sirst"]
    payload["paths"] = {
        "sam2": {"repo": str(sam2_repo), "checkpoint_root": str(checkpoint_root)},
        "artifacts": {"root": str(artifact_root)},
        "reference_results": {"root": str(path.parent / "reference_results")},
        "datasets": {key: str(value) for key, value in dataset_roots.items()},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class AutoPromptRunnerTests(unittest.TestCase):
    def test_dry_run_generates_training_config_and_learned_eval_configs(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_auto_prompt_4090x4.local.yaml"
            _write_auto_prompt_config(config_path, root / "artifacts")
            output = io.StringIO()
            progress_output = io.StringIO()

            with redirect_stdout(output), redirect_stderr(progress_output):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
                            "--suites",
                            "auto_prompt",
                            "--checkpoints",
                            "large",
                            "--modes",
                            "learned_box",
                            "--dry-run",
                            "--no-analysis",
                            "--python-bin",
                            "python",
                            "--train-batch-size",
                            "16",
                            "--train-num-workers",
                            "4",
                            "--train-prefetch-factor",
                            "4",
                            "--train-shuffle-buffer-size",
                            "512",
                            "--train-amp",
                            "--train-profile-interval",
                            "200",
                        ]
                    ),
                    0,
                )

            text = output.getvalue()
            progress_text = progress_output.getvalue()
            self.assertIn("[train] dry_run CUDA_VISIBLE_DEVICES=0", text)
            self.assertIn("gpu=1", text)
            self.assertNotIn("dataset=nuaa_sirst", progress_text)
            manifest_path = root / "artifacts" / "sam2_ir_qd_m1_auto_prompt" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_count"], 1)
            self.assertEqual(manifest["failed_count"], 0)
            self.assertEqual(manifest["train"]["status"], "dry_run")
            preflight_path = Path(manifest["dataset_preflight"]["path"])
            self.assertTrue(preflight_path.exists())
            self.assertTrue(manifest["dataset_preflight"]["summary"]["overall"]["valid"])
            self.assertEqual(manifest["dataset_preflight"]["summary"]["mode"], "fast")
            self.assertEqual(manifest["dataset_preflight"]["summary"]["train"]["dataset_count"], 1)
            self.assertEqual(manifest["dataset_preflight"]["summary"]["eval"]["dataset_count"], 1)
            train_report = manifest["dataset_preflight"]["summary"]["train"]["reports"][0]
            self.assertEqual(train_report["sample_limit"], 256)
            self.assertEqual(train_report["image_limit"], 256)
            self.assertTrue(train_report["is_limited"])

            train_config = yaml.safe_load(Path(manifest["train"]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(train_config["dataset_configs"]), 1)
            self.assertTrue(train_config["train"]["show_progress"])
            self.assertEqual(train_config["train"]["progress_backend"], "tqdm")
            self.assertEqual(train_config["train"]["batch_size"], 16)
            self.assertEqual(train_config["train"]["num_workers"], 4)
            self.assertEqual(train_config["train"]["prefetch_factor"], 4)
            self.assertEqual(train_config["train"]["shuffle_buffer_size"], 512)
            self.assertTrue(train_config["train"]["use_amp"])
            self.assertEqual(train_config["train"]["profile_interval_batches"], 200)
            first_run_config = yaml.safe_load(Path(manifest["records"][0]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(first_run_config["method"]["prompt_checkpoint"], manifest["train"]["checkpoint_path"])
            self.assertEqual(first_run_config["method"]["prompt_top_k"], 5)
            self.assertTrue(first_run_config["method"]["heatmaps"]["enabled"])
            self.assertFalse(first_run_config["runtime"]["show_progress"])
            self.assertEqual(first_run_config["runtime"]["progress_backend"], "none")

    def test_dataset_preflight_failure_stops_runner_before_train(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_auto_prompt_4090x4.local.yaml"
            _write_auto_prompt_config(config_path, root / "artifacts", write_samples=False)
            output = io.StringIO()
            progress_output = io.StringIO()

            with redirect_stdout(output), redirect_stderr(progress_output):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
                            "--suites",
                            "auto_prompt",
                            "--checkpoints",
                            "large",
                            "--modes",
                            "learned_box",
                            "--dry-run",
                            "--no-analysis",
                            "--python-bin",
                            "python",
                        ]
                    ),
                    1,
                )

            text = output.getvalue()
            self.assertIn("[preflight] failed", text)
            self.assertNotIn("[train] dry_run", text)
            manifest_path = root / "artifacts" / "sam2_ir_qd_m1_auto_prompt" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["dataset_preflight"]["summary"]["overall"]["valid"])
            self.assertEqual(manifest["failures"][0]["status"], "dataset_preflight_failed")

    def test_preflight_mode_off_skips_dataset_validation(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_auto_prompt_4090x4.local.yaml"
            _write_auto_prompt_config(config_path, root / "artifacts", write_samples=False)
            output = io.StringIO()
            progress_output = io.StringIO()

            with redirect_stdout(output), redirect_stderr(progress_output):
                self.assertEqual(
                    runner.main(
                        [
                            "--config",
                            str(config_path),
                            "--suites",
                            "auto_prompt",
                            "--checkpoints",
                            "large",
                            "--modes",
                            "learned_box",
                            "--dry-run",
                            "--no-analysis",
                            "--python-bin",
                            "python",
                            "--preflight-mode",
                            "off",
                        ]
                    ),
                    0,
                )

            text = output.getvalue()
            self.assertIn("[preflight] skip", text)
            self.assertIn("[train] dry_run", text)
            manifest_path = root / "artifacts" / "sam2_ir_qd_m1_auto_prompt" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = manifest["dataset_preflight"]["summary"]
            self.assertEqual(summary["mode"], "off")
            self.assertTrue(summary["overall"]["valid"])
            self.assertTrue(summary["train"]["reports"][0]["skipped"])
            self.assertFalse(summary["train"]["reports"][0]["is_limited"])

    def test_fast_preflight_limits_rbgt_tiny_more_aggressively(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_auto_prompt_4090x4.local.yaml"
            _write_auto_prompt_config(config_path, root / "artifacts", write_samples=False)
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["auto_prompt"]["train_datasets"] = ["rbgt_tiny_ir_box"]
            payload["suites"]["auto_prompt"]["datasets"] = ["rbgt_tiny_ir_box"]
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            preflight_limits = []

            def fake_preflight(config):
                preflight_limits.append((config.dataset.dataset_id, config.runtime.max_samples, config.runtime.max_images))
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "warning_count": 0,
                    "size_mismatch_warning_count": 0,
                    "warning_examples": [],
                    "sample_count": config.runtime.max_samples,
                    "image_count": config.runtime.max_images,
                }

            with patch.object(auto_prompt_runner, "preflight_dataset", side_effect=fake_preflight):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        runner.main(
                            [
                                "--config",
                                str(config_path),
                                "--suites",
                                "auto_prompt",
                                "--checkpoints",
                                "large",
                                "--modes",
                                "heuristic_box",
                                "--dry-run",
                                "--no-analysis",
                                "--python-bin",
                                "python",
                            ]
                        ),
                        0,
                    )

            self.assertTrue(preflight_limits)
            self.assertTrue(all(item == ("RBGT-Tiny", 64, 64) for item in preflight_limits))
            manifest_path = root / "artifacts" / "sam2_ir_qd_m1_auto_prompt" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_report = manifest["dataset_preflight"]["summary"]["train"]["reports"][0]
            self.assertEqual(train_report["sample_limit"], 64)
            self.assertEqual(train_report["image_limit"], 64)
            self.assertTrue(train_report["is_limited"])

    def test_full_preflight_does_not_limit_rbgt_tiny(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "server_auto_prompt_4090x4.local.yaml"
            _write_auto_prompt_config(config_path, root / "artifacts", write_samples=False)
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["auto_prompt"]["train_datasets"] = ["rbgt_tiny_ir_box"]
            payload["suites"]["auto_prompt"]["datasets"] = ["rbgt_tiny_ir_box"]
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            preflight_limits = []

            def fake_preflight(config):
                preflight_limits.append((config.dataset.dataset_id, config.runtime.max_samples, config.runtime.max_images))
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "warning_count": 0,
                    "size_mismatch_warning_count": 0,
                    "warning_examples": [],
                    "sample_count": 1,
                    "image_count": 1,
                }

            with patch.object(auto_prompt_runner, "preflight_dataset", side_effect=fake_preflight):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        runner.main(
                            [
                                "--config",
                                str(config_path),
                                "--suites",
                                "auto_prompt",
                                "--checkpoints",
                                "large",
                                "--modes",
                                "heuristic_box",
                                "--dry-run",
                                "--no-analysis",
                                "--python-bin",
                                "python",
                                "--preflight-mode",
                                "full",
                            ]
                        ),
                        0,
                    )

            self.assertTrue(preflight_limits)
            self.assertTrue(all(item == ("RBGT-Tiny", 0, 0) for item in preflight_limits))
            manifest_path = root / "artifacts" / "sam2_ir_qd_m1_auto_prompt" / "benchmark_manifest_latest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_report = manifest["dataset_preflight"]["summary"]["train"]["reports"][0]
            self.assertEqual(train_report["sample_limit"], 0)
            self.assertEqual(train_report["image_limit"], 0)
            self.assertFalse(train_report["is_limited"])

    def test_eval_progress_keeps_fixed_description_and_reports_queue(self):
        progress = FakeProgress()
        counts = {"completed": 1, "skipped": 0, "failed": 0, "dry_run": 0}

        auto_prompt_runner._set_eval_progress(progress, counts=counts, active=4, queued=5)
        auto_prompt_runner._advance_eval_progress(progress, status="completed", counts=counts, active=3, queued=4)

        self.assertEqual(progress.updates, 1)
        self.assertEqual(progress.postfixes[0]["active"], 4)
        self.assertEqual(progress.postfixes[0]["queued"], 5)
        self.assertEqual(progress.postfixes[-1]["completed"], 2)
        self.assertEqual(progress.postfixes[-1]["active"], 3)
        self.assertEqual(progress.postfixes[-1]["queued"], 4)

    def test_run_logged_streams_output_and_writes_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "stream.log"
            command = [sys.executable, "-c", "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr)"]
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = auto_prompt_runner._run_logged(command, env=os.environ.copy(), log_path=log_path, stream_logs=True)

            self.assertEqual(result.returncode, 0)
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("stdout-ok", log_text)
            self.assertIn("stderr-ok", log_text)
            self.assertIn("stdout-ok", stderr.getvalue())
            self.assertIn("stderr-ok", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
