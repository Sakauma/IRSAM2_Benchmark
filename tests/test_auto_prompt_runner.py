import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

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


def _write_auto_prompt_config(path: Path, artifact_root: Path) -> None:
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
            self.assertEqual(manifest["run_count"], 4)
            self.assertEqual(manifest["failed_count"], 0)
            self.assertEqual(manifest["train"]["status"], "dry_run")

            train_config = yaml.safe_load(Path(manifest["train"]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(train_config["dataset_configs"]), 4)
            self.assertTrue(train_config["train"]["show_progress"])
            self.assertEqual(train_config["train"]["progress_backend"], "tqdm")
            first_run_config = yaml.safe_load(Path(manifest["records"][0]["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(first_run_config["method"]["prompt_checkpoint"], manifest["train"]["checkpoint_path"])
            self.assertEqual(first_run_config["method"]["prompt_top_k"], 5)
            self.assertTrue(first_run_config["method"]["heatmaps"]["enabled"])
            self.assertFalse(first_run_config["runtime"]["show_progress"])
            self.assertEqual(first_run_config["runtime"]["progress_backend"], "none")

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
