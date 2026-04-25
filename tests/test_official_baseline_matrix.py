import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_matrix_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_official_baseline_matrix.py"
    spec = importlib.util.spec_from_file_location("run_official_baseline_matrix_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_config() -> dict:
    return {
        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
        "dataset": {
            "dataset_id": "dummy_dataset",
            "adapter": "generic_image_mask",
            "root": ".",
            "images_dir": "images",
            "masks_dir": "masks",
            "mask_mode": "binary",
            "class_map": {},
        },
        "runtime": {
            "artifact_root": "artifacts",
            "reference_results_root": "reference_results",
            "output_name": "out",
            "device": "cpu",
            "num_workers": 0,
            "smoke_test": True,
            "max_samples": 0,
            "max_images": 0,
            "save_visuals": True,
            "seeds": [42],
        },
        "evaluation": {
            "benchmark_version": "v1",
            "track": "track_a_image_prompted",
            "protocol": "mask_supervised",
            "inference_mode": "box",
            "prompt_policy": {
                "name": "p",
                "prompt_type": "box",
                "prompt_source": "gt",
                "prompt_budget": 1,
                "multi_mask": False,
            },
        },
        "stages": {},
        "ablations": {},
    }


def _fixtures(config_path: Path):
    model = {"alias": "tiny", "model_id": "sam2.1_hiera_tiny", "cfg": "cfg.yaml", "ckpt": "ckpt.pt"}
    baseline = {
        "name": "sam2_zero_shot",
        "alias": "box",
        "track": "track_a_image_prompted",
        "inference_mode": "box",
        "prompt_policy": {"name": "box", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1},
    }
    dataset = {
        "alias": "dummy",
        "config_path": config_path,
        "dataset_root_env": "DUMMY_DATASET_ROOT",
        "dataset_root_default": "/datasets/dummy",
    }
    return model, baseline, dataset


def _write_completed_run(matrix, output_dir: Path) -> None:
    matrix._write_json(output_dir / "summary.json", {"mean": {"mIoU": 0.75}, "std": {"mIoU": 0.0}})
    matrix._write_json(output_dir / "results.json", [])
    matrix._write_json(output_dir / "eval_reports" / "rows.json", [])
    matrix._write_json(output_dir / "run_metadata.json", {"status": "completed"})


class OfficialBaselineMatrixTests(unittest.TestCase):
    def test_resume_skips_completed_run_and_rebuilds_summary(self):
        matrix = _load_matrix_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts"
            config_path = root / "config.yaml"
            matrix._write_yaml(config_path, _base_config())
            model, baseline, dataset = _fixtures(config_path)
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)
            _write_completed_run(matrix, output_dir)

            with patch.object(matrix, "MODELS", [model]), patch.object(matrix, "BASELINES", [baseline]), patch.object(
                matrix, "DATASETS", [dataset]
            ), patch.object(matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}), patch.dict(
                os.environ,
                {
                    "ARTIFACT_ROOT": str(artifact_root),
                    "MATRIX_RESUME": "1",
                    "MATRIX_MODELS": "tiny",
                    "MATRIX_DATASETS": "dummy",
                    "MATRIX_MODES": "box",
                },
                clear=False,
            ), patch.object(matrix.subprocess, "run", side_effect=AssertionError("completed run should be skipped")):
                self.assertEqual(matrix.main(), 0)

            summary = json.loads((artifact_root / "official_baseline_matrix" / "matrix_summary.json").read_text(encoding="utf-8"))
            failures = json.loads((artifact_root / "official_baseline_matrix" / "matrix_failures.json").read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["status"], "skipped")
            self.assertEqual(summary[0]["mIoU"], 0.75)
            self.assertEqual(failures, [])

    def test_successful_run_writes_completed_metadata(self):
        matrix = _load_matrix_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts"
            config_path = root / "config.yaml"
            ckpt_path = root / "ckpt.pt"
            ckpt_path.write_bytes(b"checkpoint")
            matrix._write_yaml(config_path, _base_config())
            model, baseline, dataset = _fixtures(config_path)
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)

            def fake_run(*args, **kwargs):
                matrix._write_json(output_dir / "summary.json", {"mean": {"mIoU": 0.5}, "std": {"mIoU": 0.0}})
                matrix._write_json(output_dir / "results.json", [])
                matrix._write_json(output_dir / "eval_reports" / "rows.json", [])
                return subprocess.CompletedProcess(args=args[0], returncode=0)

            with patch.object(matrix, "MODELS", [model]), patch.object(matrix, "BASELINES", [baseline]), patch.object(
                matrix, "DATASETS", [dataset]
            ), patch.object(matrix, "_resolve_model_ckpt", return_value=str(ckpt_path)), patch.object(
                matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}
            ), patch.dict(
                os.environ,
                {
                    "ARTIFACT_ROOT": str(artifact_root),
                    "MATRIX_RESUME": "0",
                    "MATRIX_MODELS": "tiny",
                    "MATRIX_DATASETS": "dummy",
                    "MATRIX_MODES": "box",
                    "DUMMY_DATASET_ROOT": "/datasets/override",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                clear=False,
            ), patch.object(matrix.subprocess, "run", side_effect=fake_run):
                self.assertEqual(matrix.main(), 0)

            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["dataset_root"], "/datasets/override")
            self.assertEqual(metadata["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(metadata["config"]["runtime"]["update_reference_results"], False)
            self.assertEqual(metadata["checkpoint"]["path"], str(ckpt_path))
            self.assertEqual(len(metadata["checkpoint"]["sha256"]), 64)

    def test_failed_run_writes_failure_list_and_failed_metadata(self):
        matrix = _load_matrix_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts"
            config_path = root / "config.yaml"
            matrix._write_yaml(config_path, _base_config())
            model, baseline, dataset = _fixtures(config_path)
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)

            with patch.object(matrix, "MODELS", [model]), patch.object(matrix, "BASELINES", [baseline]), patch.object(
                matrix, "DATASETS", [dataset]
            ), patch.object(matrix, "_resolve_model_ckpt", return_value=str(root / "ckpt.pt")), patch.object(
                matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}
            ), patch.dict(
                os.environ,
                {
                    "ARTIFACT_ROOT": str(artifact_root),
                    "MATRIX_RESUME": "0",
                    "MATRIX_MODELS": "tiny",
                    "MATRIX_DATASETS": "dummy",
                    "MATRIX_MODES": "box",
                },
                clear=False,
            ), patch.object(matrix.subprocess, "run", side_effect=subprocess.CalledProcessError(7, ["python"])):
                self.assertEqual(matrix.main(), 1)

            failures = json.loads((artifact_root / "official_baseline_matrix" / "matrix_failures.json").read_text(encoding="utf-8"))
            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["returncode"], 7)
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["failure"]["returncode"], 7)


if __name__ == "__main__":
    unittest.main()
