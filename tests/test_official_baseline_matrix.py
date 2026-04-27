import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


def _load_matrix_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_official_baseline_matrix.py"
    spec = importlib.util.spec_from_file_location("run_official_baseline_matrix_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _complete_config(root: Path, artifact_root: Path, ckpt_path: Path) -> dict:
    return {
        "benchmark": {"suite_name": "test", "artifact_subdir": "paper_5090"},
        "paths": {
            "sam2": {"repo": str(root / "sam2"), "checkpoint_root": str(ckpt_path.parent)},
            "artifacts": {"root": str(artifact_root)},
            "reference_results": {"root": str(root / "reference_results")},
            "datasets": {"dummy": "/datasets/dummy"},
        },
        "runtime_defaults": {
            "artifact_root": "artifacts",
            "reference_results_root": "reference_results",
            "output_name": "out",
            "device": "cpu",
            "num_workers": 0,
            "smoke_test": True,
            "max_samples": 0,
            "max_images": 0,
            "save_visuals": True,
            "visual_limit": 24,
            "update_reference_results": False,
            "seeds": [42],
        },
        "runtime": {"device": "cpu"},
        "evaluation_defaults": {
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
        "stage_defaults": {},
        "datasets": {
            "dummy": {
                "config": {
                    "dataset_id": "dummy_dataset",
                    "adapter": "generic_image_mask",
                    "root": ".",
                    "images_dir": "images",
                    "masks_dir": "masks",
                    "mask_mode": "binary",
                    "class_map": {},
                }
            }
        },
        "methods": {
            "sam2_box_oracle": {
                "baseline": "sam2_pretrained_box_prompt",
                "method": {"name": "sam2_box_oracle", "family": "sam2_oracle_prompt", "modality": "ir"},
                "modules": {},
                "evaluation": {
                    "track": "track_a_image_prompted",
                    "inference_mode": "box",
                    "prompt_policy": {
                        "name": "box",
                        "prompt_type": "box",
                        "prompt_source": "gt",
                        "prompt_budget": 1,
                        "multi_mask": False,
                    },
                },
            }
        },
        "models": [
            {
                "alias": "tiny",
                "model_id": "sam2.1_hiera_tiny",
                "family": "sam2",
                "cfg": "cfg.yaml",
                "ckpt": ckpt_path.name,
            }
        ],
        "modes": [{"method": "sam2_box_oracle", "alias": "box"}],
        "suites": {
            "mask": {
                "experiment_id": "T",
                "modes": ["sam2_box_oracle"],
                "datasets": ["dummy"],
                "run_analysis": False,
            }
        },
        "official_matrix": {
            "artifact_subdir": "official_baseline_matrix",
            "datasets": ["dummy"],
            "models": ["tiny"],
            "methods": ["sam2_box_oracle"],
            "seeds": [42],
            "visual_limit": 24,
            "resume": True,
        },
    }


def _write_config(path: Path, root: Path, artifact_root: Path, ckpt_path: Path) -> None:
    path.write_text(yaml.safe_dump(_complete_config(root, artifact_root, ckpt_path), sort_keys=False), encoding="utf-8")


def _fixtures():
    model = {"alias": "tiny", "model_id": "sam2.1_hiera_tiny"}
    baseline = {"alias": "box"}
    dataset = {"alias": "dummy"}
    return model, baseline, dataset


def _write_completed_run(matrix, output_dir: Path) -> None:
    matrix._write_json(output_dir / "summary.json", {"mean": {"mIoU": 0.75}, "std": {"mIoU": 0.0}})
    matrix._write_json(output_dir / "results.json", [])
    matrix._write_json(output_dir / "eval_reports" / "rows.json", [{"sample_id": "s0"}])
    matrix._write_json(output_dir / "run_metadata.json", {"status": "completed"})


class OfficialBaselineMatrixTests(unittest.TestCase):
    def test_resume_skips_completed_run_and_rebuilds_summary(self):
        matrix = _load_matrix_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts"
            ckpt_path = root / "checkpoints" / "ckpt.pt"
            ckpt_path.parent.mkdir(parents=True)
            ckpt_path.write_bytes(b"checkpoint")
            config_path = root / "server_benchmark_full.local.yaml"
            _write_config(config_path, root, artifact_root, ckpt_path)
            model, baseline, dataset = _fixtures()
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)
            _write_completed_run(matrix, output_dir)

            with patch.object(matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}), patch.dict(
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
                self.assertEqual(matrix.main(["--config", str(config_path)]), 0)

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
            ckpt_path = root / "checkpoints" / "ckpt.pt"
            ckpt_path.parent.mkdir(parents=True)
            ckpt_path.write_bytes(b"checkpoint")
            config_path = root / "server_benchmark_full.local.yaml"
            _write_config(config_path, root, artifact_root, ckpt_path)
            model, baseline, dataset = _fixtures()
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)

            def fake_run(*args, **kwargs):
                matrix._write_json(output_dir / "summary.json", {"mean": {"mIoU": 0.5}, "std": {"mIoU": 0.0}})
                matrix._write_json(output_dir / "results.json", [])
                matrix._write_json(output_dir / "eval_reports" / "rows.json", [{"sample_id": "s0"}])
                return subprocess.CompletedProcess(args=args[0], returncode=0)

            with patch.object(matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}), patch.dict(
                os.environ,
                {
                    "ARTIFACT_ROOT": str(artifact_root),
                    "MATRIX_RESUME": "0",
                    "MATRIX_MODELS": "tiny",
                    "MATRIX_DATASETS": "dummy",
                    "MATRIX_MODES": "box",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                clear=False,
            ), patch.object(matrix.subprocess, "run", side_effect=fake_run):
                self.assertEqual(matrix.main(["--config", str(config_path)]), 0)

            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["dataset_root"], "/datasets/dummy")
            self.assertEqual(metadata["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(metadata["config"]["runtime"]["update_reference_results"], False)
            self.assertEqual(metadata["checkpoint"]["path"], str(ckpt_path))
            self.assertEqual(len(metadata["checkpoint"]["sha256"]), 64)

    def test_failed_run_writes_failure_list_and_failed_metadata(self):
        matrix = _load_matrix_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_root = root / "artifacts"
            ckpt_path = root / "checkpoints" / "ckpt.pt"
            ckpt_path.parent.mkdir(parents=True)
            ckpt_path.write_bytes(b"checkpoint")
            config_path = root / "server_benchmark_full.local.yaml"
            _write_config(config_path, root, artifact_root, ckpt_path)
            model, baseline, dataset = _fixtures()
            output_dir = artifact_root / matrix._run_output_name(dataset, model, baseline)

            with patch.object(matrix, "_collect_common_metadata", return_value={"git": {"commit": "abc", "dirty": False}}), patch.dict(
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
                self.assertEqual(matrix.main(["--config", str(config_path)]), 1)

            failures = json.loads((artifact_root / "official_baseline_matrix" / "matrix_failures.json").read_text(encoding="utf-8"))
            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["returncode"], 7)
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["failure"]["returncode"], 7)


if __name__ == "__main__":
    unittest.main()
