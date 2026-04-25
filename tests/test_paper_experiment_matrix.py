import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml


def _load_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_experiments.py"
    spec = importlib.util.spec_from_file_location("run_paper_experiments_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PaperExperimentMatrixTests(unittest.TestCase):
    def test_dry_run_expands_p0_auto_prompt(self):
        runner = _load_runner()
        matrix_path = Path(__file__).resolve().parents[1] / "configs" / "paper_experiments_v1.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            paths_path = Path(temp_dir) / "paths.yaml"
            paths_path.write_text(
                yaml.safe_dump(
                    {
                        "sam2": {"repo": "/path/to/sam2", "checkpoint_root": "/path/to/sam2/checkpoints"},
                        "artifacts": {"root": str(Path(temp_dir) / "artifacts")},
                        "datasets": {"multimodal": "/path/to/MultiModal"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    runner.main(
                        [
                            "--matrix",
                            str(matrix_path),
                            "--paths",
                            str(paths_path),
                            "--group",
                            "p0_auto_prompt",
                            "--dry-run",
                            "--python-bin",
                            "python",
                        ]
                    ),
                    0,
                )
            text = output.getvalue()
            self.assertIn("sam2_no_prompt_auto_mask", text)
            self.assertIn("sam2_physics_auto_prompt", text)
            self.assertIn("/path/to/MultiModal", text)
            self.assertIn("paper_v1/generated/run_configs/p0_auto_prompt", text)
            self.assertNotIn("DATASET_ROOT=", text)
            self.assertTrue((Path(temp_dir) / "artifacts" / "paper_v1" / "generated" / "run_configs" / "p0_auto_prompt").exists())

    def test_paths_yaml_overrides_dataset_and_sam2_paths(self):
        runner = _load_runner()
        matrix_path = Path(__file__).resolve().parents[1] / "configs" / "paper_experiments_v1.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            paths_path = Path(temp_dir) / "paths.yaml"
            paths_path.write_text(
                yaml.safe_dump(
                    {
                        "sam2": {"repo": "/custom/sam2", "checkpoint_root": "/custom/checkpoints"},
                        "artifacts": {"root": "/custom/artifacts"},
                        "datasets": {"multimodal": "/custom/MultiModal"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            matrix = runner._load_yaml(matrix_path)
            paths = runner._load_yaml(paths_path)
            experiment = [item for item in matrix["experiments"] if item["experiment_id"] == "T2_ir_auto_prompt"][0]
            config = runner._build_app_config(matrix, experiment, "multimodal", "sam2_physics_auto_prompt", paths)
            self.assertEqual(config["dataset"]["root"], "/custom/MultiModal")
            self.assertEqual(config["model"]["repo"], "/custom/sam2")
            self.assertEqual(config["model"]["ckpt"], "/custom/checkpoints/sam2.1_hiera_base_plus.pt")
            self.assertEqual(config["runtime"]["artifact_root"], "/custom/artifacts")


if __name__ == "__main__":
    unittest.main()
