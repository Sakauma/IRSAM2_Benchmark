import json
import tempfile
import unittest
from pathlib import Path

from irsam2_benchmark.config import load_app_config


class ConfigExtensionTests(unittest.TestCase):
    def test_method_modules_and_modality_are_optional(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                        "dataset": {
                            "dataset_id": "dummy",
                            "adapter": "generic_image_mask",
                            "root": ".",
                            "images_dir": "images",
                            "masks_dir": "masks",
                        },
                        "runtime": {
                            "artifact_root": "artifacts",
                            "reference_results_root": "reference_results",
                            "output_name": "out",
                            "device": "cpu",
                        },
                        "evaluation": {
                            "benchmark_version": "v1",
                            "track": "track_a_image_prompted",
                            "protocol": "mask",
                            "inference_mode": "box",
                            "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            self.assertEqual(config.dataset.modality, "ir")
            self.assertEqual(config.method, {})
            self.assertEqual(config.modules, {})

    def test_method_modules_and_modality_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                        "dataset": {
                            "dataset_id": "dummy",
                            "adapter": "generic_image_mask",
                            "root": ".",
                            "modality": "ir",
                            "images_dir": "images",
                            "masks_dir": "masks",
                        },
                        "runtime": {
                            "artifact_root": "artifacts",
                            "reference_results_root": "reference_results",
                            "output_name": "out",
                            "device": "cpu",
                        },
                        "evaluation": {
                            "benchmark_version": "v1",
                            "track": "track_a_image_prompted",
                            "protocol": "mask",
                            "inference_mode": "box",
                            "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1},
                        },
                        "method": {"name": "sam2_physics_auto_prompt", "modality": "ir"},
                        "modules": {"prior": {"name": "prior_fusion", "enabled": ["local_contrast"]}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            self.assertEqual(config.method["name"], "sam2_physics_auto_prompt")
            self.assertEqual(config.modules["prior"]["enabled"], ["local_contrast"])


if __name__ == "__main__":
    unittest.main()

