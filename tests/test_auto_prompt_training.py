import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from irsam2_benchmark.training import train_auto_prompt_from_config


def _write_dataset(root: Path) -> Path:
    data_root = root / "data"
    images = data_root / "images"
    masks = data_root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for idx in range(2):
        image = np.zeros((16, 16), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        image[6:8, 6 + idx : 8 + idx] = 255
        mask[6:8, 6 + idx : 8 + idx] = 255
        Image.fromarray(image).save(images / f"{idx}.png")
        Image.fromarray(mask).save(masks / f"{idx}.png")

    config_dir = root / "configs"
    config_dir.mkdir()
    dataset_config = config_dir / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "model": {"model_id": "dummy", "cfg": "dummy.yaml", "ckpt": "dummy.pt"},
                "dataset": {
                    "dataset_id": "synthetic",
                    "adapter": "generic_image_mask",
                    "root": "data",
                    "images_dir": "images",
                    "masks_dir": "masks",
                    "modality": "ir",
                    "mask_mode": "binary",
                },
                "runtime": {
                    "artifact_root": "artifacts",
                    "reference_results_root": "reference_results",
                    "output_name": "synthetic",
                    "device": "cpu",
                    "max_samples": 0,
                    "max_images": 0,
                    "save_visuals": False,
                    "update_reference_results": False,
                    "seeds": [42],
                },
                "evaluation": {
                    "benchmark_version": "test",
                    "track": "track_a_mask_prompt",
                    "protocol": "test",
                    "inference_mode": "box",
                    "prompt_policy": {
                        "name": "default_box_gt",
                        "prompt_type": "box",
                        "prompt_source": "gt",
                        "prompt_budget": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_config


class AutoPromptTrainingTests(unittest.TestCase):
    def test_train_auto_prompt_from_config_writes_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_config = _write_dataset(root)
            train_config = root / "auto_prompt.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "unit",
                        "output_root": str(root / "outputs"),
                        "dataset_configs": [str(dataset_config)],
                        "train": {
                            "device": "cpu",
                            "epochs": 1,
                            "batch_size": 1,
                            "learning_rate": 0.001,
                            "max_long_side": 16,
                            "max_samples": 2,
                        },
                        "model": {"hidden_channels": 4},
                        "target": {"gaussian_sigma": 1.0, "positive_radius": 1},
                    }
                ),
                encoding="utf-8",
            )

            summary = train_auto_prompt_from_config(train_config)

            checkpoint = Path(summary["checkpoint_path"])
            summary_path = Path(summary["output_dir"]) / "train_summary.json"
            self.assertTrue(checkpoint.exists())
            self.assertTrue(summary_path.exists())
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["sample_count"], 2)
            self.assertIn("final_loss", saved)
            self.assertEqual(len(saved["heatmaps"]), 2)
            self.assertTrue(Path(saved["heatmaps"][0]["overlay"]).exists())


if __name__ == "__main__":
    unittest.main()
