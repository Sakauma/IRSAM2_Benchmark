import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter
from irsam2_benchmark.data.masks import sample_mask_array
from irsam2_benchmark.evaluation.runner import build_segmentation_row


class MultiModalLazyMaskTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                    "dataset": {
                        "dataset_id": "MultiModalTiny",
                        "adapter": "multimodal_raw",
                        "root": ".",
                        "modality": "ir",
                        "images_dir": "img",
                        "annotations_dir": "",
                        "mask_mode": "auto",
                        "class_map": {},
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
                        "prompt_policy": {
                            "name": "p",
                            "prompt_type": "box",
                            "prompt_source": "gt",
                            "prompt_budget": 1,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_multimodal_keeps_polygon_lazy_but_evaluates_mask_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_dir = root / "img"
            label_dir = root / "label"
            img_dir.mkdir()
            label_dir.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(img_dir / "frame_001.png")
            (label_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "detection": {
                            "instances": [
                                {
                                    "category": "drone",
                                    "mask": [[2, 2, 5, 2, 5, 5, 2, 5]],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            config_path = self._write_config(root)

            config = load_app_config(config_path)
            dataset = build_dataset_adapter(config).load(config)

            self.assertEqual(dataset.manifest.sample_count, 1)
            sample = dataset.samples[0]
            self.assertIsNone(sample.mask_array)
            self.assertTrue(sample.has_mask())
            self.assertIsNotNone(sample.bbox_tight)
            self.assertIsNotNone(sample.bbox_loose)
            self.assertIsNotNone(sample.point_prompt)

            gt_mask = sample_mask_array(sample)
            self.assertIsNotNone(gt_mask)
            self.assertEqual(gt_mask.shape, (8, 8))
            self.assertGreater(float(gt_mask.sum()), 0.0)

            row = build_segmentation_row(sample, gt_mask, gt_mask, elapsed_ms=1.0)
            self.assertEqual(row["mIoU"], 1.0)
            self.assertEqual(row["Dice"], 1.0)
            self.assertGreater(row["GTAreaPixels"], 0.0)

    def test_multimodal_matches_jpg_images_from_configured_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_dir = root / "img"
            label_dir = root / "label"
            img_dir.mkdir()
            label_dir.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(img_dir / "frame_001.jpg")
            (label_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "detection": {
                            "instances": [
                                {
                                    "category": "drone",
                                    "mask": [[1, 1, 4, 1, 4, 4, 1, 4]],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(self._write_config(root))
            dataset = build_dataset_adapter(config).load(config)

            self.assertEqual(dataset.manifest.sample_count, 1)
            self.assertEqual(dataset.manifest.image_count, 1)
            self.assertEqual(dataset.samples[0].image_path.suffix, ".jpg")

    def test_multimodal_uses_first_valid_polygon_when_instance_has_multiple_polygons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_dir = root / "img"
            label_dir = root / "label"
            img_dir.mkdir()
            label_dir.mkdir()
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(img_dir / "frame_001.png")
            first_polygon = [1, 1, 4, 1, 4, 4, 1, 4]
            second_polygon = [6, 6, 8, 6, 8, 8, 6, 8]
            (label_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "detection": {
                            "instances": [
                                {
                                    "category": "drone",
                                    "mask": [first_polygon, second_polygon],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(self._write_config(root))
            dataset = build_dataset_adapter(config).load(config)
            sample = dataset.samples[0]

            self.assertEqual(sample.metadata["mask_source"]["points"], [float(value) for value in first_polygon])
            gt_mask = sample_mask_array(sample)
            self.assertGreater(float(gt_mask[2, 2]), 0.0)
            self.assertEqual(float(gt_mask[7, 7]), 0.0)


if __name__ == "__main__":
    unittest.main()
