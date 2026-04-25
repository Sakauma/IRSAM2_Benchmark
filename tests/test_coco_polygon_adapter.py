import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter


class CocoPolygonAdapterTests(unittest.TestCase):
    def test_coco_polygon_decodes_to_mask_prompt_and_box(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "image"
            ann_dir = root / "annotations_coco"
            image_dir.mkdir()
            ann_dir.mkdir()
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(image_dir / "frame_001.png")
            (ann_dir / "instances_train.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "frame_001.png", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "target"}],
                        "annotations": [
                            {
                                "id": 7,
                                "image_id": 1,
                                "category_id": 1,
                                "segmentation": [[[2, 2, 6, 2, 6, 6, 2, 6]]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                        "dataset": {
                            "dataset_id": "CocoPolygon",
                            "adapter": "coco_like",
                            "root": ".",
                            "modality": "ir",
                            "images_dir": "image",
                            "annotations_dir": "annotations_coco",
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
                            "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            dataset = build_dataset_adapter(load_app_config(config_path)).load(load_app_config(config_path))
            self.assertEqual(dataset.manifest.sample_count, 1)
            sample = dataset.samples[0]
            self.assertIsNotNone(sample.mask_array)
            self.assertIsNotNone(sample.bbox_tight)
            self.assertIsNotNone(sample.bbox_loose)
            self.assertIsNotNone(sample.point_prompt)
            self.assertGreater(float(sample.mask_array.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
