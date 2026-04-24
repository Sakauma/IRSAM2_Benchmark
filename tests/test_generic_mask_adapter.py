import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter


class GenericMaskAdapterTests(unittest.TestCase):
    def test_generic_binary_mask_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 3:5] = 255
            Image.fromarray(image).save(images / "sample.png")
            Image.fromarray(mask).save(masks / "sample.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "track_a_image_prompted", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}},
                  "stages": {},
                  "ablations": {}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)
            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.manifest.image_count, 1)
            self.assertEqual(loaded.samples[0].category, "foreground")
            self.assertEqual(loaded.samples[0].frame_id, "sample")
            self.assertEqual(loaded.samples[0].track_id, None)
            self.assertTrue(loaded.samples[0].sample_id.startswith("sample"))
            self.assertIsNotNone(loaded.samples[0].bbox_loose)

    def test_generic_instance_masks_keep_image_frame_id_and_unique_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.zeros((8, 8), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[1:3, 1:3] = 1
            mask[4:6, 4:6] = 2
            Image.fromarray(image).save(images / "frame_0001.png")
            Image.fromarray(mask).save(masks / "frame_0001.png")

            config_path = root / "config.json"
            config_path.write_text(
                """
                {
                  "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                  "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "instance_id", "class_map": {"1": "car", "2": "plane"}},
                  "runtime": {"artifact_root": "artifacts", "reference_results_root": "reference_results", "output_name": "out", "device": "cpu", "num_workers": 0, "smoke_test": true, "max_samples": 0, "max_images": 0, "save_visuals": false, "seeds": [42]},
                  "evaluation": {"benchmark_version": "v1", "track": "track_a_image_prompted", "protocol": "mask_supervised", "inference_mode": "box", "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": false}},
                  "stages": {},
                  "ablations": {}
                }
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)
            loaded = adapter.load(config)

            self.assertEqual(loaded.manifest.sample_count, 2)
            self.assertEqual(loaded.manifest.image_count, 1)
            self.assertEqual({sample.frame_id for sample in loaded.samples}, {"frame_0001"})
            self.assertEqual(len({sample.sample_id for sample in loaded.samples}), 2)
            self.assertEqual({sample.track_id for sample in loaded.samples}, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
