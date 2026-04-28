import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter


def _write_rbgt_config(path: Path, *, images_dir: str = "image") -> None:
    path.write_text(
        json.dumps(
            {
                "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                "dataset": {
                    "dataset_id": "RBGT-Tiny",
                    "adapter": "rbgt_tiny_ir_only",
                    "root": ".",
                    "images_dir": images_dir,
                    "annotations_dir": "annotations_coco",
                    "mask_mode": "auto",
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
                    "max_images": 10,
                    "save_visuals": False,
                    "seeds": [42],
                },
                "evaluation": {
                    "benchmark_version": "v1",
                    "track": "track_a_image_prompted",
                    "protocol": "box_only",
                    "inference_mode": "box",
                    "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": False},
                },
            }
        ),
        encoding="utf-8",
    )


class RBGTTinyAdapterTests(unittest.TestCase):
    def test_rbgt_adapter_only_reads_ir_branch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image"
            ann_root = root / "annotations_coco"
            (image_root / "DJI_0001" / "00").mkdir(parents=True)
            (image_root / "DJI_0001" / "01").mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "00" / "00000.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "01" / "00000.jpg")
            ann_root.mkdir()

            rgb_payload = {
                "images": [{"id": 1, "file_name": "DJI_0001/00/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            ir_payload = {
                "images": [{"id": 2, "file_name": "DJI_0001/01/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 2, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_00_test2017.json").write_text(json.dumps(rgb_payload), encoding="utf-8")
            (ann_root / "instances_01_test2017.json").write_text(json.dumps(ir_payload), encoding="utf-8")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                        "dataset": {
                            "dataset_id": "RBGT-Tiny",
                            "adapter": "rbgt_tiny_ir_only",
                            "root": ".",
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
                            "num_workers": 0,
                            "smoke_test": True,
                            "max_samples": 0,
                            "max_images": 10,
                            "save_visuals": False,
                            "seeds": [42],
                        },
                        "evaluation": {
                            "benchmark_version": "v1",
                            "track": "track_a_image_prompted",
                            "protocol": "box_only",
                            "inference_mode": "box",
                            "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": False},
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(config_path)
            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.adapter_name, "rbgt_tiny_ir_only")
            self.assertEqual(loaded.manifest.image_count, 1)
            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertTrue(all("/01/" in sample.image_path.as_posix() for sample in loaded.samples))

    def test_rbgt_adapter_reads_generic_annotation_with_ir_branch_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image"
            ann_root = root / "annotations_coco"
            (image_root / "DJI_0001" / "00").mkdir(parents=True)
            (image_root / "DJI_0001" / "01").mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "00" / "00000.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "01" / "00000.jpg")
            ann_root.mkdir()
            payload = {
                "images": [
                    {"id": 1, "file_name": "DJI_0001/00/00000.jpg", "width": 8, "height": 8},
                    {"id": 2, "file_name": "DJI_0001/01/00000.jpg", "width": 8, "height": 8},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 2, 3, 3]},
                ],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "config.json"
            _write_rbgt_config(config_path)
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertTrue(all("/01/" in sample.image_path.as_posix() for sample in loaded.samples))

    def test_rbgt_adapter_reads_ir_named_annotation_without_branch_in_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image"
            ann_root = root / "annotations_coco"
            image_root.mkdir()
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_01_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "config.json"
            _write_rbgt_config(config_path)
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].image_path.name, "00000.jpg")

    def test_rbgt_adapter_reads_flat_ir_only_images_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images"
            ann_root = root / "annotations_coco"
            image_root.mkdir()
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "config.json"
            _write_rbgt_config(config_path)
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].image_path.parent.name, "images")

    def test_rbgt_adapter_reads_ir_only_sequence_subdirectories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images" / "DJI_0022_1"
            ann_root = root / "annotations_coco"
            image_root.mkdir(parents=True)
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "DJI_0022_1/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "config.json"
            _write_rbgt_config(config_path)
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].image_path.parent.name, "DJI_0022_1")

    def test_rbgt_adapter_reads_ir_annotation_with_sequence_branch_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images" / "DJI_0022_1" / "01"
            ann_root = root / "annotations_coco"
            image_root.mkdir(parents=True)
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "DJI_0022_1/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_01_train2017.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "config.json"
            _write_rbgt_config(config_path)
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            self.assertEqual(loaded.samples[0].image_path.parent.name, "01")
            self.assertEqual(loaded.samples[0].image_path.parent.parent.name, "DJI_0022_1")


if __name__ == "__main__":
    unittest.main()
