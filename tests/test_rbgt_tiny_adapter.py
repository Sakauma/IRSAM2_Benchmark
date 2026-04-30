import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.convert_rbgt_coco_to_voc_ir import convert_dataset
from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter
from irsam2_benchmark.data.masks import MASK_SOURCE_KEY, sample_mask_array


def _write_rbgt_config(path: Path, *, images_dir: str = "image", annotations_dir: str = "annotations_coco", mask_mode: str = "auto") -> None:
    path.write_text(
        json.dumps(
            {
                "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                "dataset": {
                    "dataset_id": "RBGT-Tiny",
                    "adapter": "rbgt_tiny_ir_only",
                    "root": ".",
                    "images_dir": images_dir,
                    "annotations_dir": annotations_dir,
                    "mask_mode": mask_mode,
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

    def test_convert_rbgt_coco_to_voc_ir_preserves_segmentation_and_filters_visible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image"
            ann_root = root / "annotations_coco"
            (image_root / "DJI_0001" / "00").mkdir(parents=True)
            (image_root / "DJI_0001" / "01").mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "00" / "00000.jpg")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "DJI_0001" / "01" / "00000.jpg")
            ann_root.mkdir()
            ir_segmentation = [[2, 2, 5, 2, 5, 5, 2, 5]]
            payload = {
                "images": [
                    {"id": 1, "file_name": "DJI_0001/00/00000.jpg", "width": 8, "height": 8},
                    {"id": 2, "file_name": "DJI_0001/01/00000.jpg", "width": 8, "height": 8},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]},
                    {
                        "id": 20,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [2, 2, 3, 3],
                        "segmentation": ir_segmentation,
                        "area": 9,
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = convert_dataset(root=root, overwrite=True)

            self.assertEqual(summary.images_written, 1)
            self.assertEqual(summary.objects_written, 1)
            self.assertFalse((root / "annotations_voc" / "DJI_0001" / "00" / "00000.xml").exists())
            xml_path = root / "annotations_voc" / "DJI_0001" / "01" / "00000.xml"
            self.assertTrue(xml_path.exists())
            xml_root = ET.parse(xml_path).getroot()
            obj = xml_root.find("object")
            self.assertIsNotNone(obj)
            self.assertEqual(obj.findtext("name"), "target")
            self.assertEqual(json.loads(obj.findtext("coco_segmentation_json")), ir_segmentation)
            self.assertEqual(json.loads(obj.findtext("coco_annotation_json"))["id"], 20)

    def test_rbgt_adapter_reads_voc_bbox_without_mask_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image" / "DJI_0001" / "01"
            ann_root = root / "annotations_coco"
            image_root.mkdir(parents=True)
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "DJI_0001/01/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3], "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5]]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            convert_dataset(root=root, overwrite=True)
            config_path = root / "config.json"
            _write_rbgt_config(config_path, annotations_dir="annotations_voc", mask_mode="bbox")
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            sample = loaded.samples[0]
            self.assertEqual(sample.supervision_type, "bbox")
            self.assertEqual(sample.annotation_protocol_flag, "voc_bbox_only")
            self.assertEqual(sample.bbox_tight, [2.0, 2.0, 5.0, 5.0])
            self.assertIsNone(sample_mask_array(sample))
            self.assertNotIn(MASK_SOURCE_KEY, sample.metadata)

    def test_rbgt_voc_iter_samples_yields_before_consuming_all_xml_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "image" / "DJI_0001" / "01" / "00000.jpg"
            ann_root = root / "annotations_voc" / "DJI_0001" / "01"
            image_path.parent.mkdir(parents=True)
            ann_root.mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_path)
            xml_path = ann_root / "00000.xml"
            xml_path.write_text(
                """
<annotation>
  <filename>DJI_0001/01/00000.jpg</filename>
  <size><width>8</width><height>8</height></size>
  <object>
    <name>target</name>
    <bndbox><xmin>2</xmin><ymin>2</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>
  </object>
</annotation>
""".strip(),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            _write_rbgt_config(config_path, annotations_dir="annotations_voc", mask_mode="bbox")
            config = load_app_config(config_path)
            adapter = build_dataset_adapter(config)

            def guarded_xml_files(ann_dir):
                yield xml_path
                raise AssertionError("iter_samples consumed a second XML before yielding the first sample")

            adapter._iter_voc_annotation_files = guarded_xml_files
            sample = next(adapter.iter_samples(config))

            self.assertEqual(sample.sample_id, "DJI_0001/01/00000.jpg__ann_0::target::voc_bbox_only")
            self.assertEqual(sample.bbox_tight, [2.0, 2.0, 5.0, 5.0])

    def test_rbgt_adapter_reads_voc_segmentation_as_lazy_polygon(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image" / "DJI_0001" / "01"
            ann_root = root / "annotations_coco"
            image_root.mkdir(parents=True)
            ann_root.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image_root / "00000.jpg")
            payload = {
                "images": [{"id": 1, "file_name": "DJI_0001/01/00000.jpg", "width": 8, "height": 8}],
                "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3], "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5]]}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            convert_dataset(root=root, overwrite=True)
            config_path = root / "config.json"
            _write_rbgt_config(config_path, annotations_dir="annotations_voc", mask_mode="segmentation")
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            sample = loaded.samples[0]
            self.assertEqual(sample.supervision_type, "mask")
            self.assertEqual(sample.annotation_protocol_flag, "voc_coco_segmentation")
            self.assertIsNone(sample.mask_array)
            self.assertEqual(sample.metadata[MASK_SOURCE_KEY]["type"], "coco_polygon")
            self.assertGreater(float(sample_mask_array(sample).sum()), 0.0)

    def test_rbgt_adapter_reads_voc_segmentation_as_lazy_uncompressed_rle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "image" / "DJI_0001" / "01"
            ann_root = root / "annotations_coco"
            image_root.mkdir(parents=True)
            ann_root.mkdir()
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_root / "00000.jpg")
            rle = {"size": [4, 4], "counts": [0, 4, 12]}
            payload = {
                "images": [{"id": 1, "file_name": "DJI_0001/01/00000.jpg", "width": 4, "height": 4}],
                "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 4], "segmentation": rle}],
                "categories": [{"id": 1, "name": "target"}],
            }
            (ann_root / "instances_test2017.json").write_text(json.dumps(payload), encoding="utf-8")
            convert_dataset(root=root, overwrite=True)
            config_path = root / "config.json"
            _write_rbgt_config(config_path, annotations_dir="annotations_voc", mask_mode="segmentation")
            config = load_app_config(config_path)

            loaded = build_dataset_adapter(config).load(config)

            self.assertEqual(loaded.manifest.sample_count, 1)
            sample = loaded.samples[0]
            self.assertEqual(sample.supervision_type, "mask")
            self.assertEqual(sample.metadata[MASK_SOURCE_KEY]["type"], "coco_rle")
            mask = sample_mask_array(sample)
            self.assertIsNotNone(mask)
            self.assertEqual(mask.shape, (4, 4))
            self.assertEqual(float(mask.sum()), 4.0)
            self.assertTrue(np.all(mask[:, 0] == 1.0))


if __name__ == "__main__":
    unittest.main()
