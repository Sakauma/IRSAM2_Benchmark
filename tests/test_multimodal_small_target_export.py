import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.data import build_dataset_adapter


def _load_exporter():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_multimodal_small_target_coco.py"
    spec = importlib.util.spec_from_file_location("export_multimodal_small_target_coco_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiModalSmallTargetExportTests(unittest.TestCase):
    def test_export_filters_multimodal_polygons_to_coco_small_targets(self):
        exporter = _load_exporter()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_dir = root / "img"
            label_dir = root / "label"
            img_dir.mkdir()
            label_dir.mkdir()
            Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(img_dir / "frame_001.png")
            (label_dir / "frame_001.json").write_text(
                json.dumps(
                    {
                        "detection": {
                            "instances": [
                                {"category": "large_ship", "mask": [[0, 0, 80, 0, 80, 80, 0, 80]]},
                                {"category": "tiny_ship", "mask": [[10, 10, 16, 10, 16, 16, 10, 16]]},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = exporter.export_multimodal_small_target_coco(root=root, overwrite=True)

            self.assertEqual(summary.instances_seen, 2)
            self.assertEqual(summary.annotations_kept, 1)
            self.assertEqual(summary.skipped_too_large, 1)
            payload = json.loads(Path(summary.output_path).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["images"]), 1)
            self.assertEqual(len(payload["annotations"]), 1)
            self.assertEqual(payload["categories"][0]["name"], "tiny_ship")

            config_path = root / "coco_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                        "dataset": {
                            "dataset_id": "MultiModal-SmallTarget",
                            "adapter": "coco_like",
                            "root": ".",
                            "modality": "ir",
                            "images_dir": "img",
                            "annotations_dir": "annotations_coco_small_target",
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
            config = load_app_config(config_path)
            dataset = build_dataset_adapter(config).load(config)

            self.assertEqual(dataset.manifest.sample_count, 1)
            self.assertEqual(dataset.samples[0].category, "tiny_ship")
            self.assertEqual(dataset.samples[0].target_scale, "small")


if __name__ == "__main__":
    unittest.main()
