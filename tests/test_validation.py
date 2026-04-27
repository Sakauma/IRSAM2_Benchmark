import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.adapters import DatasetManifest, LoadedDataset
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.pipeline.runner import run_command
from irsam2_benchmark.validation import preflight_dataset, validate_run_artifacts


def _write_generic_dataset(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir()
    masks.mkdir()
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:5] = 255
    Image.fromarray(image).save(images / "sample.png")
    Image.fromarray(mask).save(masks / "sample.png")


def _write_config(root: Path, *, update_reference_results: bool = False) -> Path:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                "dataset": {
                    "dataset_id": "generic",
                    "adapter": "generic_image_mask",
                    "root": ".",
                    "images_dir": "images",
                    "masks_dir": "masks",
                    "mask_mode": "binary",
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
                    "max_images": 0,
                    "save_visuals": False,
                    "update_reference_results": update_reference_results,
                    "seeds": [42],
                    "max_failure_rate": 0.05,
                },
                "evaluation": {
                    "benchmark_version": "v1",
                    "track": "track_a_image_prompted",
                    "protocol": "mask_supervised",
                    "inference_mode": "box",
                    "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": False},
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_valid_run_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "eval_reports").mkdir()
    (run_dir / "benchmark_spec.json").write_text(json.dumps({"inference_mode": "box"}), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "expected_sample_count": 1,
                "expected_eval_units": 1,
                "expected_row_count": 1,
                "row_count": 1,
                "error_count": 0,
                "missing_row_count": 0,
                "failure_rate": 0.0,
                "failure_rate_threshold": 0.05,
                "mean": {"mIoU": 0.5},
                "std": {"mIoU": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(json.dumps([{"seed": 42, "mIoU": 0.5}]), encoding="utf-8")
    (run_dir / "eval_reports" / "rows.json").write_text(
        json.dumps([{"sample_id": "s0", "frame_id": "frame", "sequence_id": "seq", "eval_unit": "instance", "mIoU": 0.5}]),
        encoding="utf-8",
    )


class ValidationTests(unittest.TestCase):
    def test_preflight_dataset_reports_counts_and_mask_area(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_generic_dataset(root)
            config = load_app_config(_write_config(root))

            report = preflight_dataset(config)

            self.assertTrue(report["valid"])
            self.assertEqual(report["adapter_name"], "generic_image_mask")
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(report["category_counts"], {"foreground": 1})
            self.assertEqual(report["annotation_protocol_counts"], {"generic_binary_mask": 1})
            self.assertEqual(report["area_pixels"]["count"], 1)
            self.assertEqual(report["area_pixels"]["min"], 8.0)
            self.assertEqual(report["missing_bbox_count"], 0)
            self.assertEqual(report["missing_point_count"], 0)

    def test_validate_run_artifacts_accepts_minimal_complete_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            _write_valid_run_artifacts(run_dir)

            report = validate_run_artifacts(run_dir)

            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["row_count"], 1)
            self.assertEqual(report["mean_metric_count"], 1)

    def test_validate_run_artifacts_rejects_missing_file_and_nonfinite_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            _write_valid_run_artifacts(run_dir)
            (run_dir / "results.json").unlink()
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "expected_sample_count": 1,
                        "expected_eval_units": 1,
                        "expected_row_count": 1,
                        "row_count": 1,
                        "error_count": 0,
                        "missing_row_count": 0,
                        "failure_rate": 0.0,
                        "failure_rate_threshold": 0.05,
                        "mean": {"mIoU": float("nan")},
                    }
                ),
                encoding="utf-8",
            )

            report = validate_run_artifacts(run_dir)

            self.assertFalse(report["valid"])
            self.assertTrue(any("Missing required artifact file: results.json" in item for item in report["errors"]))
            self.assertTrue(any("non-finite numeric value" in item for item in report["errors"]))

    def test_validate_run_artifacts_rejects_failure_rate_over_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            _write_valid_run_artifacts(run_dir)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["failure_rate"] = 0.5
            summary["failure_rate_threshold"] = 0.05
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            report = validate_run_artifacts(run_dir)

            self.assertFalse(report["valid"])
            self.assertTrue(any("failure_rate=0.5000 exceeds" in item for item in report["errors"]))

    def test_run_command_writes_config_fingerprints_and_runtime_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_generic_dataset(root)
            config = load_app_config(_write_config(root))
            sample = Sample(
                image_path=root / "images" / "sample.png",
                sample_id="sample::foreground::generic_binary_mask",
                frame_id="sample",
                sequence_id="__single_sequence__",
                frame_index=0,
                temporal_key="sample",
                width=8,
                height=8,
                category="foreground",
                target_scale="small",
                device_source="unknown",
                annotation_protocol_flag="generic_binary_mask",
                supervision_type="mask",
                bbox_tight=[3.0, 2.0, 5.0, 6.0],
                bbox_loose=[2.0, 1.0, 6.0, 7.0],
                point_prompt=[3.5, 3.5],
                mask_array=np.ones((8, 8), dtype=np.float32),
            )
            dataset = LoadedDataset(
                manifest=DatasetManifest(
                    adapter_name="generic_image_mask",
                    dataset_id="generic",
                    root=str(root),
                    sample_count=1,
                    image_count=1,
                    sequence_count=1,
                    category_count=1,
                    notes="test",
                ),
                samples=[sample],
            )

            class DummyMethod:
                inference_mode = InferenceMode.BOX

                def predict_sample(self, _sample):
                    return {"mask": np.ones((8, 8), dtype=np.float32), "score": 1.0}

                def predict_samples(self, samples):
                    return {item.sample_id: self.predict_sample(item) for item in samples}

            def fake_build_dataset_adapter(_config):
                class DummyAdapter:
                    def load(self, __config):
                        return dataset

                return DummyAdapter()

            def fake_build_baseline_registry(_config):
                return {"bbox_rect": DummyMethod()}

            def fake_evaluate_method(**_kwargs):
                return (
                    {"mIoU": 1.0, "LatencyMs": 0.1},
                    [{"sample_id": "sample", "frame_id": "sample", "sequence_id": "__single_sequence__", "eval_unit": "instance", "mIoU": 1.0}],
                )

            with patch("irsam2_benchmark.pipeline.runner.build_dataset_adapter", side_effect=fake_build_dataset_adapter), patch(
                "irsam2_benchmark.pipeline.runner.build_baseline_registry", side_effect=fake_build_baseline_registry
            ), patch("irsam2_benchmark.pipeline.runner.evaluate_method", side_effect=fake_evaluate_method):
                run_command(config, "baseline", baseline_name="bbox_rect")

            benchmark_spec = json.loads((config.output_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
            metadata = json.loads((config.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            summary = json.loads((config.output_dir / "summary.json").read_text(encoding="utf-8"))
            expected_hash = config.fingerprints["config_file_sha256"]
            self.assertEqual(benchmark_spec["fingerprints"]["config_file_sha256"], expected_hash)
            self.assertEqual(metadata["fingerprints"]["config_file_sha256"], expected_hash)
            self.assertIn("runtime_resources", summary)
            self.assertIn("wall_time_s", summary["runtime_resources"])
            self.assertIn("cuda_available", metadata["runtime_resources"])


if __name__ == "__main__":
    unittest.main()
