import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from irsam2_benchmark.config import load_app_config
from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.adapters import DatasetManifest, LoadedDataset
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.pipeline.runner import run_command, set_global_seed


def _build_test_config(root: Path, *, save_visuals: bool, seeds: list[int], update_reference_results: bool = True, visual_limit: int = 24):
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"model_id": "dummy", "family": "sam2", "cfg": "cfg", "ckpt": "ckpt", "repo": ""},
                "dataset": {"dataset_id": "generic", "adapter": "generic_image_mask", "root": ".", "images_dir": "images", "masks_dir": "masks", "mask_mode": "binary", "class_map": {}},
                "runtime": {
                    "artifact_root": "artifacts",
                    "reference_results_root": "reference_results",
                    "output_name": "out",
                    "device": "cpu",
                    "num_workers": 0,
                    "smoke_test": True,
                    "max_samples": 0,
                    "max_images": 0,
                    "save_visuals": save_visuals,
                    "visual_limit": visual_limit,
                    "update_reference_results": update_reference_results,
                    "seeds": seeds,
                },
                "evaluation": {
                    "benchmark_version": "v1",
                    "track": "track_a_image_prompted",
                    "protocol": "mask_supervised",
                    "inference_mode": "box",
                    "prompt_policy": {"name": "p", "prompt_type": "box", "prompt_source": "gt", "prompt_budget": 1, "multi_mask": False},
                },
                "stages": {},
                "ablations": {},
            }
        ),
        encoding="utf-8",
    )
    return load_app_config(config_path)


def _build_test_dataset(root: Path) -> LoadedDataset:
    sample = Sample(
        image_path=root / "dummy.png",
        sample_id="frame_0::foreground::generic_binary_mask",
        frame_id="frame_0",
        sequence_id="seq",
        frame_index=0,
        temporal_key="frame_0",
        width=4,
        height=4,
        category="foreground",
        target_scale="small",
        device_source="cam",
        annotation_protocol_flag="mask",
        supervision_type="mask",
        bbox_tight=[0, 0, 2, 2],
        bbox_loose=[0, 0, 2, 2],
        point_prompt=[0.5, 0.5],
        mask_array=np.ones((4, 4), dtype=np.float32),
    )
    return LoadedDataset(
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


class PipelineRunnerSeedTests(unittest.TestCase):
    def test_set_global_seed_resets_random_sources(self):
        set_global_seed(7)
        values_a = (random.random(), float(np.random.random()))
        set_global_seed(7)
        values_b = (random.random(), float(np.random.random()))
        self.assertEqual(values_a, values_b)

    def test_run_command_rebuilds_method_for_each_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_test_config(root, save_visuals=False, seeds=[11, 12])
            dataset = _build_test_dataset(root)

            called_tokens = []

            class DummyMethod:
                inference_mode = InferenceMode.BOX

                def __init__(self, token):
                    self.token = token

            def fake_build_dataset_adapter(_config):
                class DummyAdapter:
                    def load(self, __config):
                        return dataset

                return DummyAdapter()

            def fake_build_baseline_registry(_config):
                token = (round(random.random(), 6), round(float(np.random.random()), 6))
                method = DummyMethod(token)
                return {"bbox_rect": method, "sam2_zero_shot": method}

            def fake_evaluate_method(**kwargs):
                called_tokens.append(kwargs["method"].token)
                return {"LatencyMs": kwargs["method"].token[0]}, [{"sample_id": "frame_0", "frame_id": "frame_0", "sequence_id": "seq", "LatencyMs": kwargs["method"].token[0]}]

            with patch("irsam2_benchmark.pipeline.runner.build_dataset_adapter", side_effect=fake_build_dataset_adapter), patch(
                "irsam2_benchmark.pipeline.runner.build_baseline_registry", side_effect=fake_build_baseline_registry
            ), patch("irsam2_benchmark.pipeline.runner.evaluate_method", side_effect=fake_evaluate_method):
                run_command(config, "baseline", baseline_name="bbox_rect")

            expected_tokens = []
            for seed in [11, 12]:
                set_global_seed(seed)
                expected_tokens.append((round(random.random(), 6), round(float(np.random.random()), 6)))
            self.assertEqual(called_tokens, expected_tokens)
            self.assertNotEqual(called_tokens[0], called_tokens[1])

    def test_run_command_saves_visuals_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_test_config(root, save_visuals=True, seeds=[42], visual_limit=1)
            dataset = _build_test_dataset(root)

            class DummyMethod:
                inference_mode = InferenceMode.BOX

                def predict_sample(self, _sample):
                    return {"mask": np.ones((4, 4), dtype=np.float32), "score": 1.0}

            visual_calls = []

            def fake_build_dataset_adapter(_config):
                class DummyAdapter:
                    def load(self, __config):
                        return dataset

                return DummyAdapter()

            def fake_build_baseline_registry(_config):
                method = DummyMethod()
                return {"bbox_rect": method, "sam2_zero_shot": method}

            def fake_evaluate_method(**kwargs):
                return {"LatencyMs": 1.0}, [{"sample_id": "frame_0", "frame_id": "frame_0", "sequence_id": "seq", "LatencyMs": 1.0}]

            def fake_save_visualizations(**kwargs):
                visual_calls.append(kwargs)
                return [kwargs["output_dir"] / "visuals" / kwargs["method_name"] / f"seed_{kwargs['seed']}" / "000.png"]

            with patch("irsam2_benchmark.pipeline.runner.build_dataset_adapter", side_effect=fake_build_dataset_adapter), patch(
                "irsam2_benchmark.pipeline.runner.build_baseline_registry", side_effect=fake_build_baseline_registry
            ), patch("irsam2_benchmark.pipeline.runner.evaluate_method", side_effect=fake_evaluate_method), patch(
                "irsam2_benchmark.pipeline.runner.save_visualizations", side_effect=fake_save_visualizations
            ):
                run_command(config, "baseline", baseline_name="bbox_rect")

            self.assertEqual(len(visual_calls), 1)
            self.assertEqual(visual_calls[0]["method_name"], "bbox_rect")
            self.assertEqual(visual_calls[0]["seed"], 42)
            self.assertEqual(len(visual_calls[0]["visual_records"]), 1)

    def test_run_command_updates_reference_results_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_test_config(root, save_visuals=False, seeds=[42])
            dataset = _build_test_dataset(root)

            class DummyMethod:
                inference_mode = InferenceMode.BOX

            def fake_build_dataset_adapter(_config):
                class DummyAdapter:
                    def load(self, __config):
                        return dataset

                return DummyAdapter()

            def fake_build_baseline_registry(_config):
                method = DummyMethod()
                return {"bbox_rect": method, "sam2_zero_shot": method}

            def fake_evaluate_method(**kwargs):
                return {"LatencyMs": 1.0}, [{"sample_id": "frame_0", "frame_id": "frame_0", "sequence_id": "seq", "LatencyMs": 1.0}]

            with patch("irsam2_benchmark.pipeline.runner.build_dataset_adapter", side_effect=fake_build_dataset_adapter), patch(
                "irsam2_benchmark.pipeline.runner.build_baseline_registry", side_effect=fake_build_baseline_registry
            ), patch("irsam2_benchmark.pipeline.runner.evaluate_method", side_effect=fake_evaluate_method), patch(
                "irsam2_benchmark.pipeline.runner._snapshot_reference_outputs"
            ) as snapshot_mock:
                run_command(config, "baseline", baseline_name="bbox_rect")

            snapshot_mock.assert_called_once_with(config, "bbox_rect", config.output_dir)

    def test_run_command_can_skip_reference_results_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_test_config(root, save_visuals=False, seeds=[42], update_reference_results=False)
            dataset = _build_test_dataset(root)

            class DummyMethod:
                inference_mode = InferenceMode.BOX

            def fake_build_dataset_adapter(_config):
                class DummyAdapter:
                    def load(self, __config):
                        return dataset

                return DummyAdapter()

            def fake_build_baseline_registry(_config):
                method = DummyMethod()
                return {"bbox_rect": method, "sam2_zero_shot": method}

            def fake_evaluate_method(**kwargs):
                return {"LatencyMs": 1.0}, [{"sample_id": "frame_0", "frame_id": "frame_0", "sequence_id": "seq", "LatencyMs": 1.0}]

            with patch("irsam2_benchmark.pipeline.runner.build_dataset_adapter", side_effect=fake_build_dataset_adapter), patch(
                "irsam2_benchmark.pipeline.runner.build_baseline_registry", side_effect=fake_build_baseline_registry
            ), patch("irsam2_benchmark.pipeline.runner.evaluate_method", side_effect=fake_evaluate_method), patch(
                "irsam2_benchmark.pipeline.runner._snapshot_reference_outputs"
            ) as snapshot_mock:
                run_command(config, "baseline", baseline_name="bbox_rect")

            snapshot_mock.assert_not_called()
            metadata = json.loads((config.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["baseline_name"], "bbox_rect")
            self.assertEqual(metadata["config_path"], str(config.config_path))
            self.assertIn("git", metadata)
            self.assertIn("torch", metadata)


if __name__ == "__main__":
    unittest.main()
