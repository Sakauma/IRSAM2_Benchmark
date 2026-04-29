import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from irsam2_benchmark.baselines.methods import ExternalPredictionMaskBaseline, build_baseline_registry
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.evaluation.runner import evaluate_method


def _sample(frame_id: str = "seq/img.v1") -> Sample:
    gt = np.zeros((4, 4), dtype=np.float32)
    gt[1:3, 1:3] = 1.0
    return Sample(
        image_path=Path("dummy.png"),
        sample_id=f"{frame_id}::foreground::mask",
        frame_id=frame_id,
        sequence_id="seq",
        frame_index=0,
        temporal_key=frame_id,
        width=4,
        height=4,
        category="foreground",
        target_scale="small",
        device_source="cam",
        annotation_protocol_flag="mask",
        supervision_type="mask",
        bbox_tight=[1, 1, 3, 3],
        bbox_loose=[1, 1, 3, 3],
        point_prompt=[1.5, 1.5],
        mask_array=gt,
    )


def _config(root: Path, prediction_root: Path):
    return SimpleNamespace(
        root=root,
        sam2_repo=root,
        config_path=root / "config.yaml",
        output_dir=root / "artifacts" / "out",
        dataset=SimpleNamespace(dataset_id="NUAA-SIRST", modality="ir"),
        model=SimpleNamespace(model_id="external"),
        runtime=SimpleNamespace(
            image_batch_size=1,
            reuse_image_embedding=False,
            batch_oom_fallback=True,
            show_progress=False,
            progress_update_interval_s=0.0,
        ),
        method={
            "name": "dnanet_external_prediction",
            "prediction_root": str(prediction_root),
            "prediction_threshold": 0.5,
            "external_model_name": "DNANet",
        },
    )


class ExternalPredictionMaskBaselineTests(unittest.TestCase):
    def test_registry_build_does_not_require_external_prediction_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, root / "predictions")
            config.method = {}

            registry = build_baseline_registry(config)

            self.assertIn("external_prediction_mask", registry)

    def test_loads_mask_and_manifest_latency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_root = root / "predictions"
            dataset_dir = prediction_root / "NUAA-SIRST" / "seq"
            dataset_dir.mkdir(parents=True)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 255
            Image.fromarray(mask, mode="L").save(dataset_dir / "img.v1.png")
            manifest = prediction_root / "NUAA-SIRST" / "manifest.jsonl"
            manifest.write_text(json.dumps({"frame_id": "seq/img.v1", "latency_ms": 12.5}) + "\n", encoding="utf-8")

            config = _config(root, prediction_root)
            method = ExternalPredictionMaskBaseline(config)
            sample = _sample()
            pred = method.predict_sample(sample)

            self.assertEqual(pred["mask"].shape, (4, 4))
            self.assertEqual(float(pred["mask"].sum()), 4.0)
            self.assertEqual(pred["LatencyMs"], 12.5)

            _, rows = evaluate_method(
                method=method,
                samples=[sample],
                config=config,
                track_name="track_a_image_prompted",
                inference_mode=method.inference_mode,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["LatencyMs"], 12.5)
            self.assertEqual(rows[0]["ExternalPredictionModel"], "DNANet")
            self.assertEqual(rows[0]["PromptProtocol"], "external_prediction_import_v1")

    def test_missing_prediction_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_root = root / "predictions"
            method = ExternalPredictionMaskBaseline(_config(root, prediction_root))

            with self.assertRaises(FileNotFoundError):
                method.predict_sample(_sample())


if __name__ == "__main__":
    unittest.main()
