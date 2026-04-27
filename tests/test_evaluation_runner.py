import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from irsam2_benchmark.core.interfaces import InferenceMode
from irsam2_benchmark.data.sample import Sample
from irsam2_benchmark.evaluation.runner import align_mask_to_sample, build_segmentation_row, box_to_area, evaluate_method


def make_sample(
    *,
    sample_id: str,
    frame_id: str,
    frame_index: int = 0,
    sequence_id: str = "seq",
    track_id: str | None = None,
    category: str = "target",
    target_scale: str = "small",
    bbox_tight: list[float] | None = None,
    bbox_loose: list[float] | None = None,
    mask_array: np.ndarray | None = None,
    supervision_type: str = "mask",
) -> Sample:
    return Sample(
        image_path=Path("dummy.png"),
        sample_id=sample_id,
        frame_id=frame_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
        temporal_key=frame_id,
        width=4,
        height=4,
        category=category,
        target_scale=target_scale,
        device_source="cam",
        annotation_protocol_flag="mask",
        supervision_type=supervision_type,
        track_id=track_id,
        bbox_tight=bbox_tight,
        bbox_loose=bbox_loose,
        point_prompt=[0.5, 0.5],
        mask_array=mask_array,
    )


class DummyConfig:
    class dataset:
        modality = "ir"

    class runtime:
        image_batch_size = 1
        batch_oom_fallback = True


class LoggingConfig:
    def __init__(self, output_dir: Path, *, image_batch_size: int = 1, show_progress: bool = False):
        self.output_dir = output_dir
        self.config_path = output_dir / "config.yaml"
        self.dataset = type("Dataset", (), {"modality": "ir", "dataset_id": "dummy_dataset"})()
        self.model = type("Model", (), {"model_id": "dummy_model"})()
        self.runtime = type(
            "Runtime",
            (),
            {"image_batch_size": image_batch_size, "batch_oom_fallback": True, "show_progress": show_progress, "progress_update_interval_s": 0.0},
        )()


def read_error_records(config: LoggingConfig) -> list[dict]:
    path = config.output_dir / "eval_reports" / "error_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EvaluationRunnerTests(unittest.TestCase):
    def test_bbox_iou_uses_predicted_mask_box(self):
        gt_mask = box_to_area([0, 0, 2, 2], 4, 4)
        pred_mask = box_to_area([2, 0, 4, 2], 4, 4)
        item = make_sample(
            sample_id="frame_0__inst_0",
            frame_id="frame_0",
            bbox_tight=[0, 0, 2, 2],
            bbox_loose=[0, 0, 4, 4],
            mask_array=gt_mask,
        )

        row = build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms=1.0)

        self.assertEqual(row["BBoxIoU"], 0.0)
        self.assertGreater(row["LooseBoxMaskIoU"], 0.0)
        self.assertEqual(row["eval_unit"], "instance")
        self.assertEqual(row["supervision_type"], "mask")
        self.assertIn("BoundaryF1Exact", row)
        self.assertIn("BoundaryF1Tol1", row)

    def test_bbox_only_row_does_not_emit_mask_metrics(self):
        pred_mask = box_to_area([0, 0, 2, 2], 4, 4)
        item = make_sample(
            sample_id="frame_0__box_0",
            frame_id="frame_0",
            bbox_tight=[0, 0, 2, 2],
            bbox_loose=[0, 0, 3, 3],
            mask_array=None,
            supervision_type="bbox",
        )

        row = build_segmentation_row(
            item,
            pred_mask,
            np.zeros((4, 4), dtype=np.float32),
            elapsed_ms=1.0,
            prompt={"box": [0, 0, 2, 2], "point": [1, 1]},
        )

        self.assertEqual(row["supervision_type"], "bbox")
        self.assertEqual(row["BBoxIoU"], 1.0)
        self.assertNotIn("mIoU", row)
        self.assertNotIn("Dice", row)
        self.assertNotIn("BoundaryF1", row)
        self.assertNotIn("TargetRecallIoU25", row)
        self.assertEqual(row["PromptBoxBBoxIoU"], 1.0)
        self.assertEqual(row["PromptPointInBBox"], 1.0)

    def test_pred_mask_alignment_records_resize_metadata(self):
        gt_mask = box_to_area([0, 0, 4, 4], 4, 4)
        pred_mask = np.ones((2, 2), dtype=np.float32)
        item = make_sample(
            sample_id="frame_0__inst_0",
            frame_id="frame_0",
            bbox_tight=[0, 0, 4, 4],
            bbox_loose=[0, 0, 4, 4],
            mask_array=gt_mask,
        )

        aligned_mask, metadata = align_mask_to_sample(pred_mask, item)
        row = build_segmentation_row(item, aligned_mask, gt_mask, elapsed_ms=1.0, mask_alignment=metadata)

        self.assertEqual(aligned_mask.shape, (4, 4))
        self.assertTrue(row["PredMaskWasResized"])
        self.assertEqual(row["PredMaskOriginalHeight"], 2)
        self.assertEqual(row["PredMaskOriginalWidth"], 2)
        self.assertEqual(row["PredMaskAlignedHeight"], 4)
        self.assertEqual(row["PredMaskAlignedWidth"], 4)
        self.assertEqual(row["mIoU"], 1.0)

    def test_evaluate_method_aligns_pred_mask_before_metrics(self):
        gt_mask = box_to_area([0, 0, 4, 4], 4, 4)
        item = make_sample(
            sample_id="frame_0__inst_0",
            frame_id="frame_0",
            bbox_tight=[0, 0, 4, 4],
            bbox_loose=[0, 0, 4, 4],
            mask_array=gt_mask,
        )

        class DummyMethod:
            def predict_sample(self, sample):
                return {"mask": np.ones((2, 2), dtype=np.float32), "prompt": None}

        _, rows = evaluate_method(
            method=DummyMethod(),
            samples=[item],
            config=None,
            track_name="track_a_mask_prompt",
            inference_mode=InferenceMode.BOX,
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["PredMaskWasResized"])
        self.assertEqual(rows[0]["PredMaskAlignedHeight"], 4)
        self.assertEqual(rows[0]["PredMaskAlignedWidth"], 4)

    def test_evaluate_method_batches_prompted_samples_by_runtime_size(self):
        mask = box_to_area([0, 0, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id=f"frame_{idx}__inst_0", frame_id=f"frame_{idx}", mask_array=mask)
            for idx in range(5)
        ]

        class Config(DummyConfig):
            class runtime:
                image_batch_size = 2
                batch_oom_fallback = True

        class DummyBatchMethod:
            def __init__(self):
                self.calls = []

            def predict_samples(self, batch):
                self.calls.append([sample.sample_id for sample in batch])
                return {
                    sample.sample_id: {"mask": np.ones((4, 4), dtype=np.float32), "prompt": None}
                    for sample in batch
                }

        method = DummyBatchMethod()
        _, rows = evaluate_method(
            method=method,
            samples=samples,
            config=Config(),
            track_name="track_a_mask_prompt",
            inference_mode=InferenceMode.BOX,
        )

        self.assertEqual(method.calls, [["frame_0__inst_0", "frame_1__inst_0"], ["frame_2__inst_0", "frame_3__inst_0"], ["frame_4__inst_0"]])
        self.assertEqual([row["sample_id"] for row in rows], [sample.sample_id for sample in samples])
        self.assertEqual([row["BatchSize"] for row in rows], [2, 2, 2, 2, 1])
        self.assertEqual([row["BatchIndex"] for row in rows], [0, 0, 1, 1, 2])
        self.assertEqual([row["BatchItemIndex"] for row in rows], [0, 1, 0, 1, 0])
        self.assertTrue(all("BatchLatencyMs" in row for row in rows))

    def test_evaluate_method_reports_tqdm_progress_when_enabled(self):
        mask = box_to_area([0, 0, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id=f"frame_{idx}__inst_0", frame_id=f"frame_{idx}", mask_array=mask)
            for idx in range(2)
        ]

        class DummyBatchMethod:
            def predict_samples(self, batch):
                return {
                    sample.sample_id: {"mask": np.ones((4, 4), dtype=np.float32), "prompt": None}
                    for sample in batch
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(Path(temp_dir) / "out", image_batch_size=1, show_progress=True)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                _, rows = evaluate_method(
                    method=DummyBatchMethod(),
                    samples=samples,
                    config=config,
                    track_name="track_a_mask_prompt",
                    inference_mode=InferenceMode.BOX,
                )

        self.assertEqual(len(rows), 2)
        progress_text = stderr.getvalue()
        self.assertIn("dummy_dataset", progress_text)
        self.assertIn("2/2", progress_text)

    def test_evaluate_method_logs_batch_prediction_key_mismatch_and_skips_samples(self):
        mask = box_to_area([0, 0, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__inst_0", frame_id="frame_0", mask_array=mask),
            make_sample(sample_id="frame_1__inst_0", frame_id="frame_1", mask_array=mask),
        ]

        class BadBatchMethod:
            def predict_samples(self, batch):
                del batch
                return {"wrong_id": {"mask": np.ones((4, 4), dtype=np.float32), "prompt": None}}

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(Path(temp_dir) / "out", image_batch_size=2)
            _, rows = evaluate_method(
                method=BadBatchMethod(),
                samples=samples,
                config=config,
                track_name="track_a_mask_prompt",
                inference_mode=InferenceMode.BOX,
            )

            records = read_error_records(config)

        self.assertEqual(rows, [])
        self.assertEqual({record["sample_id"] for record in records}, {sample.sample_id for sample in samples})
        self.assertTrue(all(record["stage"] == "prompted_batch_prediction" for record in records))
        self.assertTrue(all("Batch prediction/sample_id mismatch" in record["error_message"] for record in records))

    def test_evaluate_method_logs_metric_shape_errors_and_continues(self):
        bad_gt_mask = np.ones((2, 2), dtype=np.float32)
        good_gt_mask = box_to_area([0, 0, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__inst_0", frame_id="frame_0", mask_array=bad_gt_mask),
            make_sample(sample_id="frame_1__inst_0", frame_id="frame_1", mask_array=good_gt_mask),
        ]

        class DummyBatchMethod:
            def predict_samples(self, batch):
                return {
                    sample.sample_id: {"mask": np.ones((4, 4), dtype=np.float32), "prompt": None}
                    for sample in batch
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(Path(temp_dir) / "out", image_batch_size=2)
            _, rows = evaluate_method(
                method=DummyBatchMethod(),
                samples=samples,
                config=config,
                track_name="track_a_mask_prompt",
                inference_mode=InferenceMode.BOX,
            )

            records = read_error_records(config)

        self.assertEqual([row["sample_id"] for row in rows], ["frame_1__inst_0"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "frame_0__inst_0")
        self.assertEqual(records[0]["stage"], "prompted_row_build")

    def test_evaluate_method_splits_cuda_oom_batches(self):
        mask = box_to_area([0, 0, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id=f"frame_{idx}__inst_0", frame_id=f"frame_{idx}", mask_array=mask)
            for idx in range(3)
        ]

        class Config(DummyConfig):
            class runtime:
                image_batch_size = 3
                batch_oom_fallback = True

        class OOMThenSingleMethod:
            def __init__(self):
                self.calls = []

            def predict_samples(self, batch):
                self.calls.append(len(batch))
                if len(batch) > 1:
                    raise RuntimeError("CUDA out of memory")
                sample = batch[0]
                return {sample.sample_id: {"mask": np.ones((4, 4), dtype=np.float32), "prompt": None}}

        method = OOMThenSingleMethod()
        _, rows = evaluate_method(
            method=method,
            samples=samples,
            config=Config(),
            track_name="track_a_mask_prompt",
            inference_mode=InferenceMode.BOX,
        )

        self.assertEqual([row["sample_id"] for row in rows], [sample.sample_id for sample in samples])
        self.assertEqual([row["BatchSize"] for row in rows], [1, 1, 1])
        self.assertEqual(method.calls, [3, 1, 2, 1, 1])

    def test_auto_mask_evaluates_once_per_image(self):
        mask_a = box_to_area([0, 0, 2, 2], 4, 4)
        mask_b = box_to_area([2, 2, 4, 4], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__inst_0", frame_id="frame_0", mask_array=mask_a, category="drone"),
            make_sample(sample_id="frame_0__inst_1", frame_id="frame_0", mask_array=mask_b, category="plane"),
        ]

        class DummyAutoMaskMethod:
            def __init__(self):
                self.calls = 0

            def predict_sample(self, sample):
                self.calls += 1
                return {
                    "instances": [
                        {"mask": mask_a, "score": 0.9},
                        {"mask": mask_b, "score": 0.8},
                    ]
                }

        method = DummyAutoMaskMethod()
        summary, rows = evaluate_method(
            method=method,
            samples=samples,
            config=None,
            track_name="track_b_auto_mask",
            inference_mode=InferenceMode.NO_PROMPT_AUTO_MASK,
        )

        self.assertEqual(method.calls, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "frame_0")
        self.assertEqual(rows[0]["eval_unit"], "image")
        self.assertEqual(rows[0]["supervision_type"], "mask")
        self.assertEqual(rows[0]["category_name"], "multiple")
        self.assertEqual(summary["instance_precision"], 1.0)
        self.assertEqual(summary["instance_recall"], 1.0)

    def test_auto_mask_prediction_errors_are_logged_and_skipped(self):
        mask = box_to_area([0, 0, 2, 2], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__inst_0", frame_id="frame_0", mask_array=mask),
            make_sample(sample_id="frame_0__inst_1", frame_id="frame_0", mask_array=mask),
        ]

        class FailingAutoMaskMethod:
            def predict_sample(self, sample):
                del sample
                raise ValueError("bad auto mask image")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(Path(temp_dir) / "out")
            _, rows = evaluate_method(
                method=FailingAutoMaskMethod(),
                samples=samples,
                config=config,
                track_name="track_b_auto_mask",
                inference_mode=InferenceMode.NO_PROMPT_AUTO_MASK,
            )

            records = read_error_records(config)

        self.assertEqual(rows, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "auto_mask_prediction")
        self.assertEqual(records[0]["group_sample_ids"], ["frame_0__inst_0", "frame_0__inst_1"])

if __name__ == "__main__":
    unittest.main()
