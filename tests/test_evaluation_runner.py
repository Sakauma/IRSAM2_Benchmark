import unittest
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

    def test_video_propagation_groups_by_track(self):
        mask = box_to_area([0, 0, 2, 2], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__track_a", frame_id="frame_0", frame_index=0, track_id="track_a", mask_array=mask),
            make_sample(sample_id="frame_1__track_a", frame_id="frame_1", frame_index=1, track_id="track_a", mask_array=mask),
            make_sample(sample_id="frame_0__track_b", frame_id="frame_0", frame_index=0, track_id="track_b", mask_array=mask),
            make_sample(sample_id="frame_1__track_b", frame_id="frame_1", frame_index=1, track_id="track_b", mask_array=mask),
        ]

        class DummyVideoMethod:
            def __init__(self):
                self.calls = []

            def predict_sequence(self, items):
                self.calls.append([(item.frame_id, item.track_id) for item in items])
                return {item.sample_id: np.asarray(item.mask_array, dtype=np.float32) for item in items}

        method = DummyVideoMethod()
        summary, rows = evaluate_method(
            method=method,
            samples=samples,
            config=None,
            track_name="track_c_video_propagation",
            inference_mode=InferenceMode.VIDEO_PROPAGATION,
        )

        self.assertEqual(len(method.calls), 2)
        self.assertEqual(method.calls[0], [("frame_0", "track_a"), ("frame_1", "track_a")])
        self.assertEqual(method.calls[1], [("frame_0", "track_b"), ("frame_1", "track_b")])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["track_id"] in {"track_a", "track_b"} for row in rows))
        self.assertIn("temporal_iou_mean", summary)

    def test_video_propagation_requires_track_id(self):
        mask = box_to_area([0, 0, 2, 2], 4, 4)
        samples = [
            make_sample(sample_id="frame_0__inst_0", frame_id="frame_0", frame_index=0, track_id=None, mask_array=mask),
        ]

        class DummyVideoMethod:
            def predict_sequence(self, items):
                return {item.sample_id: np.asarray(item.mask_array, dtype=np.float32) for item in items}

        with self.assertRaises(RuntimeError):
            evaluate_method(
                method=DummyVideoMethod(),
                samples=samples,
                config=None,
                track_name="track_c_video_propagation",
                inference_mode=InferenceMode.VIDEO_PROPAGATION,
            )


if __name__ == "__main__":
    unittest.main()
