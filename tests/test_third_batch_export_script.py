import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_third_batch_predictions.py"
SPEC = importlib.util.spec_from_file_location("export_third_batch_predictions", SCRIPT_PATH)
export_third_batch_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = export_third_batch_predictions
SPEC.loader.exec_module(export_third_batch_predictions)


class ThirdBatchExportScriptTests(unittest.TestCase):
    def test_discovers_images_and_preserves_nested_frame_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "seq"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "frame.001.bmp"
            Image.fromarray(np.zeros((3, 5), dtype=np.uint8), mode="L").save(image_path)

            images = export_third_batch_predictions.discover_images(root / "images", (".bmp",))
            frame_id = export_third_batch_predictions.frame_id_from_image(image_path, root / "images")

            self.assertEqual(images, [image_path])
            self.assertEqual(frame_id, "seq/frame.001")
            self.assertEqual(
                export_third_batch_predictions.prediction_path_from_frame_id(root / "predictions", frame_id),
                root / "predictions" / "seq" / "frame.001.png",
            )

    def test_bgm_preprocess_pads_and_preserves_original_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.png"
            Image.fromarray(np.full((3, 5), 10, dtype=np.uint8), mode="L").save(image_path)

            tensor, original_hw = export_third_batch_predictions.preprocess_bgm_image(
                image_path,
                dataset_id="unknown",
                mean=10.0,
                std=2.0,
            )

            self.assertEqual(tuple(tensor.shape), (1, 1, 32, 32))
            self.assertEqual(original_hw, (3, 5))
            self.assertTrue(torch.allclose(tensor[:, :, :3, :5], torch.zeros((1, 1, 3, 5))))

    def test_logits_to_binary_mask_crops_to_original_size(self):
        logits = torch.tensor([[[[0.1, 0.6, 0.2], [0.7, 0.1, 0.8], [0.9, 0.9, 0.9]]]])

        mask = export_third_batch_predictions.logits_to_binary_mask(logits, original_hw=(2, 3), threshold=0.5)

        self.assertEqual(mask.tolist(), [[0, 255, 0], [255, 0, 255]])

    def test_mask_to_prompt_box_uses_loose_box_protocol(self):
        mask = np.zeros((10, 12), dtype=np.float32)
        mask[4:6, 5:7] = 1.0

        tight = export_third_batch_predictions.mask_to_prompt_box(mask, width=12, height=10, variant="tight")
        loose = export_third_batch_predictions.mask_to_prompt_box(mask, width=12, height=10, variant="loose")

        self.assertEqual(tight, [5.0, 4.0, 7.0, 6.0])
        self.assertLessEqual(loose[0], tight[0])
        self.assertLessEqual(loose[1], tight[1])
        self.assertGreaterEqual(loose[2], tight[2])
        self.assertGreaterEqual(loose[3], tight[3])

    def test_find_mask_path_accepts_nuaa_pixels0_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_root = Path(temp_dir)
            mask_path = mask_root / "Misc_1_pixels0.png"
            Image.fromarray(np.zeros((3, 5), dtype=np.uint8), mode="L").save(mask_path)

            resolved = export_third_batch_predictions.find_mask_path(mask_root, "Misc_1", (".png",))

            self.assertEqual(resolved, mask_path)

    def test_multimodal_prompt_jobs_match_adapter_sample_id_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "img"
            label_root = root / "label"
            image_root.mkdir()
            label_root.mkdir()
            Image.fromarray(np.zeros((8, 10), dtype=np.uint8), mode="L").save(image_root / "frame_001.jpg")
            (label_root / "frame_001.json").write_text(
                json.dumps(
                    {
                        "detection": {
                            "instances": [
                                {"category": "drone", "mask": [[2, 2, 5, 2, 5, 5, 2, 5]]},
                                {"category": "bird", "mask": []},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            jobs = export_third_batch_predictions.build_multimodal_prompt_jobs(
                image_root=image_root,
                label_root=label_root,
                image_extensions=(".jpg",),
                max_images=0,
            )

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].frame_id, "frame_001")
            self.assertEqual(jobs[0].sample_id, "frame_001__inst_0::drone::polygon_mask")
            self.assertEqual(jobs[0].prompt_mask.shape, (8, 10))
            self.assertGreater(float(jobs[0].prompt_mask.sum()), 0.0)
            self.assertEqual(
                export_third_batch_predictions.safe_sample_filename(jobs[0].sample_id),
                "frame_001__inst_0__drone__polygon_mask.png",
            )

    def test_default_checkpoint_policy_records_fallback_source(self):
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("mshnet", "multimodal"),
            "IRSTD-1K",
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("drpcanet", "NUAA-SIRST"),
            "SIRSTv1",
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("rpcanet_pp", "NUDT-SIRST"),
            "NUDT-SIRST",
        )
        self.assertTrue(str(export_third_batch_predictions.default_checkpoint("hdnet", "NUAA-SIRST")).endswith("HDNet_IRSTD1k.pkl"))
        self.assertTrue(
            str(export_third_batch_predictions.default_repo("sam_vit_b")).endswith("sources/segment-anything")
        )
        self.assertTrue(str(export_third_batch_predictions.default_repo("hq_sam_vit_b")).endswith("sources/sam-hq"))
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("sam_vit_b", "NUAA-SIRST")).endswith(
                "checkpoints/segment-anything/sam_vit_b_01ec64.pth"
            )
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("hq_sam_vit_b", "NUAA-SIRST")).endswith(
                "checkpoints/sam-hq/sam_hq_vit_b.pth"
            )
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("sam_vit_b", "multimodal"),
            "general_sam_pretraining",
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("hq_sam_vit_b", "multimodal"),
            "general_sam_pretraining",
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_repo("sam2_unet_cod")).endswith("sources/SAM2-UNet")
        )
        self.assertTrue(str(export_third_batch_predictions.default_repo("serankdet")).endswith("sources/SeRankDet"))
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("sam2_unet_cod", "NUAA-SIRST")).endswith(
                "checkpoints/SAM2-UNet/COD/SAM2UNet-COD.pth"
            )
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("serankdet", "NUAA-SIRST")).endswith(
                "checkpoints/SeRankDet/NUAA-SIRST/SIRST_mIoU.pth.tar"
            )
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("sam2_unet_cod", "multimodal"),
            "COD",
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("serankdet", "multimodal"),
            "IRSTD-1K",
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_repo("pconv_mshnet_p43")).endswith("sources/PConv-SDloss")
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_repo("pconv_yolov8n_p2_p43_boxmask")).endswith("sources/PConv-SDloss")
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("pconv_mshnet_p43", "NUAA-SIRST")).endswith(
                "checkpoints/PConv-SDloss/MSHNet-IRSTD-1K-P43.zip"
            )
        )
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("pconv_yolov8n_p2_p43_boxmask", "NUAA-SIRST")).endswith(
                "checkpoints/PConv-SDloss/yolov8n-p2-IRSTD-1K-P43-0.5.zip"
            )
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("pconv_mshnet_p43", "multimodal"),
            "IRSTD-1K",
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("pconv_yolov8n_p2_p43_boxmask", "multimodal"),
            "IRSTD-1K",
        )
        self.assertTrue(str(export_third_batch_predictions.default_repo("uiu_net")).endswith("sources/UIU-Net"))
        self.assertTrue(
            str(export_third_batch_predictions.default_checkpoint("uiu_net", "NUAA-SIRST")).endswith(
                "checkpoints/UIU-Net/UIU-Net_saved_models.zip"
            )
        )
        self.assertEqual(
            export_third_batch_predictions.checkpoint_source_dataset("uiu_net", "multimodal"),
            "generic_uiu_net_release",
        )

    def test_serankdet_input_size_follows_dataset_configs(self):
        self.assertEqual(export_third_batch_predictions.serankdet_input_size("NUDT-SIRST"), 256)
        self.assertEqual(export_third_batch_predictions.serankdet_input_size("NUAA-SIRST"), 512)
        self.assertEqual(export_third_batch_predictions.serankdet_input_size("IRSTD-1K"), 512)
        self.assertEqual(export_third_batch_predictions.serankdet_input_size("multimodal"), 512)

    def test_method_threshold_defaults_preserve_efficientsam_logit_protocol(self):
        self.assertEqual(export_third_batch_predictions.resolve_threshold("bgm", None), 0.5)
        self.assertEqual(export_third_batch_predictions.resolve_threshold("fastsam", None), 0.5)
        self.assertEqual(export_third_batch_predictions.resolve_threshold("efficient_sam_vitt", None), 0.0)
        self.assertEqual(export_third_batch_predictions.resolve_threshold("pconv_yolov8n_p2_p43_boxmask", None), 0.25)
        self.assertEqual(export_third_batch_predictions.resolve_threshold("efficient_sam_vitt", 0.25), 0.25)

    def test_efficient_sam_box_prompt_tensors_use_box_corner_labels(self):
        points, labels = export_third_batch_predictions.efficient_sam_box_prompt_tensors(
            [1.0, 2.0, 5.0, 6.0],
            torch.device("cpu"),
        )

        self.assertEqual(tuple(points.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(labels.shape), (1, 1, 2))
        self.assertEqual(labels.tolist(), [[[2, 3]]])
        self.assertEqual(points.tolist(), [[[[1.0, 2.0], [5.0, 6.0]]]])

    def test_sam_predictor_branch_uses_plain_box_prompt(self):
        class FakePredictor:
            def __init__(self):
                self.kwargs = None

            def predict(self, **kwargs):
                self.kwargs = kwargs
                masks = np.stack(
                    [
                        np.zeros((4, 5), dtype=bool),
                        np.ones((4, 5), dtype=bool),
                    ]
                )
                return masks, np.asarray([0.1, 0.9]), None

        predictor = FakePredictor()

        mask = export_third_batch_predictions.predict_promptable_box_mask(
            method="sam_vit_b",
            model=predictor,
            image_state=None,
            prompt_box=[1.0, 2.0, 3.0, 4.0],
            width=5,
            height=4,
            threshold=0.5,
            device=torch.device("cpu"),
        )

        self.assertEqual(set(np.unique(mask).tolist()), {255})
        self.assertNotIn("hq_token_only", predictor.kwargs)
        self.assertEqual(predictor.kwargs["multimask_output"], True)
        self.assertEqual(predictor.kwargs["box"].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_hq_sam_predictor_branch_uses_official_quantitative_protocol(self):
        class FakePredictor:
            def __init__(self):
                self.kwargs = None

            def predict(self, **kwargs):
                self.kwargs = kwargs
                masks = np.stack(
                    [
                        np.zeros((4, 5), dtype=bool),
                        np.ones((4, 5), dtype=bool),
                    ]
                )
                return masks, np.asarray([0.2, 0.8]), None

        predictor = FakePredictor()

        mask = export_third_batch_predictions.predict_promptable_box_mask(
            method="hq_sam_vit_b",
            model=predictor,
            image_state=None,
            prompt_box=[1.0, 2.0, 3.0, 4.0],
            width=5,
            height=4,
            threshold=0.5,
            device=torch.device("cpu"),
        )

        self.assertEqual(set(np.unique(mask).tolist()), {255})
        self.assertEqual(predictor.kwargs["hq_token_only"], False)
        self.assertEqual(predictor.kwargs["multimask_output"], True)
        self.assertEqual(predictor.kwargs["box"].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_tensor_to_binary_mask_resizes_and_sigmoids_logits(self):
        logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])

        mask = export_third_batch_predictions.tensor_to_binary_mask(
            logits,
            original_hw=(4, 4),
            threshold=0.5,
            apply_sigmoid=True,
        )

        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})

    def test_sam2_unet_dense_export_uses_first_logits_and_original_size(self):
        class FakeSAM2UNet(torch.nn.Module):
            def forward(self, tensor):
                self.input_shape = tuple(tensor.shape)
                logits = torch.full((1, 1, 352, 352), 10.0)
                return logits, torch.zeros_like(logits), torch.zeros_like(logits)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            output_path = root / "out.png"
            Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8), mode="RGB").save(image_path)
            model = FakeSAM2UNet()

            latency_ms, original_hw = export_third_batch_predictions.export_local_dense_one(
                method="sam2_unet_cod",
                model=model,
                image_path=image_path,
                output_path=output_path,
                dataset_id="NUAA-SIRST",
                threshold=0.5,
                device=torch.device("cpu"),
            )

            self.assertEqual(model.input_shape, (1, 3, 352, 352))
            self.assertEqual(original_hw, (5, 7))
            self.assertGreaterEqual(latency_ms, 0.0)
            with Image.open(output_path) as output:
                arr = np.asarray(output)
            self.assertEqual(arr.shape, (5, 7))
            self.assertEqual(set(np.unique(arr).tolist()), {255})

    def test_serankdet_dense_export_uses_dataset_size_and_last_logits(self):
        class FakeSeRankDet(torch.nn.Module):
            def forward(self, tensor):
                self.input_shape = tuple(tensor.shape)
                negative = torch.full((1, 1, 256, 256), -10.0)
                positive = torch.full((1, 1, 256, 256), 10.0)
                return [negative, positive]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            output_path = root / "out.png"
            Image.fromarray(np.zeros((6, 4, 3), dtype=np.uint8), mode="RGB").save(image_path)
            model = FakeSeRankDet()

            latency_ms, original_hw = export_third_batch_predictions.export_local_dense_one(
                method="serankdet",
                model=model,
                image_path=image_path,
                output_path=output_path,
                dataset_id="NUDT-SIRST",
                threshold=0.5,
                device=torch.device("cpu"),
            )

            self.assertEqual(model.input_shape, (1, 3, 256, 256))
            self.assertEqual(original_hw, (6, 4))
            self.assertGreaterEqual(latency_ms, 0.0)
            with Image.open(output_path) as output:
                arr = np.asarray(output)
            self.assertEqual(arr.shape, (6, 4))
            self.assertEqual(set(np.unique(arr).tolist()), {255})

    def test_pconv_mshnet_dense_export_uses_mshnet_protocol(self):
        class FakePConvMSHNet(torch.nn.Module):
            def forward(self, tensor, warm_flag):
                self.input_shape = tuple(tensor.shape)
                self.warm_flag = warm_flag
                logits = torch.full((1, 1, 256, 256), 10.0)
                return [], logits

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            output_path = root / "out.png"
            Image.fromarray(np.zeros((9, 11, 3), dtype=np.uint8), mode="RGB").save(image_path)
            model = FakePConvMSHNet()

            latency_ms, original_hw = export_third_batch_predictions.export_local_dense_one(
                method="pconv_mshnet_p43",
                model=model,
                image_path=image_path,
                output_path=output_path,
                dataset_id="NUAA-SIRST",
                threshold=0.5,
                device=torch.device("cpu"),
            )

            self.assertEqual(model.input_shape, (1, 3, 256, 256))
            self.assertFalse(model.warm_flag)
            self.assertEqual(original_hw, (9, 11))
            self.assertGreaterEqual(latency_ms, 0.0)
            with Image.open(output_path) as output:
                arr = np.asarray(output)
            self.assertEqual(arr.shape, (9, 11))
            self.assertEqual(set(np.unique(arr).tolist()), {255})

    def test_yolo_boxmask_export_fills_predicted_boxes(self):
        class FakeBoxes:
            xyxy = torch.tensor([[1.2, 2.1, 4.8, 5.9]])

        class FakeResult:
            boxes = FakeBoxes()

        class FakeYOLO:
            def predict(self, image, conf, device, verbose):
                self.image_shape = image.shape
                self.conf = conf
                self.device = device
                self.verbose = verbose
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            output_path = root / "out.png"
            Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8), mode="RGB").save(image_path)
            model = FakeYOLO()

            latency_ms, original_hw = export_third_batch_predictions.export_local_dense_one(
                method="pconv_yolov8n_p2_p43_boxmask",
                model=model,
                image_path=image_path,
                output_path=output_path,
                dataset_id="NUAA-SIRST",
                threshold=0.25,
                device=torch.device("cpu"),
            )

            self.assertEqual(model.image_shape, (8, 10, 3))
            self.assertEqual(model.conf, 0.25)
            self.assertEqual(original_hw, (8, 10))
            self.assertGreaterEqual(latency_ms, 0.0)
            with Image.open(output_path) as output:
                arr = np.asarray(output)
            self.assertEqual(arr.shape, (8, 10))
            self.assertEqual(set(np.unique(arr).tolist()), {0, 255})
            self.assertTrue(np.all(arr[2:6, 1:5] == 255))

    def test_single_channel_conv_expansion_preserves_repeated_gray_response(self):
        conv = torch.nn.Conv2d(1, 2, kernel_size=(1, 3), padding=(0, 1), bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 1, 1, 3))
        expanded = export_third_batch_predictions.expand_single_channel_conv_to_rgb(conv)
        gray = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        rgb = gray.repeat(1, 3, 1, 1)

        self.assertEqual(expanded.in_channels, 3)
        self.assertTrue(torch.allclose(conv(gray), expanded(rgb), atol=1e-5))

    def test_zip_state_dict_loader_reads_weight_member(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "weights.zip"
            state_dict = {"layer.weight": torch.ones((1, 1))}
            payload_path = root / "weight.pkl"
            torch.save(state_dict, payload_path)
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(payload_path, "nested/weight.pkl")

            loaded = export_third_batch_predictions.load_zip_state_dict(zip_path, "weight.pkl")

            self.assertEqual(list(loaded), ["layer.weight"])
            self.assertTrue(torch.equal(loaded["layer.weight"], state_dict["layer.weight"]))

    def test_uiu_dense_export_uses_320_input_and_minmax_output(self):
        class FakeUIUNet(torch.nn.Module):
            def forward(self, tensor):
                self.input_shape = tuple(tensor.shape)
                logits = torch.zeros((1, 1, 320, 320))
                logits[:, :, 100:220, 100:220] = 10.0
                return logits, logits, logits, logits, logits, logits, logits

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.png"
            output_path = root / "out.png"
            Image.fromarray(np.full((7, 9, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
            model = FakeUIUNet()

            latency_ms, original_hw = export_third_batch_predictions.export_local_dense_one(
                method="uiu_net",
                model=model,
                image_path=image_path,
                output_path=output_path,
                dataset_id="NUAA-SIRST",
                threshold=0.5,
                device=torch.device("cpu"),
            )

            self.assertEqual(model.input_shape, (1, 3, 320, 320))
            self.assertEqual(original_hw, (7, 9))
            self.assertGreaterEqual(latency_ms, 0.0)
            with Image.open(output_path) as output:
                arr = np.asarray(output)
            self.assertEqual(arr.shape, (7, 9))
            self.assertEqual(set(np.unique(arr).tolist()), {0, 255})


if __name__ == "__main__":
    unittest.main()
