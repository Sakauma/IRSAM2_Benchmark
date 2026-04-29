#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def default_dnanet_repo() -> Path:
    benchmark_root = Path(__file__).resolve().parents[1]
    project_root = benchmark_root.parent
    return project_root / "external_repos" / "ir_std" / "DNANet"


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        extensions.append(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("At least one image extension is required.")
    return tuple(extensions)


def discover_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def frame_id_from_image(image_path: Path, image_root: Path) -> str:
    return image_path.relative_to(image_root).with_suffix("").as_posix()


def prediction_path_from_frame_id(dataset_dir: Path, frame_id: str, suffix: str = ".png") -> Path:
    return dataset_dir / Path(f"{frame_id}{suffix}")


def preprocess_image(image_path: Path, input_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((input_size, input_size), BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor, (height, width)


def binary_mask_from_logits(logits: torch.Tensor, threshold: float) -> np.ndarray:
    arr = logits.detach().squeeze().cpu().numpy()
    return ((arr > threshold).astype(np.uint8) * 255)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def load_dnanet(repo: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not repo.exists():
        raise FileNotFoundError(f"DNANet repo does not exist: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"DNANet checkpoint does not exist: {checkpoint}")
    repo_text = str(repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from model.load_param_data import load_param
    from model.model_DNANet import DNANet, Res_CBAM_block

    nb_filter, num_blocks = load_param("three", "resnet_18")
    model = DNANet(
        num_classes=1,
        input_channels=3,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision="True",
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_one(
    *,
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    input_size: int,
    threshold: float,
    device: torch.device,
) -> tuple[float, tuple[int, int]]:
    tensor, original_hw = preprocess_image(image_path, input_size)
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = F.interpolate(logits, size=original_hw, mode="bilinear", align_corners=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
    mask = binary_mask_from_logits(logits, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return latency_ms, original_hw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DNANet masks as external predictions for IRSAM2_Benchmark.")
    default_repo = default_dnanet_repo()
    parser.add_argument("--repo", type=Path, default=default_repo, help="Path to the DNANet repository.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to pretrain_DNANet_model.tar.")
    parser.add_argument("--dataset-id", required=True, help="Dataset id used by the benchmark, e.g. NUAA-SIRST.")
    parser.add_argument("--image-root", required=True, type=Path, help="Directory containing benchmark images.")
    parser.add_argument("--output-root", required=True, type=Path, help="Root directory for exported predictions.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images to export; 0 means all images.")
    parser.add_argument("--input-size", type=int, default=256, help="DNANet square input size.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Logit threshold used by the original DNANet test code.")
    parser.add_argument(
        "--image-extensions",
        default=".bmp,.png,.jpg,.jpeg,.tif,.tiff",
        help="Comma-separated image extensions to scan recursively.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = args.repo.expanduser().resolve()
    checkpoint = (args.checkpoint or (repo / "pretrain_DNANet_model.tar")).expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_dir = output_root / args.dataset_id
    manifest_path = dataset_dir / "manifest.jsonl"

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    device = resolve_device(args.device)
    images = discover_images(image_root, parse_extensions(args.image_extensions))
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {image_root}")

    print(f"[dnanet-export] loading model checkpoint={checkpoint} device={device}", flush=True)
    model = load_dnanet(repo, checkpoint, device)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(images)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, image_path in enumerate(images, start=1):
            frame_id = frame_id_from_image(image_path, image_root)
            output_path = prediction_path_from_frame_id(dataset_dir, frame_id)
            latency_ms, original_hw = export_one(
                model=model,
                image_path=image_path,
                output_path=output_path,
                input_size=int(args.input_size),
                threshold=float(args.threshold),
                device=device,
            )
            height, width = original_hw
            record = {
                "dataset_id": args.dataset_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "prediction_path": str(output_path),
                "original_width": width,
                "original_height": height,
                "input_size": int(args.input_size),
                "threshold": float(args.threshold),
                "latency_ms": latency_ms,
                "checkpoint": str(checkpoint),
                "model_name": "DNANet_resnet18_channel_three_deep_supervision",
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if index == total or index % 25 == 0:
                print(f"[dnanet-export] {index}/{total} exported last={frame_id} latency_ms={latency_ms:.2f}", flush=True)

    mean_latency = float(np.mean([record["latency_ms"] for record in records]))
    print(f"[dnanet-export] done count={total} output={dataset_dir} manifest={manifest_path} mean_latency_ms={mean_latency:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
