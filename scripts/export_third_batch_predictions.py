#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import json
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent
BENCHMARK_SRC = BENCHMARK_ROOT / "src"
WORKSPACE_ROOT = BENCHMARK_ROOT / "artifacts" / "comparison_workspace_20260429"
if str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SRC))

from irsam2_benchmark.data.masks import polygon_to_mask  # noqa: E402
from irsam2_benchmark.data.prompt_synthesis import clamp_box_xyxy, expand_box_xyxy, mask_to_tight_box  # noqa: E402


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

BGM_NORM_CONFIG: dict[str, dict[str, float]] = {
    "NUAA-SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST": {"mean": 107.80905151367188, "std": 33.02274703979492},
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
    "IRSTD-1k": {"mean": 87.4661865234375, "std": 39.71953201293945},
}

BGM_CHECKPOINTS: dict[str, Path] = {
    "IRSTD-1K": WORKSPACE_ROOT
    / "checkpoints"
    / "BGM"
    / "weights"
    / "BasicUNet_plus-2024-09-20-19-40-03-IRSTD-1K"
    / "best_mIoU_on_IRSTD-1K.pth.tar",
    "IRSTD-1k": WORKSPACE_ROOT
    / "checkpoints"
    / "BGM"
    / "weights"
    / "BasicUNet_plus-2024-09-20-19-40-03-IRSTD-1K"
    / "best_mIoU_on_IRSTD-1K.pth.tar",
    "NUAA-SIRST": WORKSPACE_ROOT
    / "checkpoints"
    / "BGM"
    / "weights"
    / "BasicUNet_plus-2024-09-20-21-17-03-NUAA-SIRST"
    / "best_mIoU_on_NUAA-SIRST.pth.tar",
    "NUDT-SIRST": WORKSPACE_ROOT
    / "checkpoints"
    / "BGM"
    / "weights"
    / "BasicUNet_plus-2024-09-20-21-47-11-NUDT-SIRST"
    / "best_mIoU_on_NUDT-SIRST.pth.tar",
}
BGM_CHECKPOINTS["multimodal"] = BGM_CHECKPOINTS["IRSTD-1K"]
BGM_CHECKPOINTS["MultiModal"] = BGM_CHECKPOINTS["IRSTD-1K"]

DENSE_MASK_METHODS = {
    "bgm",
    "sctransnet",
    "mshnet",
    "drpcanet",
    "rpcanet_pp",
    "hdnet",
    "sam2_unet_cod",
    "serankdet",
    "pconv_mshnet_p43",
    "uiu_net",
}
BOX_FILL_METHODS = {"pconv_yolov8n_p2_p43_boxmask"}
BOX_PROMPT_METHODS = {"fastsam", "mobile_sam", "efficient_sam_vitt", "edge_sam", "sam_vit_b", "hq_sam_vit_b"}
SUPPORTED_METHODS = sorted(DENSE_MASK_METHODS | BOX_FILL_METHODS | BOX_PROMPT_METHODS)

SCTRANSNET_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "SCTransNet" / "SCTransNet_SIRST3.pth"
MSHNET_CHECKPOINTS: dict[str, tuple[Path, str]] = {
    "NUDT-SIRST": (WORKSPACE_ROOT / "checkpoints" / "MSHNet" / "MSHNet_NUDT_SIRST.tar", "NUDT-SIRST"),
    "IRSTD-1K": (WORKSPACE_ROOT / "checkpoints" / "MSHNet" / "MSHNet_IRSTD1k_new.tar", "IRSTD-1K"),
    "IRSTD-1k": (WORKSPACE_ROOT / "checkpoints" / "MSHNet" / "MSHNet_IRSTD1k_new.tar", "IRSTD-1K"),
}
DRPCANET_CHECKPOINTS: dict[str, tuple[Path, str]] = {
    "NUDT-SIRST": (WORKSPACE_ROOT / "sources" / "DRPCA-Net" / "checkpoints" / "NUDTIRSTD_mIoU_94.16.pkl", "NUDT-SIRST"),
    "IRSTD-1K": (WORKSPACE_ROOT / "sources" / "DRPCA-Net" / "checkpoints" / "IRSTD1K_mIoU_64.14.pkl", "IRSTD-1K"),
    "IRSTD-1k": (WORKSPACE_ROOT / "sources" / "DRPCA-Net" / "checkpoints" / "IRSTD1K_mIoU_64.14.pkl", "IRSTD-1K"),
    "NUAA-SIRST": (WORKSPACE_ROOT / "sources" / "DRPCA-Net" / "checkpoints" / "SIRSTv1_mIoU_75.52.pkl", "SIRSTv1"),
}
RPCANET_PP_CHECKPOINTS: dict[str, tuple[Path, str]] = {
    "NUDT-SIRST": (WORKSPACE_ROOT / "sources" / "RPCANet" / "result" / "ISTD" / "NUDT" / "RPCANet++_s6.pkl", "NUDT-SIRST"),
    "IRSTD-1K": (WORKSPACE_ROOT / "sources" / "RPCANet" / "result" / "ISTD" / "1K" / "RPCANet++_s6.pkl", "IRSTD-1K"),
    "IRSTD-1k": (WORKSPACE_ROOT / "sources" / "RPCANet" / "result" / "ISTD" / "1K" / "RPCANet++_s6.pkl", "IRSTD-1K"),
    "NUAA-SIRST": (WORKSPACE_ROOT / "sources" / "RPCANet" / "result" / "ISTD" / "SIRST" / "RPCANet++_s6.pkl", "SIRST"),
}
HDNET_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "HDNet" / "HDNet_IRSTD1k.pkl"
MOBILE_SAM_CHECKPOINT = WORKSPACE_ROOT / "sources" / "MobileSAM" / "weights" / "mobile_sam.pt"
EFFICIENT_SAM_VITT_CHECKPOINT = WORKSPACE_ROOT / "sources" / "EfficientSAM" / "weights" / "efficient_sam_vitt.pt"
EDGESAM_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "EdgeSAM" / "edge_sam.pth"
SAM_VIT_B_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "segment-anything" / "sam_vit_b_01ec64.pth"
HQ_SAM_VIT_B_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "sam-hq" / "sam_hq_vit_b.pth"
SAM2_UNET_COD_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "SAM2-UNet" / "COD" / "SAM2UNet-COD.pth"
PCONV_MSHNET_P43_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "PConv-SDloss" / "MSHNet-IRSTD-1K-P43.zip"
PCONV_YOLOV8N_P2_P43_CHECKPOINT = (
    WORKSPACE_ROOT / "checkpoints" / "PConv-SDloss" / "yolov8n-p2-IRSTD-1K-P43-0.5.zip"
)
UIU_NET_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "UIU-Net" / "UIU-Net_saved_models.zip"
EXTRACTED_CHECKPOINT_ROOT = WORKSPACE_ROOT / "checkpoints_extracted"
SERANKDET_CHECKPOINTS: dict[str, tuple[Path, str]] = {
    "NUAA-SIRST": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "NUAA-SIRST" / "SIRST_mIoU.pth.tar", "NUAA-SIRST"),
    "NUDT-SIRST": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "NUDT-SIRST" / "NUDT_mIoU.pth.tar", "NUDT-SIRST"),
    "IRSTD-1K": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "IRSTD-1K" / "IRSTDK_mIoU.pth.tar", "IRSTD-1K"),
    "IRSTD-1k": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "IRSTD-1K" / "IRSTDK_mIoU.pth.tar", "IRSTD-1K"),
    "multimodal": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "IRSTD-1K" / "IRSTDK_mIoU.pth.tar", "IRSTD-1K"),
    "MultiModal": (WORKSPACE_ROOT / "checkpoints" / "SeRankDet" / "IRSTD-1K" / "IRSTDK_mIoU.pth.tar", "IRSTD-1K"),
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PromptJob:
    image_path: Path
    frame_id: str
    sample_id: str
    polygon: tuple[float, ...]
    width: int
    height: int
    category: str
    instance_index: int

    @property
    def prompt_mask(self) -> np.ndarray:
        return polygon_to_mask(self.polygon, height=self.height, width=self.width).astype(np.float32)


def default_repo(method: str) -> Path:
    if method == "bgm":
        return WORKSPACE_ROOT / "sources" / "BGM"
    if method == "fastsam":
        return WORKSPACE_ROOT / "sources" / "FastSAM"
    if method == "sctransnet":
        return WORKSPACE_ROOT / "sources" / "SCTransNet"
    if method == "mshnet":
        return WORKSPACE_ROOT / "sources" / "MSHNet"
    if method == "drpcanet":
        return WORKSPACE_ROOT / "sources" / "DRPCA-Net"
    if method == "rpcanet_pp":
        return WORKSPACE_ROOT / "sources" / "RPCANet"
    if method == "hdnet":
        return WORKSPACE_ROOT / "sources" / "HDNet"
    if method == "mobile_sam":
        return WORKSPACE_ROOT / "sources" / "MobileSAM"
    if method == "efficient_sam_vitt":
        return WORKSPACE_ROOT / "sources" / "EfficientSAM"
    if method == "edge_sam":
        return WORKSPACE_ROOT / "sources" / "EdgeSAM"
    if method == "sam_vit_b":
        return WORKSPACE_ROOT / "sources" / "segment-anything"
    if method == "hq_sam_vit_b":
        return WORKSPACE_ROOT / "sources" / "sam-hq"
    if method == "sam2_unet_cod":
        return WORKSPACE_ROOT / "sources" / "SAM2-UNet"
    if method == "serankdet":
        return WORKSPACE_ROOT / "sources" / "SeRankDet"
    if method in {"pconv_mshnet_p43", "pconv_yolov8n_p2_p43_boxmask"}:
        return WORKSPACE_ROOT / "sources" / "PConv-SDloss"
    if method == "uiu_net":
        return WORKSPACE_ROOT / "sources" / "UIU-Net"
    raise ValueError(f"Unsupported method: {method}")


def default_checkpoint(method: str, dataset_id: str) -> Path | None:
    if method == "bgm":
        return BGM_CHECKPOINTS.get(dataset_id)
    if method == "fastsam":
        fastsam_x = WORKSPACE_ROOT / "checkpoints" / "FastSAM" / "FastSAM.pt"
        fastsam_s = WORKSPACE_ROOT / "checkpoints" / "FastSAM" / "FastSAM-s.pt"
        return fastsam_x if fastsam_x.exists() else fastsam_s
    if method == "sctransnet":
        return SCTRANSNET_CHECKPOINT
    if method == "mshnet":
        return MSHNET_CHECKPOINTS.get(dataset_id, MSHNET_CHECKPOINTS["IRSTD-1K"])[0]
    if method == "drpcanet":
        return DRPCANET_CHECKPOINTS.get(dataset_id, DRPCANET_CHECKPOINTS["IRSTD-1K"])[0]
    if method == "rpcanet_pp":
        return RPCANET_PP_CHECKPOINTS.get(dataset_id, RPCANET_PP_CHECKPOINTS["IRSTD-1K"])[0]
    if method == "hdnet":
        return HDNET_CHECKPOINT
    if method == "mobile_sam":
        return MOBILE_SAM_CHECKPOINT
    if method == "efficient_sam_vitt":
        return EFFICIENT_SAM_VITT_CHECKPOINT
    if method == "edge_sam":
        return EDGESAM_CHECKPOINT
    if method == "sam_vit_b":
        return SAM_VIT_B_CHECKPOINT
    if method == "hq_sam_vit_b":
        return HQ_SAM_VIT_B_CHECKPOINT
    if method == "sam2_unet_cod":
        return SAM2_UNET_COD_CHECKPOINT
    if method == "serankdet":
        return SERANKDET_CHECKPOINTS.get(dataset_id, SERANKDET_CHECKPOINTS["IRSTD-1K"])[0]
    if method == "pconv_mshnet_p43":
        return PCONV_MSHNET_P43_CHECKPOINT
    if method == "pconv_yolov8n_p2_p43_boxmask":
        return PCONV_YOLOV8N_P2_P43_CHECKPOINT
    if method == "uiu_net":
        return UIU_NET_CHECKPOINT
    raise ValueError(f"Unsupported method: {method}")


def checkpoint_source_dataset(method: str, dataset_id: str, checkpoint_was_overridden: bool = False) -> str:
    if checkpoint_was_overridden:
        return "custom"
    if method == "bgm":
        return "IRSTD-1K" if dataset_id in {"multimodal", "MultiModal"} else dataset_id
    if method == "fastsam":
        return "SA-1B/general"
    if method == "sctransnet":
        return "SIRST3"
    if method == "mshnet":
        return MSHNET_CHECKPOINTS.get(dataset_id, MSHNET_CHECKPOINTS["IRSTD-1K"])[1]
    if method == "drpcanet":
        return DRPCANET_CHECKPOINTS.get(dataset_id, DRPCANET_CHECKPOINTS["IRSTD-1K"])[1]
    if method == "rpcanet_pp":
        return RPCANET_PP_CHECKPOINTS.get(dataset_id, RPCANET_PP_CHECKPOINTS["IRSTD-1K"])[1]
    if method == "hdnet":
        return "IRSTD-1K"
    if method == "sam2_unet_cod":
        return "COD"
    if method == "serankdet":
        return SERANKDET_CHECKPOINTS.get(dataset_id, SERANKDET_CHECKPOINTS["IRSTD-1K"])[1]
    if method in {"pconv_mshnet_p43", "pconv_yolov8n_p2_p43_boxmask"}:
        return "IRSTD-1K"
    if method == "uiu_net":
        return "generic_uiu_net_release"
    if method in {"mobile_sam", "efficient_sam_vitt", "edge_sam", "sam_vit_b", "hq_sam_vit_b"}:
        return "general_sam_pretraining"
    raise ValueError(f"Unsupported method: {method}")


def resolve_threshold(method: str, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if method == "efficient_sam_vitt":
        return 0.0
    if method == "pconv_yolov8n_p2_p43_boxmask":
        return 0.25
    return 0.5


def serankdet_input_size(dataset_id: str) -> int:
    return 256 if dataset_id == "NUDT-SIRST" else 512


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        extensions.append(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("At least one extension is required.")
    return tuple(extensions)


def discover_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def frame_id_from_image(image_path: Path, image_root: Path) -> str:
    return image_path.relative_to(image_root).with_suffix("").as_posix()


def prediction_path_from_frame_id(dataset_dir: Path, frame_id: str, suffix: str = ".png") -> Path:
    return dataset_dir / Path(f"{frame_id}{suffix}")


def safe_sample_filename(sample_id: str, suffix: str = ".png") -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in sample_id)
    return f"{safe}{suffix}"


def image_index_by_stem(root: Path, extensions: Iterable[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in discover_images(root, extensions):
        index.setdefault(path.stem, path)
        index.setdefault(path.stem.lower(), path)
    return index


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_multimodal_prompt_jobs(
    *,
    image_root: Path,
    label_root: Path,
    image_extensions: Iterable[str],
    max_images: int,
) -> list[PromptJob]:
    image_index = image_index_by_stem(image_root, image_extensions)
    jobs: list[PromptJob] = []
    seen_images: set[str] = set()
    for label_path in sorted(label_root.glob("*.json")):
        stem = label_path.stem
        if max_images > 0 and len(seen_images) >= max_images and stem not in seen_images:
            break
        image_path = image_index.get(stem) or image_index.get(stem.lower())
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        data = read_json(label_path)
        detection = data.get("detection") if isinstance(data, dict) else None
        instances = detection.get("instances", []) if isinstance(detection, dict) else []
        added_for_image = False
        for inst_idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            masks = inst.get("mask", [])
            polygon = masks[0] if masks and len(masks[0]) >= 6 else None
            if polygon is None:
                continue
            polygon_values = tuple(float(value) for value in polygon)
            if polygon_to_tight_box(polygon_values) is None:
                continue
            category = str(inst.get("category", "unknown"))
            jobs.append(
                PromptJob(
                    image_path=image_path,
                    frame_id=stem,
                    sample_id=f"{stem}__inst_{inst_idx}::{category}::polygon_mask",
                    polygon=polygon_values,
                    width=width,
                    height=height,
                    category=category,
                    instance_index=inst_idx,
                )
            )
            added_for_image = True
        if added_for_image:
            seen_images.add(stem)
    return jobs


def find_mask_path(mask_root: Path, frame_id: str, extensions: Iterable[str]) -> Path:
    rel = Path(frame_id)
    candidates = []
    for ext in extensions:
        candidates.append(mask_root / Path(f"{rel.as_posix()}{ext}"))
        candidates.append(mask_root / Path(f"{rel.as_posix()}_pixels0{ext}"))
    if rel.suffix:
        candidates.append(mask_root / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"GT mask not found for frame_id={frame_id!r}. Tried: {tried}")


def read_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)


def mask_to_prompt_box(mask: np.ndarray, width: int, height: int, variant: str) -> list[float]:
    tight = mask_to_tight_box(mask)
    if variant == "tight":
        return clamp_box_xyxy(tight, width=width, height=height)
    if variant == "loose":
        return expand_box_xyxy(tight, width=width, height=height)
    raise ValueError(f"Unsupported prompt box variant: {variant}")


def polygon_to_tight_box(polygon: Iterable[float]) -> list[float] | None:
    coords = [float(value) for value in polygon]
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def polygon_to_prompt_box(polygon: Iterable[float], width: int, height: int, variant: str) -> list[float]:
    tight = polygon_to_tight_box(polygon)
    if tight is None:
        return [0.0, 0.0, 1.0, 1.0]
    if variant == "tight":
        return clamp_box_xyxy(tight, width=width, height=height)
    if variant == "loose":
        return expand_box_xyxy(tight, width=width, height=height)
    raise ValueError(f"Unsupported prompt box variant: {variant}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def prepare_repo_import(repo: Path, prefixes: Iterable[str]) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"Third-party repo does not exist: {repo}")
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
    repo_text = str(repo.resolve())
    sys.path = [item for item in sys.path if item != repo_text]
    sys.path.insert(0, repo_text)


class _UnavailableOptionalDependency:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("This optional EdgeSAM RPN dependency is unavailable and should not be used for plain box-prompt inference.")


def install_edgesam_optional_dependency_stubs() -> None:
    dense_heads = types.ModuleType("mmdet.models.dense_heads")
    dense_heads.RPNHead = _UnavailableOptionalDependency
    dense_heads.CenterNetUpdateHead = _UnavailableOptionalDependency
    necks = types.ModuleType("mmdet.models.necks")
    necks.FPN = _UnavailableOptionalDependency
    models = types.ModuleType("mmdet.models")
    models.dense_heads = dense_heads
    models.necks = necks
    mmdet = types.ModuleType("mmdet")
    mmdet.models = models

    efficientdet = types.SimpleNamespace(
        BiFPN=_UnavailableOptionalDependency,
        EfficientDetSepBNHead=_UnavailableOptionalDependency,
    )
    efficient_det_pkg = types.ModuleType("projects.EfficientDet")
    efficient_det_pkg.efficientdet = efficientdet
    projects = types.ModuleType("projects")
    projects.EfficientDet = efficient_det_pkg

    class ConfigDict(dict):
        pass

    mmengine = types.ModuleType("mmengine")
    mmengine.ConfigDict = ConfigDict

    sys.modules.setdefault("mmdet", mmdet)
    sys.modules.setdefault("mmdet.models", models)
    sys.modules.setdefault("mmdet.models.dense_heads", dense_heads)
    sys.modules.setdefault("mmdet.models.necks", necks)
    sys.modules.setdefault("projects", projects)
    sys.modules.setdefault("projects.EfficientDet", efficient_det_pkg)
    sys.modules.setdefault("mmengine", mmengine)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def conv_autopad(k: int | tuple[int, int], p: int | tuple[int, int] | None = None, d: int = 1) -> int | tuple[int, int]:
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (value - 1) + 1 for value in k)
    if p is None:
        return k // 2 if isinstance(k, int) else tuple(value // 2 for value in k)
    return p


class _PConvInnerConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 1,
        s: int = 1,
        p: int | tuple[int, int] | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, conv_autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _PinwheelConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int) -> None:
        super().__init__()
        pads = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList(nn.ZeroPad2d(padding=pad) for pad in pads)
        self.cw = _PConvInnerConv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = _PConvInnerConv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = _PConvInnerConv(c2, c2, 2, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


def register_ultralytics_pconv() -> None:
    import importlib
    import ultralytics
    import ultralytics.nn.modules.conv as ultralytics_conv

    module_aliases = {
        "ultralytics.utils": "ultralytics.yolo.utils",
        "ultralytics.utils.loss": "ultralytics.yolo.utils.loss",
        "ultralytics.utils.tal": "ultralytics.yolo.utils.tal",
    }
    for alias, target in module_aliases.items():
        module = importlib.import_module(target)
        sys.modules.setdefault(alias, module)
        if alias == "ultralytics.utils":
            setattr(ultralytics, "utils", module)

    if not hasattr(ultralytics_conv, "PConv"):
        setattr(ultralytics_conv, "PConv", _PinwheelConv)


def expand_single_channel_conv_to_rgb(conv: nn.Conv2d) -> nn.Conv2d:
    if conv.in_channels != 1:
        return conv
    expanded = nn.Conv2d(
        in_channels=3,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    expanded = expanded.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        expanded.weight.copy_(conv.weight.repeat(1, 3, 1, 1) / 3.0)
        if conv.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(conv.bias)
    return expanded


def adapt_yolov8_pconv_first_layer_to_rgb(model: object) -> None:
    detection_model = getattr(model, "model", None)
    layers = getattr(detection_model, "model", None)
    if not layers:
        return
    first_layer = layers[0]
    for branch_name in ("cw", "ch"):
        branch = getattr(first_layer, branch_name, None)
        conv = getattr(branch, "conv", None)
        if isinstance(conv, nn.Conv2d):
            branch.conv = expand_single_channel_conv_to_rgb(conv)


class _PConvMSHNetFirstBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _PinwheelConv(in_channels, out_channels, 4, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _PinwheelConv(out_channels, out_channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module | None = None
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.ca = _PConvChannelAttention(out_channels)
        self.sa = _PConvSpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        return self.relu(out)


class _PConvChannelAttention(nn.Module):
    def __init__(self, in_planes: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class _PConvSpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


def load_zip_state_dict(checkpoint: Path, member_suffix: str) -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(checkpoint) as archive:
        matches = [name for name in archive.namelist() if name.endswith(member_suffix)]
        if not matches:
            raise FileNotFoundError(f"No zip member ending with {member_suffix!r} in {checkpoint}")
        payload = torch.load(BytesIO(archive.read(matches[0])), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, dict) and "net" in payload and isinstance(payload["net"], dict):
        payload = payload["net"]
    if not isinstance(payload, dict):
        raise TypeError(f"Zip member {member_suffix!r} in {checkpoint} did not contain a state dict.")
    return payload


def materialize_zip_member(checkpoint: Path, member_suffix: str) -> Path:
    with zipfile.ZipFile(checkpoint) as archive:
        matches = [name for name in archive.namelist() if name.endswith(member_suffix)]
        if not matches:
            raise FileNotFoundError(f"No zip member ending with {member_suffix!r} in {checkpoint}")
        member = matches[0]
        output_path = EXTRACTED_CHECKPOINT_ROOT / checkpoint.stem / member
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(archive.read(member))
        return output_path


def pad_to_multiple(arr: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = arr.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
    return arr, (height, width)


def preprocess_bgm_image(image_path: Path, dataset_id: str, mean: float | None, std: float | None) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("I"), dtype=np.float32)
    norm = BGM_NORM_CONFIG.get(dataset_id)
    if mean is None:
        mean = norm["mean"] if norm is not None else float(arr.mean())
    if std is None:
        std = norm["std"] if norm is not None else float(arr.std())
    std = max(float(std), 1e-6)
    arr = (arr - float(mean)) / std
    arr, original_hw = pad_to_multiple(arr, multiple=32)
    tensor = torch.from_numpy(arr[None, None, :, :]).contiguous()
    return tensor, original_hw


def logits_to_binary_mask(logits: torch.Tensor, original_hw: tuple[int, int], threshold: float) -> np.ndarray:
    height, width = original_hw
    arr = logits.detach().squeeze().cpu().numpy()
    arr = arr[:height, :width]
    return ((arr > threshold).astype(np.uint8) * 255)


def tensor_to_binary_mask(
    logits: torch.Tensor,
    *,
    original_hw: tuple[int, int],
    threshold: float,
    apply_sigmoid: bool,
) -> np.ndarray:
    height, width = original_hw
    if logits.shape[-2:] != (height, width):
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    if apply_sigmoid:
        logits = torch.sigmoid(logits)
    arr = logits.detach().squeeze().cpu().numpy()
    return ((arr > threshold).astype(np.uint8) * 255)


def yolo_boxes_to_binary_mask(result: object, *, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return mask
    xyxy = boxes.xyxy.detach().cpu().numpy() if isinstance(boxes.xyxy, torch.Tensor) else np.asarray(boxes.xyxy)
    for raw_box in xyxy:
        x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
        left = int(max(0, min(width, np.floor(x1))))
        top = int(max(0, min(height, np.floor(y1))))
        right = int(max(0, min(width, np.ceil(x2))))
        bottom = int(max(0, min(height, np.ceil(y2))))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
    return mask


def preprocess_rgb_square_image(image_path: Path, input_size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((input_size, input_size), BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None, :, :, :]).contiguous()
    return tensor, (height, width)


def preprocess_uiu_image(image_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((320, 320), BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    max_value = max(float(arr.max()), 1e-6)
    arr = arr / max_value
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None, :, :, :]).contiguous()
    return tensor, (height, width)


def preprocess_gray_square_image(image_path: Path, input_size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        resized = gray.resize((input_size, input_size), BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr[None, None, :, :]).contiguous()
    return tensor, (height, width)


def fastsam_masks_to_binary(masks: object, width: int, height: int, threshold: float) -> np.ndarray:
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    if arr.shape != (height, width):
        arr = np.asarray(Image.fromarray(arr.astype(np.float32)).resize((width, height), NEAREST))
    return ((arr > threshold).astype(np.uint8) * 255)


def load_bgm_model(repo: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not repo.exists():
        raise FileNotFoundError(f"BGM repo does not exist: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"BGM checkpoint does not exist: {checkpoint}")
    prepare_repo_import(repo, prefixes=("net",))
    from net import Net

    model = Net("BasicUNet_plus")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_fastsam_model(repo: Path, checkpoint: Path):
    if not repo.exists():
        raise FileNotFoundError(f"FastSAM repo does not exist: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"FastSAM checkpoint does not exist: {checkpoint}")
    prepare_repo_import(repo, prefixes=("fastsam",))
    from fastsam import FastSAM, FastSAMPrompt

    return FastSAM(str(checkpoint)), FastSAMPrompt


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


def load_local_dense_model(method: str, repo: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"{method} checkpoint does not exist: {checkpoint}")
    if method == "sctransnet":
        prepare_repo_import(repo, prefixes=("model", "dataset", "metrics", "utils", "warmup_scheduler"))
        import model.Config as config
        from model.SCTransNet import SCTransNet

        model = SCTransNet(config.get_SCTrans_config(), mode="test", deepsuper=True)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "mshnet":
        prepare_repo_import(repo, prefixes=("model", "utils"))
        from model.MSHNet import MSHNet

        model = MSHNet(3)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "pconv_mshnet_p43":
        mshnet_repo = WORKSPACE_ROOT / "sources" / "MSHNet"
        prepare_repo_import(mshnet_repo, prefixes=("model", "utils"))
        from model.MSHNet import MSHNet

        model = MSHNet(3)
        model.encoder_0[0] = _PConvMSHNetFirstBlock(16, 16)
        state_dict = load_zip_state_dict(checkpoint, "weight.pkl")
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "hdnet":
        if device.type != "cuda":
            raise RuntimeError("HDNet vendor code initializes CUDA tensors at import time; run it with --device cuda or cuda:0.")
        prepare_repo_import(repo, prefixes=("model", "utils"))
        from model.HDNet import HDNet

        model = HDNet(3)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method in {"drpcanet", "rpcanet_pp"}:
        prepare_repo_import(repo, prefixes=("models", "utils", "run_config"))
        from models import get_model

        model_name = "Drpcanet" if method == "drpcanet" else "rpcanet_pp"
        model = get_model(model_name)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "sam2_unet_cod":
        if device.type != "cuda":
            raise RuntimeError("SAM2-UNet vendor code builds SAM2 on CUDA by default; run it with --device cuda or cuda:0.")
        torch.cuda.set_device(device)
        prepare_repo_import(repo, prefixes=("SAM2UNet", "sam2", "sam2_configs"))
        from SAM2UNet import SAM2UNet

        model = SAM2UNet()
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "serankdet":
        prepare_repo_import(repo, prefixes=("model", "build", "utils"))
        from model.build_segmentor import Model

        cfg = types.SimpleNamespace(
            model={
                "name": "UNet",
                "type": "EncoderDecoder",
                "pretrained": None,
                "backbone": {"type": None},
                "decode_head": {"type": "SeRankDet", "deep_supervision": True},
                "loss": {"type": "SoftIoULoss"},
            }
        )
        model = Model(cfg)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    elif method == "pconv_yolov8n_p2_p43_boxmask":
        register_ultralytics_pconv()
        from ultralytics import YOLO

        best_pt = materialize_zip_member(checkpoint, "weights/best.pt")
        model = YOLO(str(best_pt))
        adapt_yolov8_pconv_first_layer_to_rgb(model)
        return model
    elif method == "uiu_net":
        prepare_repo_import(repo, prefixes=("model", "data_loader", "utils"))
        from model import UIUNET

        model = UIUNET(3, 1)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    else:
        raise ValueError(f"Unsupported dense local method: {method}")
    model.to(device)
    model.eval()
    return model


def load_box_prompt_model(method: str, repo: Path, checkpoint: Path, device: torch.device) -> Any:
    if not checkpoint.exists():
        raise FileNotFoundError(f"{method} checkpoint does not exist: {checkpoint}")
    if method == "mobile_sam":
        prepare_repo_import(repo, prefixes=("mobile_sam",))
        from mobile_sam import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)
    if method == "edge_sam":
        prepare_repo_import(repo, prefixes=("edge_sam",))
        install_edgesam_optional_dependency_stubs()
        from edge_sam import SamPredictor, sam_model_registry

        sam = sam_model_registry["edge_sam"](checkpoint=str(checkpoint))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)
    if method == "efficient_sam_vitt":
        prepare_repo_import(repo, prefixes=("efficient_sam",))
        from efficient_sam.efficient_sam import build_efficient_sam

        model = build_efficient_sam(encoder_patch_embed_dim=192, encoder_num_heads=3, checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        return model
    if method in {"sam_vit_b", "hq_sam_vit_b"}:
        prepare_repo_import(repo, prefixes=("segment_anything",))
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)
    raise ValueError(f"Unsupported box-prompt local method: {method}")


def export_bgm_one(
    *,
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    dataset_id: str,
    threshold: float,
    device: torch.device,
    mean: float | None,
    std: float | None,
) -> tuple[float, tuple[int, int]]:
    tensor, original_hw = preprocess_bgm_image(image_path, dataset_id=dataset_id, mean=mean, std=std)
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        synchronize_if_cuda(device)
        start = time.perf_counter()
        logits = model(tensor)
        synchronize_if_cuda(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
    mask = logits_to_binary_mask(logits, original_hw, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return latency_ms, original_hw


def export_local_dense_one(
    *,
    method: str,
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    dataset_id: str,
    threshold: float,
    device: torch.device,
) -> tuple[float, tuple[int, int]]:
    if method == "pconv_yolov8n_p2_p43_boxmask":
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            rgb_np = np.asarray(rgb, dtype=np.uint8)
        synchronize_if_cuda(device)
        start = time.perf_counter()
        results = model.predict(rgb_np, conf=threshold, device=str(device), verbose=False)
        synchronize_if_cuda(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        mask = yolo_boxes_to_binary_mask(results[0], width=width, height=height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(output_path)
        return latency_ms, (height, width)
    if method == "sctransnet":
        tensor, original_hw = preprocess_bgm_image(image_path, dataset_id=dataset_id, mean=None, std=None)
    elif method in {"mshnet", "hdnet", "pconv_mshnet_p43"}:
        tensor, original_hw = preprocess_rgb_square_image(image_path, input_size=256)
    elif method in {"drpcanet", "rpcanet_pp"}:
        tensor, original_hw = preprocess_gray_square_image(image_path, input_size=256)
    elif method == "sam2_unet_cod":
        tensor, original_hw = preprocess_rgb_square_image(image_path, input_size=352)
    elif method == "serankdet":
        tensor, original_hw = preprocess_rgb_square_image(image_path, input_size=serankdet_input_size(dataset_id))
    elif method == "uiu_net":
        tensor, original_hw = preprocess_uiu_image(image_path)
    else:
        raise ValueError(f"Unsupported dense local method: {method}")
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        synchronize_if_cuda(device)
        start = time.perf_counter()
        if method in {"mshnet", "hdnet", "pconv_mshnet_p43"}:
            _, logits = model(tensor, False)
            mask = tensor_to_binary_mask(logits, original_hw=original_hw, threshold=threshold, apply_sigmoid=True)
        elif method in {"drpcanet", "rpcanet_pp"}:
            _, logits = model(tensor)
            mask = tensor_to_binary_mask(logits, original_hw=original_hw, threshold=threshold, apply_sigmoid=True)
        elif method == "sam2_unet_cod":
            logits, _, _ = model(tensor)
            mask = tensor_to_binary_mask(logits, original_hw=original_hw, threshold=threshold, apply_sigmoid=True)
        elif method == "serankdet":
            outputs = model(tensor)
            logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            mask = tensor_to_binary_mask(logits, original_hw=original_hw, threshold=threshold, apply_sigmoid=True)
        elif method == "uiu_net":
            outputs = model(tensor)
            logits = outputs[0][:, 0:1, :, :] if isinstance(outputs, (list, tuple)) else outputs[:, 0:1, :, :]
            pred = logits.detach()
            pred = (pred - pred.amin(dim=(-2, -1), keepdim=True)) / (
                pred.amax(dim=(-2, -1), keepdim=True) - pred.amin(dim=(-2, -1), keepdim=True) + 1e-6
            )
            mask = tensor_to_binary_mask(pred, original_hw=original_hw, threshold=threshold, apply_sigmoid=False)
        else:
            logits = model(tensor)
            mask = tensor_to_binary_mask(logits, original_hw=original_hw, threshold=threshold, apply_sigmoid=False)
        synchronize_if_cuda(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return latency_ms, original_hw


def export_fastsam_one(
    *,
    model,
    prompt_cls,
    image_path: Path,
    mask_path: Path | None,
    prompt_mask: np.ndarray | None,
    output_path: Path,
    prompt_box_variant: str,
    imgsz: int,
    conf: float,
    iou: float,
    retina_masks: bool,
    threshold: float,
    device: torch.device,
) -> tuple[float, tuple[int, int], list[float]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    gt_mask = prompt_mask if prompt_mask is not None else read_binary_mask(mask_path)  # type: ignore[arg-type]
    prompt_box = mask_to_prompt_box(gt_mask, width=width, height=height, variant=prompt_box_variant)
    with torch.no_grad():
        synchronize_if_cuda(device)
        start = time.perf_counter()
        everything_results = model(
            rgb,
            device=str(device),
            retina_masks=retina_masks,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        prompt = prompt_cls(rgb, everything_results, device=str(device))
        masks = prompt.box_prompt(bboxes=[prompt_box])
        synchronize_if_cuda(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
    mask = fastsam_masks_to_binary(masks, width=width, height=height, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return latency_ms, (height, width), prompt_box


def export_fastsam_prompt_jobs(
    *,
    model,
    prompt_cls,
    prompt_jobs: list[PromptJob],
    dataset_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    checkpoint_source: str,
    dataset_id: str,
    prompt_box_variant: str,
    imgsz: int,
    conf: float,
    iou: float,
    retina_masks: bool,
    threshold: float,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    jobs = sorted(prompt_jobs, key=lambda item: (str(item.image_path), item.instance_index))
    image_groups = [(image_path, list(group)) for image_path, group in groupby(jobs, key=lambda item: item.image_path)]
    total_images = len(image_groups)
    total_jobs = len(jobs)
    completed_jobs = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for image_index, (image_path, grouped_jobs) in enumerate(image_groups, start=1):
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
            with torch.no_grad():
                synchronize_if_cuda(device)
                inference_start = time.perf_counter()
                everything_results = model(
                    rgb,
                    device=str(device),
                    retina_masks=retina_masks,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                )
                prompt = prompt_cls(rgb, everything_results, device=str(device))
                synchronize_if_cuda(device)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
            shared_inference_ms = inference_ms / max(1, len(grouped_jobs))
            for job in grouped_jobs:
                prompt_box = polygon_to_prompt_box(job.polygon, width=width, height=height, variant=prompt_box_variant)
                prompt_start = time.perf_counter()
                masks = prompt.box_prompt(bboxes=[prompt_box])
                synchronize_if_cuda(device)
                prompt_ms = (time.perf_counter() - prompt_start) * 1000.0
                latency_ms = shared_inference_ms + prompt_ms
                output_path = dataset_dir / "samples" / safe_sample_filename(job.sample_id)
                mask = fastsam_masks_to_binary(masks, width=width, height=height, threshold=threshold)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask, mode="L").save(output_path)
                record: dict[str, object] = {
                    "dataset_id": dataset_id,
                    "frame_id": job.frame_id,
                    "sample_id": job.sample_id,
                    "category": job.category,
                    "instance_index": job.instance_index,
                    "image_path": str(image_path),
                    "prediction_path": str(output_path),
                    "threshold": float(threshold),
                    "checkpoint": str(checkpoint),
                    "checkpoint_source_dataset": checkpoint_source,
                    "method": "fastsam",
                    "prompt_source": "multimodal_polygon_json",
                    "prompt_box": prompt_box,
                    "prompt_box_variant": prompt_box_variant,
                    "fastsam_imgsz": int(imgsz),
                    "fastsam_conf": float(conf),
                    "fastsam_iou": float(iou),
                    "original_width": width,
                    "original_height": height,
                    "latency_ms": latency_ms,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed_jobs += 1
            if image_index == total_images or image_index % 25 == 0:
                print(
                    f"[third-batch-export] images={image_index}/{total_images} jobs={completed_jobs}/{total_jobs} "
                    f"last={grouped_jobs[-1].frame_id} image_latency_ms={inference_ms:.2f}",
                    flush=True,
                )
    return records


def bool_mask_to_binary(mask: object, width: int, height: int, threshold: float) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    if arr.shape != (height, width):
        arr = np.asarray(Image.fromarray(arr.astype(np.float32)).resize((width, height), NEAREST))
    return ((arr > threshold).astype(np.uint8) * 255)


def efficient_sam_box_prompt_tensors(prompt_box: list[float], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = [float(value) for value in prompt_box]
    points = torch.tensor([[[[x1, y1], [x2, y2]]]], dtype=torch.float32, device=device)
    labels = torch.tensor([[[2, 3]]], dtype=torch.int64, device=device)
    return points, labels


def predict_promptable_box_mask(
    *,
    method: str,
    model: Any,
    image_state: Any,
    prompt_box: list[float],
    width: int,
    height: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    if method == "mobile_sam":
        masks, scores, _ = model.predict(box=np.asarray(prompt_box, dtype=np.float32), multimask_output=True)
        best_idx = int(np.asarray(scores).argmax()) if np.asarray(scores).size else 0
        return bool_mask_to_binary(masks[best_idx], width=width, height=height, threshold=threshold)
    if method == "sam_vit_b":
        masks, scores, _ = model.predict(box=np.asarray(prompt_box, dtype=np.float32), multimask_output=True)
        best_idx = int(np.asarray(scores).argmax()) if np.asarray(scores).size else 0
        return bool_mask_to_binary(masks[best_idx], width=width, height=height, threshold=threshold)
    if method == "hq_sam_vit_b":
        masks, scores, _ = model.predict(
            box=np.asarray(prompt_box, dtype=np.float32),
            multimask_output=True,
            hq_token_only=False,
        )
        best_idx = int(np.asarray(scores).argmax()) if np.asarray(scores).size else 0
        return bool_mask_to_binary(masks[best_idx], width=width, height=height, threshold=threshold)
    if method == "edge_sam":
        masks, scores, _ = model.predict(box=np.asarray(prompt_box, dtype=np.float32), num_multimask_outputs=3)
        best_idx = int(np.asarray(scores).argmax()) if np.asarray(scores).size else 0
        return bool_mask_to_binary(masks[best_idx], width=width, height=height, threshold=threshold)
    if method == "efficient_sam_vitt":
        points, labels = efficient_sam_box_prompt_tensors(prompt_box, device)
        image_embeddings = image_state
        masks, iou_predictions = model.predict_masks(
            image_embeddings,
            points,
            labels,
            multimask_output=True,
            input_h=height,
            input_w=width,
            output_h=height,
            output_w=width,
        )
        best_idx = int(torch.argmax(iou_predictions[0, 0]).item())
        return bool_mask_to_binary(masks[0, 0, best_idx], width=width, height=height, threshold=threshold)
    raise ValueError(f"Unsupported promptable method: {method}")


def prepare_promptable_image_state(
    *,
    method: str,
    model: Any,
    rgb: Image.Image,
    device: torch.device,
) -> tuple[Any, float]:
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    width, height = rgb.size
    with torch.no_grad():
        synchronize_if_cuda(device)
        start = time.perf_counter()
        if method in {"mobile_sam", "edge_sam", "sam_vit_b", "hq_sam_vit_b"}:
            model.set_image(rgb_np)
            image_state = None
        elif method == "efficient_sam_vitt":
            arr = rgb_np.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)[None, :, :, :]).contiguous().to(device)
            image_state = model.get_image_embeddings(tensor)
        else:
            raise ValueError(f"Unsupported promptable method: {method}")
        synchronize_if_cuda(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
    _ = (width, height)
    return image_state, latency_ms


def export_box_prompt_one(
    *,
    method: str,
    model: Any,
    image_path: Path,
    mask_path: Path | None,
    prompt_mask: np.ndarray | None,
    output_path: Path,
    prompt_box_variant: str,
    threshold: float,
    device: torch.device,
) -> tuple[float, tuple[int, int], list[float]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    gt_mask = prompt_mask if prompt_mask is not None else read_binary_mask(mask_path)  # type: ignore[arg-type]
    prompt_box = mask_to_prompt_box(gt_mask, width=width, height=height, variant=prompt_box_variant)
    image_state, image_latency_ms = prepare_promptable_image_state(method=method, model=model, rgb=rgb, device=device)
    with torch.no_grad():
        synchronize_if_cuda(device)
        start = time.perf_counter()
        mask = predict_promptable_box_mask(
            method=method,
            model=model,
            image_state=image_state,
            prompt_box=prompt_box,
            width=width,
            height=height,
            threshold=threshold,
            device=device,
        )
        synchronize_if_cuda(device)
        prompt_latency_ms = (time.perf_counter() - start) * 1000.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return image_latency_ms + prompt_latency_ms, (height, width), prompt_box


def export_box_prompt_jobs(
    *,
    method: str,
    model: Any,
    prompt_jobs: list[PromptJob],
    dataset_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    checkpoint_source: str,
    dataset_id: str,
    prompt_box_variant: str,
    threshold: float,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    jobs = sorted(prompt_jobs, key=lambda item: (str(item.image_path), item.instance_index))
    image_groups = [(image_path, list(group)) for image_path, group in groupby(jobs, key=lambda item: item.image_path)]
    total_images = len(image_groups)
    total_jobs = len(jobs)
    completed_jobs = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for image_index, (image_path, grouped_jobs) in enumerate(image_groups, start=1):
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
            image_state, image_latency_ms = prepare_promptable_image_state(method=method, model=model, rgb=rgb, device=device)
            shared_image_latency_ms = image_latency_ms / max(1, len(grouped_jobs))
            for job in grouped_jobs:
                prompt_box = polygon_to_prompt_box(job.polygon, width=width, height=height, variant=prompt_box_variant)
                with torch.no_grad():
                    synchronize_if_cuda(device)
                    start = time.perf_counter()
                    mask = predict_promptable_box_mask(
                        method=method,
                        model=model,
                        image_state=image_state,
                        prompt_box=prompt_box,
                        width=width,
                        height=height,
                        threshold=threshold,
                        device=device,
                    )
                    synchronize_if_cuda(device)
                    prompt_latency_ms = (time.perf_counter() - start) * 1000.0
                latency_ms = shared_image_latency_ms + prompt_latency_ms
                output_path = dataset_dir / "samples" / safe_sample_filename(job.sample_id)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask, mode="L").save(output_path)
                record: dict[str, object] = {
                    "dataset_id": dataset_id,
                    "frame_id": job.frame_id,
                    "sample_id": job.sample_id,
                    "category": job.category,
                    "instance_index": job.instance_index,
                    "image_path": str(image_path),
                    "prediction_path": str(output_path),
                    "threshold": float(threshold),
                    "checkpoint": str(checkpoint),
                    "checkpoint_source_dataset": checkpoint_source,
                    "method": method,
                    "prompt_source": "multimodal_polygon_json",
                    "prompt_box": prompt_box,
                    "prompt_box_variant": prompt_box_variant,
                    "original_width": width,
                    "original_height": height,
                    "latency_ms": latency_ms,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed_jobs += 1
            if image_index == total_images or image_index % 25 == 0:
                print(
                    f"[third-batch-export] images={image_index}/{total_images} jobs={completed_jobs}/{total_jobs} "
                    f"last={grouped_jobs[-1].frame_id} image_latency_ms={image_latency_ms:.2f}",
                    flush=True,
                )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export third-batch comparison masks for IRSAM2_Benchmark.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--repo", type=Path, default=None, help="Path to the third-party repository.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to the model checkpoint.")
    parser.add_argument("--dataset-id", required=True, help="Benchmark dataset id, e.g. NUDT-SIRST.")
    parser.add_argument("--image-root", required=True, type=Path, help="Directory containing benchmark images.")
    parser.add_argument("--mask-root", type=Path, default=None, help="Directory containing GT masks for prompted models.")
    parser.add_argument("--label-root", type=Path, default=None, help="Directory containing MultiModal label JSON files for polygon-prompt export.")
    parser.add_argument("--output-root", required=True, type=Path, help="Root directory for exported predictions.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images to export; 0 means all images.")
    parser.add_argument("--threshold", type=float, default=None, help="Probability/mask threshold for binary PNG export.")
    parser.add_argument("--image-extensions", default=".bmp,.png,.jpg,.jpeg,.tif,.tiff")
    parser.add_argument("--mask-extensions", default=".png,.bmp,.tif,.tiff")
    parser.add_argument("--bgm-mean", type=float, default=None, help="Override BGM image normalization mean.")
    parser.add_argument("--bgm-std", type=float, default=None, help="Override BGM image normalization std.")
    parser.add_argument("--prompt-box-variant", choices=["loose", "tight"], default="loose")
    parser.add_argument("--fastsam-imgsz", type=int, default=1024)
    parser.add_argument("--fastsam-conf", type=float, default=0.4)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    parser.add_argument("--fastsam-retina", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = (args.repo or default_repo(args.method)).expanduser().resolve()
    checkpoint_was_overridden = args.checkpoint is not None
    checkpoint = args.checkpoint or default_checkpoint(args.method, args.dataset_id)
    if checkpoint is None:
        raise FileNotFoundError(f"No default checkpoint is known for method={args.method!r}, dataset_id={args.dataset_id!r}.")
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_source = checkpoint_source_dataset(
        args.method,
        args.dataset_id,
        checkpoint_was_overridden=checkpoint_was_overridden,
    )
    threshold = resolve_threshold(args.method, args.threshold)
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_dir = output_root / args.dataset_id
    manifest_path = dataset_dir / "manifest.jsonl"

    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    images = discover_images(image_root, parse_extensions(args.image_extensions))
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {image_root}")

    device = resolve_device(args.device)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    bgm_model: torch.nn.Module | None = None
    dense_model: torch.nn.Module | None = None
    fastsam_model = None
    prompt_cls = None
    box_prompt_model: Any | None = None
    mask_root = args.mask_root.expanduser().resolve() if args.mask_root is not None else None
    label_root = args.label_root.expanduser().resolve() if args.label_root is not None else None

    if args.method == "bgm":
        print(f"[third-batch-export] loading BGM checkpoint={checkpoint} device={device}", flush=True)
        bgm_model = load_bgm_model(repo, checkpoint, device)
    elif args.method == "fastsam":
        if mask_root is None and label_root is None:
            raise ValueError("Either --mask-root or --label-root is required for FastSAM box-prompt export.")
        if mask_root is not None and not mask_root.exists():
            raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
        if label_root is not None and not label_root.exists():
            raise FileNotFoundError(f"Label root does not exist: {label_root}")
        print(f"[third-batch-export] loading FastSAM checkpoint={checkpoint} device={device}", flush=True)
        fastsam_model, prompt_cls = load_fastsam_model(repo, checkpoint)
    elif args.method in DENSE_MASK_METHODS | BOX_FILL_METHODS:
        print(f"[third-batch-export] loading {args.method} checkpoint={checkpoint} device={device}", flush=True)
        dense_model = load_local_dense_model(args.method, repo, checkpoint, device)
    else:
        if mask_root is None and label_root is None:
            raise ValueError(f"Either --mask-root or --label-root is required for {args.method} box-prompt export.")
        if mask_root is not None and not mask_root.exists():
            raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
        if label_root is not None and not label_root.exists():
            raise FileNotFoundError(f"Label root does not exist: {label_root}")
        print(f"[third-batch-export] loading {args.method} checkpoint={checkpoint} device={device}", flush=True)
        box_prompt_model = load_box_prompt_model(args.method, repo, checkpoint, device)

    prompt_jobs: list[PromptJob] = []
    if args.method in BOX_PROMPT_METHODS and label_root is not None:
        prompt_jobs = build_multimodal_prompt_jobs(
            image_root=image_root,
            label_root=label_root,
            image_extensions=parse_extensions(args.image_extensions),
            max_images=int(args.max_images),
        )
        if not prompt_jobs:
            raise RuntimeError(f"No polygon prompt jobs found under label root {label_root}")

    total = len(prompt_jobs) if prompt_jobs else len(images)
    records = []
    if prompt_jobs:
        if args.method == "fastsam":
            assert fastsam_model is not None and prompt_cls is not None
            records = export_fastsam_prompt_jobs(
                model=fastsam_model,
                prompt_cls=prompt_cls,
                prompt_jobs=prompt_jobs,
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                checkpoint=checkpoint,
                checkpoint_source=checkpoint_source,
                dataset_id=args.dataset_id,
                prompt_box_variant=args.prompt_box_variant,
                imgsz=int(args.fastsam_imgsz),
                conf=float(args.fastsam_conf),
                iou=float(args.fastsam_iou),
                retina_masks=bool(args.fastsam_retina),
                threshold=threshold,
                device=device,
            )
        else:
            assert box_prompt_model is not None
            records = export_box_prompt_jobs(
                method=args.method,
                model=box_prompt_model,
                prompt_jobs=prompt_jobs,
                dataset_dir=dataset_dir,
                manifest_path=manifest_path,
                checkpoint=checkpoint,
                checkpoint_source=checkpoint_source,
                dataset_id=args.dataset_id,
                prompt_box_variant=args.prompt_box_variant,
                threshold=threshold,
                device=device,
            )
        mean_latency = float(np.mean([float(record["latency_ms"]) for record in records]))
        print(
            f"[third-batch-export] done method={args.method} count={total} output={dataset_dir} "
            f"manifest={manifest_path} mean_latency_ms={mean_latency:.2f}",
            flush=True,
        )
        return 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        iterable = prompt_jobs if prompt_jobs else images
        for index, item in enumerate(iterable, start=1):
            if isinstance(item, PromptJob):
                image_path = item.image_path
                frame_id = item.frame_id
                output_path = dataset_dir / "samples" / safe_sample_filename(item.sample_id)
            else:
                image_path = item
                frame_id = frame_id_from_image(image_path, image_root)
                output_path = prediction_path_from_frame_id(dataset_dir, frame_id)
            record: dict[str, object] = {
                "dataset_id": args.dataset_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "prediction_path": str(output_path),
                "threshold": threshold,
                "checkpoint": str(checkpoint),
                "checkpoint_source_dataset": checkpoint_source,
                "method": args.method,
            }
            if args.method == "bgm":
                assert bgm_model is not None
                latency_ms, original_hw = export_bgm_one(
                    model=bgm_model,
                    image_path=image_path,
                    output_path=output_path,
                    dataset_id=args.dataset_id,
                    threshold=threshold,
                    device=device,
                    mean=args.bgm_mean,
                    std=args.bgm_std,
                )
            elif args.method == "fastsam":
                assert fastsam_model is not None and prompt_cls is not None
                if isinstance(item, PromptJob):
                    mask_path = None
                    prompt_mask = item.prompt_mask
                    record.update(
                        {
                            "sample_id": item.sample_id,
                            "category": item.category,
                            "instance_index": item.instance_index,
                            "prompt_source": "multimodal_polygon_json",
                        }
                    )
                else:
                    if mask_root is None:
                        raise ValueError("--mask-root is required for non-MultiModal FastSAM export.")
                    mask_path = find_mask_path(mask_root, frame_id, parse_extensions(args.mask_extensions))
                    prompt_mask = None
                latency_ms, original_hw, prompt_box = export_fastsam_one(
                    model=fastsam_model,
                    prompt_cls=prompt_cls,
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt_mask=prompt_mask,
                    output_path=output_path,
                    prompt_box_variant=args.prompt_box_variant,
                    imgsz=int(args.fastsam_imgsz),
                    conf=float(args.fastsam_conf),
                    iou=float(args.fastsam_iou),
                    retina_masks=bool(args.fastsam_retina),
                    threshold=threshold,
                    device=device,
                )
                record.update(
                    {
                        "prompt_box": prompt_box,
                        "prompt_box_variant": args.prompt_box_variant,
                        "fastsam_imgsz": int(args.fastsam_imgsz),
                        "fastsam_conf": float(args.fastsam_conf),
                        "fastsam_iou": float(args.fastsam_iou),
                    }
                )
                if mask_path is not None:
                    record["mask_path"] = str(mask_path)
            elif args.method in DENSE_MASK_METHODS | BOX_FILL_METHODS:
                assert dense_model is not None
                latency_ms, original_hw = export_local_dense_one(
                    method=args.method,
                    model=dense_model,
                    image_path=image_path,
                    output_path=output_path,
                    dataset_id=args.dataset_id,
                    threshold=threshold,
                    device=device,
                )
            else:
                assert box_prompt_model is not None
                if mask_root is None:
                    raise ValueError(f"--mask-root is required for non-MultiModal {args.method} export.")
                mask_path = find_mask_path(mask_root, frame_id, parse_extensions(args.mask_extensions))
                latency_ms, original_hw, prompt_box = export_box_prompt_one(
                    method=args.method,
                    model=box_prompt_model,
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt_mask=None,
                    output_path=output_path,
                    prompt_box_variant=args.prompt_box_variant,
                    threshold=threshold,
                    device=device,
                )
                record.update(
                    {
                        "prompt_box": prompt_box,
                        "prompt_box_variant": args.prompt_box_variant,
                        "prompt_source": "mask_derived_box",
                        "mask_path": str(mask_path),
                    }
                )
            height, width = original_hw
            record.update({"original_width": width, "original_height": height, "latency_ms": latency_ms})
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if index == total or index % 25 == 0:
                print(f"[third-batch-export] {index}/{total} exported last={frame_id} latency_ms={latency_ms:.2f}", flush=True)

    mean_latency = float(np.mean([float(record["latency_ms"]) for record in records]))
    print(
        f"[third-batch-export] done method={args.method} count={total} output={dataset_dir} "
        f"manifest={manifest_path} mean_latency_ms={mean_latency:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
