from __future__ import annotations

import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml
from PIL import Image

from ..config import load_app_config
from ..data import build_dataset_adapter
from ..data.prompt_synthesis import clamp_box_xyxy
from ..data.sample import Sample
from ..evaluation.heatmaps import write_heatmap_artifact
from ..models import AutoPromptModelConfig, build_ir_prompt_net, ir_prior_stack, save_auto_prompt_checkpoint


def _require_torch():
    try:
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("auto prompt training requires PyTorch.") from exc
    return torch, F, DataLoader, Dataset


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return payload


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [base / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_training_samples(config_path: Path, *, max_samples: int = 0) -> tuple[list[Sample], dict[str, int]]:
    raw = _read_yaml(config_path)
    dataset_configs = raw.get("dataset_configs", [])
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError("auto prompt training config requires a non-empty dataset_configs list.")

    samples: list[Sample] = []
    counts: dict[str, int] = {}
    for item in dataset_configs:
        dataset_config = _resolve_path(config_path.parent, str(item))
        app_config = load_app_config(dataset_config)
        loaded = build_dataset_adapter(app_config).load(app_config)
        usable = [sample for sample in loaded.samples if sample.bbox_tight is not None or sample.bbox_loose is not None]
        for sample in usable:
            sample.metadata.setdefault("dataset_id", app_config.dataset.dataset_id)
            counts[f"dataset:{app_config.dataset.dataset_id}"] = counts.get(f"dataset:{app_config.dataset.dataset_id}", 0) + 1
            counts[f"supervision:{sample.supervision_type}"] = counts.get(f"supervision:{sample.supervision_type}", 0) + 1
        samples.extend(usable)
        if max_samples > 0 and len(samples) >= max_samples:
            clipped = samples[:max_samples]
            clipped_counts: dict[str, int] = {}
            for sample in clipped:
                clipped_counts[f"supervision:{sample.supervision_type}"] = clipped_counts.get(f"supervision:{sample.supervision_type}", 0) + 1
            return clipped, {**counts, **{f"used_{key}": value for key, value in clipped_counts.items()}}
    return samples, counts


def _bool_setting(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _training_dataset_paths(config_path: Path, *, key: str = "dataset_configs") -> list[str]:
    raw = _read_yaml(config_path)
    dataset_configs = raw.get(key, [])
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError(f"auto prompt training config requires a non-empty {key} list.")
    return [str(item) for item in dataset_configs]


def _iter_training_samples(
    config_path: Path,
    *,
    dataset_configs: list[str] | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Iterator[Sample]:
    if dataset_configs is None:
        dataset_configs = _training_dataset_paths(config_path)
    shard_text = f" shard={shard_id}/{num_shards}" if num_shards > 1 else ""
    for item in dataset_configs:
        dataset_config = _resolve_path(config_path.parent, str(item))
        app_config = load_app_config(dataset_config)
        adapter = build_dataset_adapter(app_config)
        started_at = time.perf_counter()
        print(
            f"[train-load] stream_start dataset={app_config.dataset.dataset_id} "
            f"adapter={adapter.adapter_name} max_samples={app_config.runtime.max_samples} "
            f"max_images={app_config.runtime.max_images}{shard_text}",
            file=sys.stderr,
            flush=True,
        )
        loaded_count = 0
        usable_count = 0
        first_usable_reported = False
        for sample in adapter.iter_samples(app_config, shard_id=shard_id, num_shards=num_shards):
            loaded_count += 1
            if sample.bbox_tight is None and sample.bbox_loose is None:
                continue
            usable_count += 1
            sample.metadata.setdefault("dataset_id", app_config.dataset.dataset_id)
            if not first_usable_reported:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[train-load] first_sample dataset={app_config.dataset.dataset_id} "
                    f"elapsed={elapsed:.1f}s sample_id={sample.sample_id}{shard_text}",
                    file=sys.stderr,
                    flush=True,
                )
                first_usable_reported = True
            yield sample
        elapsed = time.perf_counter() - started_at
        print(
            f"[train-load] stream_done dataset={app_config.dataset.dataset_id} "
            f"loaded={loaded_count} usable={usable_count} elapsed={elapsed:.1f}s{shard_text}",
            file=sys.stderr,
            flush=True,
        )


def _shuffle_buffer(samples: Iterable[Sample], *, buffer_size: int, rng: random.Random) -> Iterator[Sample]:
    if buffer_size <= 1:
        yield from samples
        return
    buffer: list[Sample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _limit_samples(samples: Iterable[Sample], *, limit: int) -> Iterator[Sample]:
    if limit <= 0:
        yield from samples
        return
    iterator = iter(samples)
    for _ in range(limit):
        try:
            yield next(iterator)
        except StopIteration:
            return


def _load_resized_gray_and_box(sample: Sample, max_long_side: int) -> tuple[np.ndarray, list[float]]:
    with Image.open(sample.image_path) as image:
        gray_image = image.convert("L")
        src_w, src_h = gray_image.size
        scale = 1.0
        if max_long_side > 0 and max(src_h, src_w) > max_long_side:
            scale = float(max_long_side) / float(max(src_h, src_w))
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            gray_image = gray_image.resize((dst_w, dst_h), Image.BILINEAR)
        arr = np.asarray(gray_image, dtype=np.float32)

    raw_box = sample.bbox_tight or sample.bbox_loose
    if raw_box is None:
        raise ValueError(f"Sample {sample.sample_id!r} has no training box.")
    box = [float(value) * scale for value in raw_box[:4]]
    return arr, clamp_box_xyxy(box, width=arr.shape[1], height=arr.shape[0])


def _target_from_box(
    *,
    box: list[float],
    height: int,
    width: int,
    gaussian_sigma: float,
    positive_radius: int,
    min_box_side: float,
    hard_negative_weight: float,
    hard_negative_percentile: float,
    prior_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = clamp_box_xyxy(box, width=width, height=height)
    cx = float(np.clip(0.5 * (x1 + x2), 0.0, max(0.0, width - 1.0)))
    cy = float(np.clip(0.5 * (y1 + y2), 0.0, max(0.0, height - 1.0)))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if gaussian_sigma > 0.0:
        objectness = np.exp(-dist2 / (2.0 * gaussian_sigma * gaussian_sigma)).astype(np.float32)
    else:
        objectness = np.zeros((height, width), dtype=np.float32)
        objectness[int(round(cy)), int(round(cx))] = 1.0
    positive = (dist2 <= float(max(0, positive_radius)) ** 2).astype(np.float32)
    if float(positive.sum()) <= 0.0:
        positive[int(round(cy)), int(round(cx))] = 1.0
    size = np.zeros((2, height, width), dtype=np.float32)
    size[0, :, :] = max(float(min_box_side), float(x2 - x1))
    size[1, :, :] = max(float(min_box_side), float(y2 - y1))
    objectness_weight = np.ones((height, width), dtype=np.float32)
    if hard_negative_weight > 1.0 and prior_score is not None:
        x1i, y1i, x2i, y2i = [int(round(value)) for value in (x1, y1, x2, y2)]
        inside = np.zeros((height, width), dtype=bool)
        inside[max(0, y1i) : min(height, y2i), max(0, x1i) : min(width, x2i)] = True
        prior = np.asarray(prior_score, dtype=np.float32)
        threshold = float(np.percentile(prior, min(max(float(hard_negative_percentile), 0.0), 100.0)))
        hard_negative = np.logical_and(~inside, prior >= threshold)
        objectness_weight[hard_negative] = float(hard_negative_weight)
    return objectness[None], size, positive[None], objectness_weight[None]


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    def pad(arr: np.ndarray, channels: int) -> np.ndarray:
        out = np.zeros((channels, max_h, max_w), dtype=np.float32)
        out[:, : arr.shape[1], : arr.shape[2]] = arr
        return out

    return {
        "image": torch.from_numpy(np.stack([pad(item["image"], 3) for item in batch])),
        "objectness": torch.from_numpy(np.stack([pad(item["objectness"], 1) for item in batch])),
        "objectness_weight": torch.from_numpy(np.stack([pad(item["objectness_weight"], 1) for item in batch])),
        "box_size": torch.from_numpy(np.stack([pad(item["box_size"], 2) for item in batch])),
        "box_weight": torch.from_numpy(np.stack([pad(item["box_weight"], 1) for item in batch])),
        "sample_ids": [item["sample_id"] for item in batch],
        "dataset_ids": [item["dataset_id"] for item in batch],
        "supervision_types": [item["supervision_type"] for item in batch],
        "samples": [item["sample"] for item in batch],
    }


def _sample_to_training_item(
    sample: Sample,
    *,
    max_long_side: int,
    target_config: dict[str, Any],
    model_config: AutoPromptModelConfig,
) -> dict[str, Any]:
    gray, box = _load_resized_gray_and_box(sample, max_long_side)
    image = ir_prior_stack(gray, use_local_contrast=model_config.use_local_contrast, use_top_hat=model_config.use_top_hat)
    prior_score = np.maximum.reduce(image)
    objectness, box_size, box_weight, objectness_weight = _target_from_box(
        box=box,
        height=int(image.shape[1]),
        width=int(image.shape[2]),
        gaussian_sigma=float(target_config.get("gaussian_sigma", 2.0)),
        positive_radius=int(target_config.get("positive_radius", 1)),
        min_box_side=float(model_config.min_box_side),
        hard_negative_weight=float(target_config.get("hard_negative_weight", 1.0)),
        hard_negative_percentile=float(target_config.get("hard_negative_percentile", 95.0)),
        prior_score=prior_score,
    )
    return {
        "image": image,
        "objectness": objectness,
        "objectness_weight": objectness_weight,
        "box_size": box_size,
        "box_weight": box_weight,
        "sample_id": sample.sample_id,
        "dataset_id": str(sample.metadata.get("dataset_id", "unknown")),
        "supervision_type": sample.supervision_type,
        "sample": sample,
    }


def _load_resized_gray_uint8_and_box(sample: Sample, max_long_side: int) -> tuple[np.ndarray, list[float]]:
    gray, box = _load_resized_gray_and_box(sample, max_long_side)
    return np.clip(gray, 0.0, 255.0).astype(np.uint8), box


def _torch_normalize_maps(torch: Any, x: Any) -> Any:
    flat = x.flatten(2)
    min_v = flat.min(dim=2).values.view(x.shape[0], x.shape[1], 1, 1)
    max_v = flat.max(dim=2).values.view(x.shape[0], x.shape[1], 1, 1)
    return (x - min_v) / (max_v - min_v).clamp_min(1e-6)


def _pad_for_pool(torch: Any, x: Any, radius: int) -> Any:
    if radius <= 0:
        return x
    mode = "reflect" if x.shape[-2] > radius and x.shape[-1] > radius else "replicate"
    return torch.nn.functional.pad(x, (radius, radius, radius, radius), mode=mode)


def _avg_pool_reflect(torch: Any, F: Any, x: Any, radius: int) -> Any:
    kernel = 2 * int(radius) + 1
    return F.avg_pool2d(_pad_for_pool(torch, x, int(radius)), kernel_size=kernel, stride=1)


def _min_pool_reflect(torch: Any, F: Any, x: Any, kernel: int) -> Any:
    radius = int(kernel) // 2
    return -F.max_pool2d(-_pad_for_pool(torch, x, radius), kernel_size=int(kernel), stride=1)


def _max_pool_reflect(torch: Any, F: Any, x: Any, kernel: int) -> Any:
    radius = int(kernel) // 2
    return F.max_pool2d(_pad_for_pool(torch, x, radius), kernel_size=int(kernel), stride=1)


def _ir_prior_stack_batch(torch: Any, F: Any, gray: Any, *, use_local_contrast: bool = True, use_top_hat: bool = True) -> Any:
    arr = _torch_normalize_maps(torch, gray)
    if use_local_contrast:
        small_mean = _avg_pool_reflect(torch, F, arr, radius=2)
        large_mean = _avg_pool_reflect(torch, F, arr, radius=8)
        local_contrast = _torch_normalize_maps(torch, (small_mean - large_mean).clamp_min(0.0))
    else:
        local_contrast = torch.zeros_like(arr)
    if use_top_hat:
        hats = []
        for kernel in (3, 5, 9):
            opened = _max_pool_reflect(torch, F, _min_pool_reflect(torch, F, arr, kernel), kernel)
            hats.append((arr - opened).clamp_min(0.0))
        top_hat = _torch_normalize_maps(torch, torch.stack(hats, dim=0).max(dim=0).values)
    else:
        top_hat = torch.zeros_like(arr)
    return torch.cat([arr, local_contrast, top_hat], dim=1).float()


def _target_from_box_batch(
    torch: Any,
    *,
    boxes: Any,
    height: int,
    width: int,
    gaussian_sigma: float,
    positive_radius: int,
    min_box_side: float,
    hard_negative_weight: float,
    hard_negative_percentile: float,
    prior_score: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    device = boxes.device
    boxes = boxes.float()
    x1 = boxes[:, 0].clamp(0.0, max(0.0, float(width - 1)))
    y1 = boxes[:, 1].clamp(0.0, max(0.0, float(height - 1)))
    x2 = boxes[:, 2].clamp(0.0, float(width))
    y2 = boxes[:, 3].clamp(0.0, float(height))
    cx = (0.5 * (x1 + x2)).clamp(0.0, max(0.0, float(width - 1)))
    cy = (0.5 * (y1 + y2)).clamp(0.0, max(0.0, float(height - 1)))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    dist2 = (xx - cx.view(-1, 1, 1)) ** 2 + (yy - cy.view(-1, 1, 1)) ** 2
    if gaussian_sigma > 0.0:
        objectness = torch.exp(-dist2 / (2.0 * float(gaussian_sigma) * float(gaussian_sigma)))
    else:
        objectness = torch.zeros_like(dist2)
        objectness[
            torch.arange(boxes.shape[0], device=device),
            cy.round().long().clamp(0, height - 1),
            cx.round().long().clamp(0, width - 1),
        ] = 1.0
    positive = (dist2 <= float(max(0, int(positive_radius))) ** 2).float()
    empty_positive = positive.flatten(1).sum(dim=1) <= 0
    if bool(empty_positive.any()):
        positive[
            torch.where(empty_positive)[0],
            cy[empty_positive].round().long().clamp(0, height - 1),
            cx[empty_positive].round().long().clamp(0, width - 1),
        ] = 1.0
    box_w = (x2 - x1).clamp_min(float(min_box_side)).view(-1, 1, 1)
    box_h = (y2 - y1).clamp_min(float(min_box_side)).view(-1, 1, 1)
    box_size = torch.stack([box_w.expand(-1, height, width), box_h.expand(-1, height, width)], dim=1)
    objectness_weight = torch.ones((boxes.shape[0], height, width), device=device, dtype=torch.float32)
    if hard_negative_weight > 1.0 and prior_score is not None:
        inside = (
            (xx >= x1.view(-1, 1, 1).round())
            & (xx < x2.view(-1, 1, 1).round())
            & (yy >= y1.view(-1, 1, 1).round())
            & (yy < y2.view(-1, 1, 1).round())
        )
        q = min(max(float(hard_negative_percentile) / 100.0, 0.0), 1.0)
        threshold = torch.quantile(prior_score.float().flatten(1), q=q, dim=1).view(-1, 1, 1)
        objectness_weight[(~inside) & (prior_score.float() >= threshold)] = float(hard_negative_weight)
    return objectness[:, None], box_size, positive[:, None], objectness_weight[:, None]


def _positive_balance_weight(torch: Any, positive: Any, *, max_positive_weight: float) -> Any:
    positive = positive.float()
    negative = 1.0 - positive
    total = positive.flatten(1).shape[1]
    pos_count = positive.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    neg_count = negative.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    pos_weight = (float(total) / (2.0 * pos_count)).clamp(max=float(max_positive_weight))
    neg_weight = float(total) / (2.0 * neg_count)
    return positive * pos_weight + negative * neg_weight


def _objectness_loss(
    *,
    torch: Any,
    F: Any,
    logits: Any,
    target: Any,
    positive: Any,
    objectness_weight: Any,
    train_cfg: dict[str, Any],
) -> Any:
    mode = str(train_cfg.get("objectness_loss", "weighted_bce")).strip().lower()
    base = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = objectness_weight.float()
    if mode in {"balanced_bce", "balanced_focal", "focal_balanced"}:
        weight = weight * _positive_balance_weight(
            torch,
            (positive > 0.0).float(),
            max_positive_weight=float(train_cfg.get("max_positive_weight", 256.0)),
        )
    if mode in {"focal", "balanced_focal", "focal_balanced"}:
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        alpha = float(train_cfg.get("focal_alpha", 0.75))
        pt = torch.exp(-base)
        alpha_factor = torch.where((positive > 0.0), torch.full_like(base, alpha), torch.full_like(base, 1.0 - alpha))
        base = alpha_factor * (1.0 - pt).clamp_min(0.0).pow(gamma) * base
    return (base * weight).mean()


def _ranking_loss(
    *,
    torch: Any,
    logits: Any,
    positive: Any,
    train_cfg: dict[str, Any],
) -> Any:
    weight = float(train_cfg.get("ranking_loss_weight", 0.0))
    if weight <= 0.0:
        return logits.new_tensor(0.0)
    margin = float(train_cfg.get("ranking_margin", 1.0))
    negative_top_k = max(1, int(train_cfg.get("ranking_negative_top_k", 32)))
    flat_logits = logits.flatten(1)
    flat_positive = (positive > 0.0).flatten(1)
    losses = []
    for sample_logits, sample_positive in zip(flat_logits, flat_positive):
        if not bool(sample_positive.any()):
            continue
        pos_score = sample_logits[sample_positive].max()
        neg_logits = sample_logits[~sample_positive]
        if int(neg_logits.numel()) <= 0:
            continue
        top_k = min(negative_top_k, int(neg_logits.numel()))
        neg_score = torch.topk(neg_logits, k=top_k).values.mean()
        losses.append((margin + neg_score - pos_score).clamp_min(0.0))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _heuristic_distill_loss(*, F: Any, logits: Any, image: Any, train_cfg: dict[str, Any]) -> Any:
    weight = float(train_cfg.get("heuristic_distill_weight", 0.0))
    if weight <= 0.0:
        return logits.new_tensor(0.0)
    teacher = image.float().max(dim=1, keepdim=True).values.clamp(0.0, 1.0)
    return F.mse_loss(logits.sigmoid(), teacher)


def _compute_auto_prompt_losses(
    *,
    torch: Any,
    F: Any,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    objectness = batch["objectness"]
    objectness_weight = batch["objectness_weight"]
    box_size = batch["box_size"]
    box_weight = batch["box_weight"]
    logits = outputs["objectness_logits"]
    objectness_loss = _objectness_loss(
        torch=torch,
        F=F,
        logits=logits,
        target=objectness,
        positive=box_weight,
        objectness_weight=objectness_weight,
        train_cfg=train_cfg,
    )
    raw_box_loss = F.smooth_l1_loss(outputs["box_size"], box_size, reduction="none")
    weighted_box_loss = raw_box_loss * box_weight
    box_loss = weighted_box_loss.sum() / box_weight.sum().clamp_min(1.0)
    ranking_loss = _ranking_loss(torch=torch, logits=logits, positive=box_weight, train_cfg=train_cfg)
    distill_loss = _heuristic_distill_loss(F=F, logits=logits, image=batch["image"], train_cfg=train_cfg)
    loss = (
        objectness_loss
        + float(train_cfg.get("box_loss_weight", 0.1)) * box_loss
        + float(train_cfg.get("ranking_loss_weight", 0.0)) * ranking_loss
        + float(train_cfg.get("heuristic_distill_weight", 0.0)) * distill_loss
    )
    return loss, {
        "objectness_loss": objectness_loss,
        "box_loss": box_loss,
        "ranking_loss": ranking_loss,
        "heuristic_distill_loss": distill_loss,
    }


def _build_dataset_class():
    _, _, _, Dataset = _require_torch()

    class AutoPromptDataset(Dataset):
        def __init__(
            self,
            samples: list[Sample],
            *,
            max_long_side: int,
            target_config: dict[str, Any],
            model_config: AutoPromptModelConfig,
        ) -> None:
            self.samples = samples
            self.max_long_side = int(max_long_side)
            self.target_config = target_config
            self.model_config = model_config

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = self.samples[index]
            return _sample_to_training_item(
                sample,
                max_long_side=self.max_long_side,
                target_config=self.target_config,
                model_config=self.model_config,
            )

    return AutoPromptDataset


def _build_streaming_dataset_class():
    torch, _, _, _ = _require_torch()
    IterableDataset = torch.utils.data.IterableDataset

    class StreamingAutoPromptDataset(IterableDataset):
        def __init__(
            self,
            *,
            config_path: Path,
            max_long_side: int,
            target_config: dict[str, Any],
            model_config: AutoPromptModelConfig,
            shuffle_buffer_size: int,
            max_samples: int,
            seed: int,
        ) -> None:
            super().__init__()
            self.config_path = config_path
            self.max_long_side = int(max_long_side)
            self.target_config = target_config
            self.model_config = model_config
            self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
            self.max_samples = max(0, int(max_samples))
            self.seed = int(seed)
            self.iteration = 0

        def _sample_to_item(self, sample: Sample) -> dict[str, Any]:
            return _sample_to_training_item(
                sample,
                max_long_side=self.max_long_side,
                target_config=self.target_config,
                model_config=self.model_config,
            )

        def __iter__(self) -> Iterator[dict[str, Any]]:
            worker_info = torch.utils.data.get_worker_info()
            iteration = self.iteration
            self.iteration += 1
            shard_id = int(worker_info.id) if worker_info is not None else 0
            num_shards = int(worker_info.num_workers) if worker_info is not None else 1
            rng = random.Random(self.seed + iteration * max(1, num_shards) + shard_id)
            sample_limit = self.max_samples
            if sample_limit > 0 and num_shards > 1:
                sample_limit = (sample_limit + num_shards - 1) // num_shards
            sample_iterable = _shuffle_buffer(
                _limit_samples(
                    _iter_training_samples(self.config_path, shard_id=shard_id, num_shards=num_shards),
                    limit=sample_limit,
                ),
                buffer_size=self.shuffle_buffer_size,
                rng=rng,
            )
            for sample in sample_iterable:
                yield self._sample_to_item(sample)

    return StreamingAutoPromptDataset


class _LineProgress:
    def __init__(self, *, desc: str, total: int | None, unit: str, mininterval: float) -> None:
        self.desc = desc
        self.total = total
        self.unit = unit
        self.mininterval = max(0.0, float(mininterval))
        self.count = 0
        self.started_at = time.perf_counter()
        self.last_print_at = 0.0
        self.postfix: dict[str, str] = {}
        self._print(force=True)

    def set_postfix(self, **kwargs: Any) -> None:
        self.postfix = {key: str(value) for key, value in kwargs.items()}

    def update(self, n: int) -> None:
        self.count += int(n)
        self._print(force=self.total is not None and self.count >= self.total)

    def close(self) -> None:
        self._print(force=True)

    def _print(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self.last_print_at < self.mininterval:
            return
        elapsed = max(0.0, now - self.started_at)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        postfix = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        postfix_text = f" {postfix}" if postfix else ""
        total_text = str(self.total) if self.total is not None else "?"
        print(
            f"[train-progress] {self.desc} {self.count}/{total_text} {self.unit} "
            f"elapsed={elapsed:.1f}s rate={rate:.2f}/{self.unit}{postfix_text}",
            file=sys.stderr,
            flush=True,
        )
        self.last_print_at = now


def _training_progress_bar(
    *,
    desc: str,
    total: int | None,
    enabled: bool,
    backend: str,
    mininterval: float,
) -> Any | None:
    if not enabled:
        return None
    if total is not None and total <= 0:
        return None
    backend = str(backend).strip().lower()
    if backend in {"none", "off", "false", "0"}:
        return None
    if backend not in {"auto", "tqdm", "line"}:
        backend = "auto"
    if backend == "line" or (backend == "auto" and not sys.stderr.isatty()):
        return _LineProgress(desc=desc, total=total, unit="batch", mininterval=mininterval)
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(
        total=total,
        unit="batch",
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        mininterval=max(0.0, float(mininterval)),
        file=sys.stderr,
    )


def _cuda_grad_scaler(torch: Any) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler()


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


@dataclass
class DenseGpuCache:
    image: Any
    objectness: Any
    objectness_weight: Any
    box_size: Any
    box_weight: Any
    sample_ids: list[str]
    dataset_ids: list[str]
    supervision_types: list[str]
    samples: list[Sample]
    cached_gb: float

    def __len__(self) -> int:
        return len(self.sample_ids)

    def batch(self, indices: Any) -> dict[str, Any]:
        index_list = [int(value) for value in indices.detach().cpu().tolist()]
        return {
            "image": self.image[indices],
            "objectness": self.objectness[indices],
            "objectness_weight": self.objectness_weight[indices],
            "box_size": self.box_size[indices],
            "box_weight": self.box_weight[indices],
            "sample_ids": [self.sample_ids[index] for index in index_list],
            "dataset_ids": [self.dataset_ids[index] for index in index_list],
            "supervision_types": [self.supervision_types[index] for index in index_list],
            "samples": [self.samples[index] for index in index_list],
            "source": "gpu_cache",
        }


@dataclass
class LightGrayCache:
    grays: list[np.ndarray]
    boxes: list[list[float]]
    sample_ids: list[str]
    dataset_ids: list[str]
    supervision_types: list[str]
    samples: list[Sample]
    cached_gb: float

    def __len__(self) -> int:
        return len(self.sample_ids)


def _torch_cache_dtype(torch: Any, name: str) -> Any:
    text = str(name).strip().lower()
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _tensor_gb(*tensors: Any) -> float:
    total = 0
    for tensor in tensors:
        total += int(tensor.nelement()) * int(tensor.element_size())
    return total / float(1024**3)


def _ensure_cuda_cache_device(torch: Any, device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("auto-prompt gpu/light cache requires a CUDA training device.")


def _build_dense_gpu_cache(
    *,
    torch: Any,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    cache_dtype: Any,
) -> DenseGpuCache | None:
    if not dataset_configs:
        return None
    _ensure_cuda_cache_device(torch, device)
    max_samples = max(0, int(train_cfg.get("max_samples", 0)))
    max_long_side = int(train_cfg.get("max_long_side", 512))
    started_at = time.perf_counter()
    items: list[dict[str, Any]] = []
    for sample in _limit_samples(_iter_training_samples(config_path, dataset_configs=dataset_configs), limit=max_samples):
        items.append(
            _sample_to_training_item(sample, max_long_side=max_long_side, target_config=target_cfg, model_config=model_cfg)
        )
    if not items:
        return None
    max_h = max(int(item["image"].shape[1]) for item in items)
    max_w = max(int(item["image"].shape[2]) for item in items)
    image = torch.zeros((len(items), 3, max_h, max_w), device=device, dtype=cache_dtype)
    objectness = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    objectness_weight = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    box_size = torch.zeros((len(items), 2, max_h, max_w), device=device, dtype=cache_dtype)
    box_weight = torch.zeros((len(items), 1, max_h, max_w), device=device, dtype=cache_dtype)
    for index, item in enumerate(items):
        h, w = int(item["image"].shape[1]), int(item["image"].shape[2])
        image[index, :, :h, :w] = torch.from_numpy(item["image"]).to(device=device, dtype=cache_dtype)
        objectness[index, :, :h, :w] = torch.from_numpy(item["objectness"]).to(device=device, dtype=cache_dtype)
        objectness_weight[index, :, :h, :w] = torch.from_numpy(item["objectness_weight"]).to(device=device, dtype=cache_dtype)
        box_size[index, :, :h, :w] = torch.from_numpy(item["box_size"]).to(device=device, dtype=cache_dtype)
        box_weight[index, :, :h, :w] = torch.from_numpy(item["box_weight"]).to(device=device, dtype=cache_dtype)
    cached_gb = _tensor_gb(image, objectness, objectness_weight, box_size, box_weight)
    elapsed = time.perf_counter() - started_at
    dataset_counts: dict[str, int] = {}
    for item in items:
        key = str(item["dataset_id"])
        dataset_counts[key] = dataset_counts.get(key, 0) + 1
    print(
        f"[cache-build] mode=dense_gpu samples={len(items)} shape={max_h}x{max_w} "
        f"cached_gb={cached_gb:.2f} elapsed={elapsed:.1f}s datasets={dataset_counts}",
        file=sys.stderr,
        flush=True,
    )
    return DenseGpuCache(
        image=image,
        objectness=objectness,
        objectness_weight=objectness_weight,
        box_size=box_size,
        box_weight=box_weight,
        sample_ids=[str(item["sample_id"]) for item in items],
        dataset_ids=[str(item["dataset_id"]) for item in items],
        supervision_types=[str(item["supervision_type"]) for item in items],
        samples=[item["sample"] for item in items],
        cached_gb=cached_gb,
    )


def _build_light_gray_cache(
    *,
    config_path: Path,
    dataset_configs: list[str],
    train_cfg: dict[str, Any],
) -> LightGrayCache | None:
    if not dataset_configs:
        return None
    max_long_side = int(train_cfg.get("max_long_side", 512))
    max_samples = max(0, int(train_cfg.get("light_cache_max_samples", 0)))
    started_at = time.perf_counter()
    grays: list[np.ndarray] = []
    boxes: list[list[float]] = []
    sample_ids: list[str] = []
    dataset_ids: list[str] = []
    supervision_types: list[str] = []
    samples: list[Sample] = []
    for sample in _limit_samples(_iter_training_samples(config_path, dataset_configs=dataset_configs), limit=max_samples):
        gray, box = _load_resized_gray_uint8_and_box(sample, max_long_side)
        grays.append(gray)
        boxes.append(box)
        sample_ids.append(sample.sample_id)
        dataset_ids.append(str(sample.metadata.get("dataset_id", "unknown")))
        supervision_types.append(sample.supervision_type)
        samples.append(sample)
    if not grays:
        return None
    cached_gb = sum(int(gray.nbytes) for gray in grays) / float(1024**3)
    elapsed = time.perf_counter() - started_at
    dataset_counts: dict[str, int] = {}
    for dataset_id in dataset_ids:
        dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + 1
    print(
        f"[cache-build] mode=light_gray samples={len(grays)} cached_gb={cached_gb:.2f} "
        f"elapsed={elapsed:.1f}s datasets={dataset_counts}",
        file=sys.stderr,
        flush=True,
    )
    return LightGrayCache(
        grays=grays,
        boxes=boxes,
        sample_ids=sample_ids,
        dataset_ids=dataset_ids,
        supervision_types=supervision_types,
        samples=samples,
        cached_gb=cached_gb,
    )


def _light_cache_batch(
    cache: LightGrayCache,
    indices: list[int],
    *,
    torch: Any,
    F: Any,
    device: str,
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
) -> dict[str, Any]:
    selected = [cache.grays[index] for index in indices]
    max_h = max(int(gray.shape[0]) for gray in selected)
    max_w = max(int(gray.shape[1]) for gray in selected)
    if all(int(gray.shape[0]) == max_h and int(gray.shape[1]) == max_w for gray in selected):
        gray_np = np.stack(selected).astype(np.float32, copy=False)[:, None]
    else:
        gray_np = np.zeros((len(indices), 1, max_h, max_w), dtype=np.float32)
        for out_index, gray in enumerate(selected):
            h, w = int(gray.shape[0]), int(gray.shape[1])
            gray_np[out_index, 0, :h, :w] = gray.astype(np.float32)
    boxes_np = np.zeros((len(indices), 4), dtype=np.float32)
    for out_index, cache_index in enumerate(indices):
        gray = selected[out_index]
        h, w = int(gray.shape[0]), int(gray.shape[1])
        boxes_np[out_index] = np.asarray(clamp_box_xyxy(cache.boxes[cache_index], width=w, height=h), dtype=np.float32)
    gray_tensor = torch.from_numpy(gray_np).to(device=device, dtype=torch.float32, non_blocking=True)
    boxes = torch.from_numpy(boxes_np).to(device=device, dtype=torch.float32, non_blocking=True)
    image = _ir_prior_stack_batch(
        torch,
        F,
        gray_tensor,
        use_local_contrast=model_cfg.use_local_contrast,
        use_top_hat=model_cfg.use_top_hat,
    )
    prior_score = image.max(dim=1).values
    objectness, box_size, box_weight, objectness_weight = _target_from_box_batch(
        torch,
        boxes=boxes,
        height=max_h,
        width=max_w,
        gaussian_sigma=float(target_cfg.get("gaussian_sigma", 2.0)),
        positive_radius=int(target_cfg.get("positive_radius", 1)),
        min_box_side=float(model_cfg.min_box_side),
        hard_negative_weight=float(target_cfg.get("hard_negative_weight", 1.0)),
        hard_negative_percentile=float(target_cfg.get("hard_negative_percentile", 95.0)),
        prior_score=prior_score,
    )
    return {
        "image": image,
        "objectness": objectness,
        "objectness_weight": objectness_weight,
        "box_size": box_size,
        "box_weight": box_weight,
        "sample_ids": [cache.sample_ids[index] for index in indices],
        "dataset_ids": [cache.dataset_ids[index] for index in indices],
        "supervision_types": [cache.supervision_types[index] for index in indices],
        "samples": [cache.samples[index] for index in indices],
        "source": "light_cache",
    }


def _step_optimizer_after_backward(*, optimizer: Any, scaler: Any | None) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _train_tensor_batch(
    *,
    torch: Any,
    F: Any,
    model: Any,
    optimizer: Any,
    scaler: Any | None,
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str,
    non_blocking: bool,
    use_amp: bool,
    step_optimizer: bool = True,
    zero_grad: bool = True,
    loss_scale: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    step_started = time.perf_counter()
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    with _autocast_context(torch, enabled=use_amp):
        outputs = model(image)
        loss, loss_parts = _compute_auto_prompt_losses(
            torch=torch,
            F=F,
            outputs=outputs,
            batch={
                **batch,
                "image": image,
                "objectness": objectness,
                "objectness_weight": objectness_weight,
                "box_size": box_size,
                "box_weight": box_weight,
            },
            train_cfg=train_cfg,
        )
    if scaler is not None:
        scaler.scale(loss * float(loss_scale)).backward()
        if step_optimizer:
            _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
    else:
        (loss * float(loss_scale)).backward()
        if step_optimizer:
            _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
    return (
        float(loss.detach().cpu()),
        float(loss_parts["objectness_loss"].detach().cpu()),
        float(loss_parts["box_loss"].detach().cpu()),
        float(loss_parts["ranking_loss"].detach().cpu()),
        float(loss_parts["heuristic_distill_loss"].detach().cpu()),
        (time.perf_counter() - step_started) * 1000.0,
    )


def _evaluate_tensor_batch(
    *,
    torch: Any,
    F: Any,
    model: Any,
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str,
    non_blocking: bool,
    use_amp: bool,
) -> tuple[float, float, float, float, float]:
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    with torch.no_grad():
        with _autocast_context(torch, enabled=use_amp):
            outputs = model(image)
            loss, loss_parts = _compute_auto_prompt_losses(
                torch=torch,
                F=F,
                outputs=outputs,
                batch={
                    **batch,
                    "image": image,
                    "objectness": objectness,
                    "objectness_weight": objectness_weight,
                    "box_size": box_size,
                    "box_weight": box_weight,
                },
                train_cfg=train_cfg,
            )
    return (
        float(loss.detach().cpu()),
        float(loss_parts["objectness_loss"].detach().cpu()),
        float(loss_parts["box_loss"].detach().cpu()),
        float(loss_parts["ranking_loss"].detach().cpu()),
        float(loss_parts["heuristic_distill_loss"].detach().cpu()),
    )


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _autotune_light_batch_size(
    *,
    torch: Any,
    F: Any,
    model: Any,
    optimizer: Any,
    cache: LightGrayCache | None,
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    non_blocking: bool,
    use_amp: bool,
    requested: Any,
    max_batch_size: int,
) -> int:
    if cache is None or len(cache) <= 0:
        return 0
    if str(requested).strip().lower() != "auto":
        return max(1, int(requested))
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return min(max(1, int(max_batch_size)), len(cache))
    low, high, best = 1, min(max(1, int(max_batch_size)), len(cache)), 1
    probe_indices = list(range(high))
    while low <= high:
        candidate = (low + high) // 2
        try:
            batch = _light_cache_batch(
                cache,
                probe_indices[:candidate],
                torch=torch,
                F=F,
                device=device,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
            )
            _train_tensor_batch(
                torch=torch,
                F=F,
                model=model,
                optimizer=optimizer,
                scaler=None,
                batch=batch,
                train_cfg=train_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
                step_optimizer=False,
            )
            optimizer.zero_grad(set_to_none=True)
            best = candidate
            low = candidate + 1
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if _is_oom_error(exc):
                torch.cuda.empty_cache()
                high = candidate - 1
                continue
            raise
    print(f"[batch-autotune] light_cache_batch_size={best} max_candidate={max_batch_size}", file=sys.stderr, flush=True)
    return best


def _sample_light_indices(cache_size: int, *, limit: int, rng: random.Random) -> list[int]:
    if cache_size <= 0:
        return []
    if limit <= 0 or limit >= cache_size:
        indices = list(range(cache_size))
        rng.shuffle(indices)
        return indices
    return rng.sample(range(cache_size), limit)


def _record_batch_counts(batch: dict[str, Any], counts: dict[str, int]) -> None:
    for dataset_id in batch["dataset_ids"]:
        counts[f"dataset:{dataset_id}"] = counts.get(f"dataset:{dataset_id}", 0) + 1
    for supervision_type in batch["supervision_types"]:
        key = f"supervision:{supervision_type}"
        counts[key] = counts.get(key, 0) + 1


def _split_indices(count: int, *, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(max(0, int(count))))
    if len(indices) <= 1 or validation_ratio <= 0.0:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    validation_count = int(round(len(indices) * min(max(validation_ratio, 0.0), 0.9)))
    validation_count = min(len(indices) - 1, max(1, validation_count))
    validation_indices = sorted(indices[:validation_count])
    train_indices = sorted(indices[validation_count:])
    return train_indices, validation_indices


def _cache_batches_from_indices(indices: list[int], *, batch_size: int) -> Iterator[list[int]]:
    step = max(1, int(batch_size))
    for start in range(0, len(indices), step):
        yield indices[start : start + step]


def _evaluate_cache_selection_metric(
    *,
    torch: Any,
    F: Any,
    model: Any,
    dense_cache: DenseGpuCache | None,
    dense_indices: list[int],
    light_cache: LightGrayCache | None,
    light_indices: list[int],
    batch_size: int,
    light_batch_size: int,
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    device: str,
    non_blocking: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    objectness_sum = 0.0
    box_sum = 0.0
    ranking_sum = 0.0
    distill_sum = 0.0
    batch_count = 0
    sample_count = 0
    max_batches = max(0, int(train_cfg.get("validation_max_batches", 0)))

    def consume(batch: dict[str, Any]) -> bool:
        nonlocal loss_sum, objectness_sum, box_sum, ranking_sum, distill_sum, batch_count, sample_count
        if max_batches > 0 and batch_count >= max_batches:
            return False
        loss_value, objectness_value, box_value, ranking_value, distill_value = _evaluate_tensor_batch(
            torch=torch,
            F=F,
            model=model,
            batch=batch,
            train_cfg=train_cfg,
            device=device,
            non_blocking=non_blocking,
            use_amp=use_amp,
        )
        loss_sum += loss_value
        objectness_sum += objectness_value
        box_sum += box_value
        ranking_sum += ranking_value
        distill_sum += distill_value
        batch_count += 1
        sample_count += len(batch["sample_ids"])
        return True

    if dense_cache is not None and dense_indices:
        for batch_indices in _cache_batches_from_indices(dense_indices, batch_size=batch_size):
            tensor_indices = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
            if not consume(dense_cache.batch(tensor_indices)):
                break
    if light_cache is not None and light_indices and (max_batches <= 0 or batch_count < max_batches):
        for batch_indices in _cache_batches_from_indices(light_indices, batch_size=max(1, light_batch_size)):
            batch = _light_cache_batch(
                light_cache,
                batch_indices,
                torch=torch,
                F=F,
                device=device,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
            )
            if not consume(batch):
                break

    if batch_count <= 0:
        return {}
    denom = max(1, batch_count)
    return {
        "val_loss": loss_sum / denom,
        "val_objectness_loss": objectness_sum / denom,
        "val_box_loss": box_sum / denom,
        "val_ranking_loss": ranking_sum / denom,
        "val_heuristic_distill_loss": distill_sum / denom,
        "val_batch_count": float(batch_count),
        "val_sample_count": float(sample_count),
    }


def _checkpoint_epoch_record(
    *,
    output_dir: Path,
    epoch: int,
    model: Any,
    model_cfg: AutoPromptModelConfig,
    metadata: dict[str, Any],
    metric_name: str,
    metric_value: float,
) -> dict[str, Any]:
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    save_auto_prompt_checkpoint(checkpoint_path, model, config=model_cfg, metadata={**metadata, "epoch": epoch})
    return {
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
    }


def _finalize_auto_prompt_training(
    *,
    config_path: Path,
    raw: dict[str, Any],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
    model: Any,
    requested_device: str,
    device: str,
    first_epoch_sample_count: int,
    first_epoch_sample_counts: dict[str, int],
    trained_sample_events: int,
    heatmap_samples: list[Sample],
    history: list[dict[str, float]],
    checkpoint_history: list[dict[str, Any]] | None = None,
    selected_checkpoint_path: Path | None = None,
    best_checkpoint_epoch: int | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "sample_count": first_epoch_sample_count,
        "sample_counts": first_epoch_sample_counts,
        "trained_sample_events": trained_sample_events,
        "requested_device": requested_device,
        "device": device,
        "target": target_cfg,
        "train": train_cfg,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    save_auto_prompt_checkpoint(checkpoint_path, model, config=model_cfg, metadata=metadata)
    if selected_checkpoint_path is None:
        selected_checkpoint_path = checkpoint_path
    if selected_checkpoint_path != checkpoint_path and not selected_checkpoint_path.exists():
        save_auto_prompt_checkpoint(selected_checkpoint_path, model, config=model_cfg, metadata=metadata)
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    heatmap_outputs = _write_training_heatmaps(
        output_dir=output_dir,
        model=model,
        samples=heatmap_samples,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
        limit=heatmap_limit,
        experiment_id=experiment_id,
    )
    summary = {
        **metadata,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "final_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "best_checkpoint_path": str(selected_checkpoint_path),
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "checkpoint_history": checkpoint_history or [],
        "model": asdict(model_cfg),
        "history": history,
        "final_loss": history[-1]["loss"],
        "heatmaps": heatmap_outputs,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _train_auto_prompt_cached_from_config(
    *,
    config_path: Path,
    raw: dict[str, Any],
    train_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    model_cfg: AutoPromptModelConfig,
) -> dict[str, Any]:
    torch, F, _, _ = _require_torch()
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _ensure_cuda_cache_device(torch, device)

    cache_dtype = _torch_cache_dtype(torch, str(train_cfg.get("cache_dtype", "float16")))
    batch_size = max(1, int(train_cfg.get("batch_size", 1)))
    light_requested_batch = train_cfg.get("light_cache_batch_size", train_cfg.get("stream_batch_size", "auto"))
    light_batch_max = max(1, int(train_cfg.get("light_cache_batch_size_max", train_cfg.get("stream_batch_size_max", 1024))))
    light_samples_per_epoch = max(0, int(train_cfg.get("light_cache_samples_per_epoch", train_cfg.get("stream_samples_per_epoch", 8192))))
    non_blocking = _bool_setting(train_cfg.get("non_blocking"), default=True)
    use_amp = _bool_setting(train_cfg.get("use_amp"), default=False) and str(device).startswith("cuda")
    profile_interval_batches = max(0, int(train_cfg.get("profile_interval_batches", 0)))
    max_steps_per_epoch = max(0, int(train_cfg.get("max_steps_per_epoch", 0)))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))

    gpu_cache_configs = [str(item) for item in raw.get("gpu_cache_dataset_configs", [])]
    light_cache_configs = [str(item) for item in raw.get("light_cache_dataset_configs", [])]
    dense_cache = _build_dense_gpu_cache(
        torch=torch,
        config_path=config_path,
        dataset_configs=gpu_cache_configs,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        device=device,
        cache_dtype=cache_dtype,
    )
    light_cache = _build_light_gray_cache(config_path=config_path, dataset_configs=light_cache_configs, train_cfg=train_cfg)
    if dense_cache is None and light_cache is None:
        raise RuntimeError("No samples were found for auto prompt cached training.")

    model = build_ir_prompt_net(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = _cuda_grad_scaler(torch) if use_amp else None
    light_batch_size = _autotune_light_batch_size(
        torch=torch,
        F=F,
        model=model,
        optimizer=optimizer,
        cache=light_cache,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        device=device,
        non_blocking=non_blocking,
        use_amp=use_amp,
        requested=light_requested_batch,
        max_batch_size=light_batch_max,
    )

    epochs = max(1, int(train_cfg.get("epochs", 1)))
    show_progress = bool(train_cfg.get("show_progress", True))
    progress_backend = str(train_cfg.get("progress_backend", "auto"))
    progress_mininterval = float(train_cfg.get("progress_update_interval_s", 1.0))
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    validation_ratio = float(train_cfg.get("validation_ratio", 0.0))
    validation_seed = int(train_cfg.get("validation_seed", seed + 1009))
    dense_train_indices, dense_val_indices = _split_indices(
        len(dense_cache) if dense_cache is not None else 0,
        validation_ratio=validation_ratio,
        seed=validation_seed,
    )
    light_train_indices, light_val_indices = _split_indices(
        len(light_cache) if light_cache is not None else 0,
        validation_ratio=validation_ratio,
        seed=validation_seed + 17,
    )
    checkpoint_interval = max(0, int(train_cfg.get("checkpoint_interval_epochs", 0)))
    select_best_checkpoint = _bool_setting(train_cfg.get("select_best_checkpoint"), default=checkpoint_interval > 0)
    selection_metric_name = str(train_cfg.get("selection_metric", "val_loss")).strip() or "val_loss"
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / str(train_cfg.get("best_checkpoint_name", "checkpoint_best.pt"))
    checkpoint_history: list[dict[str, Any]] = []
    selected_checkpoint_path: Path | None = None
    best_checkpoint_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    history: list[dict[str, float]] = []
    first_epoch_sample_counts: dict[str, int] = {}
    first_epoch_sample_count = 0
    trained_sample_events = 0
    heatmap_samples: list[Sample] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        objectness_sum = 0.0
        box_sum = 0.0
        ranking_sum = 0.0
        distill_sum = 0.0
        batch_count = 0
        pending_accumulation_batches = 0
        epoch_sample_count = 0
        dense_batches = (len(dense_train_indices) + batch_size - 1) // batch_size if dense_cache is not None else 0
        light_train_size = len(light_train_indices) if light_cache is not None else 0
        light_limit = min(light_samples_per_epoch, light_train_size) if light_cache is not None and light_samples_per_epoch > 0 else light_train_size
        light_batches = (light_limit + max(1, light_batch_size) - 1) // max(1, light_batch_size) if light_cache is not None else 0
        progress_total = max_steps_per_epoch if max_steps_per_epoch > 0 else dense_batches + light_batches
        progress = _training_progress_bar(
            desc=f"auto-prompt epoch {epoch + 1}/{epochs}",
            total=progress_total,
            enabled=show_progress,
            backend=progress_backend,
            mininterval=progress_mininterval,
        )

        def consume_batch(batch: dict[str, Any], *, source: str, data_wait_ms: float = 0.0) -> bool:
            nonlocal loss_sum, objectness_sum, box_sum, ranking_sum, distill_sum
            nonlocal batch_count, pending_accumulation_batches, epoch_sample_count, trained_sample_events, heatmap_samples
            if max_steps_per_epoch > 0 and batch_count >= max_steps_per_epoch:
                return False
            zero_grad = pending_accumulation_batches == 0
            pending_accumulation_batches += 1
            step_optimizer = pending_accumulation_batches >= gradient_accumulation_steps
            loss_value, objectness_value, box_value, ranking_value, distill_value, step_ms = _train_tensor_batch(
                torch=torch,
                F=F,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                train_cfg=train_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
                step_optimizer=step_optimizer,
                zero_grad=zero_grad,
                loss_scale=1.0 / float(gradient_accumulation_steps),
            )
            if step_optimizer:
                pending_accumulation_batches = 0
            batch_count += 1
            batch_sample_count = len(batch["sample_ids"])
            epoch_sample_count += batch_sample_count
            trained_sample_events += batch_sample_count
            loss_sum += loss_value
            objectness_sum += objectness_value
            box_sum += box_value
            ranking_sum += ranking_value
            distill_sum += distill_value
            if epoch == 0:
                _record_batch_counts(batch, first_epoch_sample_counts)
            if len(heatmap_samples) < heatmap_limit:
                needed = heatmap_limit - len(heatmap_samples)
                heatmap_samples.extend(batch["samples"][:needed])
            if profile_interval_batches > 0 and batch_count % profile_interval_batches == 0:
                step_seconds = max(step_ms / 1000.0, 1e-9)
                print(
                    f"[train-profile] epoch={epoch + 1} batch={batch_count} source={source} "
                    f"data_wait_ms={data_wait_ms:.1f} step_ms={step_ms:.1f} "
                    f"samples_per_s={batch_sample_count / step_seconds:.2f} batch_size={batch_sample_count} "
                    f"accum={gradient_accumulation_steps} amp={use_amp}",
                    file=sys.stderr,
                    flush=True,
                )
            if progress is not None:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    objectness=f"{objectness_value:.4f}",
                    box=f"{box_value:.4f}",
                    rank=f"{ranking_value:.4f}",
                    distill=f"{distill_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    accum=f"{pending_accumulation_batches}/{gradient_accumulation_steps}",
                    source=source,
                )
                progress.update(1)
            return True

        try:
            if dense_cache is not None and dense_train_indices:
                train_tensor = torch.as_tensor(dense_train_indices, device=device, dtype=torch.long)
                order = torch.randperm(len(train_tensor), device=device)
                for start in range(0, len(train_tensor), batch_size):
                    if not consume_batch(dense_cache.batch(train_tensor[order[start : start + batch_size]]), source="gpu_cache"):
                        break
            if light_cache is not None and (max_steps_per_epoch <= 0 or batch_count < max_steps_per_epoch):
                rng = random.Random(seed + epoch)
                light_positions = _sample_light_indices(len(light_train_indices), limit=light_samples_per_epoch, rng=rng)
                light_epoch_indices = [light_train_indices[position] for position in light_positions]
                for start in range(0, len(light_epoch_indices), max(1, light_batch_size)):
                    batch_indices = light_epoch_indices[start : start + max(1, light_batch_size)]
                    data_started = time.perf_counter()
                    batch = _light_cache_batch(
                        light_cache,
                        batch_indices,
                        torch=torch,
                        F=F,
                        device=device,
                        target_cfg=target_cfg,
                        model_cfg=model_cfg,
                    )
                    data_wait_ms = (time.perf_counter() - data_started) * 1000.0
                    if not consume_batch(batch, source="light_cache", data_wait_ms=data_wait_ms):
                        break
            if pending_accumulation_batches > 0:
                _step_optimizer_after_backward(optimizer=optimizer, scaler=scaler)
                pending_accumulation_batches = 0
        finally:
            if progress is not None:
                progress.close()
        if batch_count <= 0:
            raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
        if epoch == 0:
            first_epoch_sample_count = epoch_sample_count
            print(
                f"[train-mix] epoch=1 gpu_cache_batches={dense_batches} light_cache_batches={light_batches} "
                f"usable={first_epoch_sample_count}",
                file=sys.stderr,
                flush=True,
            )
        denom = max(1, batch_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / denom,
                "objectness_loss": objectness_sum / denom,
                "box_loss": box_sum / denom,
                "ranking_loss": ranking_sum / denom,
                "heuristic_distill_loss": distill_sum / denom,
            }
        )
        history_entry = history[-1]
        epoch_number = epoch + 1
        checkpoint_due = checkpoint_interval > 0 and (epoch_number % checkpoint_interval == 0 or epoch_number == epochs)
        if checkpoint_due:
            validation = _evaluate_cache_selection_metric(
                torch=torch,
                F=F,
                model=model,
                dense_cache=dense_cache,
                dense_indices=dense_val_indices,
                light_cache=light_cache,
                light_indices=light_val_indices,
                batch_size=batch_size,
                light_batch_size=light_batch_size,
                train_cfg=train_cfg,
                target_cfg=target_cfg,
                model_cfg=model_cfg,
                device=device,
                non_blocking=non_blocking,
                use_amp=use_amp,
            )
            history_entry.update(validation)
            metric_name = selection_metric_name if selection_metric_name in history_entry else ("val_loss" if "val_loss" in history_entry else "loss")
            metric_value = float(history_entry[metric_name])
            checkpoint_metadata = {
                "experiment_id": experiment_id,
                "config_path": str(config_path),
                "epoch": epoch_number,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "target": target_cfg,
                "train": train_cfg,
            }
            checkpoint_record = _checkpoint_epoch_record(
                output_dir=output_dir,
                epoch=epoch_number,
                model=model,
                model_cfg=model_cfg,
                metadata=checkpoint_metadata,
                metric_name=metric_name,
                metric_value=metric_value,
            )
            checkpoint_record.update({key: value for key, value in history_entry.items() if key.startswith("val_")})
            checkpoint_history.append(checkpoint_record)
            if select_best_checkpoint and (best_metric_value is None or metric_value < best_metric_value):
                save_auto_prompt_checkpoint(best_checkpoint_path, model, config=model_cfg, metadata=checkpoint_metadata)
                selected_checkpoint_path = best_checkpoint_path
                best_checkpoint_epoch = epoch_number
                best_metric_name = metric_name
                best_metric_value = metric_value
            print(
                f"[checkpoint] epoch={epoch_number} path={checkpoint_record['checkpoint_path']} "
                f"{metric_name}={metric_value:.6f} best_epoch={best_checkpoint_epoch}",
                file=sys.stderr,
                flush=True,
            )

    return _finalize_auto_prompt_training(
        config_path=config_path,
        raw=raw,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        model=model,
        requested_device=requested_device,
        device=device,
        first_epoch_sample_count=first_epoch_sample_count,
        first_epoch_sample_counts=first_epoch_sample_counts,
        trained_sample_events=trained_sample_events,
        heatmap_samples=heatmap_samples,
        history=history,
        checkpoint_history=checkpoint_history,
        selected_checkpoint_path=selected_checkpoint_path,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        extra_metadata={
            "cache": {
                "mode": "mixed_gpu_light",
                "dense_gpu_samples": len(dense_cache) if dense_cache is not None else 0,
                "dense_gpu_train_samples": len(dense_train_indices),
                "dense_gpu_validation_samples": len(dense_val_indices),
                "dense_gpu_cached_gb": dense_cache.cached_gb if dense_cache is not None else 0.0,
                "light_cache_samples": len(light_cache) if light_cache is not None else 0,
                "light_cache_train_samples": len(light_train_indices),
                "light_cache_validation_samples": len(light_val_indices),
                "light_cache_cached_gb": light_cache.cached_gb if light_cache is not None else 0.0,
                "light_cache_batch_size": light_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_light_cache_batch_size": light_batch_size * gradient_accumulation_steps,
                "light_cache_samples_per_epoch": light_samples_per_epoch,
            }
        },
    )


def train_auto_prompt_from_config(config_path: str | Path) -> dict[str, Any]:
    torch, F, DataLoader, _ = _require_torch()
    config_path = Path(config_path).resolve()
    raw = _read_yaml(config_path)
    train_cfg = dict(raw.get("train", {}))
    target_cfg = dict(raw.get("target", {}))
    model_cfg = AutoPromptModelConfig(**dict(raw.get("model", {})))
    if raw.get("gpu_cache_dataset_configs") or raw.get("light_cache_dataset_configs"):
        return _train_auto_prompt_cached_from_config(
            config_path=config_path,
            raw=raw,
            train_cfg=train_cfg,
            target_cfg=target_cfg,
            model_cfg=model_cfg,
        )
    max_samples = int(train_cfg.get("max_samples", 0))
    max_steps_per_epoch = max(0, int(train_cfg.get("max_steps_per_epoch", 0)))
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    batch_size = max(1, int(train_cfg.get("batch_size", 1)))
    num_workers = max(0, int(train_cfg.get("num_workers", 0)))
    pin_memory = _bool_setting(train_cfg.get("pin_memory"), default=device.startswith("cuda"))
    prefetch_factor = max(1, int(train_cfg.get("prefetch_factor", 2)))
    persistent_workers = _bool_setting(train_cfg.get("persistent_workers"), default=num_workers > 0)
    non_blocking = _bool_setting(train_cfg.get("non_blocking"), default=pin_memory and device.startswith("cuda"))
    use_amp = _bool_setting(train_cfg.get("use_amp"), default=False) and device.startswith("cuda")
    profile_interval_batches = max(0, int(train_cfg.get("profile_interval_batches", 0)))
    heatmap_limit = int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8
    dataset_cls = _build_streaming_dataset_class()
    dataset = dataset_cls(
        config_path=config_path,
        max_long_side=int(train_cfg.get("max_long_side", 512)),
        target_config=target_cfg,
        model_config=model_cfg,
        shuffle_buffer_size=int(train_cfg.get("shuffle_buffer_size", 256)),
        max_samples=max_samples,
        seed=seed,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)

    model: Any | None = None
    optimizer: Any | None = None
    scaler: Any | None = None
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    show_progress = bool(train_cfg.get("show_progress", True))
    progress_backend = str(train_cfg.get("progress_backend", "auto"))
    progress_mininterval = float(train_cfg.get("progress_update_interval_s", 1.0))
    checkpoint_interval = max(0, int(train_cfg.get("checkpoint_interval_epochs", 0)))
    select_best_checkpoint = _bool_setting(train_cfg.get("select_best_checkpoint"), default=checkpoint_interval > 0)
    selection_metric_name = str(train_cfg.get("selection_metric", "loss")).strip() or "loss"
    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / str(train_cfg.get("best_checkpoint_name", "checkpoint_best.pt"))
    checkpoint_history: list[dict[str, Any]] = []
    selected_checkpoint_path: Path | None = None
    best_checkpoint_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    history: list[dict[str, float]] = []
    first_epoch_sample_counts: dict[str, int] = {}
    first_epoch_sample_count = 0
    trained_sample_events = 0
    heatmap_samples: list[Sample] = []
    for epoch in range(epochs):
        if model is not None:
            model.train()
        loss_sum = 0.0
        objectness_sum = 0.0
        box_sum = 0.0
        ranking_sum = 0.0
        distill_sum = 0.0
        batch_count = 0
        epoch_sample_count = 0
        progress_total = max_steps_per_epoch if max_steps_per_epoch > 0 else None
        if progress_total is None and max_samples > 0:
            progress_total = (max_samples + batch_size - 1) // batch_size
        progress = _training_progress_bar(
            desc=f"auto-prompt epoch {epoch + 1}/{epochs}",
            total=progress_total,
            enabled=show_progress,
            backend=progress_backend,
            mininterval=progress_mininterval,
        )
        try:
            loader_iter = iter(loader)
            data_wait_started = time.perf_counter()
            while True:
                if max_steps_per_epoch > 0 and batch_count >= max_steps_per_epoch:
                    break
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                batch_ready_at = time.perf_counter()
                data_wait_ms = (batch_ready_at - data_wait_started) * 1000.0
                if model is None:
                    model = build_ir_prompt_net(model_cfg).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(train_cfg.get("learning_rate", 3e-4)),
                        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                    )
                    if use_amp:
                        scaler = _cuda_grad_scaler(torch)
                model.train()
                if optimizer is None:
                    raise RuntimeError("Optimizer was not initialized after model creation.")
                step_started = time.perf_counter()
                image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                objectness = batch["objectness"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                box_size = batch["box_size"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                box_weight = batch["box_weight"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(torch, enabled=use_amp):
                    outputs = model(image)
                    loss, loss_parts = _compute_auto_prompt_losses(
                        torch=torch,
                        F=F,
                        outputs=outputs,
                        batch={
                            **batch,
                            "image": image,
                            "objectness": objectness,
                            "objectness_weight": objectness_weight,
                            "box_size": box_size,
                            "box_weight": box_weight,
                        },
                        train_cfg=train_cfg,
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                step_ms = (time.perf_counter() - step_started) * 1000.0

                loss_value = float(loss.detach().cpu())
                objectness_value = float(loss_parts["objectness_loss"].detach().cpu())
                box_value = float(loss_parts["box_loss"].detach().cpu())
                ranking_value = float(loss_parts["ranking_loss"].detach().cpu())
                distill_value = float(loss_parts["heuristic_distill_loss"].detach().cpu())
                loss_sum += loss_value
                objectness_sum += objectness_value
                box_sum += box_value
                ranking_sum += ranking_value
                distill_sum += distill_value
                batch_count += 1
                batch_sample_count = len(batch["sample_ids"])
                epoch_sample_count += batch_sample_count
                trained_sample_events += batch_sample_count
                if epoch == 0:
                    for dataset_id in batch["dataset_ids"]:
                        first_epoch_sample_counts[f"dataset:{dataset_id}"] = first_epoch_sample_counts.get(f"dataset:{dataset_id}", 0) + 1
                    for supervision_type in batch["supervision_types"]:
                        key = f"supervision:{supervision_type}"
                        first_epoch_sample_counts[key] = first_epoch_sample_counts.get(key, 0) + 1
                if len(heatmap_samples) < heatmap_limit:
                    needed = heatmap_limit - len(heatmap_samples)
                    heatmap_samples.extend(batch["samples"][:needed])
                if profile_interval_batches > 0 and batch_count % profile_interval_batches == 0:
                    step_seconds = max(step_ms / 1000.0, 1e-9)
                    print(
                        f"[train-profile] epoch={epoch + 1} batch={batch_count} "
                        f"data_wait_ms={data_wait_ms:.1f} step_ms={step_ms:.1f} "
                        f"samples_per_s={batch_sample_count / step_seconds:.2f} "
                        f"batch_size={batch_sample_count} num_workers={num_workers} amp={use_amp}",
                        file=sys.stderr,
                        flush=True,
                    )
                if progress is not None:
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        objectness=f"{objectness_value:.4f}",
                        box=f"{box_value:.4f}",
                        rank=f"{ranking_value:.4f}",
                        distill=f"{distill_value:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                    progress.update(1)
                data_wait_started = time.perf_counter()
        finally:
            if progress is not None:
                progress.close()
        if batch_count <= 0:
            raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
        if epoch == 0:
            first_epoch_sample_count = epoch_sample_count
            print(
                f"[train-load] total usable={first_epoch_sample_count} batches={batch_count}",
                file=sys.stderr,
                flush=True,
            )
        denom = max(1, batch_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / denom,
                "objectness_loss": objectness_sum / denom,
                "box_loss": box_sum / denom,
                "ranking_loss": ranking_sum / denom,
                "heuristic_distill_loss": distill_sum / denom,
            }
        )
        history_entry = history[-1]
        epoch_number = epoch + 1
        checkpoint_due = checkpoint_interval > 0 and (epoch_number % checkpoint_interval == 0 or epoch_number == epochs)
        if checkpoint_due and model is not None:
            metric_name = selection_metric_name if selection_metric_name in history_entry else "loss"
            metric_value = float(history_entry[metric_name])
            checkpoint_metadata = {
                "experiment_id": experiment_id,
                "config_path": str(config_path),
                "epoch": epoch_number,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "target": target_cfg,
                "train": train_cfg,
            }
            checkpoint_record = _checkpoint_epoch_record(
                output_dir=output_dir,
                epoch=epoch_number,
                model=model,
                model_cfg=model_cfg,
                metadata=checkpoint_metadata,
                metric_name=metric_name,
                metric_value=metric_value,
            )
            checkpoint_history.append(checkpoint_record)
            if select_best_checkpoint and (best_metric_value is None or metric_value < best_metric_value):
                save_auto_prompt_checkpoint(best_checkpoint_path, model, config=model_cfg, metadata=checkpoint_metadata)
                selected_checkpoint_path = best_checkpoint_path
                best_checkpoint_epoch = epoch_number
                best_metric_name = metric_name
                best_metric_value = metric_value

    if model is None:
        raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
    return _finalize_auto_prompt_training(
        config_path=config_path,
        raw=raw,
        train_cfg=train_cfg,
        target_cfg=target_cfg,
        model_cfg=model_cfg,
        model=model,
        requested_device=requested_device,
        device=device,
        first_epoch_sample_count=first_epoch_sample_count,
        first_epoch_sample_counts=first_epoch_sample_counts,
        trained_sample_events=trained_sample_events,
        heatmap_samples=heatmap_samples,
        history=history,
        checkpoint_history=checkpoint_history,
        selected_checkpoint_path=selected_checkpoint_path,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )


def _write_training_heatmaps(
    *,
    output_dir: Path,
    model: Any,
    samples: list[Sample],
    model_config: AutoPromptModelConfig,
    train_config: dict[str, Any],
    device: str,
    limit: int,
    experiment_id: str,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    torch, _, _, _ = _require_torch()
    model.eval()
    records: list[dict[str, str]] = []
    max_long_side = int(train_config.get("max_long_side", 512))
    for sample in samples[:limit]:
        gray, _ = _load_resized_gray_and_box(sample, max_long_side)
        prior = ir_prior_stack(gray, use_local_contrast=model_config.use_local_contrast, use_top_hat=model_config.use_top_hat)
        with torch.no_grad():
            outputs = model(torch.from_numpy(prior[None]).to(device=device, dtype=torch.float32))
        objectness = torch.sigmoid(outputs["objectness_logits"][0, 0]).detach().cpu().numpy()
        paths = write_heatmap_artifact(
            root=output_dir / "heatmaps",
            experiment_id=experiment_id,
            dataset=str(sample.metadata.get("dataset_id", sample.sequence_id or "dataset")),
            sample_id=sample.sample_id,
            stage="auto_prompt_objectness_train",
            heatmap=objectness,
            image=np.moveaxis(prior[[0, 0, 0]], 0, -1) * 255.0,
            meta={
                "model": "IRPromptNetSmall",
                "sample_id": sample.sample_id,
                "supervision_type": sample.supervision_type,
                "image_path": str(sample.image_path),
            },
        )
        records.append(paths)
    return records
