from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
            gray, box = _load_resized_gray_and_box(sample, self.max_long_side)
            image = ir_prior_stack(gray, use_local_contrast=self.model_config.use_local_contrast, use_top_hat=self.model_config.use_top_hat)
            prior_score = np.maximum.reduce(image)
            objectness, box_size, box_weight, objectness_weight = _target_from_box(
                box=box,
                height=int(image.shape[1]),
                width=int(image.shape[2]),
                gaussian_sigma=float(self.target_config.get("gaussian_sigma", 2.0)),
                positive_radius=int(self.target_config.get("positive_radius", 1)),
                min_box_side=float(self.model_config.min_box_side),
                hard_negative_weight=float(self.target_config.get("hard_negative_weight", 1.0)),
                hard_negative_percentile=float(self.target_config.get("hard_negative_percentile", 95.0)),
                prior_score=prior_score,
            )
            return {
                "image": image,
                "objectness": objectness,
                "objectness_weight": objectness_weight,
                "box_size": box_size,
                "box_weight": box_weight,
                "sample_id": sample.sample_id,
            }

    return AutoPromptDataset


class _LineProgress:
    def __init__(self, *, desc: str, total: int, unit: str, mininterval: float) -> None:
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
        self._print(force=self.count >= self.total)

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
        print(
            f"[train-progress] {self.desc} {self.count}/{self.total} {self.unit} "
            f"elapsed={elapsed:.1f}s rate={rate:.2f}/{self.unit}{postfix_text}",
            file=sys.stderr,
            flush=True,
        )
        self.last_print_at = now


def _training_progress_bar(
    *,
    desc: str,
    total: int,
    enabled: bool,
    backend: str,
    mininterval: float,
) -> Any | None:
    if not enabled or total <= 0:
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


def train_auto_prompt_from_config(config_path: str | Path) -> dict[str, Any]:
    torch, F, DataLoader, _ = _require_torch()
    config_path = Path(config_path).resolve()
    raw = _read_yaml(config_path)
    train_cfg = dict(raw.get("train", {}))
    target_cfg = dict(raw.get("target", {}))
    model_cfg = AutoPromptModelConfig(**dict(raw.get("model", {})))
    max_samples = int(train_cfg.get("max_samples", 0))
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples, sample_counts = _load_training_samples(config_path, max_samples=max_samples)
    if not samples:
        raise RuntimeError("No samples with bbox_tight/bbox_loose were found for auto prompt training.")
    random.shuffle(samples)

    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    dataset_cls = _build_dataset_class()
    dataset = dataset_cls(
        samples,
        max_long_side=int(train_cfg.get("max_long_side", 512)),
        target_config=target_cfg,
        model_config=model_cfg,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(train_cfg.get("batch_size", 1))),
        shuffle=True,
        num_workers=max(0, int(train_cfg.get("num_workers", 0))),
        collate_fn=_collate_batch,
    )

    model = build_ir_prompt_net(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    show_progress = bool(train_cfg.get("show_progress", True))
    progress_backend = str(train_cfg.get("progress_backend", "auto"))
    progress_mininterval = float(train_cfg.get("progress_update_interval_s", 1.0))
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        objectness_sum = 0.0
        box_sum = 0.0
        batch_count = 0
        progress = _training_progress_bar(
            desc=f"auto-prompt epoch {epoch + 1}/{epochs}",
            total=len(loader),
            enabled=show_progress,
            backend=progress_backend,
            mininterval=progress_mininterval,
        )
        try:
            for batch in loader:
                image = batch["image"].to(device=device, dtype=torch.float32)
                objectness = batch["objectness"].to(device=device, dtype=torch.float32)
                objectness_weight = batch["objectness_weight"].to(device=device, dtype=torch.float32)
                box_size = batch["box_size"].to(device=device, dtype=torch.float32)
                box_weight = batch["box_weight"].to(device=device, dtype=torch.float32)
                outputs = model(image)
                objectness_loss = F.binary_cross_entropy_with_logits(outputs["objectness_logits"], objectness, weight=objectness_weight)
                raw_box_loss = F.smooth_l1_loss(outputs["box_size"], box_size, reduction="none")
                weighted_box_loss = raw_box_loss * box_weight
                box_loss = weighted_box_loss.sum() / box_weight.sum().clamp_min(1.0)
                loss = objectness_loss + float(train_cfg.get("box_loss_weight", 0.1)) * box_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                objectness_value = float(objectness_loss.detach().cpu())
                box_value = float(box_loss.detach().cpu())
                loss_sum += loss_value
                objectness_sum += objectness_value
                box_sum += box_value
                batch_count += 1
                if progress is not None:
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        objectness=f"{objectness_value:.4f}",
                        box=f"{box_value:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        denom = max(1, batch_count)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / denom,
                "objectness_loss": objectness_sum / denom,
                "box_loss": box_sum / denom,
            }
        )

    output_root = _resolve_path(config_path.parent, str(raw.get("output_root", "artifacts/auto_prompt")))
    experiment_id = str(raw.get("experiment_id", "auto_prompt_v1"))
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "sample_count": len(samples),
        "sample_counts": sample_counts,
        "requested_device": requested_device,
        "device": device,
        "target": target_cfg,
        "train": train_cfg,
    }
    save_auto_prompt_checkpoint(checkpoint_path, model, config=model_cfg, metadata=metadata)
    heatmap_outputs = _write_training_heatmaps(
        output_dir=output_dir,
        model=model,
        samples=samples,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
        limit=int(raw.get("heatmaps", {}).get("sample_limit", train_cfg.get("heatmap_sample_limit", 8))) if isinstance(raw.get("heatmaps", {}), dict) else 8,
        experiment_id=experiment_id,
    )
    summary = {
        **metadata,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "model": asdict(model_cfg),
        "history": history,
        "final_loss": history[-1]["loss"],
        "heatmaps": heatmap_outputs,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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
