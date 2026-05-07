from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..config import AppConfig, DatasetConfig, EvaluationConfig, ModelConfig, RuntimeConfig
from ..data.adapters import build_dataset_adapter
from ..data.masks import sample_mask_or_zeros
from ..data.sample import Sample
from ..evaluation.prompt_metrics import prompt_metrics
from ..models import AutoPromptModelConfig, decode_auto_prompt, ir_prior_stack_from_path, load_auto_prompt_model


@dataclass(frozen=True)
class ProposalSweepItem:
    top_k: int
    response_threshold: float
    nms_radius: int


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping YAML config: {path}")
    return data


def _project_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _resolve_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _as_list(raw: Any, *, default: list[Any]) -> list[Any]:
    if raw is None:
        return list(default)
    if isinstance(raw, list):
        return raw
    return [raw]


def _build_dataset_config(config_path: Path, raw: dict[str, Any], *, max_images: int) -> AppConfig:
    project_root = _project_root(config_path)
    dataset_payload = raw.get("dataset")
    if not isinstance(dataset_payload, dict):
        raise ValueError("M7 proposal diagnostic config must define a dataset mapping.")
    runtime_payload = {
        "artifact_root": "artifacts",
        "reference_results_root": "reference_results",
        "output_name": str(raw.get("output_name", "m7_proposal_diagnostic")),
        "device": str(raw.get("device", "cuda")),
        "max_samples": 0,
        "max_images": max(0, int(max_images)),
        "save_visuals": False,
        "update_reference_results": False,
        "show_progress": bool(raw.get("show_progress", True)),
    }
    return AppConfig(
        root=project_root,
        config_path=config_path,
        model=ModelConfig(model_id="proposal_only", cfg="", ckpt="", family="diagnostic"),
        dataset=DatasetConfig(**dataset_payload),
        runtime=RuntimeConfig(**runtime_payload),
        evaluation=EvaluationConfig(
            benchmark_version="m7_proposal_diagnostic",
            track="track_a_image_prompted",
            protocol="proposal_only",
            inference_mode="point",
        ),
        method={},
    )


def _load_samples(app_config: AppConfig, *, target_scales: set[str], max_samples: int) -> list[Sample]:
    adapter = build_dataset_adapter(app_config)
    loaded = adapter.load(app_config)
    samples = loaded.samples
    if target_scales:
        samples = [sample for sample in samples if str(sample.target_scale).lower() in target_scales]
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(
            f"No diagnostic samples found for dataset_id={app_config.dataset.dataset_id!r}; "
            f"target_scales={sorted(target_scales) if target_scales else 'all'}."
        )
    return samples


def _device_or_cpu(requested: str) -> str:
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return requested
    except Exception:
        pass
    print(f"[m7] requested device {requested!r} is unavailable; falling back to CPU", file=sys.stderr, flush=True)
    return "cpu"


def _first_hit_rank(candidate_points: object, gt_mask: np.ndarray) -> int:
    if not isinstance(candidate_points, list):
        return -1
    h, w = gt_mask.shape
    for index, item in enumerate(candidate_points):
        if not isinstance(item, list) or len(item) < 2:
            continue
        x = int(round(float(item[0])))
        y = int(round(float(item[1])))
        if 0 <= y < h and 0 <= x < w and gt_mask[y, x] > 0.5:
            return int(index)
    return -1


def _hit_candidate_score(candidate_points: object, rank: int) -> float | None:
    if rank < 0 or not isinstance(candidate_points, list) or rank >= len(candidate_points):
        return None
    item = candidate_points[rank]
    if isinstance(item, list) and len(item) >= 3:
        return float(item[2])
    return None


def _area_bin(area_pixels: float) -> str:
    area = float(area_pixels)
    if area < 16.0:
        return "tiny_0_15"
    if area < 64.0:
        return "tiny_16_63"
    if area < 256.0:
        return "small_64_255"
    if area < 1024.0:
        return "small_256_1023"
    return "large_1024_plus"


def proposal_metric_row(
    *,
    sample: Sample,
    prompt: dict[str, Any],
    gt_mask: np.ndarray,
    sweep: ProposalSweepItem,
    checkpoint_path: Path,
    device: str,
    inference_ms: float,
) -> dict[str, Any]:
    metrics = prompt_metrics(prompt, gt_mask)
    area_pixels = float((gt_mask > 0.5).sum())
    first_hit_rank = _first_hit_rank(prompt.get("candidate_points"), gt_mask)
    row: dict[str, Any] = {
        "dataset": sample.metadata.get("dataset_id", ""),
        "dataset_id": sample.metadata.get("dataset_id", ""),
        "sample_id": sample.sample_id,
        "frame_id": sample.frame_id,
        "image_path": str(sample.image_path),
        "category": sample.category,
        "target_scale": sample.target_scale,
        "area_bin": _area_bin(area_pixels),
        "GTAreaPixels": area_pixels,
        "GTAreaRatio": area_pixels / max(1.0, float(sample.width * sample.height)),
        "PromptTopKRequested": int(sweep.top_k),
        "PromptResponseThreshold": float(sweep.response_threshold),
        "PromptCandidateNmsRadius": int(sweep.nms_radius),
        "PromptTopKFirstHitRank": first_hit_rank,
        "PromptTopKFirstHitScore": _hit_candidate_score(prompt.get("candidate_points"), first_hit_rank),
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "ProposalInferenceMs": float(inference_ms),
    }
    row.update(metrics)
    return row


def _numeric_mean(values: Iterable[Any]) -> float | None:
    finite: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def summarize_rows(rows: list[dict[str, Any]], *, group_keys: list[str]) -> list[dict[str, Any]]:
    excluded_numeric_keys = {"sample_index"}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in group_keys and key not in excluded_numeric_keys and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    summaries: list[dict[str, Any]] = []
    for group_value, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = {key: value for key, value in zip(group_keys, group_value)}
        summary["sample_count"] = len(group_rows)
        for key in numeric_keys:
            mean = _numeric_mean(row.get(key) for row in group_rows)
            if mean is not None:
                summary[f"{key}_mean"] = mean
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sweep_items(raw: dict[str, Any]) -> list[ProposalSweepItem]:
    top_k_values = [int(value) for value in _as_list(raw.get("top_k_values"), default=[1, 5, 10, 20, 50])]
    thresholds = [float(value) for value in _as_list(raw.get("response_thresholds"), default=[0.05, 0.10, 0.15, 0.20])]
    radii = [int(value) for value in _as_list(raw.get("nms_radii"), default=[4, 8])]
    return [
        ProposalSweepItem(top_k=max(1, top_k), response_threshold=threshold, nms_radius=max(0, radius))
        for top_k in top_k_values
        for threshold in thresholds
        for radius in radii
    ]


def run_proposal_diagnostic_from_config(
    config_path: str | Path,
    *,
    max_samples: int | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(config_path).resolve()
    raw = _read_yaml(path)
    project_root = _project_root(path)
    resolved_device = _device_or_cpu(str(device or raw.get("device", "cuda")))
    sample_limit = max(0, int(max_samples if max_samples is not None else raw.get("max_samples", 0)))
    app_config = _build_dataset_config(path, raw, max_images=max(0, int(raw.get("max_images", 0))))
    target_scales = {str(item).lower() for item in _as_list(raw.get("target_scales"), default=["small"]) if str(item).strip()}
    samples = _load_samples(app_config, target_scales=target_scales, max_samples=sample_limit)

    checkpoint_path = _resolve_path(str(raw["checkpoint_path"]), project_root=project_root)
    model, checkpoint_info = load_auto_prompt_model(checkpoint_path, device=resolved_device)
    model_cfg = AutoPromptModelConfig(**dict(checkpoint_info.get("config", {})))
    use_local_contrast = bool(raw.get("use_local_contrast", model_cfg.use_local_contrast))
    use_top_hat = bool(raw.get("use_top_hat", model_cfg.use_top_hat))
    border_suppression_px = int(raw.get("border_suppression_px", 0))
    point_budget = int(raw.get("point_budget", 1))
    negative_ring = bool(raw.get("negative_ring", False))
    negative_ring_offset = float(raw.get("negative_ring_offset", model_cfg.negative_ring_offset))
    min_box_side = float(raw.get("min_box_side", model_cfg.min_box_side))
    sweep_items = _sweep_items(raw)

    import torch

    rows: list[dict[str, Any]] = []
    show_progress = bool(raw.get("show_progress", True))
    progress = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(samples), desc="m7 proposal-only", unit="sample")
        except Exception:
            progress = None
    for sample_index, sample in enumerate(samples, start=1):
        gt_mask = sample_mask_or_zeros(sample)
        inference_started = time.perf_counter()
        prior = ir_prior_stack_from_path(sample.image_path, use_local_contrast=use_local_contrast, use_top_hat=use_top_hat)
        with torch.no_grad():
            tensor = torch.from_numpy(prior[None]).to(device=resolved_device, dtype=torch.float32)
            outputs = model(tensor)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        for sweep in sweep_items:
            prompt = decode_auto_prompt(
                objectness_logits=outputs["objectness_logits"],
                box_size=outputs["box_size"],
                confidence_logit=outputs.get("confidence_logits"),
                image_width=int(prior.shape[2]),
                image_height=int(prior.shape[1]),
                min_box_side=min_box_side,
                negative_ring=negative_ring,
                negative_ring_offset=negative_ring_offset,
                top_k=sweep.top_k,
                point_budget=point_budget,
                response_threshold=sweep.response_threshold,
                nms_radius=sweep.nms_radius,
                border_suppression_px=border_suppression_px,
            )
            row = proposal_metric_row(
                sample=sample,
                prompt=prompt.metadata,
                gt_mask=gt_mask,
                sweep=sweep,
                checkpoint_path=checkpoint_path,
                device=resolved_device,
                inference_ms=inference_ms,
            )
            row["dataset"] = app_config.dataset.dataset_id
            row["dataset_id"] = app_config.dataset.dataset_id
            row["sample_index"] = sample_index
            row["checkpoint_protocol"] = checkpoint_info.get("protocol", "")
            row["model_architecture"] = checkpoint_info.get("config", {}).get("architecture", "")
            rows.append(row)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()

    out_dir = _resolve_path(str(output_dir or raw.get("output_dir", "artifacts/m7_proposal_diagnostic")), project_root=project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "proposal_rows.csv"
    summary_path = out_dir / "proposal_summary.csv"
    area_summary_path = out_dir / "proposal_area_summary.csv"
    summary_rows = summarize_rows(rows, group_keys=["PromptTopKRequested", "PromptResponseThreshold", "PromptCandidateNmsRadius"])
    area_rows = summarize_rows(rows, group_keys=["area_bin", "PromptTopKRequested", "PromptResponseThreshold", "PromptCandidateNmsRadius"])
    _write_csv(per_sample_path, rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(area_summary_path, area_rows)
    summary = {
        "config_path": str(path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(out_dir),
        "device": resolved_device,
        "sample_count": len(samples),
        "sweep_count": len(sweep_items),
        "row_count": len(rows),
        "target_scales": sorted(target_scales),
        "per_sample_csv": str(per_sample_path),
        "summary_csv": str(summary_path),
        "area_summary_csv": str(area_summary_path),
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "proposal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M7 proposal-only diagnostics without loading SAM2 decoder.")
    parser.add_argument("--config", required=True, type=Path, help="Path to M7 proposal diagnostic YAML.")
    parser.add_argument("--max-samples", type=int, help="Override diagnostic max_samples after target-scale filtering.")
    parser.add_argument("--device", help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_proposal_diagnostic_from_config(
        args.config,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
