#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irsam2_benchmark.config import load_app_config  # noqa: E402
from irsam2_benchmark.data import build_dataset_adapter  # noqa: E402
from irsam2_benchmark.data.masks import sample_mask_array  # noqa: E402
from irsam2_benchmark.evaluation.prompt_metrics import prompt_metrics  # noqa: E402
from irsam2_benchmark.models import load_auto_prompt_model, predict_learned_auto_prompt_from_path  # noqa: E402


PROMPT_METRICS = (
    "PromptHitRate",
    "TargetRecallIoU25",
    "PromptTopKHitRate",
    "PromptBoxCoverage",
    "FalseAlarmPixelsPerMP",
)
BBOX_METRICS = ("PromptPointInBBox", "PromptTopKInBBox", "PromptBoxBBoxIoU")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalise_lower_is_better(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_alarm_values = [_as_float(row.get("FalseAlarmPixelsPerMP")) for row in rows]
    scored: list[dict[str, Any]] = []
    for row in rows:
        false_alarm_norm = _normalise_lower_is_better(false_alarm_values, _as_float(row.get("FalseAlarmPixelsPerMP")))
        if any(metric in row for metric in ("PromptHitRate", "TargetRecallIoU25", "PromptBoxCoverage")):
            score = (
                0.35 * _as_float(row.get("PromptHitRate"))
                + 0.25 * _as_float(row.get("TargetRecallIoU25"))
                + 0.20 * _as_float(row.get("PromptTopKHitRate"))
                + 0.10 * _as_float(row.get("PromptBoxCoverage"))
                - 0.10 * false_alarm_norm
            )
        else:
            score = (
                0.45 * _as_float(row.get("PromptPointInBBox"))
                + 0.35 * _as_float(row.get("PromptTopKInBBox"))
                + 0.20 * _as_float(row.get("PromptBoxBBoxIoU"))
            )
        scored.append({**row, "M9SelectScore": score, "FalseAlarmPixelsPerMPNorm": false_alarm_norm})
    return scored


def _rows_from_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {path}")
    return rows


def _rows_from_train_summary(train_dir: Path) -> list[dict[str, Any]]:
    summary_path = train_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing train_summary.json: {summary_path}")
    summary = _read_json(summary_path)
    rows: list[dict[str, Any]] = []
    for record in summary.get("checkpoint_history", []):
        if not isinstance(record, dict):
            continue
        checkpoint_path = record.get("checkpoint_path")
        if not checkpoint_path:
            continue
        metric_name = str(record.get("metric_name", "loss"))
        metric_value = _as_float(record.get("metric_value"))
        rows.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "epoch": record.get("epoch", ""),
                "source_metric_name": metric_name,
                "source_metric_value": metric_value,
                "M9SelectScore": -metric_value,
            }
        )
    if not rows and summary.get("selected_checkpoint_path"):
        rows.append(
            {
                "checkpoint_path": str(summary["selected_checkpoint_path"]),
                "epoch": summary.get("best_checkpoint_epoch", ""),
                "source_metric_name": summary.get("best_metric_name", "selected"),
                "source_metric_value": summary.get("best_metric_value", 0.0),
                "M9SelectScore": 0.0,
            }
        )
    if not rows:
        raise ValueError(f"No checkpoint candidates found in {summary_path}")
    return rows


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _point_in_box(point: object, box: list[float] | None) -> float:
    if box is None or not isinstance(point, list) or len(point) < 2:
        return 0.0
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return 1.0 if x1 <= x <= x2 and y1 <= y <= y2 else 0.0


def _target_recall_from_prompt_box(prompt_box: list[float] | None, gt_box: list[float] | None, threshold: float = 0.25) -> float:
    return 1.0 if _bbox_iou(prompt_box, gt_box) >= float(threshold) else 0.0


def _checkpoint_candidates_from_summary(train_dir: Path) -> list[dict[str, Any]]:
    rows = _rows_from_train_summary(train_dir)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        path = Path(str(row["checkpoint_path"]))
        if not path.is_absolute():
            path = train_dir / path
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        candidates.append({**row, "checkpoint_path": key})
    return candidates


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _evaluate_checkpoint_prompt_metrics(
    *,
    checkpoint_path: Path,
    dataset_config_paths: list[Path],
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
) -> dict[str, float]:
    model, metadata = load_auto_prompt_model(checkpoint_path, device=device)
    cfg = metadata.get("config", {})
    metric_values: dict[str, list[float]] = {}
    sample_count = 0
    for dataset_config_path in dataset_config_paths:
        app_config = load_app_config(dataset_config_path)
        samples = build_dataset_adapter(app_config).load_samples(app_config)
        for sample in samples:
            if sample.bbox_tight is None:
                continue
            prompt = predict_learned_auto_prompt_from_path(
                model=model,
                image_path=sample.image_path,
                device=device,
                min_box_side=float(cfg.get("min_box_side", 2.0)),
                negative_ring_offset=float(cfg.get("negative_ring_offset", 4.0)),
                top_k=top_k,
                point_budget=point_budget,
                response_threshold=response_threshold,
                nms_radius=nms_radius,
                border_suppression_px=border_suppression_px,
                use_local_contrast=bool(cfg.get("use_local_contrast", True)),
                use_top_hat=bool(cfg.get("use_top_hat", True)),
            )
            prompt_dict = {"point": prompt.point, "box": prompt.box, **prompt.metadata}
            mask = sample_mask_array(sample)
            if mask is not None:
                for key, value in prompt_metrics(prompt_dict, mask).items():
                    metric_values.setdefault(key, []).append(float(value))
                metric_values.setdefault("TargetRecallIoU25", []).append(_target_recall_from_prompt_box(prompt.box, sample.bbox_tight, threshold=0.25))
            else:
                metric_values.setdefault("PromptPointInBBox", []).append(_point_in_box(prompt.point, sample.bbox_tight))
                candidates = prompt.metadata.get("candidate_points", [])
                if isinstance(candidates, list):
                    hit = any(_point_in_box(candidate, sample.bbox_tight) >= 1.0 for candidate in candidates if isinstance(candidate, list))
                    metric_values.setdefault("PromptTopKInBBox", []).append(1.0 if hit else 0.0)
                metric_values.setdefault("PromptBoxBBoxIoU", []).append(_bbox_iou(prompt.box, sample.bbox_tight))
            sample_count += 1
            if max_samples > 0 and sample_count >= max_samples:
                break
        if max_samples > 0 and sample_count >= max_samples:
            break
    output = {key: _mean(values) for key, values in sorted(metric_values.items())}
    output["SelectionSampleCount"] = float(sample_count)
    return output


def _rows_from_prompt_validation(
    *,
    train_dir: Path,
    dataset_config_paths: list[Path],
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in _checkpoint_candidates_from_summary(train_dir):
        checkpoint_path = Path(str(candidate["checkpoint_path"]))
        metrics = _evaluate_checkpoint_prompt_metrics(
            checkpoint_path=checkpoint_path,
            dataset_config_paths=dataset_config_paths,
            device=device,
            max_samples=max_samples,
            top_k=top_k,
            point_budget=point_budget,
            response_threshold=response_threshold,
            nms_radius=nms_radius,
            border_suppression_px=border_suppression_px,
        )
        rows.append({**candidate, **metrics})
    if not rows:
        raise ValueError(f"No prompt-validation checkpoint rows were produced for {train_dir}")
    return _score_rows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for preferred in ("checkpoint_path", "epoch", "M9SelectScore", *PROMPT_METRICS, "FalseAlarmPixelsPerMPNorm"):
        if any(preferred in row for row in rows) and preferred not in keys:
            keys.append(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_checkpoint(
    *,
    train_dir: Path,
    metrics_csv: Path | None,
    dataset_config_paths: list[Path] | None,
    device: str,
    max_samples: int,
    top_k: int,
    point_budget: int,
    response_threshold: float,
    nms_radius: int,
    border_suppression_px: int,
    output_name: str,
    report_name: str,
) -> dict[str, Any]:
    if dataset_config_paths:
        rows = _rows_from_prompt_validation(
            train_dir=train_dir,
            dataset_config_paths=dataset_config_paths,
            device=device,
            max_samples=max_samples,
            top_k=top_k,
            point_budget=point_budget,
            response_threshold=response_threshold,
            nms_radius=nms_radius,
            border_suppression_px=border_suppression_px,
        )
    else:
        rows = _rows_from_metrics_csv(metrics_csv) if metrics_csv is not None else _rows_from_train_summary(train_dir)
    if metrics_csv is not None and all(metric in rows[0] for metric in PROMPT_METRICS):
        rows = _score_rows(rows)
    for row in rows:
        row["M9SelectScore"] = _as_float(row.get("M9SelectScore"))
    rows = sorted(rows, key=lambda item: _as_float(item.get("M9SelectScore")), reverse=True)
    selected = rows[0]
    checkpoint_path = Path(str(selected.get("checkpoint_path", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = train_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_path}")
    output_path = train_dir / output_name
    shutil.copy2(checkpoint_path, output_path)
    report_path = train_dir / report_name
    _write_report(report_path, rows)
    summary = {
        "selected_checkpoint": str(output_path),
        "source_checkpoint": str(checkpoint_path),
        "report_path": str(report_path),
        "score": _as_float(selected.get("M9SelectScore")),
        "row": selected,
    }
    (train_dir / "checkpoint_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select an auto-prompt checkpoint for M9.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--dataset-config", action="append", type=Path, default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--point-budget", type=int, default=1)
    parser.add_argument("--response-threshold", type=float, default=0.15)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--border-suppression-px", type=int, default=4)
    parser.add_argument("--output-name", default="checkpoint_selected_m9.pt")
    parser.add_argument("--report-name", default="checkpoint_selection_report.csv")
    args = parser.parse_args()
    summary = select_checkpoint(
        train_dir=args.train_dir.resolve(),
        metrics_csv=args.metrics_csv.resolve() if args.metrics_csv else None,
        dataset_config_paths=[path.resolve() for path in args.dataset_config],
        device=args.device,
        max_samples=args.max_samples,
        top_k=args.top_k,
        point_budget=args.point_budget,
        response_threshold=args.response_threshold,
        nms_radius=args.nms_radius,
        border_suppression_px=args.border_suppression_px,
        output_name=args.output_name,
        report_name=args.report_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
