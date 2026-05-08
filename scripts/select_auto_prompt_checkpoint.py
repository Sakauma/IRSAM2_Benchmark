#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROMPT_METRICS = (
    "PromptHitRate",
    "TargetRecallIoU25",
    "PromptTopKHitRate",
    "PromptBoxCoverage",
    "FalseAlarmPixelsPerMP",
)


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
        score = (
            0.35 * _as_float(row.get("PromptHitRate"))
            + 0.25 * _as_float(row.get("TargetRecallIoU25"))
            + 0.20 * _as_float(row.get("PromptTopKHitRate"))
            + 0.10 * _as_float(row.get("PromptBoxCoverage"))
            - 0.10 * false_alarm_norm
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
    output_name: str,
    report_name: str,
) -> dict[str, Any]:
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
    parser.add_argument("--output-name", default="checkpoint_selected_m9.pt")
    parser.add_argument("--report-name", default="checkpoint_selection_report.csv")
    args = parser.parse_args()
    summary = select_checkpoint(
        train_dir=args.train_dir.resolve(),
        metrics_csv=args.metrics_csv.resolve() if args.metrics_csv else None,
        output_name=args.output_name,
        report_name=args.report_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
