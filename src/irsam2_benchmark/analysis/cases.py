from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from .io import output_pair


def _metric_value(row: Dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _ratio(row: Dict[str, Any]) -> float:
    gt = _metric_value(row, "GTAreaPixels")
    pred = _metric_value(row, "PredAreaPixels")
    return pred / max(gt, 1.0)


def select_cases(rows: List[Dict[str, Any]], primary_metric: str, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {
        "best": [],
        "median": [],
        "worst": [],
        "missed_target": [],
        "false_alarm_heavy": [],
        "over_segmented": [],
        "under_segmented": [],
    }
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset")), str(row.get("method"))), []).append(row)

    for (_, _), group_rows in groups.items():
        ranked = sorted(group_rows, key=lambda row: _metric_value(row, primary_metric))
        if not ranked:
            continue
        selected["worst"].extend(ranked[:top_k])
        selected["best"].extend(list(reversed(ranked[-top_k:])))
        median_idx = len(ranked) // 2
        half = max(1, top_k // 2)
        selected["median"].extend(ranked[max(0, median_idx - half) : median_idx + half])
        selected["missed_target"].extend([row for row in group_rows if _metric_value(row, "TargetRecallIoU10") == 0.0][:top_k])
        selected["false_alarm_heavy"].extend(sorted(group_rows, key=lambda row: _metric_value(row, "FalseAlarmPixelsPerMP"), reverse=True)[:top_k])
        selected["over_segmented"].extend(sorted(group_rows, key=_ratio, reverse=True)[:top_k])
        selected["under_segmented"].extend(sorted(group_rows, key=_ratio)[:top_k])
    return {name: _dedupe_cases(cases) for name, cases in selected.items()}


def _dedupe_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (row.get("dataset"), row.get("method"), row.get("seed"), row.get("sample_id"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def write_case_outputs(output_dir: Path, selected: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    case_root = output_dir / "error_buckets"
    figure_root = output_dir / "figures" / "qualitative_cases"
    for name, rows in selected.items():
        outputs.update(output_pair(case_root, name, rows))
        figure_dir = figure_root / name
        figure_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_existing_visuals(figure_dir, rows)
        outputs[f"{name}_figure_dir"] = str(figure_dir)
        outputs[f"{name}_copied_visuals"] = str(len(copied))
    return outputs


def _copy_existing_visuals(figure_dir: Path, rows: List[Dict[str, Any]]) -> List[Path]:
    copied: List[Path] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        sample_id = str(row.get("sample_id", "")).replace("/", "_").replace("\\", "_").replace(":", "_")
        if not run_dir.exists() or not sample_id:
            continue
        visual_root = run_dir / "visuals"
        if not visual_root.exists():
            continue
        matches = sorted(visual_root.rglob(f"*{sample_id}*.png"))
        if not matches:
            continue
        source = matches[0]
        target_name = f"{row.get('dataset')}_{row.get('method')}_{row.get('seed', 'noseed')}_{source.name}"
        target = figure_dir / target_name
        shutil.copy2(source, target)
        copied.append(target)
    return copied

