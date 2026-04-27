from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _aggregate_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    numeric: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    return {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in numeric.items()
    }


def _write_grouped_reports(eval_dir: Path, eval_rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    group_keys = [
        "device_source",
        "category_name",
        "annotation_protocol_flag",
        "target_scale",
        "sequence_id",
    ]
    for group_key in group_keys:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            grouped[str(row.get(group_key, "unknown"))].append(row)
        payload = {
            name: {
                "count": len(rows),
                "metrics": _aggregate_numeric(rows),
            }
            for name, rows in grouped.items()
        }
        path = eval_dir / f"by_{group_key}.json"
        _write_json(path, payload)
        outputs[f"by_{group_key}"] = path
    return outputs


def write_results(
    output_dir: Path,
    *,
    benchmark_spec: Dict[str, Any],
    artifact_manifest: Dict[str, Any],
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
) -> Dict[str, Path]:
    eval_dir = output_dir / "eval_reports"
    _write_json(output_dir / "benchmark_spec.json", benchmark_spec)
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "results.json", results)
    _write_json(eval_dir / "rows.json", eval_rows)
    grouped = _write_grouped_reports(eval_dir, eval_rows)
    outputs = {
        "benchmark_spec": output_dir / "benchmark_spec.json",
        "artifact_manifest": output_dir / "artifact_manifest.json",
        "summary": output_dir / "summary.json",
        "results": output_dir / "results.json",
        "eval_rows": eval_dir / "rows.json",
    }
    outputs.update(grouped)
    return outputs
