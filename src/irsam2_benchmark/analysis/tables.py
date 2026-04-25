from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def _numeric_values(rows: List[Dict[str, Any]], metric: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    assert mean is not None
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def summarize_by(rows: List[Dict[str, Any]], group_keys: Iterable[str], metrics: Iterable[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    keys = list(group_keys)
    for row in rows:
        grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)

    output = []
    for group_values, group_rows in sorted(grouped.items()):
        payload = {key: value for key, value in zip(keys, group_values)}
        payload["row_count"] = len(group_rows)
        payload["sample_count"] = len({str(row.get("sample_id", "")) for row in group_rows})
        payload["seed_count"] = len({str(row.get("seed", "")) for row in group_rows if "seed" in row})
        for metric in metrics:
            values = _numeric_values(group_rows, metric)
            payload[f"{metric}_mean"] = _mean(values)
            payload[f"{metric}_std"] = _std(values)
            payload[f"{metric}_count"] = len(values)
        output.append(payload)
    return output


def main_baseline_table(rows: List[Dict[str, Any]], metrics: Iterable[str]) -> List[Dict[str, Any]]:
    return summarize_by(rows, ["dataset", "method"], metrics)


def bucket_table(rows: List[Dict[str, Any]], metrics: Iterable[str]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    output.extend(summarize_by(rows, ["dataset", "method", "target_scale"], metrics))
    output.extend(summarize_by(rows, ["dataset", "method", "annotation_protocol_flag"], metrics))
    output.extend(summarize_by(_add_area_bucket(rows), ["dataset", "method", "area_bucket"], metrics))
    return output


def _area_bucket(area: float) -> str:
    if area < 16:
        return "lt_16_px"
    if area < 64:
        return "16_63_px"
    if area < 256:
        return "64_255_px"
    if area < 1024:
        return "256_1023_px"
    return "ge_1024_px"


def _add_area_bucket(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        area = row.get("GTAreaPixels", 0.0)
        output.append({**row, "area_bucket": _area_bucket(float(area) if isinstance(area, (int, float)) else 0.0)})
    return output
