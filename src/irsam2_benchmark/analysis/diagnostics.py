from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List


DEFAULT_AOR_METRICS = ("mIoU", "TargetRecallIoU25")
DEFAULT_PMCR_THRESHOLDS = (0.25, 0.50)
DEFAULT_FAB_BUDGETS = (1000.0, 5000.0, 10000.0)
LOW_AOR_GAP = 0.05


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else None


def _method_rows(rows: List[Dict[str, Any]]) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "unknown")), str(row.get("method", "unknown")))].append(row)
    return grouped


def _numeric_mean(rows: List[Dict[str, Any]], metric: str) -> float | None:
    return _mean(float(row[metric]) for row in rows if _is_number(row.get(metric)))


def _infer_no_prompt_method(methods: set[str]) -> str | None:
    preferred = [method for method in methods if "no_prompt" in method]
    return sorted(preferred)[0] if preferred else None


def _infer_oracle_box_method(methods: set[str]) -> str | None:
    preferred = [
        method
        for method in methods
        if ("box_oracle" in method or "oracle_box" in method or "pretrained_box" in method)
        and "tight" not in method
        and "auto" not in method
        and "point" not in method
    ]
    return sorted(preferred)[0] if preferred else None


def _infer_auto_methods(methods: set[str]) -> list[str]:
    return sorted(method for method in methods if "auto" in method and "no_prompt" not in method)


def aor_rows(rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = analysis_config.get("diagnostics", {}) if isinstance(analysis_config.get("diagnostics", {}), dict) else {}
    aor_config = diagnostics.get("aor", {}) if isinstance(diagnostics.get("aor", {}), dict) else {}
    metrics = list(aor_config.get("metrics", DEFAULT_AOR_METRICS))
    low_gap_threshold = float(aor_config.get("low_gap_threshold", LOW_AOR_GAP))
    grouped = _method_rows(rows)
    datasets = sorted({dataset for dataset, _ in grouped})
    methods_by_dataset: Dict[str, set[str]] = defaultdict(set)
    for dataset, method in grouped:
        methods_by_dataset[dataset].add(method)

    output: List[Dict[str, Any]] = []
    for dataset in datasets:
        methods = methods_by_dataset[dataset]
        no_prompt = str(aor_config.get("no_prompt_method") or _infer_no_prompt_method(methods) or "")
        oracle = str(aor_config.get("oracle_box_method") or _infer_oracle_box_method(methods) or "")
        auto_methods = [str(item) for item in aor_config.get("auto_methods", [])] or _infer_auto_methods(methods)
        for auto_method in auto_methods:
            for metric in metrics:
                no_prompt_mean = _numeric_mean(grouped.get((dataset, no_prompt), []), metric) if no_prompt else None
                oracle_mean = _numeric_mean(grouped.get((dataset, oracle), []), metric) if oracle else None
                auto_mean = _numeric_mean(grouped.get((dataset, auto_method), []), metric)
                payload: Dict[str, Any] = {
                    "diagnostic": "AOR",
                    "dataset": dataset,
                    "method": auto_method,
                    "metric": metric,
                    "no_prompt_method": no_prompt,
                    "oracle_box_method": oracle,
                    "auto_mean": auto_mean,
                    "no_prompt_mean": no_prompt_mean,
                    "oracle_box_mean": oracle_mean,
                }
                if auto_mean is None or no_prompt_mean is None or oracle_mean is None:
                    payload.update({"status": "missing_reference", "AOR": None, "value": None})
                    output.append(payload)
                    continue
                gap = oracle_mean - no_prompt_mean
                raw_aor = (auto_mean - no_prompt_mean) / (gap + 1e-8)
                payload["AORGap"] = gap
                if abs(gap) < low_gap_threshold:
                    payload.update({"status": "low_gap", "AOR": None, "value": None, "raw_AOR": raw_aor})
                else:
                    payload.update({"status": "ok", "AOR": raw_aor, "value": raw_aor})
                output.append(payload)
    return output


def pmcr_rows(rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = analysis_config.get("diagnostics", {}) if isinstance(analysis_config.get("diagnostics", {}), dict) else {}
    pmcr_config = diagnostics.get("pmcr", {}) if isinstance(diagnostics.get("pmcr", {}), dict) else {}
    thresholds = pmcr_config.get("thresholds", diagnostics.get("pmcr_thresholds", DEFAULT_PMCR_THRESHOLDS))
    thresholds = [float(value) for value in thresholds]
    output: List[Dict[str, Any]] = []
    grouped = _method_rows(rows)
    for (dataset, method), group_rows in sorted(grouped.items()):
        hit_rows = [row for row in group_rows if _is_number(row.get("PromptHitRate")) and float(row["PromptHitRate"]) >= 1.0]
        for threshold in thresholds:
            payload: Dict[str, Any] = {
                "diagnostic": "PMCR",
                "dataset": dataset,
                "method": method,
                "metric": "mIoU",
                "threshold": threshold,
                "hit_count": len(hit_rows),
                "prompt_hit_count": len(hit_rows),
            }
            if not hit_rows:
                payload.update({"status": "no_prompt_hits", "value": None, "PMCR": None})
                output.append(payload)
                continue
            converted = [row for row in hit_rows if _is_number(row.get("mIoU")) and float(row["mIoU"]) >= threshold]
            value = float(len(converted)) / float(len(hit_rows))
            payload.update({"status": "ok", "converted_count": len(converted), "value": value, "PMCR": value})
            output.append(payload)
    return output


def fab_tr_rows(rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = analysis_config.get("diagnostics", {}) if isinstance(analysis_config.get("diagnostics", {}), dict) else {}
    fab_config = diagnostics.get("fab_tr", {}) if isinstance(diagnostics.get("fab_tr", {}), dict) else {}
    budgets = fab_config.get("budgets", diagnostics.get("fab_budgets", DEFAULT_FAB_BUDGETS))
    budgets = [float(value) for value in budgets]
    output: List[Dict[str, Any]] = []
    grouped = _method_rows(rows)
    for (dataset, method), group_rows in sorted(grouped.items()):
        valid_rows = [
            row
            for row in group_rows
            if _is_number(row.get("FalseAlarmPixelsPerMP")) and _is_number(row.get("TargetRecallIoU25"))
        ]
        for budget in budgets:
            kept = [row for row in valid_rows if float(row["FalseAlarmPixelsPerMP"]) <= budget]
            payload: Dict[str, Any] = {
                "diagnostic": "FAB-TR",
                "dataset": dataset,
                "method": method,
                "metric": "TargetRecallIoU25",
                "budget_false_alarm_pixels_per_mp": budget,
                "false_alarm_budget_per_mp": budget,
                "row_count": len(valid_rows),
                "kept_count": len(kept),
                "eligible_count": len(kept),
            }
            if not valid_rows:
                payload.update({"status": "missing_metrics", "value": None, "FAB-TR": None})
            elif not kept:
                payload.update({"status": "over_budget", "value": None, "FAB-TR": None})
            else:
                value = _mean(float(row["TargetRecallIoU25"]) for row in kept)
                payload.update({"status": "ok", "value": value, "FAB-TR": value})
            output.append(payload)
    return output


def diagnostic_metric_rows(rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    output.extend(aor_rows(rows, analysis_config))
    output.extend(pmcr_rows(rows, analysis_config))
    output.extend(fab_tr_rows(rows, analysis_config))
    return output
