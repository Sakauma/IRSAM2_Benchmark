from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np


def _rank_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros_like(values, dtype=np.float64)
    sorted_values = values[order]
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and sorted_values[end] == sorted_values[cursor]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        ranks[order[cursor:end]] = avg_rank
        cursor = end
    return ranks


def wilcoxon_signed_rank(diff: np.ndarray) -> Dict[str, float]:
    nonzero = diff[diff != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"wilcoxon_w": 0.0, "wilcoxon_p": 1.0, "rank_biserial": 0.0}
    abs_diff = np.abs(nonzero)
    ranks = _rank_abs(abs_diff)
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total_rank = float(ranks.sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0.0:
        p_value = 1.0
    else:
        z = (min(w_pos, w_neg) - mean_w) / math.sqrt(var_w)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "wilcoxon_w": min(w_pos, w_neg),
        "wilcoxon_p": float(min(max(p_value, 0.0), 1.0)),
        "rank_biserial": float((w_pos - w_neg) / total_rank) if total_rank else 0.0,
    }


def bootstrap_ci(diff: np.ndarray, n_bootstrap: int, ci: float, seed: int) -> tuple[float, float]:
    if len(diff) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, int(n_bootstrap))):
        sample = rng.choice(diff, size=len(diff), replace=True)
        means.append(float(sample.mean()))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _row_key(row: Dict[str, Any]) -> tuple[str, str]:
    seed = str(row.get("seed", "__no_seed__"))
    sample_id = str(row.get("sample_id", ""))
    return seed, sample_id


def _paired_values(rows: List[Dict[str, Any]], dataset: str, baseline: str, candidate: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    baseline_rows = {}
    candidate_rows = {}
    for row in rows:
        if str(row.get("dataset")) != dataset or metric not in row:
            continue
        if not isinstance(row.get(metric), (int, float)) or isinstance(row.get(metric), bool):
            continue
        if row.get("method") == baseline:
            baseline_rows[_row_key(row)] = float(row[metric])
        if row.get("method") == candidate:
            candidate_rows[_row_key(row)] = float(row[metric])
    keys = sorted(set(baseline_rows) & set(candidate_rows))
    return np.array([baseline_rows[key] for key in keys], dtype=np.float64), np.array([candidate_rows[key] for key in keys], dtype=np.float64)


def run_paired_tests(rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats_config = analysis_config.get("statistics", {})
    comparisons = stats_config.get("comparisons", [])
    metrics = analysis_config.get("metrics", [])
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    n_bootstrap = int(stats_config.get("n_bootstrap", 10000))
    ci = float(stats_config.get("ci", 0.95))
    seed = int(stats_config.get("random_seed", 42))
    low_power_threshold = int(stats_config.get("low_power_threshold", 20))

    results: List[Dict[str, Any]] = []
    for dataset in datasets:
        for comparison in comparisons:
            baseline = comparison["baseline"]
            candidate = comparison["candidate"]
            for metric in metrics:
                baseline_values, candidate_values = _paired_values(rows, dataset, baseline, candidate, metric)
                n_pairs = int(len(baseline_values))
                payload: Dict[str, Any] = {
                    "dataset": dataset,
                    "comparison": comparison.get("name", f"{candidate}_vs_{baseline}"),
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "status": "ok" if n_pairs > 0 else "skipped_no_pairs",
                    "low_power": n_pairs < low_power_threshold,
                }
                if n_pairs == 0:
                    results.append(payload)
                    continue
                diff = candidate_values - baseline_values
                ci_low, ci_high = bootstrap_ci(diff, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
                wilcoxon = wilcoxon_signed_rank(diff)
                payload.update(
                    {
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "mean_diff": float(diff.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        **wilcoxon,
                    }
                )
                results.append(payload)
    return _holm_correct(results)


def _holm_correct(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and isinstance(row.get("wilcoxon_p"), (int, float)):
            grouped[str(row["metric"])].append(row)
    for metric_rows in grouped.values():
        ordered = sorted(metric_rows, key=lambda item: float(item["wilcoxon_p"]))
        m = len(ordered)
        running_max = 0.0
        for idx, row in enumerate(ordered):
            adjusted = min(1.0, float(row["wilcoxon_p"]) * (m - idx))
            running_max = max(running_max, adjusted)
            row["wilcoxon_p_holm"] = running_max
    for row in rows:
        if "wilcoxon_p_holm" not in row and "wilcoxon_p" in row:
            row["wilcoxon_p_holm"] = row["wilcoxon_p"]
    return rows

