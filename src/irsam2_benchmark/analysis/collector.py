from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .io import read_json


REQUIRED_RUN_FILES = (
    "benchmark_spec.json",
    "summary.json",
    "results.json",
    "eval_reports/rows.json",
)


@dataclass(frozen=True)
class RunRecord:
    experiment_id: str
    dataset: str
    method: str
    run_dir: Path
    summary: Dict[str, Any]
    rows: List[Dict[str, Any]]


def expected_runs(matrix: Dict[str, Any], experiment_ids: set[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for experiment in matrix.get("experiments", []):
        experiment_id = str(experiment["experiment_id"])
        if experiment_id not in experiment_ids:
            continue
        if str(experiment.get("status", "planned")).startswith("planned_after"):
            continue
        for dataset in experiment.get("datasets", []):
            if dataset not in matrix.get("datasets", {}):
                raise KeyError(f"Dataset {dataset!r} is not defined in the experiment matrix.")
            for method in experiment.get("methods", []):
                if method not in matrix.get("methods", {}):
                    raise KeyError(f"Method {method!r} is not defined in the experiment matrix.")
                runs.append({"experiment_id": experiment_id, "dataset": dataset, "method": method})
    return runs


def collect_runs(artifact_root: Path, matrix: Dict[str, Any], analysis_config: Dict[str, Any]) -> tuple[List[RunRecord], List[Dict[str, Any]]]:
    experiment_ids = {str(item) for item in analysis_config.get("experiment_groups", [])}
    found: List[RunRecord] = []
    missing: List[Dict[str, Any]] = []

    for run in expected_runs(matrix, experiment_ids):
        run_dir = artifact_root / run["experiment_id"] / run["dataset"] / run["method"]
        missing_files = [relative for relative in REQUIRED_RUN_FILES if not (run_dir / relative).exists()]
        if missing_files:
            missing.append({**run, "run_dir": str(run_dir), "missing_files": missing_files})
            continue
        summary = read_json(run_dir / "summary.json")
        rows = read_json(run_dir / "eval_reports" / "rows.json")
        augmented_rows = []
        for row in rows:
            augmented_rows.append(
                {
                    **row,
                    "experiment_id": run["experiment_id"],
                    "dataset": run["dataset"],
                    "method": run["method"],
                    "run_dir": str(run_dir),
                }
            )
        found.append(
            RunRecord(
                experiment_id=run["experiment_id"],
                dataset=run["dataset"],
                method=run["method"],
                run_dir=run_dir,
                summary=summary,
                rows=augmented_rows,
            )
        )
    return found, missing


def flatten_rows(runs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        rows.extend(run.rows)
    return rows
