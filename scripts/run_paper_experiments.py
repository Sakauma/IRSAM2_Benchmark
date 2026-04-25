#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_method(methods: Dict[str, Dict[str, Any]], method_id: str) -> Dict[str, Any]:
    raw = copy.deepcopy(methods[method_id])
    parent_id = raw.pop("extends", None)
    if not parent_id:
        return raw
    parent = _resolve_method(methods, parent_id)
    return _deep_merge(parent, raw)


def _selected_experiments(matrix: Dict[str, Any], group: str) -> List[Dict[str, Any]]:
    experiment_ids = set(matrix.get("groups", {}).get(group, [group]))
    return [item for item in matrix.get("experiments", []) if item["experiment_id"] in experiment_ids]


def _resolve_dataset_root(dataset_entry: Dict[str, Any], paths: Dict[str, Any], dataset_id: str) -> str:
    configured = paths.get("datasets", {}).get(dataset_id)
    if configured:
        return str(configured)
    root_env = dataset_entry.get("root_env")
    if root_env and os.environ.get(root_env):
        return str(os.environ[root_env])
    return str(dataset_entry["config"]["root"])


def _resolve_checkpoint_path(model: Dict[str, Any], paths: Dict[str, Any]) -> str:
    ckpt = Path(str(model["ckpt"]))
    if ckpt.is_absolute():
        return str(ckpt)
    checkpoint_root = paths.get("sam2", {}).get("checkpoint_root")
    if checkpoint_root:
        return str(Path(str(checkpoint_root)) / ckpt.name)
    return str(ckpt)


def _build_app_config(
    matrix: Dict[str, Any],
    experiment: Dict[str, Any],
    dataset_id: str,
    method_id: str,
    paths: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_entry = matrix["datasets"][dataset_id]
    method_entry = _resolve_method(matrix["methods"], method_id)
    model = copy.deepcopy(matrix["model_defaults"])
    if paths.get("sam2", {}).get("repo"):
        model["repo"] = str(paths["sam2"]["repo"])
    model["ckpt"] = _resolve_checkpoint_path(model, paths)
    dataset_config = copy.deepcopy(dataset_entry["config"])
    dataset_config["root"] = _resolve_dataset_root(dataset_entry, paths, dataset_id)
    runtime = copy.deepcopy(matrix["runtime_defaults"])
    if paths.get("artifacts", {}).get("root"):
        runtime["artifact_root"] = str(paths["artifacts"]["root"])
    runtime["output_name"] = f"paper_v1/{experiment['experiment_id']}/{dataset_id}/{method_id}"
    evaluation = _deep_merge(matrix["evaluation_defaults"], method_entry.get("evaluation", {}))
    return {
        "model": model,
        "dataset": dataset_config,
        "runtime": runtime,
        "evaluation": evaluation,
        "stages": copy.deepcopy(matrix.get("stage_defaults", {})),
        "ablations": {
            "experiment_id": experiment["experiment_id"],
            "dataset": dataset_id,
            "method": method_id,
            "tags": method_entry.get("ablation_tags", []),
            "metrics": experiment.get("metrics", []),
        },
        "method": copy.deepcopy(method_entry.get("method", {"name": method_id, "modality": "ir"})),
        "modules": copy.deepcopy(method_entry.get("modules", {})),
    }


def _iter_runs(matrix: Dict[str, Any], group: str, paths: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for experiment in _selected_experiments(matrix, group):
        if str(experiment.get("status", "planned")).startswith("planned_after"):
            continue
        for dataset_id in experiment.get("datasets", []):
            for method_id in experiment.get("methods", []):
                if method_id not in matrix["methods"]:
                    continue
                method_entry = _resolve_method(matrix["methods"], method_id)
                yield {
                    "experiment": experiment,
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "baseline": method_entry["baseline"],
                    "config": _build_app_config(matrix, experiment, dataset_id, method_id, paths),
                }


def _command_for(config_path: Path, baseline: str, python_bin: str) -> list[str]:
    return [
        python_bin,
        str(MAIN_PY),
        "run",
        "baseline",
        "--config",
        str(config_path),
        "--baseline",
        baseline,
    ]


def _format_dry_run(command: list[str], config: Dict[str, Any]) -> str:
    return f"# dataset_root={config['dataset']['root']}\n# sam2_repo={config['model'].get('repo', '')}\n" + " ".join(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run IR-only paper experiment matrix.")
    parser.add_argument("--matrix", type=Path, default=PROJECT_ROOT / "configs" / "paper_experiments_v1.yaml")
    parser.add_argument("--group", default="p0_all")
    parser.add_argument("--paths", type=Path, default=PROJECT_ROOT / "configs" / "local_paths.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args(argv)

    matrix = _load_yaml(args.matrix)
    paths = _load_yaml(args.paths) if args.paths.exists() else {}
    runs = list(_iter_runs(matrix, args.group, paths))
    if not runs:
        raise RuntimeError(f"No runnable experiments matched group={args.group!r}.")

    with tempfile.TemporaryDirectory(prefix="irsam2_paper_matrix_") as temp_dir:
        temp_root = Path(temp_dir)
        for idx, run in enumerate(runs, start=1):
            config_path = temp_root / f"{idx:04d}_{run['dataset_id']}_{run['method_id']}.yaml"
            _write_yaml(config_path, run["config"])
            command = _command_for(config_path, run["baseline"], args.python_bin)
            if args.dry_run:
                print(_format_dry_run(command, run["config"]))
                continue
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
