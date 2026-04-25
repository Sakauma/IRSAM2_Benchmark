#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
ANALYSIS_PY = PROJECT_ROOT / "scripts" / "analyze_paper_results.py"
REQUIRED_RUN_FILES = (
    "benchmark_spec.json",
    "summary.json",
    "results.json",
    "eval_reports/rows.json",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_method(methods: Dict[str, Dict[str, Any]], method_id: str) -> Dict[str, Any]:
    raw = copy.deepcopy(methods[method_id])
    parent_id = raw.pop("extends", None)
    if not parent_id:
        return raw
    parent = _resolve_method(methods, parent_id)
    return _deep_merge(parent, raw)


def _select_by_alias(items: List[Dict[str, Any]], selected: set[str] | None, key: str) -> List[Dict[str, Any]]:
    if selected is None:
        return items
    return [item for item in items if str(item[key]) in selected]


def _select_modes(items: List[Dict[str, Any]], selected: set[str] | None) -> List[Dict[str, Any]]:
    if selected is None:
        return items
    return [item for item in items if str(item["method"]) in selected or str(item.get("alias", "")) in selected]


def _split_filter(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _artifact_base(paths: Dict[str, Any]) -> Path:
    raw = paths.get("artifacts", {}).get("root", "artifacts")
    return _resolve_project_path(str(raw))


def _path_from_config(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _resolve_dataset_root(dataset_entry: Dict[str, Any], paths: Dict[str, Any], dataset_id: str) -> str:
    configured = paths.get("datasets", {}).get(dataset_id)
    if configured:
        return _path_from_config(str(configured))
    root_env = dataset_entry.get("root_env")
    if root_env and os.environ.get(root_env):
        return str(os.environ[root_env])
    return _path_from_config(str(dataset_entry["config"]["root"]))


def _resolve_checkpoint_path(model: Dict[str, Any], paths: Dict[str, Any]) -> str:
    ckpt = Path(str(model["ckpt"]))
    if ckpt.is_absolute():
        return str(ckpt)
    checkpoint_root = paths.get("sam2", {}).get("checkpoint_root")
    if checkpoint_root:
        return str(_resolve_project_path(str(checkpoint_root)) / ckpt.name)
    return str(ckpt)


def _run_artifact_root(artifact_base: Path, artifact_subdir: str, suite_key: str, checkpoint_alias: str) -> Path:
    return artifact_base / artifact_subdir / "runs" / suite_key / checkpoint_alias


def _run_output_dir(artifact_root: Path, experiment_id: str, dataset_id: str, method_id: str) -> Path:
    return artifact_root / experiment_id / dataset_id / method_id


def _run_is_complete(output_dir: Path) -> bool:
    return all((output_dir / relative).exists() for relative in REQUIRED_RUN_FILES)


def _runtime_config(base_runtime: Dict[str, Any], suite_config: Dict[str, Any], paths: Dict[str, Any], smoke_test: bool) -> Dict[str, Any]:
    runtime = _deep_merge(base_runtime, suite_config.get("runtime", {}))
    if smoke_test:
        runtime = _deep_merge(runtime, suite_config.get("smoke_test_runtime", {}))
    runtime = _deep_merge(runtime, paths.get("runtime", {}))
    return runtime


def _model_config(checkpoint: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    model = copy.deepcopy(checkpoint)
    model.pop("alias", None)
    if paths.get("sam2", {}).get("repo"):
        model["repo"] = _path_from_config(str(paths["sam2"]["repo"]))
    model["ckpt"] = _resolve_checkpoint_path(model, paths)
    return model


def _build_app_config(
    *,
    base_matrix: Dict[str, Any],
    suite_config: Dict[str, Any],
    paths: Dict[str, Any],
    suite_key: str,
    suite_entry: Dict[str, Any],
    checkpoint: Dict[str, Any],
    dataset_id: str,
    method_id: str,
    artifact_root: Path,
    smoke_test: bool,
) -> Dict[str, Any]:
    dataset_entry = base_matrix["datasets"][dataset_id]
    method_entry = _resolve_method(base_matrix["methods"], method_id)
    dataset_config = copy.deepcopy(dataset_entry["config"])
    dataset_config["root"] = _resolve_dataset_root(dataset_entry, paths, dataset_id)
    runtime = _runtime_config(base_matrix.get("runtime_defaults", {}), suite_config, paths, smoke_test)
    runtime["artifact_root"] = str(artifact_root)
    runtime["output_name"] = f"{suite_entry['experiment_id']}/{dataset_id}/{method_id}"
    evaluation = _deep_merge(base_matrix["evaluation_defaults"], method_entry.get("evaluation", {}))
    return {
        "model": _model_config(checkpoint, paths),
        "dataset": dataset_config,
        "runtime": runtime,
        "evaluation": evaluation,
        "stages": copy.deepcopy(base_matrix.get("stage_defaults", {})),
        "ablations": {
            "suite": suite_key,
            "experiment_id": suite_entry["experiment_id"],
            "checkpoint": checkpoint["alias"],
            "model_id": checkpoint["model_id"],
            "dataset": dataset_id,
            "method": method_id,
            "tags": method_entry.get("ablation_tags", []),
        },
        "method": copy.deepcopy(method_entry.get("method", {"name": method_id, "modality": "ir"})),
        "modules": copy.deepcopy(method_entry.get("modules", {})),
    }


def _build_generated_matrix(
    *,
    base_matrix: Dict[str, Any],
    suite_key: str,
    suite_entry: Dict[str, Any],
    checkpoint: Dict[str, Any],
    method_ids: List[str],
) -> Dict[str, Any]:
    selected_datasets = {dataset_id: copy.deepcopy(base_matrix["datasets"][dataset_id]) for dataset_id in suite_entry["datasets"]}
    selected_methods = {method_id: copy.deepcopy(base_matrix["methods"][method_id]) for method_id in method_ids}
    experiment = {
        "experiment_id": suite_entry["experiment_id"],
        "status": "planned",
        "datasets": list(suite_entry["datasets"]),
        "methods": list(method_ids),
        "metrics": list(suite_entry.get("metrics", [])),
        "purpose": suite_entry.get("purpose", ""),
        "suite": suite_key,
        "checkpoint": checkpoint["alias"],
    }
    model_defaults = copy.deepcopy(checkpoint)
    model_defaults.pop("alias", None)
    return {
        "model_defaults": model_defaults,
        "runtime_defaults": copy.deepcopy(base_matrix.get("runtime_defaults", {})),
        "evaluation_defaults": copy.deepcopy(base_matrix.get("evaluation_defaults", {})),
        "stage_defaults": copy.deepcopy(base_matrix.get("stage_defaults", {})),
        "datasets": selected_datasets,
        "methods": selected_methods,
        "groups": {suite_key: [suite_entry["experiment_id"]]},
        "experiments": [experiment],
    }


def _analysis_config(
    *,
    suite_key: str,
    suite_entry: Dict[str, Any],
    checkpoint_alias: str,
    matrix_path: Path,
    artifact_root: Path,
    analysis_root: Path,
) -> Dict[str, Any]:
    return {
        "artifact_root": str(artifact_root),
        "output_dir": str(analysis_root / suite_key / checkpoint_alias),
        "matrix": str(matrix_path),
        "experiment_groups": [suite_entry["experiment_id"]],
        "primary_metric": suite_entry.get("primary_metric", "mIoU"),
        "metrics": list(suite_entry.get("metrics", ["mIoU", "Dice", "LatencyMs"])),
        "lower_is_better": list(suite_entry.get("lower_is_better", ["LatencyMs"])),
        "group_keys": ["dataset", "method", "target_scale", "annotation_protocol_flag"],
        "case_selection": {"top_k": 8, "primary_metric": suite_entry.get("primary_metric", "mIoU")},
        "statistics": {
            "n_bootstrap": 10000,
            "ci": 0.95,
            "random_seed": 42,
            "low_power_threshold": 20,
            "comparisons": list(suite_entry.get("comparisons", [])),
        },
    }


def _command_for(config_path: Path, baseline: str, python_bin: str) -> List[str]:
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


def _analysis_command(analysis_path: Path, python_bin: str) -> List[str]:
    return [python_bin, str(ANALYSIS_PY), "--analysis", str(analysis_path)]


def _build_env(paths: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    execution = paths.get("execution", {})
    if execution.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(execution["cuda_visible_devices"])
    alloc_conf = execution.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
    if alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    return env


def _run_subprocess(command: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, text=True)


def _status_record(
    *,
    status: str,
    suite_key: str,
    checkpoint: Dict[str, Any],
    dataset_id: str,
    method_id: str,
    output_dir: Path,
    config_path: Path,
    command: List[str],
    returncode: int | None = None,
    message: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "suite": suite_key,
        "checkpoint": checkpoint["alias"],
        "model_id": checkpoint["model_id"],
        "dataset": dataset_id,
        "method": method_id,
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "command": " ".join(command),
        "returncode": returncode,
        "message": message,
    }


def _iter_requested_suites(suite_config: Dict[str, Any], selected_suites: set[str] | None) -> Iterable[tuple[str, Dict[str, Any]]]:
    explicit = selected_suites is not None
    for suite_key, suite_entry in suite_config.get("suites", {}).items():
        if selected_suites is not None and suite_key not in selected_suites:
            continue
        if not explicit and suite_entry.get("enabled", True) is False:
            continue
        yield suite_key, suite_entry


def _suite_checkpoints(all_checkpoints: List[Dict[str, Any]], suite_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = suite_entry.get("checkpoints")
    if not selected:
        return all_checkpoints
    selected_aliases = {str(item) for item in selected}
    return [checkpoint for checkpoint in all_checkpoints if str(checkpoint["alias"]) in selected_aliases]


def _suite_method_ids(global_mode_entries: List[Dict[str, Any]], suite_entry: Dict[str, Any]) -> List[str]:
    selected = suite_entry.get("modes")
    if not selected:
        return [str(item["method"]) for item in global_mode_entries]
    selected_modes = {str(item) for item in selected}
    return [str(item["method"]) for item in _select_modes(global_mode_entries, selected_modes)]


def _write_run_outputs(manifest_dir: Path, records: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    _write_json(manifest_dir / "run_manifest_latest.json", {"records": records, "failures": failures})
    _write_csv(manifest_dir / "run_manifest_latest.csv", records)
    _write_json(manifest_dir / "run_failures_latest.json", failures)
    _write_csv(manifest_dir / "run_failures_latest.csv", failures)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_output_dirs: set[str] = set()
    for record in records:
        if record["status"] not in {"completed", "skipped_existing"}:
            continue
        output_dir = Path(str(record["output_dir"]))
        if str(output_dir) in seen_output_dirs or not _run_is_complete(output_dir):
            continue
        seen_output_dirs.add(str(output_dir))
        summary = _read_json(output_dir / "summary.json")
        dataset_manifest = summary.get("dataset_manifest", {})
        row: Dict[str, Any] = {
            "suite": record["suite"],
            "checkpoint": record["checkpoint"],
            "model_id": record["model_id"],
            "dataset": record["dataset"],
            "method": record["method"],
            "output_dir": str(output_dir),
            "sample_count": dataset_manifest.get("sample_count", dataset_manifest.get("num_samples", "")),
        }
        for prefix in ("mean", "std"):
            for metric, value in summary.get(prefix, {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"{metric}_{prefix}"] = value
        rows.append(row)
    return rows


def _write_checkpoint_summary(manifest_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, str]:
    rows = _summary_rows(records)
    output_dir = manifest_dir / "analysis"
    json_path = output_dir / "checkpoint_sweep_summary.json"
    csv_path = output_dir / "checkpoint_sweep_summary.csv"
    _write_json(json_path, rows)
    _write_csv(csv_path, rows)
    return {"checkpoint_sweep_summary_json": str(json_path), "checkpoint_sweep_summary_csv": str(csv_path)}


def _write_final_manifest(manifest_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json(manifest_dir / "benchmark_manifest_latest.json", manifest)
    timestamp = str(manifest["created_at"]).replace(":", "").replace("-", "")
    _write_json(manifest_dir / f"benchmark_manifest_{timestamp}.json", manifest)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 5090 single-GPU full SAM2 IR benchmark and analysis.")
    parser.add_argument("--paths", type=Path, default=PROJECT_ROOT / "configs" / "local_paths.yaml")
    parser.add_argument("--suite-config", type=Path, default=PROJECT_ROOT / "configs" / "server_5090_full_benchmark.yaml")
    parser.add_argument("--suites", help="Comma-separated suite keys. Default: all suites from suite config.")
    parser.add_argument("--checkpoints", help="Comma-separated checkpoint aliases. Default: all four official SAM2.1 checkpoints.")
    parser.add_argument("--modes", help="Comma-separated method ids. Default: all configured modes.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even when required result files already exist.")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    if not args.paths.exists():
        raise FileNotFoundError(f"Path config not found: {args.paths}")
    suite_config = _load_yaml(args.suite_config)
    paths = _load_yaml(args.paths)
    base_matrix = _load_yaml(_resolve_project_path(suite_config.get("matrix_source", "configs/paper_experiments_v1.yaml")))
    artifact_subdir = str(suite_config.get("artifact_subdir", "paper_5090"))
    if args.smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    artifact_base = _artifact_base(paths)
    manifest_dir = artifact_base / artifact_subdir
    generated_dir = manifest_dir / "generated"
    config_dir = generated_dir / "run_configs"
    matrix_dir = generated_dir / "matrices"
    analysis_config_dir = generated_dir / "analysis_configs"
    analysis_root = manifest_dir / "analysis"
    env = _build_env(paths)

    selected_suites = _split_filter(args.suites)
    selected_checkpoints = _split_filter(args.checkpoints)
    selected_modes = _split_filter(args.modes)
    checkpoints = _select_by_alias(suite_config.get("checkpoints", []), selected_checkpoints, "alias")
    mode_entries = _select_modes(suite_config.get("modes", []), selected_modes)

    if not checkpoints:
        raise RuntimeError("No checkpoints selected.")
    if not mode_entries:
        raise RuntimeError("No modes selected.")

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    analysis_records: List[Dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()

    run_plan = []
    for suite_key, suite_entry in _iter_requested_suites(suite_config, selected_suites):
        suite_checkpoints = _suite_checkpoints(checkpoints, suite_entry)
        method_ids = _suite_method_ids(mode_entries, suite_entry)
        if not suite_checkpoints or not method_ids:
            continue
        for checkpoint in suite_checkpoints:
            artifact_root = _run_artifact_root(artifact_base, artifact_subdir, suite_key, checkpoint["alias"])
            generated_matrix = _build_generated_matrix(
                base_matrix=base_matrix,
                suite_key=suite_key,
                suite_entry=suite_entry,
                checkpoint=checkpoint,
                method_ids=method_ids,
            )
            matrix_path = matrix_dir / suite_key / f"{checkpoint['alias']}.yaml"
            _write_yaml(matrix_path, generated_matrix)
            if suite_entry.get("run_analysis", False):
                analysis_path = analysis_config_dir / suite_key / f"{checkpoint['alias']}.yaml"
                _write_yaml(
                    analysis_path,
                    _analysis_config(
                        suite_key=suite_key,
                        suite_entry=suite_entry,
                        checkpoint_alias=checkpoint["alias"],
                        matrix_path=matrix_path,
                        artifact_root=artifact_root,
                        analysis_root=analysis_root,
                    ),
                )
                analysis_records.append(
                    {
                        "suite": suite_key,
                        "checkpoint": checkpoint["alias"],
                        "analysis_config": str(analysis_path),
                        "analysis_output_dir": str(analysis_root / suite_key / checkpoint["alias"]),
                        "status": "planned",
                    }
                )
            for dataset_id in suite_entry.get("datasets", []):
                for method_id in method_ids:
                    if dataset_id not in base_matrix.get("datasets", {}):
                        raise KeyError(f"Dataset {dataset_id!r} is not defined in the base matrix.")
                    if method_id not in base_matrix.get("methods", {}):
                        raise KeyError(f"Method {method_id!r} is not defined in the base matrix.")
                    method_entry = _resolve_method(base_matrix["methods"], method_id)
                    output_dir = _run_output_dir(artifact_root, suite_entry["experiment_id"], dataset_id, method_id)
                    config_path = config_dir / suite_key / checkpoint["alias"] / f"{dataset_id}_{method_id}.yaml"
                    app_config = _build_app_config(
                        base_matrix=base_matrix,
                        suite_config=suite_config,
                        paths=paths,
                        suite_key=suite_key,
                        suite_entry=suite_entry,
                        checkpoint=checkpoint,
                        dataset_id=dataset_id,
                        method_id=method_id,
                        artifact_root=artifact_root,
                        smoke_test=args.smoke_test,
                    )
                    _write_yaml(config_path, app_config)
                    command = _command_for(config_path, method_entry["baseline"], args.python_bin)
                    run_plan.append((suite_key, suite_entry, checkpoint, dataset_id, method_id, output_dir, config_path, command))

    if not run_plan:
        raise RuntimeError("No runnable benchmark combinations were generated.")

    print(f"[plan] runs={len(run_plan)} artifact_root={manifest_dir}", flush=True)

    for index, (suite_key, _, checkpoint, dataset_id, method_id, output_dir, config_path, command) in enumerate(run_plan, start=1):
        prefix = (
            f"[{index}/{len(run_plan)}] suite={suite_key} ckpt={checkpoint['alias']} "
            f"model={checkpoint['model_id']} dataset={dataset_id} mode={method_id}"
        )
        if not args.rerun and _run_is_complete(output_dir):
            print(f"{prefix} skipped_existing", flush=True)
            records.append(
                _status_record(
                    status="skipped_existing",
                    suite_key=suite_key,
                    checkpoint=checkpoint,
                    dataset_id=dataset_id,
                    method_id=method_id,
                    output_dir=output_dir,
                    config_path=config_path,
                    command=command,
                )
            )
            _write_run_outputs(manifest_dir, records, failures)
            continue
        if args.dry_run:
            print(f"{prefix} dry_run", flush=True)
            print(" ".join(command), flush=True)
            records.append(
                _status_record(
                    status="dry_run",
                    suite_key=suite_key,
                    checkpoint=checkpoint,
                    dataset_id=dataset_id,
                    method_id=method_id,
                    output_dir=output_dir,
                    config_path=config_path,
                    command=command,
                )
            )
            continue
        print(f"{prefix} running", flush=True)
        result = _run_subprocess(command, env)
        if result.returncode == 0:
            records.append(
                _status_record(
                    status="completed",
                    suite_key=suite_key,
                    checkpoint=checkpoint,
                    dataset_id=dataset_id,
                    method_id=method_id,
                    output_dir=output_dir,
                    config_path=config_path,
                    command=command,
                    returncode=result.returncode,
                )
            )
        else:
            failure = _status_record(
                status="failed",
                suite_key=suite_key,
                checkpoint=checkpoint,
                dataset_id=dataset_id,
                method_id=method_id,
                output_dir=output_dir,
                config_path=config_path,
                command=command,
                returncode=result.returncode,
                message=f"Command failed with return code {result.returncode}.",
            )
            records.append(failure)
            failures.append(failure)
            print(f"{prefix} failed returncode={result.returncode}", flush=True)
            _write_run_outputs(manifest_dir, records, failures)
            if args.stop_on_error:
                break
        _write_run_outputs(manifest_dir, records, failures)

    if args.dry_run:
        for item in analysis_records:
            command = _analysis_command(Path(item["analysis_config"]), args.python_bin)
            print(f"[analysis dry_run] suite={item['suite']} ckpt={item['checkpoint']} {' '.join(command)}", flush=True)
    elif not args.no_analysis:
        for item in analysis_records:
            command = _analysis_command(Path(item["analysis_config"]), args.python_bin)
            print(f"[analysis] suite={item['suite']} ckpt={item['checkpoint']} running", flush=True)
            result = _run_subprocess(command, env)
            item["returncode"] = result.returncode
            item["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                item["message"] = f"Analysis failed with return code {result.returncode}."
                failures.append(
                    {
                        "status": "analysis_failed",
                        "suite": item["suite"],
                        "checkpoint": item["checkpoint"],
                        "command": " ".join(command),
                        "returncode": result.returncode,
                        "message": item["message"],
                    }
                )
                if args.stop_on_error:
                    break

    checkpoint_summary_outputs = _write_checkpoint_summary(manifest_dir, records) if not args.dry_run else {}
    manifest = {
        "created_at": created_at,
        "project_root": str(PROJECT_ROOT),
        "paths_config": str(args.paths.resolve()),
        "suite_config": str(args.suite_config.resolve()),
        "artifact_root": str(manifest_dir),
        "dry_run": args.dry_run,
        "smoke_test": args.smoke_test,
        "resume": not args.rerun,
        "run_count": len(run_plan),
        "completed_count": len([item for item in records if item["status"] == "completed"]),
        "skipped_existing_count": len([item for item in records if item["status"] == "skipped_existing"]),
        "failed_count": len(failures),
        "records": records,
        "analysis": analysis_records,
        "summary_outputs": checkpoint_summary_outputs,
        "failures": failures,
    }
    _write_final_manifest(manifest_dir, manifest)
    _write_run_outputs(manifest_dir, records, failures)
    print(
        f"[done] completed={manifest['completed_count']} skipped={manifest['skipped_existing_count']} failures={manifest['failed_count']}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
