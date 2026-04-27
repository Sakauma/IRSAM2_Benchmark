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

from ..core.fingerprints import sha256_file
from ..validation import validate_run_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_PY = PROJECT_ROOT / "main.py"
ANALYSIS_PY = PROJECT_ROOT / "scripts" / "analyze_paper_results.py"
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "server_benchmark_full.local.yaml"
REQUIRED_RUN_FILES = (
    "benchmark_spec.json",
    "run_metadata.json",
    "summary.json",
    "results.json",
    "eval_reports/rows.json",
)
DEFAULT_FAILURE_RATE_THRESHOLD = 0.05
FAILURE_LOG_TAIL_LINES = 40


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def _resolve_optional_project_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_config_from_env() -> Path | None:
    configured = os.environ.get("BENCHMARK_CONFIG")
    if configured:
        return _resolve_project_path(configured)
    return DEFAULT_BENCHMARK_CONFIG if DEFAULT_BENCHMARK_CONFIG.exists() else None


def _base_matrix_from_complete_config(raw: Dict[str, Any], source: Path) -> Dict[str, Any]:
    # 完整 YAML 同时保存 path/suite/matrix。这里抽出“实验矩阵”部分，
    # 形状保持和分析模块需要的 matrix 一致。
    required = ("runtime_defaults", "evaluation_defaults", "datasets", "methods")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Complete benchmark config {source} is missing required section(s): {', '.join(missing)}")
    return {
        "model_defaults": copy.deepcopy(raw.get("model_defaults", {})),
        "runtime_defaults": copy.deepcopy(raw["runtime_defaults"]),
        "evaluation_defaults": copy.deepcopy(raw["evaluation_defaults"]),
        "datasets": copy.deepcopy(raw["datasets"]),
        "methods": copy.deepcopy(raw["methods"]),
        "groups": copy.deepcopy(raw.get("groups", {})),
        "experiments": copy.deepcopy(raw.get("experiments", [])),
    }


def _suite_config_from_complete_config(raw: Dict[str, Any], source: Path) -> Dict[str, Any]:
    # 把完整 YAML 中的运行维度抽成 suite_config：
    # checkpoints/models 控制模型维度，modes 控制方法维度，suites 控制实验组合。
    benchmark = raw.get("benchmark", {})
    checkpoints = raw.get("checkpoints", raw.get("models", []))
    if not checkpoints:
        raise KeyError(f"Complete benchmark config {source} must define `models` or `checkpoints`.")
    if "modes" not in raw:
        raise KeyError(f"Complete benchmark config {source} must define `modes`.")
    if "suites" not in raw:
        raise KeyError(f"Complete benchmark config {source} must define `suites`.")
    return {
        "suite_name": raw.get("suite_name", benchmark.get("suite_name", source.stem)),
        "artifact_subdir": raw.get("artifact_subdir", benchmark.get("artifact_subdir", "paper_5090")),
        "runtime": copy.deepcopy(raw.get("runtime", {})),
        "smoke_test_runtime": copy.deepcopy(raw.get("smoke_test_runtime", {})),
        "checkpoints": copy.deepcopy(checkpoints),
        "modes": copy.deepcopy(raw["modes"]),
        "suites": copy.deepcopy(raw["suites"]),
        "analysis": copy.deepcopy(raw.get("analysis", {})),
    }


def _load_complete_benchmark_config(config_path: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str | None]]:
    # 推荐入口：一个 YAML 包含路径、模型、数据集、方法、suite 和分析配置。
    if not config_path.exists():
        raise FileNotFoundError(
            f"Complete benchmark config not found: {config_path}\n"
            "Create it from configs/server_benchmark_full.example.yaml."
        )
    raw = _load_yaml(config_path)
    paths = copy.deepcopy(raw.get("paths", {}))
    if raw.get("execution"):
        paths["execution"] = copy.deepcopy(raw["execution"])
    base_matrix = _base_matrix_from_complete_config(raw, config_path)
    suite_config = _suite_config_from_complete_config(raw, config_path)
    return paths, suite_config, base_matrix, {
        "mode": "complete",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "paths_config": None,
        "suite_config": None,
    }


def _artifact_base(paths: Dict[str, Any]) -> Path:
    raw = paths.get("artifacts", {}).get("root", "artifacts")
    return _resolve_project_path(str(raw))


def _reference_results_root(paths: Dict[str, Any]) -> Path | None:
    raw = paths.get("reference_results", {}).get("root")
    return _resolve_project_path(str(raw)) if raw else None


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


def _read_json_if_valid(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_is_complete(output_dir: Path) -> bool:
    # resume 不能只看目录存在；必须确认关键结果文件可读且 rows 非空。
    if not all((output_dir / relative).exists() for relative in REQUIRED_RUN_FILES):
        return False
    summary = _read_json_if_valid(output_dir / "summary.json")
    rows = _read_json_if_valid(output_dir / "eval_reports" / "rows.json")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        return False
    mean_metrics = summary.get("mean", {})
    if not isinstance(mean_metrics, dict) or not mean_metrics:
        return False
    if len(rows) <= 0:
        return False
    failure_rate = summary.get("failure_rate")
    threshold = summary.get("failure_rate_threshold", DEFAULT_FAILURE_RATE_THRESHOLD)
    if isinstance(failure_rate, (int, float)) and isinstance(threshold, (int, float)):
        return float(failure_rate) <= float(threshold)
    return True


def _runtime_config(
    base_runtime: Dict[str, Any],
    suite_config: Dict[str, Any],
    checkpoint: Dict[str, Any],
    paths: Dict[str, Any],
    smoke_test: bool,
) -> Dict[str, Any]:
    # runtime 覆盖顺序从低到高：
    # base defaults -> suite runtime -> checkpoint runtime -> smoke runtime -> machine paths runtime。
    runtime = _deep_merge(base_runtime, suite_config.get("runtime", {}))
    runtime = _deep_merge(runtime, checkpoint.get("runtime", {}))
    if smoke_test:
        runtime = _deep_merge(runtime, suite_config.get("smoke_test_runtime", {}))
    runtime = _deep_merge(runtime, paths.get("runtime", {}))
    reference_root = _reference_results_root(paths)
    if reference_root is not None:
        runtime["reference_results_root"] = str(reference_root)
    return runtime


def _model_config(checkpoint: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    model = copy.deepcopy(checkpoint)
    model.pop("alias", None)
    model.pop("runtime", None)
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
    source_config_path: Path,
    source_config_sha256: str,
) -> Dict[str, Any]:
    # 将矩阵中的一个组合展开成 main.py 可直接读取的 AppConfig YAML。
    # 这里会写入绝对 dataset/checkpoint/artifact 路径，保证 generated config 可单独复跑。
    dataset_entry = base_matrix["datasets"][dataset_id]
    method_entry = _resolve_method(base_matrix["methods"], method_id)
    dataset_config = copy.deepcopy(dataset_entry["config"])
    dataset_config["root"] = _resolve_dataset_root(dataset_entry, paths, dataset_id)
    runtime = _runtime_config(base_matrix.get("runtime_defaults", {}), suite_config, checkpoint, paths, smoke_test)
    runtime["artifact_root"] = str(artifact_root)
    runtime["output_name"] = f"{suite_entry['experiment_id']}/{dataset_id}/{method_id}"
    evaluation = _deep_merge(base_matrix["evaluation_defaults"], method_entry.get("evaluation", {}))
    return {
        "model": _model_config(checkpoint, paths),
        "dataset": dataset_config,
        "runtime": runtime,
        "evaluation": evaluation,
        "method": copy.deepcopy(method_entry.get("method", {"name": method_id, "modality": "ir"})),
        "fingerprints": {
            "source_config": str(source_config_path.resolve()),
            "source_config_sha256": source_config_sha256,
            "suite": suite_key,
            "checkpoint": checkpoint["alias"],
            "dataset_id": dataset_id,
            "method_id": method_id,
        },
    }


def _build_generated_matrix(
    *,
    base_matrix: Dict[str, Any],
    suite_key: str,
    suite_entry: Dict[str, Any],
    checkpoint: Dict[str, Any],
    method_ids: List[str],
) -> Dict[str, Any]:
    # 分析脚本仍按 paper matrix 读取输入。每个 checkpoint 生成一个裁剪后的 matrix，
    # 只包含当前 suite 会产生的 datasets/methods，避免分析误报缺失 run。
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
    model_defaults.pop("runtime", None)
    return {
        "model_defaults": model_defaults,
        "runtime_defaults": copy.deepcopy(base_matrix.get("runtime_defaults", {})),
        "evaluation_defaults": copy.deepcopy(base_matrix.get("evaluation_defaults", {})),
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
    analysis_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    # 每个 checkpoint 单独分析，统计比较在同一 checkpoint 内做 sample-level paired test。
    primary_metric = suite_entry.get("primary_metric", analysis_defaults.get("primary_metric", "mIoU"))
    case_selection = _deep_merge(
        {"top_k": 8, "primary_metric": primary_metric},
        analysis_defaults.get("case_selection", {}),
    )
    case_selection = _deep_merge(case_selection, suite_entry.get("case_selection", {}))
    case_selection["primary_metric"] = case_selection.get("primary_metric", primary_metric)
    statistics = _deep_merge(
        {"n_bootstrap": 10000, "ci": 0.95, "random_seed": 42, "low_power_threshold": 20},
        analysis_defaults.get("statistics", {}),
    )
    statistics = _deep_merge(statistics, suite_entry.get("statistics", {}))
    statistics["comparisons"] = list(suite_entry.get("comparisons", statistics.get("comparisons", [])))
    return {
        "artifact_root": str(artifact_root),
        "output_dir": str(analysis_root / suite_key / checkpoint_alias),
        "matrix": str(matrix_path),
        "experiment_groups": [suite_entry["experiment_id"]],
        "primary_metric": primary_metric,
        "metrics": list(suite_entry.get("metrics", analysis_defaults.get("metrics", ["mIoU", "Dice", "LatencyMs"]))),
        "lower_is_better": list(suite_entry.get("lower_is_better", analysis_defaults.get("lower_is_better", ["LatencyMs"]))),
        "group_keys": list(
            suite_entry.get(
                "group_keys",
                analysis_defaults.get(
                    "group_keys",
                    ["dataset", "method", "eval_unit", "supervision_type", "target_scale", "annotation_protocol_flag"],
                ),
            )
        ),
        "case_selection": case_selection,
        "statistics": statistics,
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
    # 子进程环境由 runner 统一设置，确保用户从任意 shell 启动都能 import 本项目 src。
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    execution = paths.get("execution", {})
    if execution.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(execution["cuda_visible_devices"])
    alloc_conf = execution.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
    if alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    return env


def _log_path(manifest_dir: Path, suite_key: str, checkpoint_alias: str, stem: str) -> Path:
    return manifest_dir / "logs" / suite_key / checkpoint_alias / f"{stem}.log"


def _tail_text(path: Path, max_lines: int = FAILURE_LOG_TAIL_LINES) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_subprocess(command: List[str], env: Dict[str, str], log_path: Path, *, stream_logs: bool = False) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.flush()
        if not stream_logs:
            return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False, text=True, stdout=handle, stderr=subprocess.STDOUT)
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if process.stdout is None:
            return subprocess.CompletedProcess(command, process.wait())
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            handle.write(text)
            handle.flush()
            sys.stderr.write(text)
            sys.stderr.flush()
        return subprocess.CompletedProcess(command, process.wait())


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
    config_sha256: str,
    log_path: Path | None = None,
    returncode: int | None = None,
    message: str = "",
    log_tail: str = "",
    validation_errors: List[str] | None = None,
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
        "config_sha256": config_sha256,
        "command": " ".join(command),
        "log_path": "" if log_path is None else str(log_path),
        "returncode": returncode,
        "message": message,
        "log_tail": log_tail,
        "validation_errors": [] if validation_errors is None else validation_errors,
    }


def _iter_requested_suites(suite_config: Dict[str, Any], selected_suites: set[str] | None) -> Iterable[tuple[str, Dict[str, Any]]]:
    # 未显式选择 suite 时，enabled: false 的 suite 会被跳过；
    # 显式 --suites 指定时允许强制运行某个默认关闭的 suite。
    explicit = selected_suites is not None
    for suite_key, suite_entry in suite_config.get("suites", {}).items():
        if selected_suites is not None and suite_key not in selected_suites:
            continue
        if not explicit and suite_entry.get("enabled", True) is False:
            continue
        yield suite_key, suite_entry


def _suite_checkpoints(all_checkpoints: List[Dict[str, Any]], suite_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = suite_entry.get("checkpoints", suite_entry.get("models"))
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
    # checkpoint_sweep_summary.csv 是跨 checkpoint 快速对比表；
    # 它只汇总 completed/skipped 且完整的 run，失败 run 不混入均值。
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
            "expected_row_count": summary.get("expected_row_count", ""),
            "row_count": summary.get("row_count", ""),
            "error_count": summary.get("error_count", ""),
            "failure_rate": summary.get("failure_rate", ""),
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


def validate_complete_config(
    *,
    paths: Dict[str, Any],
    suite_config: Dict[str, Any],
    base_matrix: Dict[str, Any],
    selected_suites: set[str] | None,
    selected_checkpoints: set[str] | None,
    selected_modes: set[str] | None,
) -> None:
    errors: List[str] = []

    suites = suite_config.get("suites", {})
    if selected_suites is not None:
        unknown = sorted(selected_suites - {str(key) for key in suites})
        errors.extend(f"Unknown suite: {item}" for item in unknown)

    checkpoint_entries = suite_config.get("checkpoints", [])
    checkpoint_aliases = {str(item.get("alias", "")) for item in checkpoint_entries}
    if selected_checkpoints is not None:
        unknown = sorted(selected_checkpoints - checkpoint_aliases)
        errors.extend(f"Unknown checkpoint alias: {item}" for item in unknown)
    selected_checkpoint_entries = _select_by_alias(checkpoint_entries, selected_checkpoints, "alias")
    if not selected_checkpoint_entries:
        errors.append("No checkpoints selected.")

    mode_entries = suite_config.get("modes", [])
    mode_names = {str(item.get("method", "")) for item in mode_entries} | {str(item.get("alias", "")) for item in mode_entries if item.get("alias")}
    if selected_modes is not None:
        unknown = sorted(selected_modes - mode_names)
        errors.extend(f"Unknown mode or method: {item}" for item in unknown)
    selected_mode_entries = _select_modes(mode_entries, selected_modes)
    if not selected_mode_entries:
        errors.append("No modes selected.")

    sam2_repo = paths.get("sam2", {}).get("repo")
    if sam2_repo:
        repo_path = _resolve_project_path(str(sam2_repo))
        if not repo_path.exists():
            errors.append(f"SAM2 repo does not exist: {repo_path}")

    for checkpoint in selected_checkpoint_entries:
        for key in ("alias", "model_id", "cfg", "ckpt"):
            if key not in checkpoint:
                errors.append(f"Checkpoint entry is missing {key}: {checkpoint!r}")
        if "ckpt" in checkpoint:
            ckpt_path = Path(_resolve_checkpoint_path(checkpoint, paths))
            if not ckpt_path.is_absolute():
                ckpt_path = PROJECT_ROOT / ckpt_path
            if not ckpt_path.exists():
                errors.append(f"Checkpoint file does not exist for {checkpoint.get('alias', '<unknown>')}: {ckpt_path}")

    for mode in selected_mode_entries:
        method_id = str(mode.get("method", ""))
        if method_id not in base_matrix.get("methods", {}):
            errors.append(f"Mode references undefined method: {method_id}")
            continue
        method_entry = _resolve_method(base_matrix["methods"], method_id)
        if not method_entry.get("baseline"):
            errors.append(f"Method is missing baseline: {method_id}")

    for suite_key, suite_entry in _iter_requested_suites(suite_config, selected_suites):
        if "experiment_id" not in suite_entry:
            errors.append(f"Suite {suite_key!r} is missing experiment_id.")
        for dataset_id in suite_entry.get("datasets", []):
            if dataset_id not in base_matrix.get("datasets", {}):
                errors.append(f"Suite {suite_key!r} references undefined dataset: {dataset_id}")
                continue
            dataset_root = Path(_resolve_dataset_root(base_matrix["datasets"][dataset_id], paths, dataset_id))
            if not dataset_root.exists():
                errors.append(f"Dataset root does not exist for {dataset_id}: {dataset_root}")
        for method_id in _suite_method_ids(selected_mode_entries, suite_entry):
            if method_id not in base_matrix.get("methods", {}):
                errors.append(f"Suite {suite_key!r} references undefined method: {method_id}")

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"Invalid complete benchmark config:\n{joined}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 5090 single-GPU full SAM2 IR benchmark and analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Complete benchmark YAML containing paths, models, datasets, methods, suites, runtime, and analysis settings.",
    )
    parser.add_argument("--suites", help="Comma-separated suite keys. Default: all suites from suite config.")
    parser.add_argument("--checkpoints", help="Comma-separated checkpoint aliases. Default: all four official SAM2.1 checkpoints.")
    parser.add_argument("--modes", help="Comma-separated method ids. Default: all configured modes.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even when required result files already exist.")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--stream-logs", action="store_true", help="Mirror child process logs to stderr while still writing per-run log files.")
    args = parser.parse_args(argv)

    config_path = _resolve_optional_project_path(args.config) or _default_config_from_env()
    if config_path is None:
        raise FileNotFoundError(
            "Complete benchmark config not found. Pass --config or create configs/server_benchmark_full.local.yaml "
            "from configs/server_benchmark_full.example.yaml."
        )
    source_config_path = config_path
    paths, suite_config, base_matrix, config_sources = _load_complete_benchmark_config(config_path)
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
    validate_complete_config(
        paths=paths,
        suite_config=suite_config,
        base_matrix=base_matrix,
        selected_suites=selected_suites,
        selected_checkpoints=selected_checkpoints,
        selected_modes=selected_modes,
    )
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
        # 展开顺序固定为 suite -> checkpoint -> dataset -> method，
        # 这样日志中的 [i/N] 与生成目录结构一致，便于中断后定位。
        suite_checkpoints = _suite_checkpoints(checkpoints, suite_entry)
        method_ids = _suite_method_ids(mode_entries, suite_entry)
        if not suite_checkpoints or not method_ids:
            continue
        for checkpoint in suite_checkpoints:
            # 每个 checkpoint 独立 artifact_root，避免不同模型结果互相覆盖。
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
            matrix_config_sha256 = sha256_file(matrix_path)
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
                        analysis_defaults=suite_config.get("analysis", {}),
                    ),
                )
                analysis_records.append(
                    {
                        "suite": suite_key,
                        "checkpoint": checkpoint["alias"],
                        "analysis_config": str(analysis_path),
                        "analysis_config_sha256": sha256_file(analysis_path),
                        "matrix_config_sha256": matrix_config_sha256,
                        "analysis_output_dir": str(analysis_root / suite_key / checkpoint["alias"]),
                        "log_path": str(_log_path(manifest_dir, suite_key, checkpoint["alias"], "analysis")),
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
                        source_config_path=source_config_path,
                        source_config_sha256=str(config_sources["config_sha256"]),
                    )
                    _write_yaml(config_path, app_config)
                    config_sha256 = sha256_file(config_path)
                    command = _command_for(config_path, method_entry["baseline"], args.python_bin)
                    log_path = _log_path(manifest_dir, suite_key, checkpoint["alias"], f"{dataset_id}_{method_id}")
                    run_plan.append((suite_key, suite_entry, checkpoint, dataset_id, method_id, output_dir, config_path, config_sha256, command, log_path))

    if not run_plan:
        raise RuntimeError("No runnable benchmark combinations were generated.")

    print(f"[plan] runs={len(run_plan)} artifact_root={manifest_dir}", flush=True)

    for index, (suite_key, _, checkpoint, dataset_id, method_id, output_dir, config_path, config_sha256, command, log_path) in enumerate(run_plan, start=1):
        prefix = (
            f"[{index}/{len(run_plan)}] suite={suite_key} ckpt={checkpoint['alias']} "
            f"model={checkpoint['model_id']} dataset={dataset_id} mode={method_id}"
        )
        if not args.rerun and _run_is_complete(output_dir):
            # resume 模式默认跳过完整 run；--rerun 会强制覆盖重跑。
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
                    config_sha256=config_sha256,
                    command=command,
                    log_path=log_path,
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
                    config_sha256=config_sha256,
                    command=command,
                    log_path=log_path,
                )
            )
            continue
        print(f"{prefix} running", flush=True)
        run_kwargs = {"stream_logs": True} if args.stream_logs else {}
        result = _run_subprocess(command, env, log_path, **run_kwargs)
        if result.returncode == 0:
            validation = validate_run_artifacts(output_dir)
            if validation["valid"]:
                records.append(
                    _status_record(
                        status="completed",
                        suite_key=suite_key,
                        checkpoint=checkpoint,
                        dataset_id=dataset_id,
                        method_id=method_id,
                        output_dir=output_dir,
                        config_path=config_path,
                        config_sha256=config_sha256,
                        command=command,
                        log_path=log_path,
                        returncode=result.returncode,
                    )
                )
            else:
                failure = _status_record(
                    status="failed_invalid_artifacts",
                    suite_key=suite_key,
                    checkpoint=checkpoint,
                    dataset_id=dataset_id,
                    method_id=method_id,
                    output_dir=output_dir,
                    config_path=config_path,
                    config_sha256=config_sha256,
                    command=command,
                    log_path=log_path,
                    returncode=result.returncode,
                    message="Command returned 0 but produced invalid artifacts.",
                    log_tail=_tail_text(log_path),
                    validation_errors=list(validation["errors"]),
                )
                records.append(failure)
                failures.append(failure)
                print(f"{prefix} failed invalid_artifacts", flush=True)
                _write_run_outputs(manifest_dir, records, failures)
                if args.stop_on_error:
                    break
        else:
            failure = _status_record(
                status="failed",
                suite_key=suite_key,
                checkpoint=checkpoint,
                dataset_id=dataset_id,
                method_id=method_id,
                output_dir=output_dir,
                config_path=config_path,
                config_sha256=config_sha256,
                command=command,
                log_path=log_path,
                returncode=result.returncode,
                message=f"Command failed with return code {result.returncode}.",
                log_tail=_tail_text(log_path),
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
        # 分析阶段在所有 run 结束后执行，保证统计表能看到完整 checkpoint 内的所有组合。
        for item in analysis_records:
            command = _analysis_command(Path(item["analysis_config"]), args.python_bin)
            print(f"[analysis] suite={item['suite']} ckpt={item['checkpoint']} running", flush=True)
            analysis_log_path = Path(item["log_path"])
            run_kwargs = {"stream_logs": True} if args.stream_logs else {}
            result = _run_subprocess(command, env, analysis_log_path, **run_kwargs)
            item["returncode"] = result.returncode
            item["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                item["message"] = f"Analysis failed with return code {result.returncode}."
                item["log_tail"] = _tail_text(analysis_log_path)
                failures.append(
                    {
                        "status": "analysis_failed",
                        "suite": item["suite"],
                        "checkpoint": item["checkpoint"],
                        "command": " ".join(command),
                        "log_path": str(analysis_log_path),
                        "returncode": result.returncode,
                        "message": item["message"],
                        "log_tail": item["log_tail"],
                    }
                )
                if args.stop_on_error:
                    break

    checkpoint_summary_outputs = _write_checkpoint_summary(manifest_dir, records) if not args.dry_run else {}
    manifest = {
        # benchmark_manifest_latest.json 是本次运行的审计入口：
        # 记录配置来源、展开后的 run 数、每个子进程命令和分析状态。
        "created_at": created_at,
        "project_root": str(PROJECT_ROOT),
        "config_mode": config_sources["mode"],
        "config": config_sources["config"],
        "config_sha256": config_sources["config_sha256"],
        "paths_config": config_sources["paths_config"],
        "suite_config": config_sources["suite_config"],
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
