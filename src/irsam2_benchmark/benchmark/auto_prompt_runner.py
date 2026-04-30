#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from . import full_runner as fr
from ..core.fingerprints import sha256_file
from ..validation import validate_run_artifacts


PROJECT_ROOT = fr.PROJECT_ROOT
TRAIN_AUTO_PROMPT_PY = PROJECT_ROOT / "scripts" / "train_auto_prompt.py"
DEFAULT_AUTO_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "server_auto_prompt_4090x4.local.yaml"
DEFAULT_AUTO_PROMPT_EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "server_auto_prompt_4090x4.example.yaml"

DEFAULT_LEARNED_METHODS: Dict[str, Dict[str, Any]] = {
    "sam2_learned_auto_point": {
        "baseline": "sam2_learned_auto_point_prompt",
        "method": {"name": "sam2_learned_auto_point", "family": "sam2_learned_auto_prompt", "modality": "ir"},
        "evaluation": {"inference_mode": "point", "prompt_policy": {"name": "learned_auto_point", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1}},
    },
    "sam2_learned_auto_box": {
        "baseline": "sam2_learned_auto_box_prompt",
        "method": {"name": "sam2_learned_auto_box", "family": "sam2_learned_auto_prompt", "modality": "ir"},
        "evaluation": {"inference_mode": "box", "prompt_policy": {"name": "learned_auto_box", "prompt_type": "box", "prompt_source": "synthesized", "prompt_budget": 1}},
    },
    "sam2_learned_auto_box_point": {
        "baseline": "sam2_learned_auto_box_point_prompt",
        "method": {"name": "sam2_learned_auto_box_point", "family": "sam2_learned_auto_prompt", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "learned_auto_box_point", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 2},
        },
    },
    "sam2_learned_auto_box_point_neg": {
        "baseline": "sam2_learned_auto_box_point_neg_prompt",
        "method": {"name": "sam2_learned_auto_box_point_neg", "family": "sam2_learned_auto_prompt", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "learned_auto_box_point_neg", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 6},
        },
    },
}

DEFAULT_HEURISTIC_METHODS: Dict[str, Dict[str, Any]] = {
    "sam2_heuristic_auto_point": {
        "baseline": "sam2_heuristic_auto_point_prompt",
        "method": {"name": "sam2_heuristic_auto_point", "family": "sam2_heuristic_auto_prompt", "modality": "ir"},
        "evaluation": {"inference_mode": "point", "prompt_policy": {"name": "heuristic_auto_point", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1}},
    },
    "sam2_heuristic_auto_box": {
        "baseline": "sam2_heuristic_auto_box_prompt",
        "method": {"name": "sam2_heuristic_auto_box", "family": "sam2_heuristic_auto_prompt", "modality": "ir"},
        "evaluation": {"inference_mode": "box", "prompt_policy": {"name": "heuristic_auto_box", "prompt_type": "box", "prompt_source": "synthesized", "prompt_budget": 1}},
    },
    "sam2_heuristic_auto_box_point": {
        "baseline": "sam2_heuristic_auto_box_point_prompt",
        "method": {"name": "sam2_heuristic_auto_box_point", "family": "sam2_heuristic_auto_prompt", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "heuristic_auto_box_point", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 2},
        },
    },
    "sam2_heuristic_auto_box_point_neg": {
        "baseline": "sam2_heuristic_auto_box_point_neg_prompt",
        "method": {"name": "sam2_heuristic_auto_box_point_neg", "family": "sam2_heuristic_auto_prompt", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "heuristic_auto_box_point_neg", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 6},
        },
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _default_config() -> Path:
    if DEFAULT_AUTO_PROMPT_CONFIG.exists():
        return DEFAULT_AUTO_PROMPT_CONFIG
    return DEFAULT_AUTO_PROMPT_EXAMPLE_CONFIG


def _auto_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(raw.get("auto_prompt", {}))
    config.setdefault("experiment_id", "sam2_ir_qd_m1_auto_prompt")
    config.setdefault("artifact_subdir", "sam2_ir_qd_m1_auto_prompt")
    config.setdefault("train_datasets", ["nuaa_sirst", "nudt_sirst", "irstd_1k", "rbgt_tiny_ir_box"])
    config.setdefault("eval_suites", ["auto_prompt"])
    config.setdefault("train_gpu", "0")
    config.setdefault("eval_gpus", ["1", "6", "7", "0"])
    config.setdefault("checkpoint_name", "checkpoint.pt")
    config.setdefault("train", {})
    config.setdefault("model", {})
    config.setdefault("target", {})
    config.setdefault("heatmaps", {})
    return config


def _ensure_auto_methods(base_matrix: Dict[str, Any]) -> None:
    methods = base_matrix.setdefault("methods", {})
    for method_id, payload in {**DEFAULT_HEURISTIC_METHODS, **DEFAULT_LEARNED_METHODS}.items():
        methods.setdefault(method_id, copy.deepcopy(payload))


def _inject_learned_prompt_config(
    *,
    base_matrix: Dict[str, Any],
    checkpoint_path: Path,
    auto_config: Dict[str, Any],
    heatmap_root: Path,
) -> None:
    prompt_runtime = dict(auto_config.get("prompt_runtime", {}))
    for method_id, method_entry in base_matrix.get("methods", {}).items():
        baseline = str(method_entry.get("baseline", ""))
        if not baseline.startswith("sam2_learned_auto_"):
            continue
        method_payload = method_entry.setdefault("method", {})
        method_payload["prompt_checkpoint"] = str(checkpoint_path)
        method_payload["prompt_device"] = "cuda"
        method_payload["prompt_top_k"] = int(prompt_runtime.get("top_k", auto_config.get("top_k", 1)))
        method_payload["prompt_point_budget"] = int(prompt_runtime.get("point_budget", auto_config.get("point_budget", 1)))
        method_payload["prompt_response_threshold"] = float(prompt_runtime.get("response_threshold", auto_config.get("response_threshold", 0.0)))
        method_payload["prompt_nms_radius"] = int(prompt_runtime.get("nms_radius", auto_config.get("nms_radius", 4)))
        method_payload["prompt_min_box_side"] = float(prompt_runtime.get("min_box_side", auto_config.get("model", {}).get("min_box_side", 2.0)))
        method_payload["prompt_negative_ring_offset"] = float(
            prompt_runtime.get("negative_ring_offset", auto_config.get("model", {}).get("negative_ring_offset", 4.0))
        )
        method_payload["prompt_use_local_contrast"] = bool(auto_config.get("model", {}).get("use_local_contrast", True))
        method_payload["prompt_use_top_hat"] = bool(auto_config.get("model", {}).get("use_top_hat", True))
        method_payload["heatmaps"] = {
            "enabled": bool(auto_config.get("heatmaps", {}).get("eval_enabled", True)),
            "root": str(heatmap_root),
            "experiment_id": str(auto_config["experiment_id"]),
            "sample_limit": int(auto_config.get("heatmaps", {}).get("eval_sample_limit", 16)),
        }


def _training_dataset_config(
    *,
    base_matrix: Dict[str, Any],
    suite_config: Dict[str, Any],
    paths: Dict[str, Any],
    dataset_id: str,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_entry = base_matrix["datasets"][dataset_id]
    dataset_config = copy.deepcopy(dataset_entry["config"])
    dataset_config["root"] = fr._resolve_dataset_root(dataset_entry, paths, dataset_id)
    runtime = copy.deepcopy(base_matrix.get("runtime_defaults", {}))
    runtime = fr._deep_merge(runtime, suite_config.get("runtime", {}))
    runtime = fr._deep_merge(runtime, paths.get("runtime", {}))
    runtime.setdefault("artifact_root", "artifacts")
    runtime.setdefault("reference_results_root", "reference_results")
    runtime.setdefault("output_name", f"auto_prompt_train/{dataset_id}")
    runtime["save_visuals"] = False
    runtime["update_reference_results"] = False
    return {
        "model": fr._model_config(checkpoint, paths),
        "dataset": dataset_config,
        "runtime": runtime,
        "evaluation": copy.deepcopy(base_matrix["evaluation_defaults"]),
        "method": {"name": "auto_prompt_training_dataset"},
    }


def _write_training_configs(
    *,
    config_path: Path,
    raw: Dict[str, Any],
    paths: Dict[str, Any],
    suite_config: Dict[str, Any],
    base_matrix: Dict[str, Any],
    auto_config: Dict[str, Any],
    generated_dir: Path,
    output_root: Path,
    show_progress: bool,
    progress_backend: str,
) -> tuple[Path, Path]:
    dataset_config_dir = generated_dir / "training_dataset_configs"
    checkpoints = suite_config.get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError("Auto prompt runner requires at least one SAM2 checkpoint entry for dataset config generation.")
    checkpoint = checkpoints[0]
    dataset_config_paths: list[str] = []
    for dataset_id in auto_config["train_datasets"]:
        dataset_payload = _training_dataset_config(
            base_matrix=base_matrix,
            suite_config=suite_config,
            paths=paths,
            dataset_id=str(dataset_id),
            checkpoint=checkpoint,
        )
        dataset_config_path = dataset_config_dir / f"{dataset_id}.yaml"
        fr._write_yaml(dataset_config_path, dataset_payload)
        dataset_config_paths.append(str(dataset_config_path))

    train_settings = copy.deepcopy(auto_config.get("train", {}))
    train_payload = {
        "experiment_id": auto_config["experiment_id"],
        "output_root": str(output_root),
        "dataset_configs": dataset_config_paths,
        "train": train_settings,
        "model": copy.deepcopy(auto_config.get("model", {})),
        "target": copy.deepcopy(auto_config.get("target", {})),
        "heatmaps": copy.deepcopy(auto_config.get("heatmaps", {})),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
    }
    if raw.get("auto_prompt_training"):
        train_payload = fr._deep_merge(train_payload, copy.deepcopy(raw["auto_prompt_training"]))
    train_payload.setdefault("train", {})
    train_payload["train"]["show_progress"] = bool(show_progress)
    train_payload["train"]["progress_backend"] = progress_backend if show_progress else "none"
    train_payload["train"].setdefault("progress_update_interval_s", 1.0)
    train_config_path = generated_dir / "auto_prompt_train.yaml"
    fr._write_yaml(train_config_path, train_payload)
    return train_config_path, output_root / str(auto_config["experiment_id"]) / str(auto_config["checkpoint_name"])


def _env_for_gpu(paths: Dict[str, Any], gpu: str) -> Dict[str, str]:
    env = fr._build_env(paths)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run_logged(command: list[str], *, env: Dict[str, str], log_path: Path, stream_logs: bool = False) -> subprocess.CompletedProcess[str]:
    return fr._run_subprocess(command, env, log_path, stream_logs=stream_logs)


def _tee_process_output(process: subprocess.Popen[bytes], handle: Any) -> None:
    if process.stdout is None:
        return
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        handle.write(text)
        handle.flush()
        sys.stderr.write(text)
        sys.stderr.flush()


def _start_logged(command: list[str], *, env: Dict[str, str], log_path: Path, stream_logs: bool = False) -> tuple[subprocess.Popen[Any], Any, threading.Thread | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write(f"$ {' '.join(command)}\n\n")
    handle.flush()
    if not stream_logs:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
        return process, handle, None
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    thread = threading.Thread(target=_tee_process_output, args=(process, handle), daemon=True)
    thread.start()
    return process, handle, thread


def _make_eval_progress(*, total: int, enabled: bool) -> Any | None:
    if not enabled or total <= 0:
        return None
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(total=total, unit="run", desc="eval runs", dynamic_ncols=True, leave=True, position=0, file=sys.stderr)


def _set_eval_progress(progress: Any | None, *, counts: Dict[str, int], active: int, queued: int) -> None:
    if progress is None:
        return
    progress.set_postfix(
        completed=counts.get("completed", 0),
        skipped=counts.get("skipped", 0),
        failed=counts.get("failed", 0),
        dry_run=counts.get("dry_run", 0),
        active=active,
        queued=queued,
    )


def _advance_eval_progress(progress: Any | None, *, status: str, counts: Dict[str, int], active: int, queued: int) -> None:
    if progress is None:
        return
    if status == "completed":
        counts["completed"] = counts.get("completed", 0) + 1
    elif status == "skipped_existing":
        counts["skipped"] = counts.get("skipped", 0) + 1
    elif status == "dry_run":
        counts["dry_run"] = counts.get("dry_run", 0) + 1
    else:
        counts["failed"] = counts.get("failed", 0) + 1
    progress.set_postfix(
        completed=counts.get("completed", 0),
        skipped=counts.get("skipped", 0),
        failed=counts.get("failed", 0),
        dry_run=counts.get("dry_run", 0),
        active=active,
        queued=queued,
    )
    progress.update(1)


def _command_for_train(train_config_path: Path, python_bin: str) -> list[str]:
    return [python_bin, str(TRAIN_AUTO_PROMPT_PY), "--config", str(train_config_path)]


def _build_run_plan(
    *,
    base_matrix: Dict[str, Any],
    suite_config: Dict[str, Any],
    paths: Dict[str, Any],
    config_dir: Path,
    matrix_dir: Path,
    analysis_config_dir: Path,
    analysis_root: Path,
    source_config_path: Path,
    source_config_sha256: str,
    selected_suites: set[str] | None,
    selected_checkpoints: set[str] | None,
    selected_modes: set[str] | None,
    smoke_test: bool,
    python_bin: str,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    checkpoints = fr._select_by_alias(suite_config.get("checkpoints", []), selected_checkpoints, "alias")
    mode_entries = fr._select_modes(suite_config.get("modes", []), selected_modes)
    run_plan = []
    analysis_records: list[dict[str, Any]] = []
    artifact_base = fr._artifact_base(paths)
    artifact_subdir = str(suite_config.get("artifact_subdir", "sam2_ir_qd_m1_auto_prompt"))
    if smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    for suite_key, suite_entry in fr._iter_requested_suites(suite_config, selected_suites):
        suite_checkpoints = fr._suite_checkpoints(checkpoints, suite_entry)
        method_ids = fr._suite_method_ids(mode_entries, suite_entry)
        for checkpoint in suite_checkpoints:
            artifact_root = fr._run_artifact_root(artifact_base, artifact_subdir, suite_key, checkpoint["alias"])
            generated_matrix = fr._build_generated_matrix(
                base_matrix=base_matrix,
                suite_key=suite_key,
                suite_entry=suite_entry,
                checkpoint=checkpoint,
                method_ids=method_ids,
            )
            matrix_path = matrix_dir / suite_key / f"{checkpoint['alias']}.yaml"
            fr._write_yaml(matrix_path, generated_matrix)
            if suite_entry.get("run_analysis", False):
                analysis_path = analysis_config_dir / suite_key / f"{checkpoint['alias']}.yaml"
                fr._write_yaml(
                    analysis_path,
                    fr._analysis_config(
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
                        "analysis_output_dir": str(analysis_root / suite_key / checkpoint["alias"]),
                        "log_path": str(fr._log_path(artifact_base / artifact_subdir, suite_key, checkpoint["alias"], "analysis")),
                        "status": "planned",
                    }
                )
            for dataset_id in suite_entry.get("datasets", []):
                for method_id in method_ids:
                    method_entry = fr._resolve_method(base_matrix["methods"], method_id)
                    output_dir = fr._run_output_dir(artifact_root, suite_entry["experiment_id"], dataset_id, method_id)
                    app_config_path = config_dir / suite_key / checkpoint["alias"] / f"{dataset_id}_{method_id}.yaml"
                    app_config = fr._build_app_config(
                        base_matrix=base_matrix,
                        suite_config=suite_config,
                        paths=paths,
                        suite_key=suite_key,
                        suite_entry=suite_entry,
                        checkpoint=checkpoint,
                        dataset_id=dataset_id,
                        method_id=method_id,
                        artifact_root=artifact_root,
                        smoke_test=smoke_test,
                        show_progress=False,
                        progress_backend="none",
                        source_config_path=source_config_path,
                        source_config_sha256=source_config_sha256,
                    )
                    fr._write_yaml(app_config_path, app_config)
                    command = fr._command_for(app_config_path, method_entry["baseline"], python_bin)
                    log_path = fr._log_path(artifact_base / artifact_subdir, suite_key, checkpoint["alias"], f"{dataset_id}_{method_id}")
                    run_plan.append(
                        (
                            suite_key,
                            checkpoint,
                            dataset_id,
                            method_id,
                            output_dir,
                            app_config_path,
                            sha256_file(app_config_path),
                            command,
                            log_path,
                        )
                    )
    return run_plan, analysis_records


def _run_eval_plan(
    *,
    run_plan: list[tuple[Any, ...]],
    paths: Dict[str, Any],
    eval_gpus: list[str],
    manifest_dir: Path,
    run_id: str,
    dry_run: bool,
    rerun: bool,
    stop_on_error: bool,
    show_progress: bool,
    stream_logs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    queue = list(run_plan)
    active: list[dict[str, Any]] = []
    free_gpus = list(eval_gpus)
    total = len(queue)
    completed_slots = 0
    progress = _make_eval_progress(total=total, enabled=show_progress)
    progress_counts: Dict[str, int] = {"completed": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    while queue or active:
        while queue and free_gpus and not (stop_on_error and failures):
            suite_key, checkpoint, dataset_id, method_id, output_dir, config_path, config_sha256, command, log_path = queue.pop(0)
            gpu = free_gpus.pop(0)
            prefix = f"[{completed_slots + len(active) + 1}/{total}] gpu={gpu} suite={suite_key} ckpt={checkpoint['alias']} dataset={dataset_id} mode={method_id}"
            _set_eval_progress(progress, counts=progress_counts, active=len(active), queued=len(queue))
            if not rerun and fr._run_is_complete(output_dir):
                if progress is None:
                    print(f"{prefix} skipped_existing", flush=True)
                records.append(
                    fr._status_record(
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
                completed_slots += 1
                free_gpus.append(gpu)
                fr._write_run_outputs(manifest_dir, records, failures, run_id)
                _advance_eval_progress(progress, status="skipped_existing", counts=progress_counts, active=len(active), queued=len(queue))
                continue
            if dry_run:
                print(f"{prefix} dry_run CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}", flush=True)
                records.append(
                    fr._status_record(
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
                completed_slots += 1
                free_gpus.append(gpu)
                _advance_eval_progress(progress, status="dry_run", counts=progress_counts, active=len(active), queued=len(queue))
                continue
            if progress is None:
                print(f"{prefix} running", flush=True)
            process, handle, thread = _start_logged(command, env=_env_for_gpu(paths, gpu), log_path=log_path, stream_logs=stream_logs)
            active.append(
                {
                    "process": process,
                    "handle": handle,
                    "thread": thread,
                    "gpu": gpu,
                    "prefix": prefix,
                    "suite_key": suite_key,
                    "checkpoint": checkpoint,
                    "dataset_id": dataset_id,
                    "method_id": method_id,
                    "output_dir": output_dir,
                    "config_path": config_path,
                    "config_sha256": config_sha256,
                    "command": command,
                    "log_path": log_path,
                }
            )
        if dry_run:
            break
        for item in list(active):
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            if item["thread"] is not None:
                item["thread"].join(timeout=5.0)
            item["handle"].close()
            active.remove(item)
            free_gpus.append(str(item["gpu"]))
            completed_slots += 1
            if returncode == 0 and validate_run_artifacts(item["output_dir"])["valid"]:
                if progress is None:
                    print(f"{item['prefix']} completed", flush=True)
                records.append(
                    fr._status_record(
                        status="completed",
                        suite_key=item["suite_key"],
                        checkpoint=item["checkpoint"],
                        dataset_id=item["dataset_id"],
                        method_id=item["method_id"],
                        output_dir=item["output_dir"],
                        config_path=item["config_path"],
                        config_sha256=item["config_sha256"],
                        command=item["command"],
                        log_path=item["log_path"],
                        returncode=returncode,
                    )
                )
                _advance_eval_progress(progress, status="completed", counts=progress_counts, active=len(active), queued=len(queue))
            else:
                validation = validate_run_artifacts(item["output_dir"]) if returncode == 0 else {"errors": []}
                status = "failed_invalid_artifacts" if returncode == 0 else "failed"
                print(f"{item['prefix']} {status}", flush=True)
                failure = fr._status_record(
                    status=status,
                    suite_key=item["suite_key"],
                    checkpoint=item["checkpoint"],
                    dataset_id=item["dataset_id"],
                    method_id=item["method_id"],
                    output_dir=item["output_dir"],
                    config_path=item["config_path"],
                    config_sha256=item["config_sha256"],
                    command=item["command"],
                    log_path=item["log_path"],
                    returncode=returncode,
                    message=f"Command finished with status={status}.",
                    log_tail=fr._tail_text(item["log_path"]),
                    validation_errors=list(validation.get("errors", [])),
                )
                records.append(failure)
                failures.append(failure)
                _advance_eval_progress(progress, status=status, counts=progress_counts, active=len(active), queued=len(queue))
            fr._write_run_outputs(manifest_dir, records, failures, run_id)
        if stop_on_error and failures:
            for item in active:
                item["process"].terminate()
                try:
                    item["process"].wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    item["process"].kill()
                    item["process"].wait(timeout=5.0)
                if item["thread"] is not None:
                    item["thread"].join(timeout=5.0)
                item["handle"].close()
            break
        if active:
            _set_eval_progress(progress, counts=progress_counts, active=len(active), queued=len(queue))
            time.sleep(1.0)
    if progress is not None:
        progress.close()
    fr._write_run_outputs(manifest_dir, records, failures, run_id)
    return records, failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAM2-IR-QD M1 learned auto prompt training and E2 evaluation on 4x4090.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--suites", help="Comma-separated suite keys. Default: auto_prompt from config.")
    parser.add_argument("--checkpoints", help="Comma-separated checkpoint aliases. Default: suite-selected checkpoints.")
    parser.add_argument("--modes", help="Comma-separated method ids or mode aliases.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Rerun evaluation runs even if complete artifacts exist.")
    parser.add_argument("--rerun-train", action="store_true", help="Retrain auto prompt even if checkpoint exists.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--stream-logs", action="store_true", help="Mirror child process logs to stderr while still writing log files.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-backend", choices=("auto", "tqdm", "line", "none"), default="tqdm")
    args = parser.parse_args(argv)

    config_path = (args.config if args.config is not None else _default_config()).resolve()
    raw = _load_yaml(config_path)
    auto_config = _auto_config(raw)
    paths, suite_config, base_matrix, config_sources = fr._load_complete_benchmark_config(config_path)
    _ensure_auto_methods(base_matrix)
    artifact_base = fr._artifact_base(paths)
    artifact_subdir = str(auto_config.get("artifact_subdir") or suite_config.get("artifact_subdir", "sam2_ir_qd_m1_auto_prompt"))
    if args.smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    suite_config["artifact_subdir"] = artifact_subdir
    manifest_dir = artifact_base / artifact_subdir
    generated_dir = manifest_dir / "generated"
    output_root = artifact_base / artifact_subdir / "train"
    train_config_path, auto_checkpoint_path = _write_training_configs(
        config_path=config_path,
        raw=raw,
        paths=paths,
        suite_config=suite_config,
        base_matrix=base_matrix,
        auto_config=auto_config,
        generated_dir=generated_dir,
        output_root=output_root,
        show_progress=not args.no_progress,
        progress_backend=args.progress_backend,
    )
    _inject_learned_prompt_config(
        base_matrix=base_matrix,
        checkpoint_path=auto_checkpoint_path,
        auto_config=auto_config,
        heatmap_root=artifact_base / artifact_subdir / "heatmaps",
    )
    selected_suites = fr._split_filter(args.suites) or set(str(item) for item in auto_config.get("eval_suites", ["auto_prompt"]))
    selected_checkpoints = fr._split_filter(args.checkpoints)
    selected_modes = fr._split_filter(args.modes)
    fr.validate_complete_config(
        paths=paths,
        suite_config=suite_config,
        base_matrix=base_matrix,
        selected_suites=selected_suites,
        selected_checkpoints=selected_checkpoints,
        selected_modes=selected_modes,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = fr._manifest_run_id(created_at)
    train_log = manifest_dir / "logs" / "auto_prompt_train.log"
    train_command = _command_for_train(train_config_path, args.python_bin)
    train_record = {
        "status": "planned",
        "config_path": str(train_config_path),
        "checkpoint_path": str(auto_checkpoint_path),
        "log_path": str(train_log),
        "command": " ".join(train_command),
        "gpu": str(auto_config.get("train_gpu", "0")),
    }

    print(f"[train] checkpoint={auto_checkpoint_path}", flush=True)
    if args.dry_run:
        train_record["status"] = "dry_run"
        print(f"[train] dry_run CUDA_VISIBLE_DEVICES={train_record['gpu']} {' '.join(train_command)}", flush=True)
    elif args.skip_train:
        train_record["status"] = "skipped_by_flag"
        print("[train] skipped_by_flag", flush=True)
    elif auto_checkpoint_path.exists() and not args.rerun_train:
        train_record["status"] = "skipped_existing"
        print("[train] skipped_existing", flush=True)
    else:
        result = _run_logged(
            train_command,
            env=_env_for_gpu(paths, str(auto_config.get("train_gpu", "0"))),
            log_path=train_log,
            stream_logs=args.stream_logs or not args.no_progress,
        )
        train_record["returncode"] = result.returncode
        train_record["status"] = "completed" if result.returncode == 0 and auto_checkpoint_path.exists() else "failed"
        if train_record["status"] != "completed":
            train_record["log_tail"] = fr._tail_text(train_log)
            manifest = {
                "created_at": created_at,
                "run_id": run_id,
                "config": str(config_path),
                "artifact_root": str(manifest_dir),
                "train": train_record,
                "records": [],
                "failures": [train_record],
            }
            fr._write_final_manifest(manifest_dir, manifest)
            return 1

    run_plan, analysis_records = _build_run_plan(
        base_matrix=base_matrix,
        suite_config=suite_config,
        paths=paths,
        config_dir=generated_dir / "run_configs",
        matrix_dir=generated_dir / "matrices",
        analysis_config_dir=generated_dir / "analysis_configs",
        analysis_root=manifest_dir / "analysis",
        source_config_path=config_path,
        source_config_sha256=str(config_sources["config_sha256"]),
        selected_suites=selected_suites,
        selected_checkpoints=selected_checkpoints,
        selected_modes=selected_modes,
        smoke_test=args.smoke_test,
        python_bin=args.python_bin,
    )
    print(f"[plan] eval_runs={len(run_plan)} artifact_root={manifest_dir}", flush=True)
    records, failures = _run_eval_plan(
        run_plan=run_plan,
        paths=paths,
        eval_gpus=[str(item) for item in auto_config.get("eval_gpus", ["1", "6", "7", "0"])],
        manifest_dir=manifest_dir,
        run_id=run_id,
        dry_run=args.dry_run,
        rerun=args.rerun,
        stop_on_error=args.stop_on_error,
        show_progress=not args.no_progress,
        stream_logs=args.stream_logs,
    )

    if not args.dry_run and not args.no_analysis and not failures:
        for item in analysis_records:
            command = fr._analysis_command(Path(item["analysis_config"]), args.python_bin)
            log_path = Path(item["log_path"])
            print(f"[analysis] suite={item['suite']} ckpt={item['checkpoint']} running", flush=True)
            result = _run_logged(command, env=fr._build_env(paths), log_path=log_path, stream_logs=args.stream_logs)
            item["returncode"] = result.returncode
            item["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                item["log_tail"] = fr._tail_text(log_path)
                failures.append({"status": "analysis_failed", **item})
                if args.stop_on_error:
                    break
    elif args.dry_run and not args.no_analysis:
        for item in analysis_records:
            command = fr._analysis_command(Path(item["analysis_config"]), args.python_bin)
            print(f"[analysis dry_run] suite={item['suite']} ckpt={item['checkpoint']} {' '.join(command)}", flush=True)

    summary_outputs = fr._write_checkpoint_summary(manifest_dir, records) if not args.dry_run else {}
    manifest = {
        "created_at": created_at,
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "config": str(config_path),
        "config_sha256": config_sources["config_sha256"],
        "artifact_root": str(manifest_dir),
        "dry_run": args.dry_run,
        "smoke_test": args.smoke_test,
        "train": train_record,
        "run_count": len(run_plan),
        "completed_count": len([item for item in records if item["status"] == "completed"]),
        "skipped_existing_count": len([item for item in records if item["status"] == "skipped_existing"]),
        "failed_count": len(failures),
        "records": records,
        "analysis": analysis_records,
        "summary_outputs": summary_outputs,
        "failures": failures,
    }
    fr._write_final_manifest(manifest_dir, manifest)
    fr._write_run_outputs(manifest_dir, records, failures, run_id)
    print(f"[done] completed={manifest['completed_count']} skipped={manifest['skipped_existing_count']} failures={manifest['failed_count']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
