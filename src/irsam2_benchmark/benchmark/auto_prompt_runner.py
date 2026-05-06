#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
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
from ..config import load_app_config
from ..core.fingerprints import sha256_file
from ..validation import preflight_dataset, validate_run_artifacts


PROJECT_ROOT = fr.PROJECT_ROOT
TRAIN_AUTO_PROMPT_PY = PROJECT_ROOT / "scripts" / "train_auto_prompt.py"
DEFAULT_AUTO_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "server_auto_prompt_4090x4.local.yaml"
DEFAULT_AUTO_PROMPT_EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "server_auto_prompt_4090x4.example.yaml"
DEFAULT_AUTO_PROMPT_EXPERIMENT_ID = "sam2_ir_qd_m4_fa_rerank_seeded_v1"
PREFLIGHT_MODES = ("fast", "full", "off")
DEFAULT_PREFLIGHT_SAMPLE_LIMIT = 256
DEFAULT_PREFLIGHT_IMAGE_LIMIT = 256
HEAVY_PREFLIGHT_SAMPLE_LIMIT = 64
HEAVY_PREFLIGHT_IMAGE_LIMIT = 64

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
    "sam2_learned_auto_point_rerank": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {"name": "sam2_learned_auto_point_rerank", "family": "sam2_learned_auto_prompt_m3", "modality": "ir"},
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_point_rerank_prior": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_learned_auto_point_rerank_prior",
            "family": "sam2_learned_auto_prompt_m3_ablation",
            "modality": "ir",
            "prompt_reranker": {"use_mask_feedback": False},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank_prior", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_point_rerank_no_frequency": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_learned_auto_point_rerank_no_frequency",
            "family": "sam2_learned_auto_prompt_m3_ablation",
            "modality": "ir",
            "prompt_reranker": {"use_frequency": False},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank_no_frequency", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_point_rerank_no_tophat": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_learned_auto_point_rerank_no_tophat",
            "family": "sam2_learned_auto_prompt_m3_ablation",
            "modality": "ir",
            "prompt_reranker": {"prior_weight_top_hat": 0.0},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank_no_tophat", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_point_rerank_no_local_contrast": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_learned_auto_point_rerank_no_local_contrast",
            "family": "sam2_learned_auto_prompt_m3_ablation",
            "modality": "ir",
            "prompt_reranker": {"prior_weight_local_contrast": 0.0},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank_no_local_contrast", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_point_rerank_mask_feedback_only": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_learned_auto_point_rerank_mask_feedback_only",
            "family": "sam2_learned_auto_prompt_m3_ablation",
            "modality": "ir",
            "prompt_reranker": {"final_weight_prior": 0.0, "final_weight_feedback": 1.0},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "learned_auto_point_rerank_mask_feedback_only", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_learned_auto_box_point_calibrated": {
        "baseline": "sam2_learned_auto_box_point_calibrated_prompt",
        "method": {"name": "sam2_learned_auto_box_point_calibrated", "family": "sam2_learned_auto_prompt_m3", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "learned_auto_box_point_calibrated", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 2},
        },
    },
    "sam2_learned_auto_box_point_calibrated_neg": {
        "baseline": "sam2_learned_auto_box_point_calibrated_neg_prompt",
        "method": {"name": "sam2_learned_auto_box_point_calibrated_neg", "family": "sam2_learned_auto_prompt_m3", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "learned_auto_box_point_calibrated_neg", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 6},
        },
    },
    "sam2_learned_auto_box_point_gated": {
        "baseline": "sam2_learned_auto_box_point_gated_prompt",
        "method": {"name": "sam2_learned_auto_box_point_gated", "family": "sam2_learned_auto_prompt_m3", "modality": "ir"},
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "learned_auto_box_point_gated", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 2},
        },
    },
    "sam2_ir_fa_rerank": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_ir_fa_rerank",
            "family": "sam2_ir_fa_rerank_m4",
            "modality": "ir",
            "prompt_reranker": {"prior_weight_local_contrast": 0.0},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "sam2_ir_fa_rerank", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_ir_fa_rerank_feedback_only": {
        "baseline": "sam2_learned_auto_point_rerank_prompt",
        "method": {
            "name": "sam2_ir_fa_rerank_feedback_only",
            "family": "sam2_ir_fa_rerank_m4_ablation",
            "modality": "ir",
            "prompt_reranker": {"prior_weight_local_contrast": 0.0, "final_weight_prior": 0.0, "final_weight_feedback": 1.0},
        },
        "evaluation": {
            "inference_mode": "point",
            "prompt_policy": {"name": "sam2_ir_fa_rerank_feedback_only", "prompt_type": "point", "prompt_source": "synthesized", "prompt_budget": 1},
        },
    },
    "sam2_ir_fa_rerank_gated_box_strict": {
        "baseline": "sam2_learned_auto_box_point_gated_prompt",
        "method": {
            "name": "sam2_ir_fa_rerank_gated_box_strict",
            "family": "sam2_ir_fa_rerank_m4_ablation",
            "modality": "ir",
            "prompt_reranker": {
                "prior_weight_local_contrast": 0.0,
                "box_enable_margin": 0.03,
                "box_enable_min_score": 0.72,
                "box_enable_min_point_feedback_score": 0.70,
            },
        },
        "evaluation": {
            "inference_mode": "box+point",
            "prompt_policy": {"name": "sam2_ir_fa_rerank_gated_box_strict", "prompt_type": "box+point", "prompt_source": "synthesized", "prompt_budget": 2},
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
    config.setdefault("experiment_id", DEFAULT_AUTO_PROMPT_EXPERIMENT_ID)
    config.setdefault("artifact_subdir", DEFAULT_AUTO_PROMPT_EXPERIMENT_ID)
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


def _train_seed_values(auto_config: Dict[str, Any]) -> list[int]:
    raw_seeds = auto_config.get("train_seeds")
    if raw_seeds is None:
        raw_seeds = [auto_config.get("train", {}).get("seed", 42)]
    if not isinstance(raw_seeds, list):
        raw_seeds = [raw_seeds]
    seeds: list[int] = []
    for item in raw_seeds:
        seed = int(item)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        seeds.append(int(auto_config.get("train", {}).get("seed", 42)))
    return seeds


def _method_is_learned(base_matrix: Dict[str, Any], method_id: str) -> bool:
    method_entry = fr._resolve_method(base_matrix["methods"], method_id)
    return str(method_entry.get("baseline", "")).startswith("sam2_learned_auto_")


def _apply_train_cli_overrides(auto_config: Dict[str, Any], args: argparse.Namespace) -> None:
    train_config = auto_config.setdefault("train", {})
    overrides = {
        "batch_size": args.train_batch_size,
        "num_workers": args.train_num_workers,
        "prefetch_factor": args.train_prefetch_factor,
        "shuffle_buffer_size": args.train_shuffle_buffer_size,
        "profile_interval_batches": args.train_profile_interval,
        "cache_dtype": args.train_cache_dtype,
        "light_cache_samples_per_epoch": args.train_rbgt_samples_per_epoch,
        "light_cache_batch_size": args.train_rbgt_batch_size,
        "light_cache_batch_size_max": args.train_rbgt_batch_size_max,
    }
    for key, value in overrides.items():
        if value is not None:
            train_config[key] = value
    if args.train_amp:
        train_config["use_amp"] = True
    if args.train_gpu_cache_datasets:
        auto_config["gpu_cache_datasets"] = [item.strip() for item in args.train_gpu_cache_datasets.split(",") if item.strip()]
    if args.train_light_cache_datasets:
        auto_config["light_cache_datasets"] = [item.strip() for item in args.train_light_cache_datasets.split(",") if item.strip()]


def _ensure_auto_methods(base_matrix: Dict[str, Any]) -> None:
    methods = base_matrix.setdefault("methods", {})
    for method_id, payload in {**DEFAULT_HEURISTIC_METHODS, **DEFAULT_LEARNED_METHODS}.items():
        methods.setdefault(method_id, copy.deepcopy(payload))


def _inject_learned_prompt_config(
    *,
    base_matrix: Dict[str, Any],
    checkpoint_path: Path,
    train_seed: int,
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
        method_payload["prompt_train_seed"] = int(train_seed)
        method_payload["prompt_device"] = "cuda"
        method_payload["prompt_top_k"] = int(prompt_runtime.get("top_k", auto_config.get("top_k", 1)))
        method_payload["prompt_point_budget"] = int(prompt_runtime.get("point_budget", auto_config.get("point_budget", 1)))
        method_payload["prompt_response_threshold"] = float(prompt_runtime.get("response_threshold", auto_config.get("response_threshold", 0.0)))
        method_payload["prompt_nms_radius"] = int(prompt_runtime.get("nms_radius", auto_config.get("nms_radius", 4)))
        method_payload["prompt_border_suppression_px"] = int(
            prompt_runtime.get("border_suppression_px", auto_config.get("border_suppression_px", 0))
        )
        method_payload["prompt_min_box_side"] = float(prompt_runtime.get("min_box_side", auto_config.get("model", {}).get("min_box_side", 2.0)))
        method_payload["prompt_negative_ring_offset"] = float(
            prompt_runtime.get("negative_ring_offset", auto_config.get("model", {}).get("negative_ring_offset", 4.0))
        )
        method_payload["prompt_use_local_contrast"] = bool(auto_config.get("model", {}).get("use_local_contrast", True))
        method_payload["prompt_use_top_hat"] = bool(auto_config.get("model", {}).get("use_top_hat", True))
        global_reranker = auto_config.get("prompt_reranker")
        method_reranker = method_payload.get("prompt_reranker")
        if isinstance(global_reranker, dict) or isinstance(method_reranker, dict):
            merged_reranker: Dict[str, Any] = {}
            if isinstance(global_reranker, dict):
                merged_reranker = copy.deepcopy(global_reranker)
            if isinstance(method_reranker, dict):
                merged_reranker = fr._deep_merge(merged_reranker, method_reranker)
            method_payload["prompt_reranker"] = merged_reranker
        method_payload["heatmaps"] = {
            "enabled": bool(auto_config.get("heatmaps", {}).get("eval_enabled", True)),
            "root": str(heatmap_root),
            "experiment_id": f"{auto_config['experiment_id']}_seed{int(train_seed)}",
            "train_seed": int(train_seed),
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
) -> list[dict[str, Any]]:
    dataset_config_dir = generated_dir / "training_dataset_configs"
    checkpoints = suite_config.get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError("Auto prompt runner requires at least one SAM2 checkpoint entry for dataset config generation.")
    checkpoint = checkpoints[0]
    dataset_config_paths: list[str] = []
    gpu_cache_dataset_ids = {str(item) for item in auto_config.get("gpu_cache_datasets", [])}
    light_cache_dataset_ids = {str(item) for item in auto_config.get("light_cache_datasets", [])}
    gpu_cache_dataset_config_paths: list[str] = []
    light_cache_dataset_config_paths: list[str] = []
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
        if str(dataset_id) in gpu_cache_dataset_ids:
            gpu_cache_dataset_config_paths.append(str(dataset_config_path))
        if str(dataset_id) in light_cache_dataset_ids:
            light_cache_dataset_config_paths.append(str(dataset_config_path))

    train_contexts: list[dict[str, Any]] = []
    source_sha256 = sha256_file(config_path)
    for train_seed in _train_seed_values(auto_config):
        train_experiment_id = f"{auto_config['experiment_id']}_seed{train_seed}"
        train_settings = copy.deepcopy(auto_config.get("train", {}))
        train_settings["seed"] = int(train_seed)
        train_payload = {
            "experiment_id": train_experiment_id,
            "base_experiment_id": auto_config["experiment_id"],
            "train_seed": int(train_seed),
            "output_root": str(output_root),
            "dataset_configs": dataset_config_paths,
            "train": train_settings,
            "model": copy.deepcopy(auto_config.get("model", {})),
            "target": copy.deepcopy(auto_config.get("target", {})),
            "heatmaps": copy.deepcopy(auto_config.get("heatmaps", {})),
            "source_config": str(config_path.resolve()),
            "source_config_sha256": source_sha256,
        }
        if gpu_cache_dataset_config_paths:
            train_payload["gpu_cache_dataset_configs"] = gpu_cache_dataset_config_paths
        if light_cache_dataset_config_paths:
            train_payload["light_cache_dataset_configs"] = light_cache_dataset_config_paths
        if raw.get("auto_prompt_training"):
            train_payload = fr._deep_merge(train_payload, copy.deepcopy(raw["auto_prompt_training"]))
        train_payload["experiment_id"] = train_experiment_id
        train_payload["base_experiment_id"] = auto_config["experiment_id"]
        train_payload["train_seed"] = int(train_seed)
        train_payload.setdefault("train", {})
        train_payload["train"]["seed"] = int(train_seed)
        train_payload["train"]["show_progress"] = bool(show_progress)
        train_payload["train"]["progress_backend"] = progress_backend if show_progress else "none"
        train_payload["train"].setdefault("progress_update_interval_s", 1.0)
        train_config_path = generated_dir / f"auto_prompt_train_seed{train_seed}.yaml"
        checkpoint_path = output_root / train_experiment_id / str(auto_config["checkpoint_name"])
        fr._write_yaml(train_config_path, train_payload)
        train_contexts.append(
            {
                "seed": int(train_seed),
                "experiment_id": train_experiment_id,
                "config_path": train_config_path,
                "checkpoint_path": checkpoint_path,
            }
        )
    return train_contexts


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


def _is_heavy_preflight_dataset(*, dataset_id: str, config_path: Path) -> bool:
    tokens = f"{dataset_id} {config_path.stem}".lower().replace("-", "_")
    return "rbgt" in tokens and "tiny" in tokens


def _preflight_limits(*, app_config: Any, config_path: Path, mode: str) -> tuple[int, int, bool]:
    if mode != "fast":
        return 0, 0, False
    if _is_heavy_preflight_dataset(dataset_id=app_config.dataset.dataset_id, config_path=config_path):
        return HEAVY_PREFLIGHT_SAMPLE_LIMIT, HEAVY_PREFLIGHT_IMAGE_LIMIT, True
    return DEFAULT_PREFLIGHT_SAMPLE_LIMIT, DEFAULT_PREFLIGHT_IMAGE_LIMIT, True


def _limited_preflight_config(app_config: Any, *, sample_limit: int, image_limit: int) -> Any:
    if sample_limit <= 0 and image_limit <= 0:
        return app_config
    preflight_config = copy.deepcopy(app_config)
    preflight_config.runtime.max_samples = sample_limit
    preflight_config.runtime.max_images = image_limit
    return preflight_config


def _skipped_preflight_report(app_config: Any, *, role: str, config_path: Path, mode: str) -> dict[str, Any]:
    return {
        "valid": True,
        "skipped": True,
        "errors": [],
        "warnings": [],
        "warning_count": 0,
        "size_mismatch_warning_count": 0,
        "warning_examples": [],
        "dataset_id": app_config.dataset.dataset_id,
        "root": str(app_config.dataset_root),
        "sample_count": 0,
        "image_count": 0,
        "role": role,
        "config_path": str(config_path),
        "mode": mode,
        "sample_limit": 0,
        "image_limit": 0,
        "is_limited": False,
    }


def _preflight_exception_report(*, role: str, config_path: Path, mode: str, exc: Exception) -> dict[str, Any]:
    skipped = mode == "off"
    return {
        "valid": skipped,
        "skipped": skipped,
        "errors": [] if skipped else [f"Dataset preflight failed for config {config_path}: {exc}"],
        "warnings": [f"Dataset preflight skipped before config load: {exc}"] if skipped else [],
        "warning_count": 1 if skipped else 0,
        "size_mismatch_warning_count": 0,
        "warning_examples": [f"Dataset preflight skipped before config load: {exc}"] if skipped else [],
        "dataset_id": "",
        "root": "",
        "sample_count": 0,
        "image_count": 0,
        "role": role,
        "config_path": str(config_path),
        "mode": mode,
        "sample_limit": 0,
        "image_limit": 0,
        "is_limited": False,
    }


def _preflight_config_paths(config_paths: list[Path], *, role: str, mode: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for config_path in config_paths:
        try:
            app_config = load_app_config(config_path)
            key = (app_config.dataset.dataset_id, str(app_config.dataset_root))
            if key in seen:
                continue
            seen.add(key)
            if mode == "off":
                report = _skipped_preflight_report(app_config, role=role, config_path=config_path, mode=mode)
                print(
                    f"[preflight] skip role={role} dataset={report['dataset_id']} mode=off root={report['root']}",
                    flush=True,
                )
                reports.append(report)
                continue
            sample_limit, image_limit, is_limited = _preflight_limits(app_config=app_config, config_path=config_path, mode=mode)
            print(
                f"[preflight] start role={role} dataset={app_config.dataset.dataset_id} mode={mode} "
                f"max_samples={sample_limit} max_images={image_limit}",
                flush=True,
            )
            report = preflight_dataset(
                _limited_preflight_config(app_config, sample_limit=sample_limit, image_limit=image_limit)
            )
            report["dataset_id"] = app_config.dataset.dataset_id
            report["root"] = str(app_config.dataset_root)
            report["mode"] = mode
            report["sample_limit"] = sample_limit
            report["image_limit"] = image_limit
            report["is_limited"] = is_limited
            print(
                f"[preflight] done role={role} dataset={report['dataset_id']} valid={report.get('valid')} "
                f"samples={report.get('sample_count', 0)} warnings={report.get('warning_count', 0)} "
                f"size_mismatch={report.get('size_mismatch_warning_count', 0)}",
                flush=True,
            )
        except Exception as exc:
            report = _preflight_exception_report(role=role, config_path=config_path, mode=mode, exc=exc)
        report["role"] = role
        report["config_path"] = str(config_path)
        reports.append(report)
    return reports


def _preflight_section(reports: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "dataset_count": len(reports),
        "valid_count": len([item for item in reports if item.get("valid")]),
        "invalid_count": len([item for item in reports if not item.get("valid")]),
        "sample_count": sum(int(item.get("sample_count", 0)) for item in reports),
        "warning_count": sum(int(item.get("warning_count", 0)) for item in reports),
        "size_mismatch_warning_count": sum(int(item.get("size_mismatch_warning_count", 0)) for item in reports),
        "reports": reports,
    }


def _dataset_preflight_summary(*, train_config_paths: list[Path], run_plan: list[tuple[Any, ...]], mode: str) -> dict[str, Any]:
    dataset_config_paths: list[Path] = []
    for train_config_path in train_config_paths:
        train_payload = _load_yaml(train_config_path)
        dataset_config_paths.extend(Path(item) for item in train_payload.get("dataset_configs", []))
    eval_config_paths = [Path(item[5]) for item in run_plan]
    train_reports = _preflight_config_paths(dataset_config_paths, role="train", mode=mode)
    eval_reports = _preflight_config_paths(eval_config_paths, role="eval", mode=mode)
    train_summary = _preflight_section(train_reports, mode=mode)
    eval_summary = _preflight_section(eval_reports, mode=mode)
    overall = {
        "mode": mode,
        "valid": train_summary["invalid_count"] == 0 and eval_summary["invalid_count"] == 0,
        "dataset_count": train_summary["dataset_count"] + eval_summary["dataset_count"],
        "invalid_count": train_summary["invalid_count"] + eval_summary["invalid_count"],
        "sample_count": train_summary["sample_count"] + eval_summary["sample_count"],
        "warning_count": train_summary["warning_count"] + eval_summary["warning_count"],
        "size_mismatch_warning_count": train_summary["size_mismatch_warning_count"] + eval_summary["size_mismatch_warning_count"],
    }
    return {"mode": mode, "overall": overall, "train": train_summary, "eval": eval_summary}


def _write_dataset_preflight_summary(manifest_dir: Path, summary: dict[str, Any]) -> Path:
    path = manifest_dir / "dataset_preflight_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_dataset_preflight_summary(summary: dict[str, Any]) -> None:
    for role in ("train", "eval"):
        section = summary[role]
        print(
            f"[preflight] {role} mode={section['mode']} datasets={section['dataset_count']} valid={section['valid_count']} "
            f"warnings={section['warning_count']} size_mismatch={section['size_mismatch_warning_count']}",
            flush=True,
        )


def _preflight_failure_record(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    errors: list[str] = []
    for role in ("train", "eval"):
        for report in summary[role]["reports"]:
            if not report.get("valid"):
                dataset = report.get("dataset_id") or report.get("config_path", "")
                report_errors = report.get("errors", [])
                errors.append(f"{role}:{dataset}: {'; '.join(str(item) for item in report_errors)}")
    return {
        "status": "dataset_preflight_failed",
        "message": "One or more datasets failed preflight validation.",
        "preflight_path": str(path),
        "errors": errors,
    }


def _build_run_plan(
    *,
    base_matrix: Dict[str, Any],
    suite_config: Dict[str, Any],
    paths: Dict[str, Any],
    auto_config: Dict[str, Any],
    train_contexts: list[dict[str, Any]],
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
    artifact_subdir = str(suite_config.get("artifact_subdir", DEFAULT_AUTO_PROMPT_EXPERIMENT_ID))
    if smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    for suite_key, suite_entry in fr._iter_requested_suites(suite_config, selected_suites):
        suite_checkpoints = fr._suite_checkpoints(checkpoints, suite_entry)
        method_ids = fr._suite_method_ids(mode_entries, suite_entry)
        reference_method_ids = [method_id for method_id in method_ids if not _method_is_learned(base_matrix, method_id)]
        learned_method_ids = [method_id for method_id in method_ids if _method_is_learned(base_matrix, method_id)]
        for checkpoint in suite_checkpoints:
            artifact_root = fr._run_artifact_root(artifact_base, artifact_subdir, suite_key, checkpoint["alias"])
            experiment_contexts: list[dict[str, Any]] = []
            if reference_method_ids:
                experiment_contexts.append(
                    {
                        "experiment_id": f"{suite_entry['experiment_id']}_reference",
                        "method_ids": reference_method_ids,
                        "train_seed": None,
                        "checkpoint_path": None,
                    }
                )
            for train_context in train_contexts:
                if learned_method_ids:
                    seed = int(train_context["seed"])
                    experiment_contexts.append(
                        {
                            "experiment_id": f"{suite_entry['experiment_id']}_seed{seed}",
                            "method_ids": learned_method_ids,
                            "train_seed": seed,
                            "checkpoint_path": Path(train_context["checkpoint_path"]),
                        }
                    )
            if not experiment_contexts:
                continue
            experiment_ids = [str(item["experiment_id"]) for item in experiment_contexts]
            experiments = [
                {
                    "experiment_id": str(item["experiment_id"]),
                    "status": "planned",
                    "datasets": list(suite_entry["datasets"]),
                    "methods": list(item["method_ids"]),
                    "metrics": list(suite_entry.get("metrics", [])),
                    "purpose": suite_entry.get("purpose", ""),
                    "suite": suite_key,
                    "checkpoint": checkpoint["alias"],
                    "train_seed": item["train_seed"],
                }
                for item in experiment_contexts
            ]
            generated_matrix = fr._build_generated_matrix(
                base_matrix=base_matrix,
                suite_key=suite_key,
                suite_entry=suite_entry,
                checkpoint=checkpoint,
                method_ids=method_ids,
                experiments=experiments,
                experiment_groups=experiment_ids,
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
                        experiment_groups=experiment_ids,
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
                        "experiment_groups": experiment_ids,
                    }
                )
            for experiment_context in experiment_contexts:
                context_matrix = copy.deepcopy(base_matrix)
                train_seed = experiment_context["train_seed"]
                if train_seed is not None:
                    _inject_learned_prompt_config(
                        base_matrix=context_matrix,
                        checkpoint_path=Path(experiment_context["checkpoint_path"]),
                        train_seed=int(train_seed),
                        auto_config=auto_config,
                        heatmap_root=artifact_base / artifact_subdir / "heatmaps",
                    )
                context_suite_entry = copy.deepcopy(suite_entry)
                context_suite_entry["experiment_id"] = str(experiment_context["experiment_id"])
                if train_seed is not None:
                    context_runtime = copy.deepcopy(context_suite_entry.get("runtime", {}))
                    context_runtime["seeds"] = [int(train_seed)]
                    context_suite_entry["runtime"] = context_runtime
                for dataset_id in suite_entry.get("datasets", []):
                    for method_id in experiment_context["method_ids"]:
                        method_entry = fr._resolve_method(context_matrix["methods"], method_id)
                        output_dir = fr._run_output_dir(artifact_root, context_suite_entry["experiment_id"], dataset_id, method_id)
                        seed_suffix = "reference" if train_seed is None else f"seed{train_seed}"
                        app_config_path = config_dir / suite_key / checkpoint["alias"] / seed_suffix / f"{dataset_id}_{method_id}.yaml"
                        app_config = fr._build_app_config(
                            base_matrix=context_matrix,
                            suite_config=suite_config,
                            paths=paths,
                            suite_key=suite_key,
                            suite_entry=context_suite_entry,
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
                        log_path = fr._log_path(
                            artifact_base / artifact_subdir,
                            suite_key,
                            checkpoint["alias"],
                            f"{seed_suffix}_{dataset_id}_{method_id}",
                        )
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
                                train_seed,
                                context_suite_entry["experiment_id"],
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
            suite_key, checkpoint, dataset_id, method_id, output_dir, config_path, config_sha256, command, log_path, train_seed, experiment_id = queue.pop(0)
            gpu = free_gpus.pop(0)
            seed_label = "reference" if train_seed is None else f"seed={train_seed}"
            prefix = (
                f"[{completed_slots + len(active) + 1}/{total}] gpu={gpu} suite={suite_key} ckpt={checkpoint['alias']} "
                f"dataset={dataset_id} mode={method_id} {seed_label}"
            )
            _set_eval_progress(progress, counts=progress_counts, active=len(active), queued=len(queue))
            if not rerun and fr._run_is_complete(output_dir):
                if progress is None:
                    print(f"{prefix} skipped_existing", flush=True)
                record = fr._status_record(
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
                record["train_seed"] = train_seed
                record["experiment_id"] = experiment_id
                records.append(record)
                completed_slots += 1
                free_gpus.append(gpu)
                fr._write_run_outputs(manifest_dir, records, failures, run_id)
                _advance_eval_progress(progress, status="skipped_existing", counts=progress_counts, active=len(active), queued=len(queue))
                continue
            if dry_run:
                print(f"{prefix} dry_run CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}", flush=True)
                record = fr._status_record(
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
                record["train_seed"] = train_seed
                record["experiment_id"] = experiment_id
                records.append(record)
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
                    "train_seed": train_seed,
                    "experiment_id": experiment_id,
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
                record = fr._status_record(
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
                record["train_seed"] = item["train_seed"]
                record["experiment_id"] = item["experiment_id"]
                records.append(record)
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
                failure["train_seed"] = item["train_seed"]
                failure["experiment_id"] = item["experiment_id"]
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
    parser = argparse.ArgumentParser(description="Run SAM2-IR-QD learned auto prompt training and M4 seeded evaluation on 4x4090.")
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
    parser.add_argument("--train-batch-size", type=int, help="Override auto-prompt training batch_size.")
    parser.add_argument("--train-num-workers", type=int, help="Override auto-prompt training DataLoader num_workers.")
    parser.add_argument("--train-prefetch-factor", type=int, help="Override auto-prompt training DataLoader prefetch_factor when workers are enabled.")
    parser.add_argument("--train-shuffle-buffer-size", type=int, help="Override auto-prompt streaming shuffle_buffer_size.")
    parser.add_argument("--train-amp", action="store_true", help="Enable CUDA AMP for auto-prompt training.")
    parser.add_argument("--train-profile-interval", type=int, help="Print one training throughput profile line every N batches.")
    parser.add_argument("--train-gpu-cache-datasets", help="Comma-separated train dataset ids to cache as dense tensors on the training GPU.")
    parser.add_argument("--train-light-cache-datasets", help="Comma-separated train dataset ids to cache as grayscale+box lightweight samples.")
    parser.add_argument("--train-cache-dtype", choices=("float16", "float32", "bfloat16"), help="Tensor dtype for dense GPU cache.")
    parser.add_argument("--train-rbgt-samples-per-epoch", type=int, help="Number of lightweight RBGT samples to draw per epoch.")
    parser.add_argument("--train-rbgt-batch-size", help="Lightweight RBGT batch size, or 'auto' to probe the largest safe value.")
    parser.add_argument("--train-rbgt-batch-size-max", type=int, help="Maximum candidate batch size for lightweight RBGT auto tuning.")
    parser.add_argument(
        "--preflight-mode",
        choices=PREFLIGHT_MODES,
        default="fast",
        help="Dataset preflight mode. fast checks bounded samples/images, full scans all data, off records a skipped summary.",
    )
    args = parser.parse_args(argv)

    config_path = (args.config if args.config is not None else _default_config()).resolve()
    raw = _load_yaml(config_path)
    auto_config = _auto_config(raw)
    _apply_train_cli_overrides(auto_config, args)
    paths, suite_config, base_matrix, config_sources = fr._load_complete_benchmark_config(config_path)
    _ensure_auto_methods(base_matrix)
    artifact_base = fr._artifact_base(paths)
    artifact_subdir = str(auto_config.get("artifact_subdir") or suite_config.get("artifact_subdir", DEFAULT_AUTO_PROMPT_EXPERIMENT_ID))
    if args.smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    suite_config["artifact_subdir"] = artifact_subdir
    manifest_dir = artifact_base / artifact_subdir
    generated_dir = manifest_dir / "generated"
    output_root = artifact_base / artifact_subdir / "train"
    train_contexts = _write_training_configs(
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
    primary_train_context = train_contexts[0]
    auto_checkpoint_path = Path(primary_train_context["checkpoint_path"])
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
    train_records = []
    for train_context in train_contexts:
        train_config_path = Path(train_context["config_path"])
        train_seed = int(train_context["seed"])
        train_command = _command_for_train(train_config_path, args.python_bin)
        train_records.append(
            {
                "status": "planned",
                "seed": train_seed,
                "experiment_id": str(train_context["experiment_id"]),
                "config_path": str(train_config_path),
                "checkpoint_path": str(train_context["checkpoint_path"]),
                "log_path": str(manifest_dir / "logs" / f"auto_prompt_train_seed{train_seed}.log"),
                "command": " ".join(train_command),
                "gpu": str(auto_config.get("train_gpu", "0")),
            }
        )
    train_record = train_records[0]
    run_plan, analysis_records = _build_run_plan(
        base_matrix=base_matrix,
        suite_config=suite_config,
        paths=paths,
        auto_config=auto_config,
        train_contexts=train_contexts,
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
    preflight_summary = _dataset_preflight_summary(
        train_config_paths=[Path(item["config_path"]) for item in train_contexts],
        run_plan=run_plan,
        mode=args.preflight_mode,
    )
    preflight_path = _write_dataset_preflight_summary(manifest_dir, preflight_summary)
    dataset_preflight = {"path": str(preflight_path), "summary": preflight_summary}
    _print_dataset_preflight_summary(preflight_summary)
    if not preflight_summary["overall"]["valid"]:
        failure = _preflight_failure_record(preflight_summary, preflight_path)
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
            "train_runs": train_records,
            "train_seed_count": len(train_records),
            "train_seeds": [item["seed"] for item in train_records],
            "dataset_preflight": dataset_preflight,
            "run_count": len(run_plan),
            "completed_count": 0,
            "skipped_existing_count": 0,
            "failed_count": 1,
            "records": [],
            "analysis": analysis_records,
            "failures": [failure],
        }
        fr._write_final_manifest(manifest_dir, manifest)
        print(f"[preflight] failed path={preflight_path}", flush=True)
        return 1

    train_failures = []
    for item in train_records:
        checkpoint_path = Path(str(item["checkpoint_path"]))
        train_command = _command_for_train(Path(str(item["config_path"])), args.python_bin)
        train_log = Path(str(item["log_path"]))
        print(f"[train seed={item['seed']}] checkpoint={checkpoint_path}", flush=True)
        if args.dry_run:
            item["status"] = "dry_run"
            print(f"[train seed={item['seed']}] dry_run CUDA_VISIBLE_DEVICES={item['gpu']} {' '.join(train_command)}", flush=True)
        elif args.skip_train:
            item["status"] = "skipped_by_flag"
            print(f"[train seed={item['seed']}] skipped_by_flag", flush=True)
        elif checkpoint_path.exists() and not args.rerun_train:
            item["status"] = "skipped_existing"
            print(f"[train seed={item['seed']}] skipped_existing", flush=True)
        else:
            result = _run_logged(
                train_command,
                env=_env_for_gpu(paths, str(auto_config.get("train_gpu", "0"))),
                log_path=train_log,
                stream_logs=args.stream_logs or not args.no_progress,
            )
            item["returncode"] = result.returncode
            item["status"] = "completed" if result.returncode == 0 and checkpoint_path.exists() else "failed"
            if item["status"] != "completed":
                item["log_tail"] = fr._tail_text(train_log)
                train_failures.append(item)
                if args.stop_on_error:
                    break
    if train_failures:
        manifest = {
            "created_at": created_at,
            "run_id": run_id,
            "config": str(config_path),
            "artifact_root": str(manifest_dir),
            "train": train_record,
            "train_runs": train_records,
            "train_seed_count": len(train_records),
            "train_seeds": [item["seed"] for item in train_records],
            "dataset_preflight": dataset_preflight,
            "records": [],
            "failures": train_failures,
        }
        fr._write_final_manifest(manifest_dir, manifest)
        return 1

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
        "train_runs": train_records,
        "train_seed_count": len(train_records),
        "train_seeds": [item["seed"] for item in train_records],
        "dataset_preflight": dataset_preflight,
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
