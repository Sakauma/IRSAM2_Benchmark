from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import auto_prompt_runner as apr
from . import full_runner as fr
from ..core.fingerprints import sha256_file


PROJECT_ROOT = fr.PROJECT_ROOT
TRAIN_AUTO_PROMPT_PY = PROJECT_ROOT / "scripts" / "train_auto_prompt.py"
SELECT_AUTO_PROMPT_CHECKPOINT_PY = PROJECT_ROOT / "scripts" / "select_auto_prompt_checkpoint.py"
EXPORT_RBGT_TINY_BOX_COCO_PY = PROJECT_ROOT / "scripts" / "export_rbgt_tiny_box_coco.py"
DEFAULT_M9_CONFIG = PROJECT_ROOT / "configs" / "server_auto_prompt_4090x8_m9_full.example.yaml"


@dataclass(frozen=True)
class StageSpec:
    key: str
    variant: str
    seed: int
    role: str
    architecture: str
    train_datasets: list[str]
    validation_datasets: list[str]
    model_overrides: dict[str, Any]
    train_overrides: dict[str, Any]
    target_overrides: dict[str, Any]
    init_from_key: str | None = None


@dataclass
class TrainJob:
    key: str
    variant: str
    seed: int
    role: str
    gpu: str
    config_path: Path
    output_dir: Path
    log_path: Path
    checkpoint_path: Path
    selected_checkpoint_path: Path
    epochs: int
    progress_state_path: Path
    progress_events_path: Path
    status: str = "planned"
    returncode: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable_train_job(job: TrainJob) -> dict[str, Any]:
    return job.__dict__ | {
        "config_path": str(job.config_path),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "checkpoint_path": str(job.checkpoint_path),
        "selected_checkpoint_path": str(job.selected_checkpoint_path),
        "progress_state_path": str(job.progress_state_path),
        "progress_events_path": str(job.progress_events_path),
    }


def _split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    output = {item.strip() for item in value.split(",") if item.strip()}
    return output or None


def _selected_checkpoint_from_summary(output_dir: Path) -> Path | None:
    summary_path = output_dir / "checkpoint_selection_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            candidate = Path(str(summary.get("selected_checkpoint", "")))
            if candidate.exists():
                return candidate
        except Exception:
            pass
    train_summary_path = output_dir / "train_summary.json"
    if train_summary_path.exists():
        try:
            summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
            for key in ("selected_checkpoint_path", "best_checkpoint_path", "checkpoint_path"):
                candidate = Path(str(summary.get(key, "")))
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return None


def _expected_rbgt_export_paths(root: Path) -> list[Path]:
    return [
        root / f"annotations_coco_ir_box_m9_{split}" / f"instances_rbgt_tiny_ir_box_{split}.json"
        for split in ("train", "val", "test")
    ]


def _reference_policy(m9: dict[str, Any]) -> str:
    policy = str(m9.get("reference_policy", "shared")).strip().lower()
    if policy in {"shared", "per_variant", "first_variant", "none"}:
        return policy
    return "shared"


def _json_list_count(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return len(value) if isinstance(value, list) else 0


def _validate_existing_rbgt_export(root: Path, export_cfg: dict[str, Any]) -> dict[str, Any]:
    outputs = _expected_rbgt_export_paths(root)
    issues: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    for path in outputs:
        split = path.parent.name.replace("annotations_coco_ir_box_m9_", "")
        if not path.exists():
            issues.append(f"missing split json: {path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"unreadable split json: {path}: {exc}")
            continue
        image_count = _json_list_count(payload, "images")
        annotation_count = _json_list_count(payload, "annotations")
        split_counts[split] = {"images": image_count, "annotations": annotation_count}
        if image_count <= 0:
            issues.append(f"empty image list in split json: {path}")
        if annotation_count <= 0:
            issues.append(f"empty annotation list in split json: {path}")

    summary_path = root / "annotations_coco_ir_box_m9_summary.json"
    summary: dict[str, Any] = {}
    if not summary_path.exists():
        issues.append(f"missing export summary: {summary_path}")
    else:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = payload if isinstance(payload, dict) else {}
        except Exception as exc:
            issues.append(f"unreadable export summary: {summary_path}: {exc}")
    expected = {
        "root": str(root.resolve()),
        "annotations_dir": str(export_cfg.get("annotations_dir", "annotations_voc")),
        "images_dir": str(export_cfg.get("images_dir", "images")),
        "small_target_filter": bool(export_cfg.get("small_target_filter", True)),
    }
    if summary:
        if str(Path(str(summary.get("root", ""))).resolve()) != expected["root"]:
            issues.append(f"summary root mismatch: expected={expected['root']} actual={summary.get('root')}")
        for key in ("annotations_dir", "images_dir", "small_target_filter"):
            if summary.get(key) != expected[key]:
                issues.append(f"summary {key} mismatch: expected={expected[key]!r} actual={summary.get(key)!r}")
    return {
        "valid": not issues,
        "issues": issues,
        "outputs": [str(path) for path in outputs],
        "summary_path": str(summary_path),
        "split_counts": split_counts,
        "expected": expected,
    }


def _stage_defaults(m9: dict[str, Any], role: str) -> dict[str, Any]:
    stages = m9.get("stage_defaults", {})
    if isinstance(stages, dict) and isinstance(stages.get(role), dict):
        return copy.deepcopy(stages[role])
    if role == "pretrain":
        return {"epochs": 40, "checkpoint_interval_epochs": 5, "learning_rate": 3e-4}
    if role == "finetune":
        return {"epochs": 60, "checkpoint_interval_epochs": 10, "learning_rate": 7.5e-5}
    return {"epochs": 60, "checkpoint_interval_epochs": 10, "learning_rate": 3e-4}


def _default_variant_specs(m9: dict[str, Any], seeds: list[int], selected_variants: set[str] | None) -> list[StageSpec]:
    public = [str(item) for item in m9.get("public_train_datasets", ["nuaa_sirst", "nudt_sirst", "irstd_1k"])]
    rbgt_train = str(m9.get("rbgt_train_dataset", "rbgt_tiny_ir_box_m9_train"))
    rbgt_val = str(m9.get("rbgt_val_dataset", "rbgt_tiny_ir_box_m9_val"))
    base_model = copy.deepcopy(m9.get("base_model", {}))
    v2_model = {**base_model, "architecture": "ir_prompt_v2", "hidden_channels": 32}
    v3_model = {**base_model, "architecture": "ir_prompt_v3_fpn", "hidden_channels": 48, "fpn_channels": 64, "depth": 6}
    base_target = copy.deepcopy(m9.get("base_target", {}))
    no_hard_negative = {**base_target, "hard_negative_weight": 1.0}
    specs: list[StageSpec] = []

    def enabled(variant: str) -> bool:
        return selected_variants is None or variant in selected_variants

    for seed in seeds:
        if enabled("M9-B") or enabled("M9-C"):
            specs.append(
                StageSpec(
                    key=f"v2_rbgt_pretrain_seed{seed}",
                    variant="M9-B",
                    seed=seed,
                    role="pretrain",
                    architecture="ir_prompt_v2",
                    train_datasets=[rbgt_train],
                    validation_datasets=[rbgt_val],
                    model_overrides=v2_model,
                    train_overrides=_stage_defaults(m9, "pretrain"),
                    target_overrides=base_target,
                )
            )
        if enabled("M9-E") or enabled("M9-G"):
            specs.append(
                StageSpec(
                    key=f"v3_rbgt_pretrain_seed{seed}",
                    variant="M9-E",
                    seed=seed,
                    role="pretrain",
                    architecture="ir_prompt_v3_fpn",
                    train_datasets=[rbgt_train],
                    validation_datasets=[rbgt_val],
                    model_overrides=v3_model,
                    train_overrides=_stage_defaults(m9, "pretrain"),
                    target_overrides=base_target,
                )
            )
        if enabled("M9-A"):
            specs.append(
                StageSpec(
                    key=f"M9-A_public_v2_seed{seed}",
                    variant="M9-A",
                    seed=seed,
                    role="public",
                    architecture="ir_prompt_v2",
                    train_datasets=public,
                    validation_datasets=public,
                    model_overrides=v2_model,
                    train_overrides=_stage_defaults(m9, "public"),
                    target_overrides=base_target,
                )
            )
        if enabled("M9-C"):
            specs.append(
                StageSpec(
                    key=f"M9-C_staged_v2_seed{seed}",
                    variant="M9-C",
                    seed=seed,
                    role="finetune",
                    architecture="ir_prompt_v2",
                    train_datasets=public,
                    validation_datasets=public,
                    model_overrides=v2_model,
                    train_overrides=_stage_defaults(m9, "finetune"),
                    target_overrides=base_target,
                    init_from_key=f"v2_rbgt_pretrain_seed{seed}",
                )
            )
        if enabled("M9-D"):
            specs.append(
                StageSpec(
                    key=f"M9-D_public_v3_seed{seed}",
                    variant="M9-D",
                    seed=seed,
                    role="public",
                    architecture="ir_prompt_v3_fpn",
                    train_datasets=public,
                    validation_datasets=public,
                    model_overrides=v3_model,
                    train_overrides=_stage_defaults(m9, "public"),
                    target_overrides=base_target,
                )
            )
        if enabled("M9-E"):
            specs.append(
                StageSpec(
                    key=f"M9-E_staged_v3_seed{seed}",
                    variant="M9-E",
                    seed=seed,
                    role="finetune",
                    architecture="ir_prompt_v3_fpn",
                    train_datasets=public,
                    validation_datasets=public,
                    model_overrides=v3_model,
                    train_overrides=_stage_defaults(m9, "finetune"),
                    target_overrides=base_target,
                    init_from_key=f"v3_rbgt_pretrain_seed{seed}",
                )
            )
        if enabled("M9-F"):
            specs.append(
                StageSpec(
                    key=f"M9-F_mixed_v3_seed{seed}",
                    variant="M9-F",
                    seed=seed,
                    role="mixed",
                    architecture="ir_prompt_v3_fpn",
                    train_datasets=[*public, rbgt_train],
                    validation_datasets=public,
                    model_overrides=v3_model,
                    train_overrides=_stage_defaults(m9, "mixed"),
                    target_overrides=base_target,
                )
            )
        if enabled("M9-G"):
            specs.append(
                StageSpec(
                    key=f"M9-G_staged_v3_no_hn_seed{seed}",
                    variant="M9-G",
                    seed=seed,
                    role="finetune",
                    architecture="ir_prompt_v3_fpn",
                    train_datasets=public,
                    validation_datasets=public,
                    model_overrides=v3_model,
                    train_overrides=_stage_defaults(m9, "finetune"),
                    target_overrides=no_hard_negative,
                    init_from_key=f"v3_rbgt_pretrain_seed{seed}",
                )
            )
    dedup: dict[str, StageSpec] = {}
    for spec in specs:
        dedup.setdefault(spec.key, spec)
    return list(dedup.values())


def _dataset_payload(
    *,
    base_matrix: dict[str, Any],
    suite_config: dict[str, Any],
    paths: dict[str, Any],
    dataset_id: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return apr._training_dataset_config(
        base_matrix=base_matrix,
        suite_config=suite_config,
        paths=paths,
        dataset_id=dataset_id,
        checkpoint=checkpoint,
    )


def _write_dataset_configs(
    *,
    dataset_ids: list[str],
    base_matrix: dict[str, Any],
    suite_config: dict[str, Any],
    paths: dict[str, Any],
    checkpoint: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_out: list[str] = []
    for dataset_id in dataset_ids:
        path = output_dir / f"{dataset_id}.yaml"
        fr._write_yaml(
            path,
            _dataset_payload(
                base_matrix=base_matrix,
                suite_config=suite_config,
                paths=paths,
                dataset_id=dataset_id,
                checkpoint=checkpoint,
            ),
        )
        paths_out.append(str(path))
    return paths_out


def _build_train_job(
    *,
    spec: StageSpec,
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    gpu: str,
    init_checkpoint: Path | None,
) -> TrainJob:
    checkpoint = suite_config["checkpoints"][0]
    dataset_dir = generated_dir / "training_dataset_configs"
    train_dataset_configs = _write_dataset_configs(
        dataset_ids=spec.train_datasets,
        base_matrix=base_matrix,
        suite_config=suite_config,
        paths=paths,
        checkpoint=checkpoint,
        output_dir=dataset_dir,
    )
    validation_dataset_configs = _write_dataset_configs(
        dataset_ids=spec.validation_datasets,
        base_matrix=base_matrix,
        suite_config=suite_config,
        paths=paths,
        checkpoint=checkpoint,
        output_dir=dataset_dir,
    )
    auto = copy.deepcopy(raw.get("auto_prompt", {}))
    train_cfg = copy.deepcopy(auto.get("train", {}))
    train_cfg.update(spec.train_overrides)
    train_cfg["seed"] = int(spec.seed)
    train_cfg.setdefault("show_progress", True)
    train_cfg.setdefault("progress_backend", "tqdm")
    if init_checkpoint is not None:
        train_cfg["init_checkpoint"] = str(init_checkpoint)
    model_cfg = copy.deepcopy(auto.get("model", {}))
    model_cfg.update(spec.model_overrides)
    target_cfg = copy.deepcopy(auto.get("target", {}))
    target_cfg.update(spec.target_overrides)
    output_root = artifact_root / "train"
    experiment_id = f"sam2_ir_qd_m9_{spec.key}"
    output_dir = output_root / experiment_id
    progress_state_path = output_dir / "train_progress_state.json"
    progress_events_path = output_dir / "train_progress_events.jsonl"
    train_cfg["progress_state_path"] = str(progress_state_path)
    train_cfg["progress_events_path"] = str(progress_events_path)
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "base_experiment_id": "sam2_ir_qd_m9_full_v1",
        "train_seed": int(spec.seed),
        "output_root": str(output_root),
        "dataset_configs": train_dataset_configs,
        "validation_dataset_configs": validation_dataset_configs,
        "validation_light_cache_dataset_configs": validation_dataset_configs,
        "light_cache_dataset_configs": train_dataset_configs,
        "train": train_cfg,
        "model": model_cfg,
        "target": target_cfg,
        "heatmaps": copy.deepcopy(auto.get("heatmaps", {})),
        "source_config": str(raw.get("__config_path__", "")),
        "source_config_sha256": raw.get("__config_sha256__", ""),
        "m9_variant": spec.variant,
        "m9_role": spec.role,
    }
    config_path = generated_dir / "train_configs" / f"{spec.key}.yaml"
    fr._write_yaml(config_path, payload)
    return TrainJob(
        key=spec.key,
        variant=spec.variant,
        seed=int(spec.seed),
        role=spec.role,
        gpu=str(gpu),
        config_path=config_path,
        output_dir=output_dir,
        log_path=artifact_root / "logs" / "train" / f"{spec.key}.log",
        checkpoint_path=output_dir / "checkpoint.pt",
        selected_checkpoint_path=output_dir / "checkpoint_selected_m9.pt",
        epochs=max(1, int(train_cfg.get("epochs", 1))),
        progress_state_path=progress_state_path,
        progress_events_path=progress_events_path,
    )


def _materialize_train_jobs(
    *,
    specs: list[StageSpec],
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    gpus: list[str],
    include_roles: set[str] | None = None,
) -> dict[str, TrainJob]:
    jobs: dict[str, TrainJob] = {}
    visible_specs = [spec for spec in specs if include_roles is None or spec.role in include_roles]
    for index, spec in enumerate(visible_specs):
        init_checkpoint = None
        if spec.init_from_key is not None:
            init_dir = artifact_root / "train" / f"sam2_ir_qd_m9_{spec.init_from_key}"
            init_checkpoint = _selected_checkpoint_from_summary(init_dir)
        job = _build_train_job(
            spec=spec,
            raw=raw,
            paths=paths,
            suite_config=suite_config,
            base_matrix=base_matrix,
            artifact_root=artifact_root,
            generated_dir=generated_dir,
            gpu=gpus[index % len(gpus)],
            init_checkpoint=init_checkpoint,
        )
        selected = _selected_checkpoint_from_summary(job.output_dir)
        if selected is not None:
            job.selected_checkpoint_path = selected
            job.status = "skipped_existing"
        jobs[job.key] = job
    return jobs


def _make_train_progress(*, total: int, enabled: bool) -> Any | None:
    if not enabled or total <= 0:
        return None
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm(total=total, unit="epoch", desc="m9 train epochs", dynamic_ncols=True, leave=True, position=0, file=sys.stderr)


def _progress_write(progress: Any | None, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
        return
    print(message, flush=True)


def _read_train_progress_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _remove_train_progress_files(job: TrainJob) -> None:
    for path in (job.progress_state_path, job.progress_events_path):
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _advance_train_progress_to(progress: Any | None, *, job: TrainJob, seen_epochs: dict[str, int], epoch: int) -> None:
    if progress is None:
        return
    target = min(job.epochs, max(0, int(epoch)))
    previous = seen_epochs.get(job.key, 0)
    if target > previous:
        progress.update(target - previous)
        seen_epochs[job.key] = target


def _set_train_progress_postfix(
    progress: Any | None,
    *,
    active: list[dict[str, Any]],
    seen_epochs: dict[str, int],
    counts: dict[str, int],
    queued: int,
) -> None:
    if progress is None:
        return
    postfix: dict[str, Any] = {
        "active": len(active),
        "completed": counts.get("completed", 0),
        "skipped": counts.get("skipped", 0),
        "failed": counts.get("failed", 0),
        "dry_run": counts.get("dry_run", 0),
        "queued": queued,
    }
    latest_state: dict[str, Any] | None = None
    latest_time = -1.0
    for running in active[:4]:
        job = running["job"]
        state = running.get("progress_state")
        phase = str(state.get("phase", "starting")) if isinstance(state, dict) else "starting"
        postfix[f"{job.variant}_{job.seed}"] = f"{seen_epochs.get(job.key, 0)}/{job.epochs}:{phase}"
        if isinstance(state, dict):
            updated_at = float(state.get("updated_at_unix", 0.0) or 0.0)
            if updated_at > latest_time:
                latest_time = updated_at
                latest_state = state
    if len(active) > 4:
        postfix["more"] = len(active) - 4
    if latest_state:
        metrics = latest_state.get("metrics", {})
        if isinstance(metrics, dict) and "loss" in metrics:
            postfix["loss"] = f"{float(metrics['loss']):.4f}"
        best_epoch = latest_state.get("best_checkpoint_epoch")
        if best_epoch is not None:
            postfix["best_epoch"] = best_epoch
    progress.set_postfix(postfix)


def _refresh_train_progress(
    progress: Any | None,
    *,
    active: list[dict[str, Any]],
    seen_epochs: dict[str, int],
    counts: dict[str, int],
    queued: int,
) -> None:
    if progress is None:
        return
    for running in active:
        job = running["job"]
        state = _read_train_progress_state(job.progress_state_path)
        if not state:
            continue
        running["progress_state"] = state
        _advance_train_progress_to(progress, job=job, seen_epochs=seen_epochs, epoch=int(state.get("epoch", 0) or 0))
    _set_train_progress_postfix(progress, active=active, seen_epochs=seen_epochs, counts=counts, queued=queued)


def _terminate_train_processes(active: list[dict[str, Any]]) -> None:
    for running in list(active):
        process = running["process"]
        if process.poll() is None:
            process.terminate()
    for running in list(active):
        process = running["process"]
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        handle = running.get("handle")
        if handle is not None and not handle.closed:
            handle.close()
        active.remove(running)


def _run_jobs(
    *,
    jobs: list[TrainJob],
    paths: dict[str, Any],
    gpus: list[str],
    dry_run: bool,
    rerun: bool,
    stop_on_error: bool,
    python_bin: str,
    show_progress: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    queue = list(jobs)
    active: list[dict[str, Any]] = []
    free_gpus = list(gpus)
    total_jobs = len(queue)
    total_epochs = sum(job.epochs for job in jobs)
    progress = _make_train_progress(total=total_epochs, enabled=show_progress)
    progress_counts: dict[str, int] = {"completed": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    seen_epochs: dict[str, int] = {}
    completed_slots = 0
    try:
        while queue or active:
            while queue and free_gpus and not (stop_on_error and failures):
                job = queue.pop(0)
                gpu = free_gpus.pop(0)
                job.gpu = str(gpu)
                prefix = f"[train {completed_slots + len(active) + 1}/{total_jobs} {job.key} seed={job.seed} gpu={gpu}]"
                existing = _selected_checkpoint_from_summary(job.output_dir)
                if existing is not None and not rerun:
                    job.status = "skipped_existing"
                    job.selected_checkpoint_path = existing
                    progress_counts["skipped"] += 1
                    _advance_train_progress_to(progress, job=job, seen_epochs=seen_epochs, epoch=job.epochs)
                    _progress_write(progress, f"{prefix} skipped_existing selected={existing}")
                    completed_slots += 1
                    free_gpus.append(str(gpu))
                    _set_train_progress_postfix(progress, active=active, seen_epochs=seen_epochs, counts=progress_counts, queued=len(queue))
                    continue
                command = [python_bin, str(TRAIN_AUTO_PROMPT_PY), "--config", str(job.config_path)]
                if dry_run:
                    job.status = "dry_run"
                    progress_counts["dry_run"] += 1
                    _advance_train_progress_to(progress, job=job, seen_epochs=seen_epochs, epoch=job.epochs)
                    _progress_write(progress, f"{prefix} dry_run CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}")
                    completed_slots += 1
                    free_gpus.append(str(gpu))
                    _set_train_progress_postfix(progress, active=active, seen_epochs=seen_epochs, counts=progress_counts, queued=len(queue))
                    continue
                _remove_train_progress_files(job)
                env = fr._build_env(paths)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = job.log_path.open("w", encoding="utf-8")
                handle.write(f"$ {' '.join(command)}\n\n")
                handle.flush()
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
                active.append({"job": job, "gpu": str(gpu), "process": process, "handle": handle, "prefix": prefix})
                _progress_write(progress, f"{prefix} running log={job.log_path}")
                _set_train_progress_postfix(progress, active=active, seen_epochs=seen_epochs, counts=progress_counts, queued=len(queue))
            if not active:
                if stop_on_error and failures:
                    break
                continue
            time.sleep(1.0)
            _refresh_train_progress(progress, active=active, seen_epochs=seen_epochs, counts=progress_counts, queued=len(queue))
            for running in list(active):
                process = running["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                running["handle"].close()
                active.remove(running)
                free_gpus.append(str(running["gpu"]))
                job = running["job"]
                job.returncode = int(returncode)
                selected = _selected_checkpoint_from_summary(job.output_dir)
                if returncode == 0 and selected is not None:
                    job.status = "completed"
                    job.selected_checkpoint_path = selected
                    progress_counts["completed"] += 1
                    _advance_train_progress_to(progress, job=job, seen_epochs=seen_epochs, epoch=job.epochs)
                    _progress_write(progress, f"{running['prefix']} completed selected={selected}")
                else:
                    job.status = "failed"
                    progress_counts["failed"] += 1
                    failure = {"key": job.key, "seed": job.seed, "role": job.role, "log_path": str(job.log_path), "returncode": returncode}
                    failures.append(failure)
                    _progress_write(progress, f"{running['prefix']} failed")
                completed_slots += 1
                _set_train_progress_postfix(progress, active=active, seen_epochs=seen_epochs, counts=progress_counts, queued=len(queue))
            if stop_on_error and failures:
                _terminate_train_processes(active)
                break
    finally:
        if progress is not None:
            progress.close()
    return failures


def _run_selector(
    *,
    job: TrainJob,
    validation_config_paths: list[str],
    paths: dict[str, Any],
    dry_run: bool,
    python_bin: str,
    selector: dict[str, Any],
) -> dict[str, Any]:
    command = [
        python_bin,
        str(SELECT_AUTO_PROMPT_CHECKPOINT_PY),
        "--train-dir",
        str(job.output_dir),
        "--device",
        str(selector.get("device", "cuda")),
        "--max-samples",
        str(int(selector.get("max_samples", 256))),
        "--top-k",
        str(int(selector.get("top_k", 5))),
        "--point-budget",
        str(int(selector.get("point_budget", 1))),
        "--response-threshold",
        str(float(selector.get("response_threshold", 0.15))),
        "--nms-radius",
        str(int(selector.get("nms_radius", 4))),
        "--border-suppression-px",
        str(int(selector.get("border_suppression_px", 4))),
    ]
    for dataset_config in validation_config_paths:
        command.extend(["--dataset-config", str(dataset_config)])
    if dry_run:
        print(f"[select dry-run] {' '.join(command)}", flush=True)
        return {"status": "dry_run", "job": job.key}
    env = fr._build_env(paths)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    log_path = job.output_dir / "checkpoint_selection.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
    selected = _selected_checkpoint_from_summary(job.output_dir)
    if result.returncode == 0 and selected is not None:
        job.selected_checkpoint_path = selected
        return {"status": "completed", "job": job.key, "selected_checkpoint": str(selected)}
    return {"status": "failed", "job": job.key, "log_path": str(log_path), "returncode": result.returncode}


def _run_export(*, raw: dict[str, Any], dry_run: bool, python_bin: str) -> dict[str, Any]:
    export_cfg = raw.get("m9", {}).get("export", {})
    if not export_cfg:
        return {"status": "skipped_no_config"}
    root = export_cfg.get("root")
    if not root:
        return {"status": "skipped_no_root"}
    root_path = Path(str(root)).expanduser()
    expected_outputs = _expected_rbgt_export_paths(root_path)
    overwrite = bool(export_cfg.get("overwrite", False))
    validation = _validate_existing_rbgt_export(root_path, export_cfg) if bool(export_cfg.get("validate_existing", True)) else {"valid": all(path.exists() for path in expected_outputs), "issues": [], "outputs": [str(path) for path in expected_outputs]}
    any_existing = any(path.exists() for path in expected_outputs)
    all_existing = all(path.exists() for path in expected_outputs)
    if all_existing and validation.get("valid") and not overwrite:
        return {"status": "skipped_existing", "outputs": [str(path) for path in expected_outputs], "validation": validation}
    if any_existing and not overwrite:
        if not bool(export_cfg.get("auto_reexport_invalid", False)):
            return {
                "status": "failed",
                "returncode": 1,
                "message": "Existing RBGT M9 export is missing, stale, or invalid. Set m9.export.overwrite=true after confirming it is safe to regenerate.",
                "validation": validation,
            }
        overwrite = True
    command = [
        python_bin,
        str(EXPORT_RBGT_TINY_BOX_COCO_PY),
        "--root",
        str(root),
        "--split",
    ]
    if bool(export_cfg.get("small_target_filter", True)):
        command.append("--small-target-filter")
    if overwrite:
        command.append("--overwrite")
    for key, flag in (
        ("seed", "--seed"),
        ("max_area_ratio", "--max-area-ratio"),
        ("max_box_side", "--max-box-side"),
        ("min_box_side", "--min-box-side"),
    ):
        if export_cfg.get(key) is not None:
            command.extend([flag, str(export_cfg[key])])
    for key, flag in (("annotations_dir", "--annotations-dir"), ("images_dir", "--images-dir")):
        if export_cfg.get(key):
            command.extend([flag, str(export_cfg[key])])
    if dry_run:
        print(f"[export dry-run] {' '.join(command)}", flush=True)
        return {"status": "dry_run", "command": command, "validation": validation}
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    post_validation = _validate_existing_rbgt_export(root_path, export_cfg) if result.returncode == 0 else validation
    status = "completed" if result.returncode == 0 and post_validation.get("valid") else "failed"
    return {"status": status, "returncode": result.returncode, "command": command, "validation": post_validation}


def _eval_variants_for_jobs(jobs: list[TrainJob], selected_variants: set[str] | None) -> list[str]:
    eval_m9b = (selected_variants is None or "M9-B" in selected_variants) and any(job.key.startswith("v2_rbgt") for job in jobs)
    return sorted({job.variant for job in jobs if job.role != "pretrain"} | ({"M9-B"} if eval_m9b else set()))


def _build_variant_eval_plan(
    *,
    variant: str,
    jobs: list[TrainJob],
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    dry_run: bool,
    python_bin: str,
    include_reference: bool,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    eval_roles = {"pretrain"} if variant == "M9-B" else {"public", "finetune", "mixed"}
    train_contexts = [
        {
            "seed": job.seed,
            "experiment_id": f"{variant}_seed{job.seed}",
            "checkpoint_path": job.selected_checkpoint_path,
            "final_checkpoint_path": job.checkpoint_path,
        }
        for job in jobs
        if job.variant == variant
        and job.role in eval_roles
        and (dry_run or job.selected_checkpoint_path.exists())
    ]
    if not train_contexts:
        return [], []
    m9 = raw.get("m9", {})
    public_eval = [str(item) for item in m9.get("public_eval_datasets", ["nuaa_sirst", "nudt_sirst", "irstd_1k"])]
    learned_modes = [str(item) for item in m9.get("learned_eval_modes", ["sam2_learned_auto_point", "sam2_learned_auto_point_rerank", "sam2_ir_fa_rerank", "sam2_ir_fa_rerank_feedback_only"])]
    reference_modes = [str(item) for item in m9.get("reference_eval_modes", ["sam2_heuristic_auto_box_point", "sam2_box_oracle", "sam2_box_point_oracle"])]
    suite_key = f"auto_prompt_m9_{variant.lower().replace('-', '').replace('_', '')}"
    suite_entry = {
        "enabled": True,
        "experiment_id": f"E9_{variant}",
        "purpose": f"M9 variant {variant} evaluation.",
        "checkpoints": ["large"],
        "modes": [*(reference_modes if include_reference else []), *learned_modes],
        "datasets": public_eval,
        "run_analysis": True,
        "primary_metric": "mIoU",
        "comparisons": [
            {"name": f"{variant}_fa_rerank_vs_learned_point", "baseline": "sam2_learned_auto_point", "candidate": "sam2_ir_fa_rerank"},
            {"name": f"{variant}_feedback_only_vs_fa_rerank", "baseline": "sam2_ir_fa_rerank", "candidate": "sam2_ir_fa_rerank_feedback_only"},
        ],
    }
    suite_subset = copy.deepcopy(suite_config)
    suite_subset["artifact_subdir"] = str(raw.get("m9", {}).get("artifact_subdir", "sam2_ir_qd_m9_full_v1"))
    suite_subset["suites"] = {suite_key: suite_entry}
    return apr._build_run_plan(
        base_matrix=base_matrix,
        suite_config=suite_subset,
        paths=paths,
        auto_config=raw.get("auto_prompt", {}),
        train_contexts=train_contexts,
        config_dir=generated_dir / "run_configs",
        matrix_dir=generated_dir / "matrices",
        analysis_config_dir=generated_dir / "analysis_configs",
        analysis_root=artifact_root / "analysis",
        source_config_path=Path(str(raw["__config_path__"])),
        source_config_sha256=str(raw["__config_sha256__"]),
        selected_suites={suite_key},
        selected_checkpoints=None,
        selected_modes=None,
        smoke_test=False,
        python_bin=python_bin,
    )


def _build_reference_eval_plan(
    *,
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    python_bin: str,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    m9 = raw.get("m9", {})
    public_eval = [str(item) for item in m9.get("public_eval_datasets", ["nuaa_sirst", "nudt_sirst", "irstd_1k"])]
    reference_modes = [str(item) for item in m9.get("reference_eval_modes", ["sam2_heuristic_auto_box_point", "sam2_box_oracle", "sam2_box_point_oracle"])]
    if not reference_modes:
        return [], []
    suite_key = "auto_prompt_m9_reference"
    suite_entry = {
        "enabled": True,
        "experiment_id": "E9_reference",
        "purpose": "Shared M9 heuristic/oracle reference evaluation.",
        "checkpoints": ["large"],
        "modes": reference_modes,
        "datasets": public_eval,
        "run_analysis": True,
        "primary_metric": "mIoU",
        "comparisons": [],
    }
    suite_subset = copy.deepcopy(suite_config)
    suite_subset["artifact_subdir"] = str(raw.get("m9", {}).get("artifact_subdir", "sam2_ir_qd_m9_full_v1"))
    suite_subset["suites"] = {suite_key: suite_entry}
    return apr._build_run_plan(
        base_matrix=base_matrix,
        suite_config=suite_subset,
        paths=paths,
        auto_config=raw.get("auto_prompt", {}),
        train_contexts=[],
        config_dir=generated_dir / "run_configs",
        matrix_dir=generated_dir / "matrices",
        analysis_config_dir=generated_dir / "analysis_configs",
        analysis_root=artifact_root / "analysis",
        source_config_path=Path(str(raw["__config_path__"])),
        source_config_sha256=str(raw["__config_sha256__"]),
        selected_suites={suite_key},
        selected_checkpoints=None,
        selected_modes=None,
        smoke_test=False,
        python_bin=python_bin,
    )


def _build_all_eval_plan(
    *,
    jobs: list[TrainJob],
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    dry_run: bool,
    python_bin: str,
    selected_variants: set[str] | None,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    reference_policy = _reference_policy(raw.get("m9", {}))
    run_plan: list[tuple[Any, ...]] = []
    analysis_records: list[dict[str, Any]] = []
    if reference_policy == "shared":
        records, analyses = _build_reference_eval_plan(
            raw=raw,
            paths=paths,
            suite_config=suite_config,
            base_matrix=base_matrix,
            artifact_root=artifact_root,
            generated_dir=generated_dir,
            python_bin=python_bin,
        )
        run_plan.extend(records)
        analysis_records.extend(analyses)
    include_reference_next = reference_policy == "first_variant"
    for variant in _eval_variants_for_jobs(jobs, selected_variants):
        records, analyses = _build_variant_eval_plan(
            variant=variant,
            jobs=jobs,
            raw=raw,
            paths=paths,
            suite_config=suite_config,
            base_matrix=base_matrix,
            artifact_root=artifact_root,
            generated_dir=generated_dir,
            dry_run=dry_run,
            python_bin=python_bin,
            include_reference=reference_policy == "per_variant" or include_reference_next,
        )
        include_reference_next = False
        run_plan.extend(records)
        analysis_records.extend(analyses)
    return run_plan, analysis_records


def _eval_variant(
    *,
    variant: str,
    jobs: list[TrainJob],
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    eval_gpus: list[str],
    dry_run: bool,
    rerun: bool,
    stop_on_error: bool,
    python_bin: str,
    include_reference: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_plan, analysis_records = _build_variant_eval_plan(
        variant=variant,
        jobs=jobs,
        raw=raw,
        paths=paths,
        suite_config=suite_config,
        base_matrix=base_matrix,
        artifact_root=artifact_root,
        generated_dir=generated_dir,
        dry_run=dry_run,
        python_bin=python_bin,
        include_reference=include_reference,
    )
    if not run_plan:
        return [], [], []
    records, failures = apr._run_eval_plan(
        run_plan=run_plan,
        paths=paths,
        eval_gpus=eval_gpus,
        manifest_dir=artifact_root,
        run_id=f"m9-{variant}-{int(time.time())}",
        dry_run=dry_run,
        rerun=rerun,
        stop_on_error=stop_on_error,
        show_progress=True,
        stream_logs=False,
    )
    if not dry_run and not failures:
        for item in analysis_records:
            command = fr._analysis_command(Path(item["analysis_config"]), python_bin)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=fr._build_env(paths), text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            item["status"] = "completed" if result.returncode == 0 else "failed"
            item["returncode"] = result.returncode
    return records, failures, analysis_records


def _eval_reference(
    *,
    raw: dict[str, Any],
    paths: dict[str, Any],
    suite_config: dict[str, Any],
    base_matrix: dict[str, Any],
    artifact_root: Path,
    generated_dir: Path,
    eval_gpus: list[str],
    dry_run: bool,
    rerun: bool,
    stop_on_error: bool,
    python_bin: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_plan, analysis_records = _build_reference_eval_plan(
        raw=raw,
        paths=paths,
        suite_config=suite_config,
        base_matrix=base_matrix,
        artifact_root=artifact_root,
        generated_dir=generated_dir,
        python_bin=python_bin,
    )
    if not run_plan:
        return [], [], []
    records, failures = apr._run_eval_plan(
        run_plan=run_plan,
        paths=paths,
        eval_gpus=eval_gpus,
        manifest_dir=artifact_root,
        run_id=f"m9-reference-{int(time.time())}",
        dry_run=dry_run,
        rerun=rerun,
        stop_on_error=stop_on_error,
        show_progress=True,
        stream_logs=False,
    )
    if not dry_run and not failures:
        for item in analysis_records:
            command = fr._analysis_command(Path(item["analysis_config"]), python_bin)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=fr._build_env(paths), text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            item["status"] = "completed" if result.returncode == 0 else "failed"
            item["returncode"] = result.returncode
    return records, failures, analysis_records


def _write_variant_summary(artifact_root: Path, jobs: list[TrainJob], eval_records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    variants = sorted({job.variant for job in jobs})
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.append(
            {
                "variant": variant,
                "train_jobs": len([job for job in jobs if job.variant == variant]),
                "completed_train_jobs": len([job for job in jobs if job.variant == variant and job.status in {"completed", "skipped_existing", "dry_run"}]),
                "eval_records": len(
                    [
                        record
                        for record in eval_records
                        if str(record.get("suite_key", record.get("suite", ""))).lower().endswith(variant.lower().replace("-", "").replace("_", ""))
                    ]
                ),
            }
        )
    summary_path = artifact_root / "m9_variant_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "train_jobs", "completed_train_jobs", "eval_records"])
        writer.writeheader()
        writer.writerows(rows)
    gate = {
        "status": "ready_for_result_interpretation" if not failures else "has_failures",
        "failure_count": len(failures),
        "note": "Compare M9-E against M6 after full analysis is complete.",
    }
    _write_json(artifact_root / "m9_success_gate.json", gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full SAM2-IR-QD M9 RBGT-Tiny experiment matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_M9_CONFIG)
    parser.add_argument("--stage", choices=("all", "export", "pretrain", "finetune", "select", "eval", "analysis"), default="all")
    parser.add_argument("--variants")
    parser.add_argument("--seeds")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rerun-stage", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--preflight-mode", choices=apr.PREFLIGHT_MODES, default="fast")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    raw = _load_yaml(config_path)
    raw["__config_path__"] = str(config_path)
    raw["__config_sha256__"] = sha256_file(config_path)
    paths, suite_config, base_matrix, _ = fr._load_complete_benchmark_config(config_path)
    apr._ensure_auto_methods(base_matrix)
    m9 = raw.setdefault("m9", {})
    artifact_subdir = str(m9.get("artifact_subdir", raw.get("auto_prompt", {}).get("artifact_subdir", "sam2_ir_qd_m9_full_v1")))
    if args.smoke_test:
        artifact_subdir = f"{artifact_subdir}_smoke"
    m9["artifact_subdir"] = artifact_subdir
    artifact_root = fr._artifact_base(paths) / artifact_subdir
    generated_dir = artifact_root / "generated_m9"
    seeds = [int(item) for item in (_split_csv(args.seeds) or {str(seed) for seed in m9.get("seeds", [42, 123, 456])})]
    seeds = sorted(seeds)
    variants = _split_csv(args.variants)
    gpus = [str(item) for item in m9.get("gpus", ["0", "1", "2", "3", "4", "5", "6", "7"])]
    if args.smoke_test:
        smoke_runtime = copy.deepcopy(suite_config.get("smoke_test_runtime", {}))
        if smoke_runtime:
            suite_config["runtime"] = fr._deep_merge(copy.deepcopy(suite_config.get("runtime", {})), smoke_runtime)
        raw.setdefault("auto_prompt", {}).setdefault("train", {})["epochs"] = 1
        m9.setdefault("stage_defaults", {})
        for role in ("pretrain", "public", "finetune", "mixed"):
            m9["stage_defaults"].setdefault(role, {})
            m9["stage_defaults"][role]["epochs"] = 1
            m9["stage_defaults"][role]["checkpoint_interval_epochs"] = 1
            m9["stage_defaults"][role]["batch_size"] = 1
            m9["stage_defaults"][role]["light_cache_batch_size"] = 1
            m9["stage_defaults"][role]["gradient_accumulation_steps"] = 1
            m9["stage_defaults"][role]["light_cache_max_samples"] = 32
            m9["stage_defaults"][role]["light_cache_samples_per_epoch"] = 32
            m9["stage_defaults"][role]["max_steps_per_epoch"] = 2
            m9["stage_defaults"][role]["validation_max_batches"] = 1
            m9["stage_defaults"][role]["max_long_side"] = 256
        m9.setdefault("selector", {})
        m9["selector"]["max_samples"] = min(int(m9["selector"].get("max_samples", 16)), 16)

    failures: list[dict[str, Any]] = []
    export_record: dict[str, Any] | None = None
    preflight_path: Path | None = None
    preflight_summary: dict[str, Any] | None = None
    if args.stage in {"all", "export"}:
        export_record = _run_export(raw=raw, dry_run=args.dry_run, python_bin=args.python_bin)
        if export_record.get("status") == "failed" and args.stop_on_error:
            return 1

    specs = _default_variant_specs(m9, seeds=seeds, selected_variants=variants)
    pretrain_specs = [spec for spec in specs if spec.role == "pretrain"]
    final_specs = [spec for spec in specs if spec.role != "pretrain"]
    checkpoint = suite_config["checkpoints"][0]
    dataset_config_dir = generated_dir / "training_dataset_configs"
    jobs_by_key: dict[str, TrainJob] = {}
    if args.stage in {"all", "pretrain", "finetune", "select", "eval", "analysis"}:
        preflight_jobs = _materialize_train_jobs(
            specs=specs,
            raw=raw,
            paths=paths,
            suite_config=suite_config,
            base_matrix=base_matrix,
            artifact_root=artifact_root,
            generated_dir=generated_dir,
            gpus=gpus,
        )
        preflight_run_plan, _ = _build_all_eval_plan(
            jobs=list(preflight_jobs.values()),
            raw=raw,
            paths=paths,
            suite_config=suite_config,
            base_matrix=base_matrix,
            artifact_root=artifact_root,
            generated_dir=generated_dir,
            dry_run=True,
            python_bin=args.python_bin,
            selected_variants=variants,
        )
        preflight_summary = apr._dataset_preflight_summary(
            train_config_paths=[job.config_path for job in preflight_jobs.values()],
            run_plan=preflight_run_plan,
            mode=args.preflight_mode,
        )
        preflight_path = apr._write_dataset_preflight_summary(artifact_root, preflight_summary)
        apr._print_dataset_preflight_summary(preflight_summary)
        if not bool(preflight_summary.get("overall", {}).get("valid", False)):
            failure = {"status": "preflight_failed", "preflight_path": str(preflight_path), "summary": preflight_summary.get("overall", {})}
            failures.append(failure)
            if args.stop_on_error:
                return 1
    if args.stage in {"select", "eval", "analysis"}:
        jobs_by_key.update(
            _materialize_train_jobs(
                specs=specs,
                raw=raw,
                paths=paths,
                suite_config=suite_config,
                base_matrix=base_matrix,
                artifact_root=artifact_root,
                generated_dir=generated_dir,
                gpus=gpus,
            )
        )

    if args.stage in {"all", "pretrain"}:
        pretrain_jobs = [
            _build_train_job(
                spec=spec,
                raw=raw,
                paths=paths,
                suite_config=suite_config,
                base_matrix=base_matrix,
                artifact_root=artifact_root,
                generated_dir=generated_dir,
                gpu=gpus[index % len(gpus)],
                init_checkpoint=None,
            )
            for index, spec in enumerate(pretrain_specs)
        ]
        failures.extend(
            _run_jobs(
                jobs=pretrain_jobs,
                paths=paths,
                gpus=gpus,
                dry_run=args.dry_run,
                rerun=args.rerun_stage,
                stop_on_error=args.stop_on_error,
                python_bin=args.python_bin,
                show_progress=bool(raw.get("auto_prompt", {}).get("train", {}).get("show_progress", True)),
            )
        )
        jobs_by_key.update({job.key: job for job in pretrain_jobs})
        if failures and args.stop_on_error:
            return 1

    if args.stage in {"all", "select", "pretrain"}:
        selector = m9.get("selector", {})
        pretrain_selection_jobs = [job for job in jobs_by_key.values() if job.role == "pretrain"]
        for job in pretrain_selection_jobs:
            validation_configs = _write_dataset_configs(
                dataset_ids=[str(m9.get("rbgt_val_dataset", "rbgt_tiny_ir_box_m9_val"))],
                base_matrix=base_matrix,
                suite_config=suite_config,
                paths=paths,
                checkpoint=checkpoint,
                output_dir=dataset_config_dir,
            )
            result = _run_selector(
                job=job,
                validation_config_paths=validation_configs,
                paths=paths,
                dry_run=args.dry_run,
                python_bin=args.python_bin,
                selector=selector,
            )
            if result.get("status") == "failed":
                failures.append(result)
                if args.stop_on_error:
                    return 1

    final_jobs: list[TrainJob] = []
    if args.stage in {"all", "finetune"}:
        for index, spec in enumerate(final_specs):
            init_checkpoint = None
            if spec.init_from_key is not None:
                init_job = jobs_by_key.get(spec.init_from_key)
                if init_job is None:
                    init_dir = artifact_root / "train" / f"sam2_ir_qd_m9_{spec.init_from_key}"
                    init_checkpoint = _selected_checkpoint_from_summary(init_dir)
                else:
                    init_checkpoint = init_job.selected_checkpoint_path
                if init_checkpoint is None or not Path(init_checkpoint).exists():
                    if not args.dry_run:
                        failures.append({"status": "missing_init_checkpoint", "stage": spec.key, "init_from_key": spec.init_from_key})
                        if args.stop_on_error:
                            return 1
            final_jobs.append(
                _build_train_job(
                    spec=spec,
                    raw=raw,
                    paths=paths,
                    suite_config=suite_config,
                    base_matrix=base_matrix,
                    artifact_root=artifact_root,
                    generated_dir=generated_dir,
                    gpu=gpus[index % len(gpus)],
                    init_checkpoint=Path(init_checkpoint) if init_checkpoint is not None else None,
                )
            )
        failures.extend(
            _run_jobs(
                jobs=final_jobs,
                paths=paths,
                gpus=gpus,
                dry_run=args.dry_run,
                rerun=args.rerun_stage,
                stop_on_error=args.stop_on_error,
                python_bin=args.python_bin,
                show_progress=bool(raw.get("auto_prompt", {}).get("train", {}).get("show_progress", True)),
            )
        )
        jobs_by_key.update({job.key: job for job in final_jobs})
        if failures and args.stop_on_error:
            return 1

    if args.stage in {"all", "select", "finetune"}:
        selector = m9.get("selector", {})
        public_val = [str(item) for item in m9.get("public_train_datasets", ["nuaa_sirst", "nudt_sirst", "irstd_1k"])]
        final_selection_jobs = final_jobs or [job for job in jobs_by_key.values() if job.role != "pretrain"]
        for job in final_selection_jobs:
            validation_configs = _write_dataset_configs(
                dataset_ids=public_val,
                base_matrix=base_matrix,
                suite_config=suite_config,
                paths=paths,
                checkpoint=checkpoint,
                output_dir=dataset_config_dir,
            )
            result = _run_selector(
                job=job,
                validation_config_paths=validation_configs,
                paths=paths,
                dry_run=args.dry_run,
                python_bin=args.python_bin,
                selector=selector,
            )
            if result.get("status") == "failed":
                failures.append(result)
                if args.stop_on_error:
                    return 1

    all_jobs = [*jobs_by_key.values()]
    eval_records: list[dict[str, Any]] = []
    analysis_records: list[dict[str, Any]] = []
    if args.stage in {"all", "eval", "analysis"}:
        policy = _reference_policy(m9)
        if policy == "shared":
            records, eval_failures, analyses = _eval_reference(
                raw=raw,
                paths=paths,
                suite_config=suite_config,
                base_matrix=base_matrix,
                artifact_root=artifact_root,
                generated_dir=generated_dir,
                eval_gpus=gpus,
                dry_run=args.dry_run,
                rerun=args.rerun_stage,
                stop_on_error=args.stop_on_error,
                python_bin=args.python_bin,
            )
            eval_records.extend(records)
            analysis_records.extend(analyses)
            failures.extend(eval_failures)
            if eval_failures and args.stop_on_error:
                _write_variant_summary(artifact_root, all_jobs, eval_records, failures)
                return 1
        include_reference_next = policy == "first_variant"
        for variant in _eval_variants_for_jobs(all_jobs, variants):
            records, eval_failures, analyses = _eval_variant(
                variant=variant,
                jobs=all_jobs,
                raw=raw,
                paths=paths,
                suite_config=suite_config,
                base_matrix=base_matrix,
                artifact_root=artifact_root,
                generated_dir=generated_dir,
                eval_gpus=gpus,
                dry_run=args.dry_run,
                rerun=args.rerun_stage,
                stop_on_error=args.stop_on_error,
                python_bin=args.python_bin,
                include_reference=policy == "per_variant" or include_reference_next,
            )
            include_reference_next = False
            eval_records.extend(records)
            analysis_records.extend(analyses)
            failures.extend(eval_failures)
            if eval_failures and args.stop_on_error:
                break

    _write_variant_summary(artifact_root, all_jobs, eval_records, failures)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "artifact_root": str(artifact_root),
        "dry_run": args.dry_run,
        "smoke_test": args.smoke_test,
        "stage": args.stage,
        "preflight_mode": args.preflight_mode,
        "preflight_path": str(preflight_path) if preflight_path is not None else None,
        "preflight": preflight_summary.get("overall", {}) if preflight_summary else None,
        "reference_policy": _reference_policy(m9),
        "variants": sorted(variants) if variants else "default",
        "seeds": seeds,
        "export": export_record,
        "export_validation": export_record.get("validation") if isinstance(export_record, dict) else None,
        "train_jobs": [_jsonable_train_job(job) for job in all_jobs],
        "eval_count": len(eval_records),
        "analysis_count": len(analysis_records),
        "failures": failures,
    }
    _write_json(artifact_root / "m9_manifest_latest.json", manifest)
    print(f"[m9 done] artifact_root={artifact_root} train_jobs={len(all_jobs)} eval_records={len(eval_records)} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
