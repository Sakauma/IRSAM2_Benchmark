from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import InfraredDataset, Sample, collate_fn, deterministic_source_split
from .dataset_adapters import build_dataset_adapter
from .evaluation import evaluate_samples
from .method_registry import build_method_registry
from .methods import BaseMethod, QualityFilteredPseudoMaskSelfTrainingSAM2, SAM2Teacher
from .metrics import primary_metric_from_metrics
from .reporting import build_summary, write_eval_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_method(
    method: BaseMethod,
    samples: List[Sample],
    output_dir: Path,
    prefix: str,
    save_artifacts: bool,
) -> Tuple[Dict[str, float], Optional[str]]:
    evaluation = evaluate_samples(method, samples, output_dir, prefix, save_artifacts)
    report_path: Optional[str] = None
    if save_artifacts:
        report_path = write_eval_report(
            output_dir=output_dir,
            prefix=prefix,
            aggregate_metrics=evaluation.aggregate_metrics,
            metric_rows=evaluation.metric_rows,
        )
    metrics = dict(evaluation.aggregate_metrics)
    if evaluation.visual_path is not None:
        metrics["VisualPath"] = evaluation.visual_path
    return metrics, report_path


def train_method(
    method: BaseMethod,
    train_samples: List[Sample],
    val_samples: List[Sample],
    config: ExperimentConfig,
    num_epochs: int,
) -> Dict[str, float]:
    if not method.is_trainable:
        val_metrics, _ = evaluate_method(method, val_samples, config.output_dir, prefix="val_static", save_artifacts=False)
        return {"best_val_mIoU": val_metrics["mIoU"], "best_epoch": 0}

    dataset = InfraredDataset(
        train_samples,
        require_mask=config.supervision_protocol == "mask_supervised",
        allow_gt_masks=config.supervision_protocol == "mask_supervised",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    optimizer = method.build_optimizer(config)

    best_val_miou = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(method.state_dict())

    for epoch in range(num_epochs):
        method.set_train()
        for batch in loader:
            method.training_step(batch, optimizer, config)
            if config.smoke_test:
                break
        method.set_eval()
        val_metrics, _ = evaluate_method(method, val_samples, config.output_dir, prefix=f"val_epoch_{epoch}", save_artifacts=False)
        if val_metrics["mIoU"] > best_val_miou:
            best_val_miou = val_metrics["mIoU"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(method.state_dict())
        if config.smoke_test:
            break

    method.load_state_dict(best_state)
    method.set_eval()
    return {"best_val_mIoU": float(best_val_miou), "best_epoch": int(best_epoch)}


def run_condition(
    condition: str,
    method: BaseMethod,
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    unlabeled_samples: List[Sample],
    config: ExperimentConfig,
    artifact_prefix: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_info = {"best_val_mIoU": 0.0, "best_epoch": 0, "pseudo_accept_count": 0}

    if isinstance(method, QualityFilteredPseudoMaskSelfTrainingSAM2):
        warmup_info = train_method(method, train_samples, val_samples, config, config.train_epochs)
        pseudo_samples = method.generate_pseudo_samples(unlabeled_samples)
        train_info.update(warmup_info)
        train_info["pseudo_accept_count"] = len(pseudo_samples)
        if pseudo_samples:
            finetune_samples = list(train_samples) + pseudo_samples
            finetune_info = train_method(method, finetune_samples, val_samples, config, config.pseudo_finetune_epochs)
            train_info.update(finetune_info)
    else:
        train_info.update(train_method(method, train_samples, val_samples, config, config.train_epochs))

    metrics, report_path = evaluate_method(method, test_samples, config.output_dir, artifact_prefix, save_artifacts=True)
    if report_path is not None:
        metrics["EvalReportPath"] = report_path
    return metrics, train_info


def run_experiment(config: Optional[ExperimentConfig] = None) -> None:
    config = config or load_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = config.root / "EXPERIMENT_PLAN.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf8"))
    planned_conditions = plan.get("compute_budget", {}).get("planned_conditions", {})
    all_planned = planned_conditions.get("core_conditions", []) + planned_conditions.get("ablation_conditions", [])
    active_conditions = [condition for condition in config.active_conditions if condition in all_planned]
    skipped_conditions = [condition for condition in all_planned if condition not in active_conditions]

    if not active_conditions:
        raise RuntimeError("No active experiment conditions remain after intersecting config.active_conditions with the plan.")

    dataset_adapter = build_dataset_adapter(config)
    loaded_dataset = dataset_adapter.load(config)
    samples = loaded_dataset.samples
    teacher = SAM2Teacher(config)
    registry = build_method_registry(teacher, config)
    results: List[Dict[str, float]] = []

    for budget in config.supervision_budgets:
        for seed in config.seeds:
            set_seed(seed)
            train_samples, val_samples, test_samples, unlabeled_samples = deterministic_source_split(samples, budget, config.eval_limit)
            for condition in active_conditions:
                if not registry.has(condition):
                    continue
                set_seed(seed)
                method = registry.build(condition)
                prefix = f"{condition}_{seed}_{budget}"
                print(f"running condition={condition} seed={seed} budget={budget}")
                sys.stdout.flush()
                metrics, train_info = run_condition(
                    condition=condition,
                    method=method,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    test_samples=test_samples,
                    unlabeled_samples=unlabeled_samples,
                    config=config,
                    artifact_prefix=prefix,
                )
                row = {
                    "condition": condition,
                    "seed": seed,
                    "budget": budget,
                    "supervision_protocol": config.supervision_protocol,
                    "best_val_mIoU": train_info.get("best_val_mIoU", 0.0),
                    "best_epoch": train_info.get("best_epoch", 0),
                    "pseudo_accept_count": train_info.get("pseudo_accept_count", 0),
                }
                row.update(metrics)
                row["primary_metric"] = primary_metric_from_metrics(metrics)
                results.append(row)
                print(
                    f"condition={condition} seed={seed} budget={budget} "
                    f"best_val_mIoU={row['best_val_mIoU']:.4f} "
                    f"pseudo_accept_count={row['pseudo_accept_count']} "
                    f"mIoU={row['mIoU']:.4f} "
                    f"primary_metric={row['primary_metric']:.6f}"
                )
                sys.stdout.flush()

    (config.output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf8")
    summary = build_summary(
        results=results,
        config=config,
        plan=plan,
        active_conditions=active_conditions,
        skipped_conditions=skipped_conditions,
        dataset_manifest=loaded_dataset.manifest,
        method_manifest=registry.manifest(active_conditions),
    )
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
