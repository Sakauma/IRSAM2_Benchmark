"""实验调度器。

这是 benchmark 平台的主控制流，负责把配置、数据集、方法、训练、评估和报告串起来。
如果只读一个文件想理解整个平台怎么运行，优先读这里。
"""

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
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig, load_config
from .data import InfraredDataset, Sample, collate_fn, deterministic_source_split
from .dataset_adapters import build_dataset_adapter
from .distributed import (
    DistributedConfig,
    all_gather_object,
    barrier,
    broadcast_object,
    destroy_distributed_process_group,
    init_distributed_process_group,
)
from .evaluation import evaluate_samples
from .method_registry import build_method_registry
from .methods import BaseMethod, QualityFilteredPseudoMaskSelfTrainingSAM2, SAM2Teacher
from .metrics import primary_metric_from_metrics, summarize_metric_rows
from .reporting import build_summary, write_eval_report


def set_seed(seed: int) -> None:
    """统一设置 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process(config: ExperimentConfig) -> bool:
    """只有 rank0 负责打印日志和写全局产物。"""
    return config.rank == 0


def log_main(message: str, config: ExperimentConfig) -> None:
    """主进程日志打印，避免多卡重复刷屏。"""
    if is_main_process(config):
        print(message)
        sys.stdout.flush()


def shard_samples_for_rank(samples: List[Sample], config: ExperimentConfig) -> List[Sample]:
    """把样本列表按 rank 做静态切片。

    评估和伪标签生成阶段都用这个逻辑来分担工作。
    """
    if not config.distributed:
        return samples
    return [sample for idx, sample in enumerate(samples) if idx % config.world_size == config.rank]


def gather_metric_rows(metric_rows: List[Dict[str, object]], config: ExperimentConfig) -> List[Dict[str, object]]:
    """收集所有 rank 的逐样本评估行，并按 sample_id 排序。"""
    if not config.distributed:
        return metric_rows
    gathered = all_gather_object(metric_rows)
    merged = [row for chunk in gathered for row in chunk]
    merged.sort(key=lambda row: str(row.get("sample_id", "")))
    return merged


def gather_pseudo_samples(pseudo_samples: List[Sample], config: ExperimentConfig) -> List[Sample]:
    """收集所有 rank 生成的伪标签样本。"""
    if not config.distributed:
        return pseudo_samples
    gathered = all_gather_object(pseudo_samples)
    merged = [sample for chunk in gathered for sample in chunk]
    merged.sort(key=lambda sample: sample.frame_id)
    return merged


def evaluate_method(
    method: BaseMethod,
    samples: List[Sample],
    config: ExperimentConfig,
    output_dir: Path,
    prefix: str,
    save_artifacts: bool,
) -> Tuple[Dict[str, float], Optional[str]]:
    """执行一次方法评估，并在多卡下汇总结果。"""
    local_samples = shard_samples_for_rank(samples, config)
    evaluation = evaluate_samples(
        method,
        local_samples,
        output_dir,
        prefix,
        save_artifacts=save_artifacts and is_main_process(config),
    )
    merged_rows = gather_metric_rows(evaluation.metric_rows, config)
    metrics: Dict[str, float] | None = None
    report_path: Optional[str] = None
    visual_path = evaluation.visual_path if is_main_process(config) else None
    if is_main_process(config):
        aggregate_metrics = summarize_metric_rows(merged_rows)
        metrics = dict(aggregate_metrics)
        if save_artifacts:
            # 只有主进程写 eval report，其他 rank 只参与计算不落盘。
            report_path = write_eval_report(
                output_dir=output_dir,
                prefix=prefix,
                aggregate_metrics=aggregate_metrics,
                metric_rows=merged_rows,
            )
        if visual_path is not None:
            metrics["VisualPath"] = visual_path
    metrics = broadcast_object(metrics, src=0)
    report_path = broadcast_object(report_path, src=0)
    return dict(metrics or {}), report_path


def train_method(
    method: BaseMethod,
    train_samples: List[Sample],
    val_samples: List[Sample],
    config: ExperimentConfig,
    num_epochs: int,
) -> Dict[str, float]:
    """训练一个方法，并根据验证集选择最佳 checkpoint。"""
    if not method.is_trainable:
        # 非训练方法直接在验证集上评估一次，用于统一后续流程。
        val_metrics, _ = evaluate_method(
            method,
            val_samples,
            config,
            config.output_dir,
            prefix="val_static",
            save_artifacts=False,
        )
        return {"best_val_mIoU": val_metrics["mIoU"], "best_epoch": 0}

    dataset = InfraredDataset(
        train_samples,
        # 只有 mask_supervised 协议才强制训练集必须提供 mask。
        require_mask=config.supervision_protocol == "mask_supervised",
        allow_gt_masks=config.supervision_protocol == "mask_supervised",
    )
    sampler = DistributedSampler(dataset, shuffle=True) if config.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    method.configure_distributed(config)
    optimizer = method.build_optimizer(config)

    best_val_miou = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(method.state_dict())

    for epoch in range(num_epochs):
        method.set_train()
        if sampler is not None:
            # DDP 下每轮都要刷新 sampler epoch，保证 shuffle 一致。
            sampler.set_epoch(epoch)
        for batch in loader:
            method.training_step(batch, optimizer, config)
            if config.smoke_test:
                # smoke 模式只验证链路能走通，不追求完整收敛。
                break
        method.set_eval()
        val_metrics, _ = evaluate_method(
            method,
            val_samples,
            config,
            config.output_dir,
            prefix=f"val_epoch_{epoch}",
            save_artifacts=False,
        )
        if val_metrics["mIoU"] > best_val_miou:
            # 统一按验证集 mIoU 选最佳权重。
            best_val_miou = val_metrics["mIoU"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(method.state_dict())
        if config.smoke_test:
            break
        barrier()

    # 训练结束后显式回载最佳权重，确保 test 阶段不是吃最后一个 epoch。
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
    """运行单个实验条件。"""
    train_info = {"best_val_mIoU": 0.0, "best_epoch": 0, "pseudo_accept_count": 0}

    if isinstance(method, QualityFilteredPseudoMaskSelfTrainingSAM2):
        # 伪标签路线分两段：先 warmup teacher adapter，再生成伪标签并微调。
        warmup_info = train_method(method, train_samples, val_samples, config, config.train_epochs)
        local_unlabeled_samples = shard_samples_for_rank(unlabeled_samples, config)
        pseudo_samples = gather_pseudo_samples(method.generate_pseudo_samples(local_unlabeled_samples), config)
        train_info.update(warmup_info)
        train_info["pseudo_accept_count"] = len(pseudo_samples)
        if pseudo_samples:
            # 伪标签和原始 labeled 集合并后再做第二阶段训练。
            finetune_samples = list(train_samples) + pseudo_samples
            finetune_info = train_method(method, finetune_samples, val_samples, config, config.pseudo_finetune_epochs)
            train_info.update(finetune_info)
    else:
        train_info.update(train_method(method, train_samples, val_samples, config, config.train_epochs))

    metrics, report_path = evaluate_method(
        method,
        test_samples,
        config,
        config.output_dir,
        artifact_prefix,
        save_artifacts=True,
    )
    if report_path is not None:
        metrics["EvalReportPath"] = report_path
    return metrics, train_info


def run_experiment(config: Optional[ExperimentConfig] = None) -> None:
    """benchmark 平台总入口。"""
    config = config or load_config()
    # 这里显式根据当前设备类型选择后端，保证单机多卡和 CPU 回退都能工作。
    init_distributed_process_group(
        DistributedConfig(
            enabled=config.distributed,
            rank=config.rank,
            local_rank=config.local_rank,
            world_size=config.world_size,
            backend="nccl" if config.device.type == "cuda" else "gloo",
        )
    )
    try:
        if is_main_process(config):
            config.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        plan_path = config.root / "EXPERIMENT_PLAN.yaml"
        plan = yaml.safe_load(plan_path.read_text(encoding="utf8"))
        # active_conditions 由“用户配置”和“实验计划”求交集，避免跑到未计划条件。
        planned_conditions = plan.get("compute_budget", {}).get("planned_conditions", {})
        all_planned = planned_conditions.get("core_conditions", []) + planned_conditions.get("ablation_conditions", [])
        active_conditions = [condition for condition in config.active_conditions if condition in all_planned]
        skipped_conditions = [condition for condition in all_planned if condition not in active_conditions]

        if not active_conditions:
            raise RuntimeError("No active experiment conditions remain after intersecting config.active_conditions with the plan.")

        dataset_adapter = build_dataset_adapter(config)
        loaded_dataset = dataset_adapter.load(config)
        samples = loaded_dataset.samples
        # teacher 和 registry 都在实验开始时统一创建，避免每个 condition 重复建大对象。
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
                    log_main(f"running condition={condition} seed={seed} budget={budget}", config)
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
                    log_main(
                        f"condition={condition} seed={seed} budget={budget} "
                        f"best_val_mIoU={row['best_val_mIoU']:.4f} "
                        f"pseudo_accept_count={row['pseudo_accept_count']} "
                        f"mIoU={row['mIoU']:.4f} "
                        f"primary_metric={row['primary_metric']:.6f}",
                        config,
                    )
                    barrier()

        if is_main_process(config):
            # results.json 更偏结果表；summary.json 更偏协议和整体说明。
            (config.output_dir / "results.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf8",
            )
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
        barrier()
    finally:
        destroy_distributed_process_group()
