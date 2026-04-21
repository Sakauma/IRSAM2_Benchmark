from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .config import ExperimentConfig
from .dataset_adapters import DatasetManifest
from .metrics import summarize_metric_rows


def _group_metric_rows(metric_rows: List[Dict[str, object]], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in metric_rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {
        group_name: summarize_metric_rows(group_rows)
        for group_name, group_rows in sorted(groups.items(), key=lambda item: item[0])
    }


def build_eval_report(
    aggregate_metrics: Dict[str, float],
    metric_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": "benchmark_eval_v1",
        "aggregate": aggregate_metrics,
        "by_device_source": _group_metric_rows(metric_rows, "device_source"),
        "by_target_scale": _group_metric_rows(metric_rows, "target_scale"),
        "by_category_name": _group_metric_rows(metric_rows, "category_name"),
        "by_annotation_protocol_flag": _group_metric_rows(metric_rows, "annotation_protocol_flag"),
        "samples": metric_rows,
    }


def write_eval_report(
    output_dir: Path,
    prefix: str,
    aggregate_metrics: Dict[str, float],
    metric_rows: List[Dict[str, object]],
) -> str:
    eval_dir = output_dir / "eval_reports"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_file = eval_dir / f"{prefix}_eval.json"
    report = build_eval_report(aggregate_metrics, metric_rows)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    return str(report_file)


def build_summary(
    results: List[Dict[str, float]],
    config: ExperimentConfig,
    plan: Dict,
    active_conditions: List[str],
    skipped_conditions: List[str],
    dataset_manifest: DatasetManifest,
    method_manifest: Sequence[Dict[str, str]],
) -> Dict[str, object]:
    compute_budget = plan.get("compute_budget", {})
    objectives_cfg = plan.get("objectives", {})
    risks_cfg = plan.get("risks", {})
    return {
        "schema_version": "benchmark_summary_v2",
        "experiment_phase": config.experiment_phase,
        "active_conditions": active_conditions,
        "deferred_conditions": config.deferred_conditions,
        "skipped_planned_conditions": skipped_conditions,
        "dataset_manifest": {
            "adapter_name": dataset_manifest.adapter_name,
            "dataset_name": dataset_manifest.dataset_name,
            "data_root": dataset_manifest.data_root,
            "sample_count": dataset_manifest.sample_count,
            "image_count": dataset_manifest.image_count,
            "category_count": dataset_manifest.category_count,
            "device_source_count": dataset_manifest.device_source_count,
            "annotation_protocols": dataset_manifest.annotation_protocols,
            "canonical_bbox": dataset_manifest.canonical_bbox,
            "notes": dataset_manifest.notes,
        },
        "method_manifest": list(method_manifest),
        "segmentation_metrics": [
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "budget": row["budget"],
                "mIoU": row["mIoU"],
                "Dice": row["Dice"],
                "BoundaryF1": row["BoundaryF1"],
                "LatencyMs": row["LatencyMs"],
                "BBoxIoU": row["BBoxIoU"],
                "TightBoxMaskIoU": row.get("TightBoxMaskIoU", 0.0),
                "LooseBoxMaskIoU": row.get("LooseBoxMaskIoU", 0.0),
                "best_val_mIoU": row["best_val_mIoU"],
                "EvalReportPath": row.get("EvalReportPath"),
                "VisualPath": row.get("VisualPath"),
            }
            for row in results
        ],
        "pseudo_label_metrics": [
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "budget": row["budget"],
                "pseudo_accept_count": row.get("pseudo_accept_count", 0),
            }
            for row in results
            if "Pseudo" in row["condition"]
        ],
        "evaluation_protocol": {
            "seeds_per_condition": len(config.seeds),
            "checkpoint_selection": "best validation mIoU checkpoint is loaded before test evaluation",
            "split_strategy": "deterministic device_source split",
            "supervision_protocol": config.supervision_protocol,
        },
        "runtime_strategy": compute_budget.get("runtime_strategy", {}),
        "primary_dataset": config.dataset_name,
        "dataset_root": str(config.data_root),
        "hypothesis_alignment": objectives_cfg.get("hypothesis_alignment", {}),
        "scientific_risks": risks_cfg.get("scientific_risks", []),
    }
