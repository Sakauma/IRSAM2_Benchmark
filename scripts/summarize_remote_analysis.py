#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS = [
    "mIoU",
    "Dice",
    "BoundaryF1Tol1",
    "BBoxIoU",
    "TargetRecallIoU25",
    "FalseAlarmPixelsPerMP",
    "FalseAlarmComponents",
    "LatencyMs",
    "PromptHitRate",
    "PromptDistanceToCentroid",
    "PromptBoxCoverage",
    "PromptPointInBBox",
    "PromptBoxBBoxIoU",
    "TightBoxMaskIoU",
    "LooseBoxMaskIoU",
    "AutoPromptCandidateScore",
    "AutoPromptNumPoints",
    "AutoPromptNegativePointCount",
    "NegativePromptInGtRate",
    "AutoPromptFallback",
]

CHECKPOINT_ORDER = ["tiny", "small", "base_plus", "large"]
DATASET_ORDER = ["nuaa_sirst", "nudt_sirst", "irstd_1k", "multimodal", "rbgt_tiny_ir_box"]
METHOD_LABELS = {
    "sam2_box_oracle": "Box",
    "sam2_point_oracle": "Point",
    "sam2_box_point_oracle": "Box+Point",
    "sam2_tight_box_oracle": "Tight Box",
    "sam2_no_prompt_auto_mask": "No Prompt",
    "sam2_heuristic_auto_point": "Auto Point",
    "sam2_heuristic_auto_box": "Auto Box",
    "sam2_heuristic_auto_box_point": "Auto Box+Point",
    "sam2_heuristic_auto_box_point_neg": "Auto Box+Point+Neg",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize copied remote SAM2 benchmark artifacts.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "artifacts_remote",
        help="Root containing copied remote paper_* artifact folders.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "analysis_remote_20260430",
        help="Root containing per-suite analysis outputs and destination for combined outputs.",
    )
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root.resolve()
    analysis_root = args.analysis_root.resolve()
    tables_dir = analysis_root / "tables"
    figures_dir = analysis_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inventory = collect_summary_inventory(artifact_root)
    if inventory.empty:
        raise RuntimeError(f"No summary.json files found under {artifact_root}")

    main_tables = collect_per_suite_table(analysis_root, "main_baseline_table.csv")
    significance = collect_significance_tables(analysis_root)
    manifests = collect_analysis_manifests(analysis_root)

    write_table_bundle(tables_dir / "remote_run_inventory", inventory)
    write_table_bundle(tables_dir / "key_metric_summary", key_metric_summary(inventory))
    write_table_bundle(tables_dir / "checkpoint_sweep_summary", checkpoint_sweep_summary(inventory))
    write_table_bundle(tables_dir / "per_suite_main_tables_combined", main_tables)
    write_table_bundle(tables_dir / "significance_tests_combined", significance)
    write_table_bundle(tables_dir / "analysis_manifest_summary", manifests)
    write_table_bundle(tables_dir / "best_by_suite_checkpoint_dataset", best_by_suite_checkpoint_dataset(inventory))

    figure_records = generate_figures(inventory, figures_dir)
    write_report(analysis_root / "analysis-report.md", inventory, significance, manifests, figure_records)
    write_stats_appendix(analysis_root / "stats-appendix.md", inventory, significance, manifests)
    write_figure_catalog(analysis_root / "figure-catalog.md", figure_records)

    print(f"Wrote combined tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote analysis-report.md, stats-appendix.md, and figure-catalog.md under {analysis_root}")
    return 0


def collect_summary_inventory(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifact_root.rglob("summary.json")):
        rel = summary_path.relative_to(artifact_root)
        parts = rel.parts
        if len(parts) < 8 or parts[1] != "runs":
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        mean = payload.get("mean", {}) or {}
        std = payload.get("std", {}) or {}
        manifest = payload.get("dataset_manifest", {}) or {}
        runtime = payload.get("runtime_resources", {}) or {}
        row = {
            "artifact_group": parts[0],
            "suite": parts[2],
            "checkpoint": parts[3],
            "experiment_group": parts[4],
            "dataset": parts[5],
            "method": parts[6],
            "summary_path": str(summary_path),
            "baseline_name": payload.get("baseline_name"),
            "adapter_name": manifest.get("adapter_name"),
            "dataset_id": manifest.get("dataset_id"),
            "manifest_sample_count": manifest.get("sample_count"),
            "manifest_image_count": manifest.get("image_count"),
            "manifest_sequence_count": manifest.get("sequence_count"),
            "expected_sample_count": payload.get("expected_sample_count"),
            "expected_eval_units": payload.get("expected_eval_units"),
            "expected_row_count": payload.get("expected_row_count"),
            "row_count": payload.get("row_count"),
            "error_count": payload.get("error_count"),
            "missing_row_count": payload.get("missing_row_count"),
            "failure_rate": payload.get("failure_rate"),
            "wall_time_s": runtime.get("wall_time_s"),
            "samples_per_s": runtime.get("samples_per_s"),
            "rows_per_s": runtime.get("rows_per_s"),
            "cuda_peak_memory_mb": bytes_to_mb(runtime.get("cuda_peak_memory_bytes")),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = mean.get(metric)
            row[f"{metric}_std"] = std.get(metric)
        rows.append(row)
    df = pd.DataFrame(rows)
    return sort_inventory(df)


def collect_per_suite_table(analysis_root: Path, filename: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((analysis_root / "per_suite").rglob(filename)):
        tags = parse_per_suite_path(path, analysis_root)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for key, value in tags.items():
            df.insert(0, key, value)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_significance_tables(analysis_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((analysis_root / "per_suite").rglob("significance_tests.csv")):
        tags = parse_per_suite_path(path, analysis_root)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or len(df.columns) == 0:
            continue
        for key, value in tags.items():
            df.insert(0, key, value)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_analysis_manifests(analysis_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((analysis_root / "per_suite").rglob("analysis_manifest.json")):
        tags = parse_per_suite_path(path, analysis_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = payload.get("missing_or_failed_runs", []) or []
        rows.append(
            {
                **tags,
                "manifest_path": str(path),
                "run_count": payload.get("run_count"),
                "row_count": payload.get("row_count"),
                "missing_or_failed_count": len(missing),
                "dry_run": payload.get("dry_run"),
                "analysis_config": payload.get("analysis_config"),
                "matrix_config": payload.get("matrix_config"),
                "artifact_root": payload.get("artifact_root"),
            }
        )
    return pd.DataFrame(rows)


def parse_per_suite_path(path: Path, analysis_root: Path) -> dict[str, str]:
    rel = path.relative_to(analysis_root / "per_suite")
    return {
        "artifact_group": rel.parts[0],
        "suite": rel.parts[1],
        "checkpoint": rel.parts[2],
    }


def key_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "artifact_group",
        "suite",
        "checkpoint",
        "experiment_group",
        "dataset",
        "method",
        "row_count",
        "expected_row_count",
        "error_count",
        "missing_row_count",
        "failure_rate",
        "mIoU_mean",
        "Dice_mean",
        "BoundaryF1Tol1_mean",
        "BBoxIoU_mean",
        "TargetRecallIoU25_mean",
        "FalseAlarmPixelsPerMP_mean",
        "LatencyMs_mean",
        "PromptHitRate_mean",
        "PromptBoxCoverage_mean",
        "AutoPromptNegativePointCount_mean",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()


def checkpoint_sweep_summary(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[df["suite"].eq("mask")].copy()
    columns = [
        "checkpoint",
        "dataset",
        "method",
        "row_count",
        "mIoU_mean",
        "Dice_mean",
        "BoundaryF1Tol1_mean",
        "TargetRecallIoU25_mean",
        "FalseAlarmPixelsPerMP_mean",
        "LatencyMs_mean",
    ]
    return mask[[col for col in columns if col in mask.columns]].copy()


def best_by_suite_checkpoint_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["suite", "checkpoint", "dataset"], dropna=False):
        metric = "mIoU_mean" if group["mIoU_mean"].notna().any() else "BBoxIoU_mean"
        valid = group[group[metric].notna()]
        if valid.empty:
            continue
        best = valid.sort_values(metric, ascending=False).iloc[0].to_dict()
        rows.append(
            {
                "suite": keys[0],
                "checkpoint": keys[1],
                "dataset": keys[2],
                "selection_metric": metric.removesuffix("_mean"),
                "best_method": best.get("method"),
                "best_value": best.get(metric),
                "row_count": best.get("row_count"),
                "error_count": best.get("error_count"),
            }
        )
    return pd.DataFrame(rows)


def generate_figures(df: pd.DataFrame, figures_dir: Path) -> list[dict[str, str]]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    records = []
    add_record(records, plot_checkpoint_sweep(df, figures_dir))
    add_record(records, plot_base_plus_prompt_heatmap(df, figures_dir))
    add_record(records, plot_auto_prompt(df, figures_dir))
    add_record(records, plot_rbgt(df, figures_dir))
    add_record(records, plot_latency_pareto(df, figures_dir))
    return records


def plot_checkpoint_sweep(df: pd.DataFrame, figures_dir: Path) -> dict[str, str] | None:
    data = df[(df["suite"].eq("mask")) & df["mIoU_mean"].notna()].copy()
    if data.empty:
        return None
    grouped = data.groupby(["checkpoint", "method"], as_index=False)["mIoU_mean"].mean()
    grouped["checkpoint"] = pd.Categorical(grouped["checkpoint"], CHECKPOINT_ORDER, ordered=True)
    grouped = grouped.sort_values("checkpoint")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for method, sub in grouped.groupby("method"):
        sub = sub.sort_values("checkpoint")
        ax.plot(
            sub["checkpoint"].astype(str),
            sub["mIoU_mean"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_ylabel("Mean mIoU across four datasets")
    ax.set_xlabel("SAM2.1 checkpoint")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "figure-01-checkpoint-sweep-miou")
    return {
        "filename": paths["png"],
        "pdf": paths["pdf"],
        "purpose": "Show how raw SAM2.1 checkpoint scale changes oracle-prompt segmentation quality.",
        "data_source": "paper_4090x3_mask_only_v1 / mask suite",
        "key_observation": best_checkpoint_sentence(grouped),
        "caveat": "Values are unweighted means over NUAA-SIRST, NUDT-SIRST, IRSTD-1K, and MultiModal.",
    }


def plot_base_plus_prompt_heatmap(df: pd.DataFrame, figures_dir: Path) -> dict[str, str] | None:
    data = df[(df["suite"].eq("mask")) & df["checkpoint"].eq("base_plus") & df["mIoU_mean"].notna()].copy()
    if data.empty:
        return None
    pivot = data.pivot_table(index="dataset", columns="method", values="mIoU_mean", aggfunc="mean")
    pivot = pivot.reindex([item for item in DATASET_ORDER if item in pivot.index])
    method_order = ["sam2_box_oracle", "sam2_point_oracle", "sam2_box_point_oracle"]
    pivot = pivot[[col for col in method_order if col in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=max(0.75, float(pivot.max().max())))
    ax.set_xticks(range(len(pivot.columns)), [METHOD_LABELS.get(col, col) for col in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            if not pd.isna(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.38 else "black")
    ax.set_title("Base-plus oracle prompt mIoU")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "figure-02-base-plus-prompt-heatmap")
    return {
        "filename": paths["png"],
        "pdf": paths["pdf"],
        "purpose": "Expose dataset-specific prompt-mode behavior for the checkpoint used by auto-prompt experiments.",
        "data_source": "paper_4090x3_mask_only_v1 / mask/base_plus",
        "key_observation": prompt_heatmap_sentence(pivot),
        "caveat": "These are oracle prompts and should not be interpreted as automatic segmentation performance.",
    }


def plot_auto_prompt(df: pd.DataFrame, figures_dir: Path) -> dict[str, str] | None:
    data = df[(df["suite"].eq("heuristic_auto_prompt")) & df["mIoU_mean"].notna()].copy()
    if data.empty:
        return None
    method_order = [
        "sam2_heuristic_auto_point",
        "sam2_heuristic_auto_box",
        "sam2_heuristic_auto_box_point",
        "sam2_heuristic_auto_box_point_neg",
    ]
    data["dataset"] = pd.Categorical(data["dataset"], [item for item in DATASET_ORDER if item in set(data["dataset"])], ordered=True)
    data["method"] = pd.Categorical(data["method"], [item for item in method_order if item in set(data["method"])], ordered=True)
    data = data.sort_values(["dataset", "method"])

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    datasets = list(data["dataset"].cat.categories)
    width = 0.18
    offsets = [(-1.5 + idx) * width for idx in range(len(method_order))]
    for idx, method in enumerate(method_order):
        sub = data[data["method"].eq(method)].set_index("dataset")
        values = [sub.loc[dataset, "mIoU_mean"] if dataset in sub.index else math.nan for dataset in datasets]
        ax.bar([x + offsets[idx] for x in range(len(datasets))], values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_xticks(range(len(datasets)), datasets, rotation=20, ha="right")
    ax.set_ylabel("mIoU")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "figure-03-heuristic-auto-prompt-miou")
    return {
        "filename": paths["png"],
        "pdf": paths["pdf"],
        "purpose": "Compare heuristic automatic prompt variants before any SAM2-IR adaptation.",
        "data_source": "paper_5090_auto_prompt_v1 / heuristic_auto_prompt/base_plus",
        "key_observation": auto_prompt_sentence(data),
        "caveat": "MultiModal auto-prompt has 239 failed rows per method; all other datasets have zero errors.",
    }


def plot_rbgt(df: pd.DataFrame, figures_dir: Path) -> dict[str, str] | None:
    data = df[(df["suite"].eq("rbgt_box")) & df["BBoxIoU_mean"].notna()].copy()
    if data.empty:
        return None
    method_order = ["sam2_box_oracle", "sam2_point_oracle", "sam2_box_point_oracle"]
    checkpoints = [item for item in CHECKPOINT_ORDER if item in set(data["checkpoint"])]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    width = 0.22
    offsets = [(-1 + idx) * width for idx in range(len(method_order))]
    for idx, method in enumerate(method_order):
        sub = data[data["method"].eq(method)].set_index("checkpoint")
        values = [sub.loc[ckpt, "BBoxIoU_mean"] if ckpt in sub.index else math.nan for ckpt in checkpoints]
        ax.bar([x + offsets[idx] for x in range(len(checkpoints))], values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_xticks(range(len(checkpoints)), checkpoints)
    ax.set_ylabel("BBox IoU")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "figure-04-rbgt-box-bboxiou")
    return {
        "filename": paths["png"],
        "pdf": paths["pdf"],
        "purpose": "Use RBGT-Tiny box-only annotations to probe prompt-conditioned mask tightness at scale.",
        "data_source": "paper_4090x3_rbgt_probe_v1 / rbgt_box",
        "key_observation": rbgt_sentence(data),
        "caveat": "RBGT-Tiny has box annotations only; BBoxIoU is a proxy and not mask mIoU.",
    }


def plot_latency_pareto(df: pd.DataFrame, figures_dir: Path) -> dict[str, str] | None:
    data = df[(df["suite"].eq("mask")) & df["mIoU_mean"].notna() & df["LatencyMs_mean"].notna()].copy()
    if data.empty:
        return None
    grouped = data.groupby(["checkpoint", "method"], as_index=False)[["mIoU_mean", "LatencyMs_mean"]].mean()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for method, sub in grouped.groupby("method"):
        ax.scatter(sub["LatencyMs_mean"], sub["mIoU_mean"], s=70, label=METHOD_LABELS.get(method, method))
        for _, row in sub.iterrows():
            ax.text(row["LatencyMs_mean"], row["mIoU_mean"], f" {row['checkpoint']}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Latency ms, log scale")
    ax.set_ylabel("Mean mIoU across four datasets")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "figure-05-latency-miou-pareto")
    return {
        "filename": paths["png"],
        "pdf": paths["pdf"],
        "purpose": "Show the raw SAM2 tradeoff between oracle-prompt quality and measured inference latency.",
        "data_source": "paper_4090x3_mask_only_v1 / mask suite",
        "key_observation": latency_sentence(grouped),
        "caveat": "Latency is taken from remote benchmark summaries and is not a controlled microbenchmark.",
    }


def write_report(path: Path, df: pd.DataFrame, sig: pd.DataFrame, manifests: pd.DataFrame, figures: list[dict[str, str]]) -> None:
    total_runs = len(df)
    total_rows = int(pd.to_numeric(df["row_count"], errors="coerce").fillna(0).sum())
    total_errors = int(pd.to_numeric(df["error_count"], errors="coerce").fillna(0).sum())
    manifest_missing = int(pd.to_numeric(manifests.get("missing_or_failed_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    mask_best = best_checkpoint_sentence(df[(df["suite"].eq("mask"))].groupby(["checkpoint", "method"], as_index=False)["mIoU_mean"].mean())
    auto_sentence_text = auto_prompt_sentence(df[df["suite"].eq("heuristic_auto_prompt")])
    rbgt_sentence_text = rbgt_sentence(df[df["suite"].eq("rbgt_box")])
    protocol_sentence_text = protocol_sentence(df[df["suite"].eq("prompt_box_protocol")])
    sig_count = int((sig["status"].eq("ok")).sum()) if not sig.empty and "status" in sig else 0

    lines = [
        "# Remote SAM2 Baseline Analysis Bundle",
        "",
        "## Scope",
        "",
        f"- Summarized {total_runs} completed remote runs from `artifacts_remote`.",
        f"- Aggregated {total_rows:,} evaluated rows with {total_errors:,} recorded errors.",
        f"- Re-ran local per-suite analysis for 9 configs; manifest missing/failed count is {manifest_missing}.",
        f"- Combined significance table contains {sig_count} completed paired tests.",
        "",
        "## Key Findings",
        "",
        f"- Oracle prompted SAM2 is useful but prompt-mode dependent: {mask_best}",
        f"- Heuristic auto-prompt is the main weakness before adaptation: {auto_sentence_text}",
        f"- Box protocol matters: {protocol_sentence_text}",
        f"- RBGT-Tiny can support box-prompt pseudo-mask probing: {rbgt_sentence_text}",
        "- MultiModal is the strongest diagnostic split for automatic prompting because it exposes large domain variation and the current heuristic prompt failures.",
        "",
        "## Immediate Research Implications",
        "",
        "- The teacher should not be framed as generic SAM2 fine-tuning. The evidence points to prompt discovery and prompt-response preservation as the publishable gap.",
        "- A strong method should improve automatic prompt hit rate and false-alarm control while preserving the oracle-prompt response on small targets.",
        "- RBGT-Tiny should be used as a large-scale weakly supervised source for prompt-conditioned pseudo masks, not as a direct mask-mIoU benchmark.",
        "- Quantization/distillation should report prompt-response fidelity metrics, not only final mIoU.",
        "",
        "## Primary Tables",
        "",
        "- `tables/remote_run_inventory.csv`: one row per completed remote run.",
        "- `tables/key_metric_summary.csv`: compact metrics for paper discussion.",
        "- `tables/checkpoint_sweep_summary.csv`: raw SAM2 checkpoint sweep on the four mask datasets.",
        "- `tables/significance_tests_combined.csv`: paired tests generated by the benchmark analysis code.",
        "- `tables/best_by_suite_checkpoint_dataset.csv`: best method per suite/checkpoint/dataset.",
        "",
        "## Figures",
        "",
    ]
    for idx, record in enumerate(figures, start=1):
        lines.append(f"- Figure {idx}: `{Path(record['filename']).name}`. {record['purpose']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The current runs are single-seed model evaluations. Paired tests use sample-level pairing, so they support within-dataset response comparisons but do not estimate training-seed variability.",
            "- RBGT-Tiny has box annotations only. The analysis reports BBoxIoU and descriptive summaries, not mask mIoU or Dice.",
            "- MultiModal heuristic auto-prompt has 239 failed rows per method. Treat its automatic prompt numbers as a failure-mode signal until those cases are inspected.",
            "- Latency numbers come from full benchmark runs and should be used for relative diagnosis, not final deployment claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stats_appendix(path: Path, df: pd.DataFrame, sig: pd.DataFrame, manifests: pd.DataFrame) -> None:
    suites = df.groupby(["artifact_group", "suite", "checkpoint"], dropna=False).agg(
        run_count=("method", "count"),
        row_count=("row_count", "sum"),
        error_count=("error_count", "sum"),
    )
    suite_lines = dataframe_to_markdown(suites.reset_index())
    sig_summary = pd.DataFrame()
    if not sig.empty and "status" in sig:
        sig_summary = sig.groupby(["artifact_group", "suite", "checkpoint", "status"], dropna=False).size().reset_index(name="count")

    lines = [
        "# Statistics Appendix",
        "",
        "## Unit of Analysis",
        "",
        "- Descriptive tables use evaluated rows emitted by each run.",
        "- Paired tests match rows by `seed` and `sample_id` within each dataset.",
        "- The benchmark currently has one model-evaluation seed. Therefore, significance tests are sample-level tests, not seed-level generalization tests.",
        "",
        "## Suite Inventory",
        "",
        suite_lines,
        "",
        "## Inferential Tests",
        "",
        "- For non-RBGT suites, the existing analysis runner used paired mean differences, bootstrap 95% confidence intervals, Wilcoxon signed-rank tests, rank-biserial effect sizes, and Holm correction within each metric.",
        "- RBGT paired tests were intentionally disabled in `configs/rbgt_box/*.yaml` because each checkpoint has 1,937,691 rows and box-only labels. The current RBGT result is descriptive.",
        "- Test outputs are in `tables/significance_tests_combined.csv`.",
        "",
        "## Test Row Counts",
        "",
        dataframe_to_markdown(sig_summary) if not sig_summary.empty else "No completed significance rows were found.",
        "",
        "## Missing and Failed Rows",
        "",
        failure_markdown(df),
        "",
        "## Limits",
        "",
        "- No training curves are present in these remote artifacts.",
        "- No repeated training seeds are present.",
        "- The current analysis cannot claim that one trained model distribution dominates another. It can claim paired response differences for the fixed evaluated checkpoints and prompts.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure_catalog(path: Path, records: list[dict[str, str]]) -> None:
    lines = ["# Figure Catalog", ""]
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## Figure {idx}",
                "",
                f"- File: `{Path(record['filename']).name}`",
                f"- PDF: `{Path(record['pdf']).name}`",
                f"- Purpose: {record['purpose']}",
                f"- Data source: {record['data_source']}",
                f"- Key observation: {record['key_observation']}",
                f"- Caption should state: metric, prompt protocol, checkpoint scope, and dataset scope.",
                f"- Interpretation check: {record['key_observation']}",
                f"- Caveat: {record['caveat']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_bundle(base: Path, df: pd.DataFrame) -> None:
    if df.empty:
        df.to_csv(base.with_suffix(".csv"), index=False)
        base.with_suffix(".json").write_text("[]\n", encoding="utf-8")
        return
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


def save_figure(fig: plt.Figure, base: Path) -> dict[str, str]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def add_record(records: list[dict[str, str]], record: dict[str, str] | None) -> None:
    if record is not None:
        records.append(record)


def sort_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["checkpoint_order"] = df["checkpoint"].map({name: idx for idx, name in enumerate(CHECKPOINT_ORDER)}).fillna(99)
    df["dataset_order"] = df["dataset"].map({name: idx for idx, name in enumerate(DATASET_ORDER)}).fillna(99)
    df = df.sort_values(["artifact_group", "suite", "checkpoint_order", "dataset_order", "method"]).drop(columns=["checkpoint_order", "dataset_order"])
    return df.reset_index(drop=True)


def bytes_to_mb(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def best_checkpoint_sentence(grouped: pd.DataFrame) -> str:
    if grouped.empty or "mIoU_mean" not in grouped:
        return "no mIoU rows were available."
    best = grouped.sort_values("mIoU_mean", ascending=False).iloc[0]
    label = METHOD_LABELS.get(str(best.get("method")), str(best.get("method")))
    return f"best unweighted mean mIoU is {best['mIoU_mean']:.3f} with checkpoint `{best['checkpoint']}` and method `{label}`."


def prompt_heatmap_sentence(pivot: pd.DataFrame) -> str:
    if pivot.empty:
        return "no base-plus prompt heatmap data were available."
    stacked = pivot.stack().sort_values(ascending=False)
    dataset, method = stacked.index[0]
    return f"highest base-plus mIoU is {stacked.iloc[0]:.3f} on `{dataset}` with `{METHOD_LABELS.get(method, method)}`."


def auto_prompt_sentence(df: pd.DataFrame) -> str:
    data = df[df["mIoU_mean"].notna()].copy() if "mIoU_mean" in df else pd.DataFrame()
    if data.empty:
        return "no heuristic auto-prompt mIoU rows were available."
    avg = data.groupby("method", as_index=False, observed=False)["mIoU_mean"].mean().sort_values("mIoU_mean", ascending=False)
    best = avg.iloc[0]
    return f"best average heuristic auto-prompt mIoU is {best['mIoU_mean']:.3f} from `{METHOD_LABELS.get(best['method'], best['method'])}`."


def rbgt_sentence(df: pd.DataFrame) -> str:
    data = df[df["BBoxIoU_mean"].notna()].copy() if "BBoxIoU_mean" in df else pd.DataFrame()
    if data.empty:
        return "no RBGT BBoxIoU rows were available."
    best = data.sort_values("BBoxIoU_mean", ascending=False).iloc[0]
    return f"best RBGT BBoxIoU is {best['BBoxIoU_mean']:.3f} with checkpoint `{best['checkpoint']}` and method `{METHOD_LABELS.get(best['method'], best['method'])}`."


def protocol_sentence(df: pd.DataFrame) -> str:
    data = df[df["mIoU_mean"].notna()].copy() if "mIoU_mean" in df else pd.DataFrame()
    if data.empty:
        return "no prompt-box protocol rows were available."
    avg = data.groupby("method", as_index=False)["mIoU_mean"].mean().sort_values("mIoU_mean", ascending=False)
    best = avg.iloc[0]
    return f"`{METHOD_LABELS.get(best['method'], best['method'])}` has the best average protocol mIoU at {best['mIoU_mean']:.3f}."


def latency_sentence(df: pd.DataFrame) -> str:
    if df.empty:
        return "no latency rows were available."
    best = df.sort_values(["mIoU_mean", "LatencyMs_mean"], ascending=[False, True]).iloc[0]
    return f"highest average mIoU point is `{best['checkpoint']}` / `{METHOD_LABELS.get(best['method'], best['method'])}` at {best['mIoU_mean']:.3f} mIoU and {best['LatencyMs_mean']:.1f} ms."


def failure_markdown(df: pd.DataFrame) -> str:
    failures = df[(pd.to_numeric(df["error_count"], errors="coerce").fillna(0) > 0) | (pd.to_numeric(df["missing_row_count"], errors="coerce").fillna(0) > 0)]
    cols = ["artifact_group", "suite", "checkpoint", "dataset", "method", "row_count", "expected_row_count", "error_count", "missing_row_count", "failure_rate"]
    if failures.empty:
        return "No failed or missing rows were recorded in summary files."
    return dataframe_to_markdown(failures[[col for col in cols if col in failures.columns]])


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "No rows."
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return view.to_markdown(index=False)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
