from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .cases import select_cases, write_case_outputs
from .collector import collect_runs, flatten_rows
from .io import output_pair, read_yaml, write_json
from .reports import write_reports
from .stats import run_paired_tests
from .tables import bucket_table, main_baseline_table, multimodal_size_table


def _analysis_root(analysis_path: Path) -> Path:
    # 仓库内 configs/*.yaml 以项目根作为相对路径基准；generated analysis config 以自身目录为基准。
    return analysis_path.parent.parent if analysis_path.parent.name == "configs" else analysis_path.parent


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def run_analysis(analysis_path: str | Path, *, dry_run: bool = False) -> Dict[str, Any]:
    # 分析阶段只消费 artifacts，不重新加载模型；适合在服务器跑完后单独反复生成表格。
    analysis_path = Path(analysis_path).resolve()
    root = _analysis_root(analysis_path)
    analysis_config = read_yaml(analysis_path)
    matrix_path = _resolve_path(root, analysis_config.get("matrix", "configs/paper_experiments_v1.yaml"))
    matrix = read_yaml(matrix_path)
    artifact_root = _resolve_path(root, analysis_config.get("artifact_root", "artifacts/paper_v1"))
    output_dir = _resolve_path(root, analysis_config.get("output_dir", "artifacts/paper_v1/analysis"))

    runs, missing = collect_runs(artifact_root, matrix, analysis_config)
    rows = flatten_rows(runs)
    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_config": str(analysis_path),
        "matrix_config": str(matrix_path),
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(root),
        "run_count": len(runs),
        "row_count": len(rows),
        "missing_or_failed_runs": missing,
        "dry_run": dry_run,
    }

    if dry_run:
        # dry-run 用于检查路径和矩阵展开，不创建分析表。
        print(f"[dry-run] analysis_config={analysis_path}")
        print(f"[dry-run] matrix_config={matrix_path}")
        print(f"[dry-run] artifact_root={artifact_root}")
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] found_runs={len(runs)} missing_runs={len(missing)} rows={len(rows)}")
        return manifest

    metrics = list(analysis_config.get("metrics", []))
    primary_metric = str(analysis_config.get("primary_metric", "mIoU"))
    case_config = analysis_config.get("case_selection", {})
    top_k = int(case_config.get("top_k", 8))

    tables_dir = output_dir / "tables"
    table_outputs: Dict[str, str] = {}
    # 这些表分别对应主表、MultiModal 大小目标分表、自动 prompt/no-prompt 表、消融表和面积桶表。
    table_outputs.update(output_pair(tables_dir, "main_baseline_table", main_baseline_table(rows, metrics)))
    table_outputs.update(output_pair(tables_dir, "multimodal_size_table", multimodal_size_table(rows, metrics)))
    table_outputs.update(output_pair(tables_dir, "auto_prompt_table", _filter_methods(rows, {"sam2_no_prompt_auto_mask", "sam2_physics_auto_prompt"}, metrics)))
    table_outputs.update(output_pair(tables_dir, "ablation_table", _filter_prefix(rows, "physics_", metrics)))
    table_outputs.update(output_pair(tables_dir, "bucket_table", bucket_table(rows, metrics)))

    significance_rows = run_paired_tests(rows, analysis_config)
    table_outputs.update(output_pair(tables_dir, "significance_tests", significance_rows))

    selected_cases = select_cases(rows, primary_metric=primary_metric, top_k=top_k)
    # case selection 输出最好/最差样本索引，供后续手动回看可视化。
    case_outputs = write_case_outputs(output_dir, selected_cases)

    manifest["outputs"] = {**table_outputs, **case_outputs}
    manifest_path = write_json(output_dir / "analysis_manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    write_json(output_dir / "analysis_manifest.json", manifest)
    report_outputs = write_reports(output_dir, manifest=manifest, table_outputs=table_outputs, case_outputs=case_outputs, significance_rows=significance_rows)
    manifest["outputs"].update(report_outputs)
    write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def _filter_methods(rows: list[Dict[str, Any]], methods: set[str], metrics: list[str]) -> list[Dict[str, Any]]:
    from .tables import summarize_by

    return summarize_by([row for row in rows if row.get("method") in methods], ["dataset", "method", "eval_unit"], metrics)


def _filter_prefix(rows: list[Dict[str, Any]], prefix: str, metrics: list[str]) -> list[Dict[str, Any]]:
    from .tables import summarize_by

    return summarize_by(
        [row for row in rows if str(row.get("method", "")).startswith(prefix) or row.get("method") == "sam2_physics_auto_prompt"],
        ["dataset", "method", "eval_unit"],
        metrics,
    )
