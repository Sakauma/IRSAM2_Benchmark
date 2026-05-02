from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .io import write_text


def write_reports(
    output_dir: Path,
    *,
    manifest: Dict[str, Any],
    table_outputs: Dict[str, str],
    case_outputs: Dict[str, str],
    significance_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    outputs = {}
    outputs["analysis_report"] = str(write_text(output_dir / "analysis-report.md", _analysis_report(manifest, table_outputs)))
    outputs["stats_appendix"] = str(write_text(output_dir / "stats-appendix.md", _stats_appendix(manifest, significance_rows)))
    outputs["figure_catalog"] = str(write_text(output_dir / "figure-catalog.md", _figure_catalog(case_outputs)))
    return outputs


def _analysis_report(manifest: Dict[str, Any], table_outputs: Dict[str, str]) -> str:
    lines = [
        "# Analysis Report",
        "",
        "## Scope",
        "",
        f"- Runs found: {manifest.get('run_count', 0)}",
        f"- Missing runs: {len(manifest.get('missing_or_failed_runs', []))}",
        f"- Sample-level rows: {manifest.get('row_count', 0)}",
        "",
        "## Generated Tables",
        "",
    ]
    for name, path in sorted(table_outputs.items()):
        lines.append(f"- `{name}`: `{path}`")
    if manifest.get("missing_or_failed_runs"):
        lines.extend(["", "## Missing Or Failed Runs", ""])
        for item in manifest["missing_or_failed_runs"]:
            lines.append(_missing_or_failed_run_line(item))
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- This report summarizes benchmark artifacts only.",
            "- Significance tests are paired sample-level tests on the current benchmark rows.",
            "- They do not prove seed-level training stability unless multiple independent training runs are later added.",
        ]
    )
    return "\n".join(lines) + "\n"


def _missing_or_failed_run_line(item: Dict[str, Any]) -> str:
    run_id = f"{item.get('experiment_id', '')}/{item.get('dataset', '')}/{item.get('method', '')}"
    if item.get("missing_files"):
        reason = f"missing {item['missing_files']}"
    elif item.get("validation_errors"):
        reason = f"failed validation: {item['validation_errors']}"
    elif item.get("invalid_artifacts"):
        reason = "failed validation"
    else:
        reason = "unavailable"
    run_dir = item.get("run_dir")
    suffix = f" (`{run_dir}`)" if run_dir else ""
    return f"- `{run_id}` {reason}{suffix}"


def _stats_appendix(manifest: Dict[str, Any], significance_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stats Appendix",
        "",
        "## Method",
        "",
        "- Unit: paired sample-level metric rows matched by dataset, method, seed, and sample_id.",
        "- CI: paired bootstrap confidence interval for mean difference.",
        "- Test: Wilcoxon signed-rank normal approximation.",
        "- Multiple comparisons: Holm correction within each metric family.",
        "",
        "## Results",
        "",
    ]
    for row in significance_rows:
        if row.get("status") != "ok":
            lines.append(f"- `{row['dataset']} / {row['comparison']} / {row['metric']}` skipped: {row['status']}.")
            continue
        lines.append(
            "- "
            f"`{row['dataset']} / {row['comparison']} / {row['metric']}`: "
            f"n={row['n_pairs']}, diff={row['mean_diff']:.6f}, "
            f"CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
            f"p={row['wilcoxon_p']:.6g}, p_holm={row['wilcoxon_p_holm']:.6g}, "
            f"effect={row['rank_biserial']:.6f}, low_power={row['low_power']}."
        )
    if not significance_rows:
        lines.append("- No paired comparisons were available.")
    lines.extend(["", "## Analysis Manifest", "", f"- Manifest path: `{manifest.get('manifest_path', '')}`"])
    return "\n".join(lines) + "\n"


def _figure_catalog(case_outputs: Dict[str, str]) -> str:
    lines = [
        "# Figure Catalog",
        "",
        "The analysis copies available per-run visualization PNG files for automatically selected cases.",
        "If no PNG is copied for a case, inspect the corresponding CSV/JSON case list.",
        "",
    ]
    for name, path in sorted(case_outputs.items()):
        if name.endswith("_figure_dir"):
            lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"
