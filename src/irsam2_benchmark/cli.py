from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import run_analysis
from .baselines import CANONICAL_BASELINE_NAMES
from .config import load_app_config
from .pipeline.runner import run_command
from .validation import preflight_dataset, validate_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    # CLI 只负责把命令和配置路径分发到 pipeline/analysis。
    # 具体实验矩阵、模型、数据集和 seed 应尽量放在 YAML 中维护。
    parser = argparse.ArgumentParser(prog="irsam2-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark command.")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze completed paper benchmark artifacts.")
    analyze.add_argument("--analysis", required=True, type=Path)
    analyze.add_argument("--dry-run", action="store_true")

    validate = sub.add_parser("validate", help="Validate datasets or completed run artifacts.")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    validate_dataset = validate_sub.add_parser("dataset", help="Preflight a dataset config without model inference.")
    validate_dataset.add_argument("--config", required=True, type=Path)
    validate_artifacts = validate_sub.add_parser("artifacts", help="Validate a completed run artifact directory.")
    validate_artifacts.add_argument("--run-dir", required=True, type=Path)

    baseline = run_sub.add_parser("baseline")
    baseline.add_argument("--config", required=True, type=Path)
    baseline.add_argument(
        "--baseline",
        required=True,
        choices=CANONICAL_BASELINE_NAMES,
        metavar="NAME",
        help="Baseline name.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        # 分析命令读取已完成 run 的 artifacts，不会重新触发模型推理。
        run_analysis(args.analysis, dry_run=args.dry_run)
        return
    if args.command == "validate":
        if args.validate_command == "dataset":
            report = preflight_dataset(load_app_config(args.config))
        elif args.validate_command == "artifacts":
            report = validate_run_artifacts(args.run_dir)
        else:  # pragma: no cover - argparse prevents this branch
            raise ValueError(f"Unknown validate command: {args.validate_command}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit(1)
        return
    config = load_app_config(args.config)
    run_command(config=config, command=args.run_command, baseline_name=getattr(args, "baseline", None))


if __name__ == "__main__":
    main()
