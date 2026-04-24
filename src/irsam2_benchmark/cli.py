from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_app_config
from .pipeline.runner import run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irsam2-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark command.")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    for name in ["transfer", "adapt", "distill", "quantize", "evaluate", "pipeline", "ablation-grid", "baseline"]:
        child = run_sub.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        if name == "baseline":
            child.add_argument(
                "--baseline",
                required=True,
                choices=[
                    "bbox_rect",
                    "sam2_zero_shot",
                    "sam2_zero_shot_point",
                    "sam2_zero_shot_box_point",
                    "sam2_no_prompt_auto_mask",
                    "sam2_video_propagation",
                    "reference_adaptation",
                    "reference_pseudo",
                    "reference_student",
                    "reference_quantized_student",
                ],
            )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_app_config(args.config)
    run_command(config=config, command=args.run_command, baseline_name=getattr(args, "baseline", None))


if __name__ == "__main__":
    main()
