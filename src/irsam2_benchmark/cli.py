"""命令行入口定义。

Author: Egor Izmaylov

这里把 benchmark 的所有公开命令统一收口到一套 argparse CLI 中。
这样做的好处是：
- 用户入口稳定；
- 脚本、服务器和 CI 都可以走同一条调用链；
- 不同 stage/baseline 的参数约束可以集中维护。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_app_config
from .pipeline.runner import run_command


def build_parser() -> argparse.ArgumentParser:
    """构造项目统一 CLI 解析器。"""
    parser = argparse.ArgumentParser(prog="irsam2-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行 benchmark 的某个命令。")
    run_sub = run.add_subparsers(dest="run_command", required=True)

    # 所有公开命令都集中在这里，方便后续扩展和文档冻结。
    for name in ["transfer", "adapt", "distill", "quantize", "evaluate", "pipeline", "ablation-grid", "baseline"]:
        child = run_sub.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        if name == "baseline":
            # baseline 名称在 CLI 层冻结，避免运行时传入未注册名称。
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
    """CLI 主入口。

    流程很简单：
    1. 解析参数；
    2. 加载结构化配置；
    3. 交给 pipeline runner 统一执行。
    """
    parser = build_parser()
    args = parser.parse_args()
    config = load_app_config(args.config)
    run_command(config=config, command=args.run_command, baseline_name=getattr(args, "baseline", None))


if __name__ == "__main__":
    main()
