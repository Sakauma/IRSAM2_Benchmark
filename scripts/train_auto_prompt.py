#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from irsam2_benchmark.training import train_auto_prompt_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the learned IR auto prompt v1 proposal model.")
    parser.add_argument("--config", required=True, type=Path, help="Path to auto prompt training YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Parse arguments without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps({"config": str(args.config), "dry_run": True}, ensure_ascii=False, indent=2))
        return 0
    summary = train_auto_prompt_from_config(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
