#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irsam2_benchmark.analysis import run_analysis  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze completed IR-only paper benchmark artifacts.")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_analysis(args.analysis, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
