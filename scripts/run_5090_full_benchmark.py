#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irsam2_benchmark.benchmark import full_runner as _full_runner  # noqa: E402
from irsam2_benchmark.benchmark.full_runner import *  # noqa: F403,E402
from irsam2_benchmark.benchmark.full_runner import main  # noqa: E402


def __getattr__(name: str):
    return getattr(_full_runner, name)


if __name__ == "__main__":
    raise SystemExit(main())
