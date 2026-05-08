#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irsam2_benchmark.benchmark.auto_prompt_runner import main as auto_prompt_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(auto_prompt_main())
