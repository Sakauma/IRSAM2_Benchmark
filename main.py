from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from irsam2_benchmark.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
