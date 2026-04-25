#!/usr/bin/env python

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_5090_full_benchmark import PROJECT_ROOT, _load_yaml, _write_yaml, main as full_main


def _extract_suite_config(args: List[str]) -> tuple[Path, List[str]]:
    suite_config = PROJECT_ROOT / "configs" / "server_5090_full_benchmark.yaml"
    stripped: List[str] = []
    idx = 0
    while idx < len(args):
        item = args[idx]
        if item == "--suite-config":
            if idx + 1 >= len(args):
                raise ValueError("--suite-config requires a path argument.")
            suite_config = Path(args[idx + 1])
            idx += 2
            continue
        if item.startswith("--suite-config="):
            suite_config = Path(item.split("=", 1)[1])
            idx += 1
            continue
        stripped.append(item)
        idx += 1
    return suite_config, stripped


def _micro_suite_config(source: Path) -> dict:
    config = _load_yaml(source)
    config["artifact_subdir"] = "paper_5090_micro"
    runtime = dict(config.get("runtime", {}))
    runtime.update(
        {
            "smoke_test": False,
            "max_images": 24,
            "max_samples": 0,
            "visual_limit": 24,
        }
    )
    config["runtime"] = runtime
    return config


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    suite_config, passthrough_args = _extract_suite_config(args)
    suite_config = suite_config if suite_config.is_absolute() else PROJECT_ROOT / suite_config
    with tempfile.TemporaryDirectory(prefix="irsam2_5090_micro_") as temp_dir:
        generated_config = Path(temp_dir) / "server_5090_micro.yaml"
        _write_yaml(generated_config, _micro_suite_config(suite_config))
        return full_main(["--suite-config", str(generated_config), *passthrough_args])


if __name__ == "__main__":
    raise SystemExit(main())
