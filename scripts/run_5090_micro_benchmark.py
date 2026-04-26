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
from run_5090_full_benchmark import DEFAULT_BENCHMARK_CONFIG, DEFAULT_LEGACY_SUITE_CONFIG


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _extract_config_args(args: List[str]) -> tuple[Path | None, Path | None, List[str]]:
    config: Path | None = None
    suite_config: Path | None = None
    stripped: List[str] = []
    idx = 0
    while idx < len(args):
        item = args[idx]
        if item == "--config":
            if idx + 1 >= len(args):
                raise ValueError("--config requires a path argument.")
            config = Path(args[idx + 1])
            idx += 2
            continue
        if item.startswith("--config="):
            config = Path(item.split("=", 1)[1])
            idx += 1
            continue
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
    return config, suite_config, stripped


def _micro_config(source: Path) -> dict:
    config = _load_yaml(source)
    config["artifact_subdir"] = config.get("micro_artifact_subdir", "paper_5090_micro")
    runtime = dict(config.get("runtime", {}))
    runtime.update(config.get("micro_runtime", {"smoke_test": False, "max_images": 24, "max_samples": 0, "visual_limit": 24}))
    config["runtime"] = runtime
    return config


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    config, suite_config, passthrough_args = _extract_config_args(args)
    config = _resolve_project_path(config) if config is not None else (DEFAULT_BENCHMARK_CONFIG if DEFAULT_BENCHMARK_CONFIG.exists() else None)
    suite_config = _resolve_project_path(suite_config) if suite_config is not None else DEFAULT_LEGACY_SUITE_CONFIG
    with tempfile.TemporaryDirectory(prefix="irsam2_5090_micro_") as temp_dir:
        generated_config = Path(temp_dir) / ("server_benchmark_micro.yaml" if config is not None else "server_5090_micro.yaml")
        _write_yaml(generated_config, _micro_config(config or suite_config))
        config_arg = "--config" if config is not None else "--suite-config"
        return full_main([config_arg, str(generated_config), *passthrough_args])


if __name__ == "__main__":
    raise SystemExit(main())
