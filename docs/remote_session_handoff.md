# Remote Session Handoff

Author: Egor Izmaylov

## Purpose

This file is a handoff note for a fresh Codex session opened inside a remote VS Code window.
If the remote Codex session does not share the local conversation history, read this file first
and continue from the state recorded below.

## Repository State

- Repository root: `/root/autodl-tmp/IRSAM2_Benchmark` on the rented server
- Latest expected commit: `2f05405 feat: add autodl bootstrap scripts`
- Old repo/workflow is historical only; the active benchmark code is the new package-based platform
- Current benchmark repo on the local machine lives at:
  - `D:/workspace/sam_ir/ir_sam2_bench`

## What Has Been Implemented

The benchmark platform itself is already usable.

Implemented:

- package-based benchmark structure
- dataset adapters
- generic `images/ + masks/` support
- prompt synthesis from mask-only datasets
- baseline registry
- `bbox_rect`
- `sam2_zero_shot`
- `sam2_zero_shot_point`
- `sam2_zero_shot_box_point`
- `sam2_no_prompt_auto_mask`
- `sam2_video_propagation` interface
- frozen artifact/report schema
- Linux helper scripts
- AutoDL bootstrap scripts

Not yet fully implemented as paper methods:

- `adapt`
- `distill`
- `quantize`

These three currently exist as stable stage scaffolds, not final training methods.

## AutoDL Server Assumptions

The rented server uses:

- `/root/autodl-tmp` as the fast data/runtime disk
- `/root/autodl-fs` as the persistent archive/storage disk

The user said the uploaded dataset archives are:

- `MultiModalCOCOClean.zip`
- `RBGT-Tiny.tar.gz`

They are expected under:

- `/root/autodl-fs/MultiModalCOCOClean.zip`
- `/root/autodl-fs/RBGT-Tiny.tar.gz`

The user already has:

- `/root/sam2`
- `/root/sam-hq`
- `/root/miniconda3`

## Recommended Server Layout

Use these runtime locations:

- repo: `/root/autodl-tmp/IRSAM2_Benchmark`
- datasets: `/root/autodl-tmp/datasets`
- outputs: `/root/autodl-tmp/runs`
- checkpoints: `/root/autodl-tmp/checkpoints`

Do not use `/` as the main runtime/output location because the system disk is small.

## New Helper Scripts Added For AutoDL

- `scripts/setup_autodl_server.sh`
- `scripts/run_autodl_smoke.sh`
- `configs/benchmark_smoke_rbgt_tiny.yaml`
- `configs/benchmark_v1_rbgt_tiny.yaml`

Also updated:

- `scripts/run_baseline.sh`
- `scripts/run_tests.sh`
- `README.md`

## Expected First Commands On The Server

If the repo is not cloned yet:

```bash
cd /root/autodl-tmp
git clone git@github.com:Sakauma/IRSAM2_Benchmark.git
cd IRSAM2_Benchmark
```

If it is already cloned:

```bash
cd /root/autodl-tmp/IRSAM2_Benchmark
git pull
```

Then initialize the server workspace:

```bash
bash scripts/setup_autodl_server.sh
source .autodl_env.sh
```

Then run smoke baselines:

```bash
bash scripts/run_autodl_smoke.sh multimodal bbox_rect
bash scripts/run_autodl_smoke.sh multimodal sam2_zero_shot
bash scripts/run_autodl_smoke.sh rbgt sam2_zero_shot
```

If a specific Python interpreter is needed:

```bash
export PYTHON_BIN=/root/miniconda3/bin/python
```

or use the target environment's `python` path instead.

## What To Do Next

After the server bootstrap is working, continue in this order:

1. verify `nvidia-smi`
2. verify Python environment imports
3. run the smoke baselines above
4. check generated `artifacts/` and `reference_results/`
5. only then move to formal benchmark runs

## Important Context For The Next Codex Session

- The user prefers a complete paper-grade benchmark platform
- SAM2 baseline support is mandatory
- generic mask-only datasets are mandatory
- image and sequence/video evaluation are both required
- `adapt / distill / quantize` are the next real implementation targets
- right now the immediate practical task is getting the server environment ready and running smoke tests

## Instruction To The Next Session

If you are the new remote Codex session, start by:

1. reading this file
2. checking whether the repo is at commit `2f05405`
3. checking whether `/root/autodl-fs/MultiModalCOCOClean.zip` and `/root/autodl-fs/RBGT-Tiny.tar.gz` exist
4. running the AutoDL bootstrap flow
5. reporting any server-side errors precisely
