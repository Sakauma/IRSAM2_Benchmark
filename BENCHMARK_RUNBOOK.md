# Benchmark Runbook

## 1. Environment
Recommended runtime:

```bash
source /path/to/conda.sh
conda run --no-capture-output -p /path/to/env python main.py
```

## 2. Required Variables
At minimum set:

```bash
DATASET_ROOT=/path/to/datasets
SAM2_REPO=/path/to/sam2
```

Common optional variables:

```bash
DATASET_NAME=MultiModal
PYTHON_BIN=/path/to/your_env/bin/python
OUTPUT_DIR=/path/to/outputs/run_v1
EXPERIMENT_SEEDS=42
SUPERVISION_BUDGETS=0.1
TRAIN_EPOCHS=2
PSEUDO_FINETUNE_EPOCHS=1
MAX_SAMPLES=0
MAX_IMAGES=0
EVAL_LIMIT=0
NUM_WORKERS=0
```

`RBGT-Tiny` defaults to the IR-only `01` branch. On large multi-object datasets prefer `MAX_IMAGES` over `MAX_SAMPLES`.

## 3. Start Commands
MultiModal:

```bash
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=MultiModal
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/multimodal_benchmark_v1

bash scripts/run_full_benchmark.sh
```

RBGT-Tiny:

```bash
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=RBGT-Tiny
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/rbgt_tiny_benchmark_v1
export MAX_IMAGES=512

bash scripts/run_full_benchmark.sh
```

## 4. Background Run Templates
`tmux`:

```bash
tmux new -s irbench
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=MultiModal
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/multimodal_benchmark_v1

bash scripts/run_full_benchmark.sh
```

Common `tmux` commands:

```bash
tmux attach -t irbench
tmux detach
tmux ls
```

`nohup`:

```bash
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=RBGT-Tiny
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/rbgt_tiny_benchmark_v1
export MAX_IMAGES=512

nohup bash scripts/run_full_benchmark.sh > run.log 2>&1 &
echo $!
```

## 5. Recommended Run Profiles
Choose one of these before the first formal run.

### Light
Use this on a single mid-range GPU when you want the run to finish quickly and only need a first pass.

MultiModal:
```bash
export EXPERIMENT_SEEDS=42
export SUPERVISION_BUDGETS=0.1
export TRAIN_EPOCHS=2
export PSEUDO_FINETUNE_EPOCHS=1
export MAX_SAMPLES=0
export MAX_IMAGES=0
export EVAL_LIMIT=0
export NUM_WORKERS=0
```

RBGT-Tiny:
```bash
export EXPERIMENT_SEEDS=42
export SUPERVISION_BUDGETS=0.1
export TRAIN_EPOCHS=2
export PSEUDO_FINETUNE_EPOCHS=1
export MAX_SAMPLES=0
export MAX_IMAGES=512
export EVAL_LIMIT=0
export NUM_WORKERS=0
```

### Standard
Use this when you want a benchmark run that is still practical on a normal server and is suitable for model selection.

MultiModal:
```bash
export EXPERIMENT_SEEDS=42,123
export SUPERVISION_BUDGETS=0.1,0.2
export TRAIN_EPOCHS=4
export PSEUDO_FINETUNE_EPOCHS=2
export MAX_SAMPLES=0
export MAX_IMAGES=0
export EVAL_LIMIT=0
export NUM_WORKERS=2
```

RBGT-Tiny:
```bash
export EXPERIMENT_SEEDS=42,123
export SUPERVISION_BUDGETS=0.1,0.2
export TRAIN_EPOCHS=4
export PSEUDO_FINETUNE_EPOCHS=2
export MAX_SAMPLES=0
export MAX_IMAGES=1024
export EVAL_LIMIT=0
export NUM_WORKERS=2
```

### Heavy
Use this only on a stronger GPU server when you want the closest thing to a full benchmark pass.

MultiModal:
```bash
export EXPERIMENT_SEEDS=42,123,456
export SUPERVISION_BUDGETS=0.1,0.2,0.5
export TRAIN_EPOCHS=6
export PSEUDO_FINETUNE_EPOCHS=4
export MAX_SAMPLES=0
export MAX_IMAGES=0
export EVAL_LIMIT=0
export NUM_WORKERS=4
```

RBGT-Tiny:
```bash
export EXPERIMENT_SEEDS=42,123,456
export SUPERVISION_BUDGETS=0.1,0.2,0.5
export TRAIN_EPOCHS=6
export PSEUDO_FINETUNE_EPOCHS=4
export MAX_SAMPLES=0
export MAX_IMAGES=2048
export EVAL_LIMIT=0
export NUM_WORKERS=4
```

Profile selection rules:
- If the first goal is only to validate plumbing and relative ranking, start with `Light`.
- If the first goal is to compare adaptation and pseudo conditions with some stability, use `Standard`.
- Only use `Heavy` after `Light` or `Standard` has already shown that the platform, data path, and SAM2 path are all stable.
- On multi-object datasets, prefer capping `MAX_IMAGES` instead of `MAX_SAMPLES`.

## 6. Preflight Checks
Syntax check:

```bash
python -m py_compile main.py experiment_core/*.py scripts/*.py
```

Smoke run:

```bash
SMOKE_TEST=1 bash scripts/run_full_benchmark.sh
unset SMOKE_TEST
```

## 7. Live Monitoring
Process check:

```bash
ps -ef | grep ir_sam2_bench | grep -v grep
```

Log tail:

```bash
tail -f run.log
```

Output growth:

```bash
watch -n 30 'ls -lh /path/to/outputs/run_v1'
```

## 8. Result Checks
Inspect `summary.json`:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('/path/to/outputs/summary.json')
data = json.loads(path.read_text(encoding='utf-8'))
print('dataset_manifest =', data.get('dataset_manifest'))
print('method_count =', len(data.get('method_manifest', [])))
print('active_conditions =', data.get('active_conditions'))
PY
```

Inspect `results.json`:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('/path/to/outputs/results.json')
rows = json.loads(path.read_text(encoding='utf-8'))
for row in rows:
    print(row['condition'], 'mIoU=', row.get('mIoU'), 'best_val_mIoU=', row.get('best_val_mIoU'))
PY
```

Count eval reports:

```bash
find /path/to/outputs/eval_reports -name '*_eval.json' | wc -l
```

Quick comparison:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('/path/to/outputs/results.json').read_text(encoding='utf-8'))
rows = {row['condition']: row for row in rows}
for key in [
    'BBoxRectMaskBaseline',
    'ZeroShotSAM2BoxPromptIR',
    'CleanBoxPEFTSAM2Adapter',
    'NoisyBoxPromptRobustSAM2Adapter',
    'QualityFilteredPseudoMaskSelfTrainingSAM2',
]:
    if key in rows:
        print(key, rows[key].get('mIoU'))
PY
```

## 9. Visualization
Render MultiModalCOCO tight/loose box visualizations:

```bash
python scripts/visualize_coco_boxes.py \
  --src-root /path/to/MultiModalCOCO \
  --output-root ./visualizations/multimodal_coco_boxes_v1
```

Outputs:
- `tight/`
- `loose/`
- `comparison/`

## 10. First Server Run Recommendation
If this is the first formal run on a new server, use one fixed `Standard` profile first instead of choosing among multiple profiles.

Recommended reasons:
- It is strong enough to test relative ranking between zero-shot, adaptation, and pseudo conditions.
- It is still bounded enough that failures are easier to debug than a full heavy run.
- It avoids wasting long GPU time before confirming that the dataset path, SAM2 path, and report outputs are all correct.

Recommended MultiModal command:

```bash
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=MultiModal
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/multimodal_server_first_run_v1

export EXPERIMENT_SEEDS=42,123
export SUPERVISION_BUDGETS=0.1,0.2
export TRAIN_EPOCHS=4
export PSEUDO_FINETUNE_EPOCHS=2
export MAX_SAMPLES=0
export MAX_IMAGES=0
export EVAL_LIMIT=0
export NUM_WORKERS=2

python -m py_compile main.py experiment_core/*.py scripts/*.py
SMOKE_TEST=1 bash scripts/run_full_benchmark.sh
unset SMOKE_TEST
bash scripts/run_full_benchmark.sh
```

Equivalent helper script:

```bash
bash scripts/run_multimodal_server_first.sh
```

Recommended RBGT-Tiny command:

```bash
cd /path/to/ir_sam2_bench

export DATASET_ROOT=/path/to/datasets
export DATASET_NAME=RBGT-Tiny
export SAM2_REPO=/path/to/sam2
export PYTHON_BIN=/path/to/your_env/bin/python
export OUTPUT_DIR=/path/to/outputs/rbgt_tiny_server_first_run_v1

export EXPERIMENT_SEEDS=42,123
export SUPERVISION_BUDGETS=0.1,0.2
export TRAIN_EPOCHS=4
export PSEUDO_FINETUNE_EPOCHS=2
export MAX_SAMPLES=0
export MAX_IMAGES=1024
export EVAL_LIMIT=0
export NUM_WORKERS=2

python -m py_compile main.py experiment_core/*.py scripts/*.py
SMOKE_TEST=1 bash scripts/run_full_benchmark.sh
unset SMOKE_TEST
bash scripts/run_full_benchmark.sh
```

Equivalent helper script:

```bash
bash scripts/run_rbgt_server_first.sh
```

After this first formal run finishes:
- If the pipeline is stable and the ranking is sensible, then move to `Heavy`.
- If the pipeline is unstable, keep the same profile and debug one variable at a time.
