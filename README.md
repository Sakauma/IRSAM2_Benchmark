# IRSAM2_Benchmark

`IRSAM2_Benchmark` is a fresh benchmark platform for evaluating SAM2-based infrared segmentation pipelines. It replaces the earlier monolithic repository layout with a package-based benchmark runtime and frozen reporting schema.

This v0.1 implementation focuses on four things:

- a clean package structure and CLI
- dataset adapters for raw MultiModal, COCO-like, RBGT-Tiny, and generic `images/ + masks/`
- first-class SAM2 baselines, including prompt-free and sequence-aware evaluation hooks
- frozen result schema, manifests, and paper-oriented reporting

## Current Scope

This codebase fully implements the benchmark platform skeleton, dataset ingestion, result schema, baseline registry, and evaluation stack. The baseline layer is functional. The full-chain training stages (`adapt`, `distill`, `quantize`) are implemented as pipeline stages with stable artifacts and interfaces, but they are still reference-stage scaffolds rather than finalized paper methods.

That split is deliberate:

- benchmark infrastructure must be stable before methods are swapped in
- baseline and evaluation behavior must be reproducible before claiming pipeline gains

## Directory Layout

- `src/irsam2_benchmark/`: runtime package
- `configs/`: YAML benchmark configs
- `docs/`: benchmark specifications
- `tests/`: unit tests
- `artifacts/`: default output root
- `reference_results/`: frozen per-baseline reference snapshots for regression checks

## CLI

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_box_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_no_prompt_auto_mask
python -m irsam2_benchmark.cli run evaluate --config configs/benchmark_v1.yaml
```

For fast validation, use the smoke config:

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

The smoke config is intentionally small enough to validate:

- dataset adapter loading
- prompt synthesis
- SAM2 checkpoint wiring
- report schema and grouped eval outputs
- reference snapshot generation under `reference_results/`

## Linux Scripts

For Linux servers, the `scripts/` directory now includes shell entrypoints:

```bash
bash scripts/run_smoke.sh
bash scripts/run_smoke.sh sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1.yaml sam2_zero_shot_point
bash scripts/run_tests.sh
```

For AutoDL-style servers, there is also a bootstrap workflow:

```bash
bash scripts/setup_autodl_server.sh
source .autodl_env.sh
bash scripts/run_autodl_smoke.sh multimodal sam2_zero_shot
bash scripts/run_autodl_smoke.sh rbgt sam2_zero_shot
```

The AutoDL helper assumes:

- dataset archives are stored under `/root/autodl-fs`
- extracted datasets, outputs, and checkpoints live under `/root/autodl-tmp`
- the SAM2 repository is available at `/root/sam2` unless `SAM2_REPO` is overridden

## Machine Paths

Machine-specific paths are configured through:

- `DATASET_ROOT`
- `SAM2_REPO`
- `ARTIFACT_ROOT`

The benchmark config keeps experimental meaning in YAML. Environment variables are used only for path resolution.

---

## 中文说明

`IRSAM2_Benchmark` 是一个面向 `SAM2` 红外分割链路的全新 benchmark 平台。它替换了旧的单体脚本式仓库结构，改为更清晰的 package 化运行时、冻结的评测协议和统一的结果 schema。

当前 `v0.1` 版本主要聚焦四件事：

- 清晰的 Python package 结构与 CLI 入口
- 原始 `MultiModal`、COCO-like、`RBGT-Tiny` 和通用 `images/ + masks/` 数据集适配
- 一等公民化的 `SAM2` 基线，包括无 prompt 和序列/视频评测入口
- 面向论文的冻结结果格式、manifest 和评测输出

### 当前范围

当前代码已经实现：

- benchmark 平台骨架
- 数据接入层
- 统一结果 schema
- baseline registry
- 评估与报告层

其中 baseline 层已经可以运行；完整链路中的 `adapt`、`distill`、`quantize` 已经定义为稳定的 pipeline stage，并具备 artifact 接口，但目前仍然是 reference-stage scaffold，还不是最终论文方法。

这是刻意的分层：

- benchmark 基础设施要先稳定
- baseline 和评测行为要先可复现
- 在此基础上再逐步接入真正的方法实现

### 目录结构

- `src/irsam2_benchmark/`：运行时代码包
- `configs/`：YAML benchmark 配置
- `docs/`：benchmark 规范文档
- `tests/`：单元测试
- `artifacts/`：默认输出目录
- `reference_results/`：用于回归检查的冻结基线结果快照

### CLI

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_box_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_no_prompt_auto_mask
python -m irsam2_benchmark.cli run evaluate --config configs/benchmark_v1.yaml
```

如果只是做快速验证，建议使用 smoke 配置：

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

这个 smoke 配置足够验证：

- 数据集 adapter 加载
- prompt synthesis
- `SAM2` checkpoint 接线是否正确
- report schema 与分组评测输出
- `reference_results/` 下的快照生成

### Linux 脚本

现在 `scripts/` 目录里也提供了 Linux 下可直接运行的 shell 脚本：

```bash
bash scripts/run_smoke.sh
bash scripts/run_smoke.sh sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1.yaml sam2_zero_shot_point
bash scripts/run_tests.sh
```

### 机器相关路径

机器相关路径通过以下环境变量配置：

- `DATASET_ROOT`
- `SAM2_REPO`
- `ARTIFACT_ROOT`

实验语义和 benchmark 协议都保留在 YAML 配置中；环境变量只负责路径解析，不承载实验定义本身。
