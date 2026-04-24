# IRSAM2_Benchmark

`IRSAM2_Benchmark/` 是一个面向 `SAM2` 红外分割链路的 benchmark 平台。它与旧仓库 `ir_sam2_bench` 有意保持独立，后者仅作为只读历史参考，不再承担当前平台的实现职责。

当前 `v0.1` 版本重点覆盖四件事：

- 清晰的包结构与统一 CLI
- `MultiModal`、COCO-like、`RBGT-Tiny` 与通用 `images/ + masks/` 数据接入
- 一等公民化的 `SAM2` baseline，包括无 prompt 和时序传播模式
- 冻结的结果 schema、artifact manifest 与面向论文的报告输出

## 当前范围

这套代码已经实现了 benchmark 平台骨架、数据接入、结果 schema、baseline registry 与评估栈。baseline 层是可运行的；完整链路中的 `adapt`、`distill`、`quantize` 虽然已经进入统一 pipeline 和 artifact 接口，但目前仍然是 reference-stage scaffold，不是最终论文方法。

这样的拆分是刻意为之：

- benchmark 基础设施需要先稳定下来
- baseline 与评估语义需要先做到可复现

## 目录结构

- `src/irsam2_benchmark/`：运行时代码包
- `configs/`：YAML benchmark 配置
- `docs/`：benchmark 规范与说明
- `tests/`：单元测试
- `artifacts/`：默认输出目录
- `reference_results/`：用于回归对比的冻结参考结果

仓库入口索引：

- `docs/README.md`：文档索引
- `configs/README.md`：配置索引
- `scripts/README.md`：脚本索引

## CLI

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_box_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_no_prompt_auto_mask
python -m irsam2_benchmark.cli run evaluate --config configs/benchmark_v1.yaml
```

如果只想做快速验证，优先使用 smoke 配置：

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

如果只想跑 10 张图左右的轻量检查，使用 quick 配置：

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_quick10.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_quick10_rbgt_tiny.yaml --baseline sam2_zero_shot
```

`smoke` 配置主要用于验证：

- 数据集 adapter 是否能正常加载
- prompt synthesis 是否正确
- `SAM2` checkpoint 接线是否正常
- 报告 schema 与分组评估输出是否正确
- `reference_results/` 下的快照是否能正常生成

`quick10` 配置用于更短时长的 sanity check：

- `configs/benchmark_quick10.yaml`：`MultiModalCOCOClean`
- `configs/benchmark_quick10_rbgt_tiny.yaml`：`RBGT-Tiny` 红外分支
- 固定 `max_images: 10`
- 固定 `seeds: [42]`
- 固定 `save_visuals: false`

如果数据集不在仓库默认相对路径下，请显式设置 `DATASET_ROOT`。这在 `configs/benchmark_quick10_rbgt_tiny.yaml` 上尤其常见。

## 文档

请把 `docs/README.md` 作为文档入口。最重要的 benchmark 规范包括：

- `docs/track_definitions.md`
- `docs/metric_cards.md`
- `docs/dataset_cards.md`
- `docs/artifact_schema_spec.md`

根 README 只保留执行入口和使用说明，详细 benchmark 协议统一放在 `docs/` 下。

## Linux 脚本

针对 Linux 环境，`scripts/` 目录提供了以下入口：

```bash
bash scripts/run_smoke.sh
bash scripts/run_smoke.sh sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1.yaml sam2_zero_shot_point
bash scripts/run_tests.sh
```

`RBGT-Tiny` 也有独立的 smoke 和 v1 配置：

```bash
bash scripts/run_baseline.sh configs/benchmark_smoke_rbgt_tiny.yaml sam2_zero_shot
bash scripts/run_baseline.sh configs/benchmark_v1_rbgt_tiny.yaml sam2_zero_shot
```

官方 baseline matrix 入口也已恢复：

```bash
bash scripts/run_official_baseline_matrix.sh
MATRIX_MODELS=tiny,small MATRIX_DATASETS=multimodal MATRIX_MODES=box,point bash scripts/run_official_baseline_matrix.sh
```

如果官方 `SAM2.1` checkpoint 不在 `${SAM2_REPO}/checkpoints` 下，请额外指定 checkpoint 根目录：

```bash
export SAM2_CKPT_ROOT=/path/to/official_sam2_checkpoints
bash scripts/run_official_baseline_matrix.sh
```

对于 AutoDL 风格环境，可以使用：

```bash
bash scripts/setup_autodl_server.sh
source .autodl_env.sh
bash scripts/run_autodl_smoke.sh multimodal sam2_zero_shot
bash scripts/run_autodl_smoke.sh rbgt sam2_zero_shot
```

每个脚本的简要说明可见 `scripts/README.md`。

## 配置

请通过 `configs/README.md` 在 quick、smoke 和完整 benchmark 配置之间做选择。

## 机器相关路径

机器相关路径统一通过以下环境变量配置：

- `DATASET_ROOT`
- `SAM2_REPO`
- `ARTIFACT_ROOT`

实验语义保留在 YAML 配置中，环境变量只负责路径解析，不承载实验定义本身。
