# IRSAM2_Benchmark

`IRSAM2_Benchmark` 是面向红外目标分割实验的 SAM2 benchmark 工程。

用完整 YAML 展开 SAM2 baseline 矩阵，运行 prompted 与 no-prompt 自动掩码评估，随后基于 artifacts 生成分析表格。

## 当前能力

- 支持 SAM2 box、tight box、point、box+point 显式 prompt baseline。
- 支持 SAM2 no-prompt automatic mask baseline。
- 支持 `NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K`、`MultiModal` 四个 mask-supervised 数据集。
- 支持 `RBGT-Tiny` 红外分支作为弱标注补充数据集。
- 支持总体、小目标、大目标分组指标。
- 支持单 YAML 配置路径、模型、方法、数据集、seed、batch、suite 和分析参数。

## 安装

先按本机 CUDA 环境安装 PyTorch，再安装本仓库：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

如果 CUDA 版本不是 `cu124`，请用 PyTorch 官网生成的安装命令替换第一行。

SAM2 官方源码以 Git submodule 形式固定在 `sam2/`。首次 clone 后执行：

```bash
git submodule update --init --recursive
```

如果需要使用外部 SAM2 checkout，也可以在 YAML 的 `paths.sam2.repo` 中覆盖路径。

## 配置

复制完整配置模板：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
```

编辑 `configs/server_benchmark_full.local.yaml`：

- `paths.sam2.repo`：SAM2 官方仓库路径。
- `paths.sam2.checkpoint_root`：SAM2 checkpoint 目录。
- `paths.datasets.*`：各数据集根目录。
- `paths.artifacts.root`：benchmark 输出目录。
- `runtime`：正式运行的 batch、seed、visual 等参数。
- `smoke_test_runtime`：`--smoke-test` 使用的轻量参数。
- `models`：要评估的 SAM2 checkpoint。
- `suites`：要展开的数据集和方法组合。

`configs/*.local.yaml` 被 `.gitignore` 排除，不应提交。

## 运行完整基准

先 dry-run 检查展开结果：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --dry-run
```

跑轻量 smoke：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --smoke-test \
  --stop-on-error
```

正式运行：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml
```

常用筛选参数：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny,small \
  --modes box,point
```

`--rerun` 会强制重跑已完成组合；默认会跳过完整 artifacts。

## 单条 baseline

矩阵脚本会自动生成单 run YAML。需要手动复跑某个生成配置时，可以直接调用 CLI：

```bash
python -m irsam2_benchmark.cli run baseline \
  --config artifacts/paper_5090/generated/run_configs/mask/tiny/nuaa_sirst_sam2_box_oracle.yaml \
  --baseline sam2_pretrained_box_prompt
```

当前合法 baseline 名称：

- `bbox_rect`
- `sam2_pretrained_box_prompt`
- `sam2_pretrained_tight_box_prompt`
- `sam2_pretrained_point_prompt`
- `sam2_pretrained_box_point_prompt`
- `sam2_no_prompt_auto_mask`

## 数据集约定

主实验使用四个 mask-supervised 数据集：

- `NUAA-SIRST`
- `NUDT-SIRST`
- `IRSTD-1K`
- `MultiModal`

`MultiModal` 使用 `img/ + label/` 结构，平台会把 JSON polygon 解码为 mask。当前协议对每个 instance 只使用第一个有效 polygon；多 polygon 合并需要单独声明为新协议。

`RBGT-Tiny` 只使用红外分支和 COCO 风格标注。它是补充 suite，不和四个 mask 主数据集混在同一主表中解释。

对 mask-supervised 数据集，box 和 point prompt 都从 GT mask 确定性生成：

- `box`：GT mask tight box 外扩得到 adaptive loose box。
- `tight_box`：GT mask 前景最小外接矩形。
- `point`：GT mask 前景像素质心。

## 结果输出

完整 benchmark 默认写入：

```text
artifacts/
└── paper_5090/
    ├── benchmark_manifest_latest.json
    ├── run_manifest_latest.csv
    ├── generated/
    │   ├── run_configs/
    │   ├── matrices/
    │   └── analysis_configs/
    ├── runs/
    └── analysis/
```

每个 run 至少包含：

- `benchmark_spec.json`
- `run_metadata.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`

如果单样本失败，会写入 `eval_reports/error_log.jsonl`。

## 分析

`run_5090_full_benchmark.py` 会在 suite 配置 `run_analysis: true` 时自动生成并执行分析配置。也可以手动运行：

```bash
python scripts/analyze_paper_results.py \
  --analysis artifacts/paper_5090/generated/analysis_configs/mask/tiny.yaml
```

分析输出包括主表、MultiModal 大小目标分表、no-prompt 自动掩码表、面积桶表、显著性检验和案例索引。

## 测试

运行单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行主配置 dry-run 测试：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny \
  --modes box \
  --dry-run
```

## 文档

- `configs/README.md`：配置文件说明。
- `scripts/README.md`：脚本说明。
- `docs/server_5090_benchmark.md`：完整基准运行方案。
- `docs/server_benchmark_runbook.md`：服务器运行和验收清单。
- `docs/metric_cards.md`：指标定义。
- `docs/dataset_cards.md`：数据集约定。
- `docs/artifact_schema_spec.md`：artifact schema。
