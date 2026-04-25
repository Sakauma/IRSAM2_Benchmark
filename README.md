# IRSAM2_Benchmark

`IRSAM2_Benchmark` 是面向红外小目标分割论文实验的 `SAM2` benchmark 平台。

当前平台只服务红外图像实验。真实 RGB 图像不会作为输入、对比项或融合分支进入实验流程。`SAM2` 需要三通道输入时，平台只会把单通道红外图像归一化后复制为三通道。

## 当前能力

- 支持 `SAM2` box、point、box+point、no-prompt automatic mask baseline。
- 支持 IR physics-prior 自动 prompt baseline：`sam2_physics_auto_prompt`。
- 支持 COCO polygon、COCO box、`images/ + masks/`、`RBGT-Tiny` 红外分支数据接入。
- 支持 `MultiModal` 的 polygon 标注作为 mask supervision。
- 支持小目标指标和 prompt 质量指标。
- 支持论文实验矩阵 YAML，一条命令展开多数据集、多方法、多消融实验。
- 预留 adapter、decoder、loss、distiller、quantizer 的模块化接口，后续训练、蒸馏、量化可以按配置消融。

## 安装

平台运行最基本的 `SAM2` baseline 也需要 PyTorch，因此 `torch` 和 `torchvision` 是核心依赖，不是可选训练依赖。

建议先按本机 CUDA 环境安装 PyTorch，再安装本仓库：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

如果 CUDA 版本不是 `cu124`，请用 PyTorch 官网生成的安装命令替换第一行。

`SAM2` 官方源码不复制进本仓库。你需要在本机保留一个 `facebookresearch/sam2` 仓库，并在 `configs/local_paths.yaml` 中配置它的位置。

## 路径配置

论文实验不推荐使用 `export DATASET_ROOT=...` 这种方式。请使用 YAML 记录机器相关路径。

先复制模板：

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

然后编辑 `configs/local_paths.yaml`：

```yaml
sam2:
  repo: "/path/to/sam2"
  checkpoint_root: "/path/to/sam2/checkpoints"

artifacts:
  root: "artifacts"

datasets:
  nuaa_sirst: "/path/to/NUAA-SIRST"
  nudt_sirst: "/path/to/NUDT-SIRST"
  irstd_1k: "/path/to/IRSTD-1K"
  multimodal: "/path/to/MultiModal"
  rbgt_tiny_ir_box: "/path/to/RBGT-Tiny"

execution:
  cuda_visible_devices: "0"
  pytorch_cuda_alloc_conf: "expandable_segments:True"

runtime:
  seeds: [42]
```

`configs/local_paths.yaml` 应该是本机私有配置，不建议提交到远端。模板文件 `configs/local_paths.example.yaml` 可以提交和共享。

## 论文实验入口

机器可读实验矩阵：

- `configs/paper_experiments_v1.yaml`

人工可读实验清单：

- `docs/paper_experiment_matrix.md`

先 dry-run 检查展开后的命令和路径：

```bash
python scripts/run_paper_experiments.py \
  --matrix configs/paper_experiments_v1.yaml \
  --paths configs/local_paths.yaml \
  --group p0_all \
  --dry-run
```

确认无误后运行 P0 全部实验：

```bash
python scripts/run_paper_experiments.py \
  --matrix configs/paper_experiments_v1.yaml \
  --paths configs/local_paths.yaml \
  --group p0_all
```

实验跑完后生成论文级分析：

```bash
python scripts/analyze_paper_results.py \
  --analysis configs/paper_analysis_v1.yaml
```

分析产物默认写入 `artifacts/paper_v1/analysis/`，包括论文表格、显著性检验、错误分桶、案例索引和 Markdown 报告。

可选实验组：

- `p0_baselines`：oracle prompt baseline。
- `p0_auto_prompt`：no-prompt SAM2 与 physics auto-prompt。
- `p0_ablation`：physics prior 和 prompt 后处理消融。
- `p0_all`：以上全部 P0 实验。

## 5090 单卡完整基准

如果要在单张 RTX 5090 服务器上跑完整基准，推荐使用专用入口：

```bash
python scripts/run_5090_full_benchmark.py --paths configs/local_paths.yaml --dry-run
python scripts/run_5090_full_benchmark.py --paths configs/local_paths.yaml --smoke-test
python scripts/run_5090_full_benchmark.py --paths configs/local_paths.yaml
```

该入口默认展开：

- 4 个 SAM2.1 checkpoint：`tiny`、`small`、`base_plus`、`large`
- 4 种主 prompt policy：`box`、`point`、`box+point`、`no_prompt`
- 4 个 mask 主数据集和单独的 `RBGT-Tiny` box-only 补充数据集
- `base_plus` 上额外运行 `tight_box` vs `loose_box`，检查 mask-derived box 生成策略是否影响结论

完整运行方案见 `docs/server_5090_benchmark.md`。服务器运行、验收和结果归档清单见 `docs/server_benchmark_runbook.md`。

## 单条 baseline 命令

如果只想手动跑某个配置，可以直接使用 CLI：

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_tight_box
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot_box_point
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_no_prompt_auto_mask
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_v1.yaml --baseline sam2_physics_auto_prompt
```

快速 smoke test：

```bash
python -m irsam2_benchmark.cli run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

## 数据集约定

论文主实验使用四个 mask-supervised 红外数据集：

- `NUAA-SIRST`
- `NUDT-SIRST`
- `IRSTD-1K`
- `MultiModal`

`MultiModal` 使用 `img/ + label/` 结构，平台通过 `MultiModalAdapter` 将 JSON polygon 解码为 mask。

`RBGT-Tiny` 只使用红外分支和 box 标注。它不进入主 mask mIoU 表，主要用于 box-only weak supervision 或后续 prompt 相关实验。

对 mask-supervised 数据集，`box` 和 `point` prompt 不是原生标注。平台会从 GT mask 确定性生成 prompt：

- `box` 默认是 `mask_derived_adaptive_loose_box_centroid_point_v2` 中的 adaptive loose box：先取 GT mask 最小外接矩形，再按 `pad_ratio=0.15`、`min_pad=2px` 外扩，但最终边长不超过 tight box 的 2 倍。
- `tight_box` 是 GT mask 前景的最小外接矩形，仅用于 prompt 协议敏感性诊断。
- `point` 是 GT mask 前景像素质心。

## 结果输出

默认输出目录由 `configs/local_paths.yaml` 中的 `artifacts.root` 控制。

每次运行会写出：

- `benchmark_spec.json`
- `artifact_manifest.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`
- 可选可视化结果

论文实验矩阵会按以下结构组织输出：

```text
artifacts/
└── paper_v1/
    ├── T1_oracle_prompt_baselines/
    ├── T2_ir_auto_prompt/
    └── T3_prior_prompt_ablation/
```

## 指标

常规分割指标：

- `mIoU`
- `Dice`
- `BoundaryF1`
- `LatencyMs`
- `BBoxIoU`

红外小目标指标：

- `TargetRecallIoU10`
- `TargetRecallIoU25`
- `TargetRecallIoU50`
- `FalseAlarmPixelsPerMP`
- `FalseAlarmComponents`
- `GTAreaPixels`
- `PredAreaPixels`

Prompt 质量指标：

- `PromptHitRate`
- `PromptDistanceToCentroid`
- `PromptBoxCoverage`

## 模块化接口

当前已预留以下 factory：

- `PriorFactory`
- `PromptFactory`
- `AdapterFactory`
- `DecoderFactory`
- `LossFactory`
- `DistillerFactory`
- `QuantizerFactory`

P0 已实现：

- `prior_fusion`
- `heuristic_physics`
- `sam2_physics_auto_prompt`

训练、蒸馏、量化阶段后续应复用同一套 `modules` 配置接口，不要另起一套实验入口。

## 测试

运行完整单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前测试覆盖：

- 数据集 adapter。
- COCO polygon mask 解码。
- `RBGT-Tiny` 红外分支读取。
- `SAM2` adapter 行为。
- physics prior/prompt 模块。
- 小目标指标和 prompt 指标。
- 论文实验矩阵 dry-run。

## 文档索引

- `docs/README.md`：文档总入口。
- `docs/paper_experiment_matrix.md`：论文实验矩阵说明。
- `docs/eval_v2_roadmap.md`：评估模块后续改进计划。
- `docs/metric_cards.md`：指标定义。
- `docs/dataset_cards.md`：数据集约定。
- `docs/artifact_schema_spec.md`：输出 schema。
- `configs/README.md`：配置文件说明。
- `scripts/README.md`：脚本说明。
