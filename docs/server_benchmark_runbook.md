# 服务器基准测试运行与归档清单

本文档记录在远程服务器上运行 IR-only SAM2 基准测试的步骤，以及运行完成后需要保存到本地的文件夹。

## 运行目标

默认完整基准会运行 88 个组合：

- `mask` 主实验：4 个 mask 数据集 × 3 个 prompted mode × 4 个 SAM2 checkpoint，共 48 个 run。
- `auto_mask` Track B 实验：4 个 mask 数据集 × `no_prompt` × 4 个 checkpoint，共 16 个 run。
- `rbgt_box` 补充实验：`RBGT-Tiny` 红外分支 × 4 个 prompt mode × 4 个 checkpoint，共 16 个 run。
- `prompt_box_protocol` 协议诊断：4 个 mask 数据集 × `adaptive loose box` 和 `tight box` × `base_plus`，共 8 个 run。

主实验只针对红外图像。`no_prompt` 是图像级 Track B automatic-mask 评估，不和 Track A prompted segmentation 做 paired test。`RBGT-Tiny` 只作为弱标注补充证据，不应混入 mask 主表。

当前 `box` prompt 协议是 `mask_derived_adaptive_loose_box_centroid_point_v2`：

- tight box：GT mask 前景像素的最小外接矩形。
- adaptive loose box：tight box 每边按 `pad_ratio=0.15` 外扩，最少扩 `2px`，但最终边长不超过 tight box 对应边长的 `2.0x`。
- point：GT mask 前景像素质心。

## 服务器准备

进入仓库：

```bash
cd /path/to/IRSAM2_Benchmark
```

安装依赖。PyTorch 安装命令需要匹配服务器 CUDA 环境：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

确认 GPU 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

确认 SAM2 仓库和 checkpoint 存在：

```bash
ls /path/to/sam2
ls /path/to/sam2/checkpoints
```

## 完整配置

复制完整 benchmark 配置模板：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
```

编辑 `configs/server_benchmark_full.local.yaml`。服务器运行的路径、模型、方法、数据集、seed、suite 和分析参数都在这个文件中：

```yaml
paths:
  sam2:
    repo: "/root/sam2"
    checkpoint_root: "/root/autodl-tmp/sam2/checkpoints"
  artifacts:
    root: "/root/autodl-tmp/irsam2_artifacts"
  datasets:
    nuaa_sirst: "/root/autodl-tmp/datasets/NUAA-SIRST"
    nudt_sirst: "/root/autodl-tmp/datasets/NUDT-SIRST"
    irstd_1k: "/root/autodl-tmp/datasets/IRSTD-1K"
    multimodal: "/root/autodl-tmp/datasets/MultiModal"
    rbgt_tiny_ir_box: "/root/autodl-tmp/datasets/RBGT-Tiny"

execution:
  cuda_visible_devices: "0"
  pytorch_cuda_alloc_conf: "expandable_segments:True"

runtime:
  seeds: [42]
```

`configs/server_benchmark_full.local.yaml` 是机器私有文件，建议保存在 artifact 中用于追溯，但不要提交到公开仓库。

## 推荐运行流程

先检查展开后的运行计划：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --dry-run
```

先跑 10 张图的 smoke test：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --smoke-test
```

正式运行全部 checkpoint：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml
```

如果服务器上只准备了 `base_plus` 和 `large` 两个 checkpoint：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --checkpoints base_plus,large
```

如果中断，直接重复正式命令。脚本会跳过已经完整生成 `benchmark_spec.json`、`run_metadata.json`、`summary.json`、`results.json`、`eval_reports/rows.json` 的 run。

强制重跑时才使用：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --rerun
```

## 运行完成后的验收

假设 `paths.artifacts.root` 是 `/root/autodl-tmp/irsam2_artifacts`，正式输出目录是：

```text
/root/autodl-tmp/irsam2_artifacts/paper_5090/
```

检查失败列表：

```bash
cat /root/autodl-tmp/irsam2_artifacts/paper_5090/run_failures_latest.json
```

理想状态是空列表 `[]`。

检查最终 manifest：

```bash
cat /root/autodl-tmp/irsam2_artifacts/paper_5090/benchmark_manifest_latest.json
```

验收标准：

- `failed_count` 等于 `0`。
- `completed_count + skipped_existing_count` 等于 `run_count`。
- `analysis/checkpoint_sweep_summary.csv` 存在。
- `analysis/mask/{checkpoint}/tables/main_baseline_table.csv` 存在。
- `analysis/mask/{checkpoint}/tables/significance_tests.csv` 存在。
- `analysis/auto_mask/{checkpoint}/tables/main_baseline_table.csv` 存在。
- `analysis/prompt_box_protocol/base_plus/tables/significance_tests.csv` 存在。

如果 `run_failures_latest.json` 非空，先不要删除任何 run 目录。优先保留完整 artifact，后续可以根据 `config_path` 和 `command` 定位失败组合。

## 需要保存到本地的内容

最稳妥的做法是保存整个正式 artifact 目录：

```text
/root/autodl-tmp/irsam2_artifacts/paper_5090/
```

这个目录包含论文分析需要的原始结果、逐样本指标、可视化、自动分析报告、显著性检验、生成配置和运行 manifest。

如果空间有限，至少保存以下内容：

- `paper_5090/run_manifest_latest.json`
- `paper_5090/run_manifest_latest.csv`
- `paper_5090/run_failures_latest.json`
- `paper_5090/benchmark_manifest_latest.json`
- `paper_5090/benchmark_manifest_*.json`
- `paper_5090/generated/`
- `paper_5090/analysis/`
- `paper_5090/runs/**/benchmark_spec.json`
- `paper_5090/runs/**/run_metadata.json`
- `paper_5090/runs/**/summary.json`
- `paper_5090/runs/**/results.json`
- `paper_5090/runs/**/eval_reports/rows.json`
- `paper_5090/runs/**/visuals/`

建议额外保存这几个配置文件，方便之后复现实验：

- `configs/server_benchmark_full.local.yaml`
- `configs/server_benchmark_full.example.yaml`

建议把代码状态也写入 artifact：

```bash
git rev-parse HEAD > /root/autodl-tmp/irsam2_artifacts/paper_5090/code_revision.txt
git status --short > /root/autodl-tmp/irsam2_artifacts/paper_5090/code_status.txt
git diff > /root/autodl-tmp/irsam2_artifacts/paper_5090/code_uncommitted.diff
```

## 拷回本地

推荐从本地 WSL 执行 `rsync`：

```bash
mkdir -p /home/sakauma/irsam2_artifacts_from_server
rsync -avh --info=progress2 \
  user@server:/root/autodl-tmp/irsam2_artifacts/paper_5090/ \
  /home/sakauma/irsam2_artifacts_from_server/paper_5090/
```

如果网络不稳定，可以先在服务器上打包：

```bash
cd /root/autodl-tmp/irsam2_artifacts
tar -czf paper_5090_artifacts.tar.gz paper_5090
```

再从本地下载：

```bash
scp user@server:/root/autodl-tmp/irsam2_artifacts/paper_5090_artifacts.tar.gz \
  /home/sakauma/irsam2_artifacts_from_server/
```

## 不需要作为论文 artifact 保存的内容

以下内容通常不需要跟随每次实验结果一起保存：

- `paper_5090_smoke/`：只用于链路检查，不作为论文正式结果。
- SAM2 checkpoint 文件：体积大，单独管理即可。
- 原始公开数据集：单独管理数据版本即可。
- Python 虚拟环境或 conda 环境目录：用依赖记录复现，不要直接归档环境目录。

如果 `MultiModal` 是自有数据集，应单独备份原始图像、mask 标注和数据版本说明。它不应只依赖 benchmark artifact 保存。

## 本地论文分析入口

拷回本地后，优先从这些文件开始分析：

- `paper_5090/analysis/checkpoint_sweep_summary.csv`
- `paper_5090/analysis/mask/{checkpoint}/analysis-report.md`
- `paper_5090/analysis/mask/{checkpoint}/stats-appendix.md`
- `paper_5090/analysis/mask/{checkpoint}/tables/main_baseline_table.csv`
- `paper_5090/analysis/mask/{checkpoint}/tables/significance_tests.csv`
- `paper_5090/analysis/auto_mask/{checkpoint}/tables/main_baseline_table.csv`
- `paper_5090/analysis/prompt_box_protocol/base_plus/tables/significance_tests.csv`
- `paper_5090/analysis/*/*/figures/qualitative_cases/`
- `paper_5090/runs/**/eval_reports/rows.json`

论文主表优先使用 `analysis/mask/{checkpoint}/tables/main_baseline_table.csv`。逐样本失败分析和后续消融对比优先使用各 run 的 `eval_reports/rows.json`。
