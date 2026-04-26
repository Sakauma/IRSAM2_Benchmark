# 5090 单卡完整基准测试

本文档记录在单张 RTX 5090 服务器上运行 IR-only SAM2 基准测试的推荐入口。

## 目标

默认脚本执行以下组合：

- checkpoint：`tiny`、`small`、`base_plus`、`large`
- Track A prompted policy：`box`、`point`、`box_point`
- Track B automatic-mask policy：`no_prompt`
- mask 主数据集：`NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K`、`MultiModal`
- 弱标注补充数据集：`RBGT-Tiny` 的 IR 分支
- 额外诊断：`base_plus` 上运行 `tight_box` vs `loose_box`，检查 mask-derived box 生成策略是否改变结论

mask 数据集会进入自动分析和显著性检验。Track A 和 Track B 分开分析，默认不做跨 track paired test。`RBGT-Tiny` 只有 box 标注，脚本会保存原始结果，但不会把它混入 mask 指标统计。

主 `box` policy 使用 `mask_derived_adaptive_loose_box_centroid_point_v2` 协议：

- tight box：GT mask 前景像素的最小外接矩形
- loose box：tight box 每边按 `pad_ratio=0.15` 扩展，最少扩 `2` 像素，但最终边长不超过 tight box 对应边长的 `2.0` 倍
- point：GT mask 前景像素质心

## 完整配置

复制完整配置模板：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
```

修改 `configs/server_benchmark_full.local.yaml`。服务器运行所需的路径、模型、方法、数据集、seed、batch、suite 和分析参数都在这个文件中。

```yaml
paths:
  sam2:
    repo: "/root/sam2"
    checkpoint_root: "/root/autodl-tmp/checkpoints"
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

## 推荐流程

先检查命令展开：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --dry-run
```

先跑小样本 smoke test：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --smoke-test
```

再跑 24 张图像级 micro test。它会展开和正式实验相同的组合，但每个数据集只跑前 24 张图像，输出到独立的 `paper_5090_micro/`：

```bash
python scripts/run_5090_micro_benchmark.py --config configs/server_benchmark_full.local.yaml --dry-run
python scripts/run_5090_micro_benchmark.py --config configs/server_benchmark_full.local.yaml --stop-on-error
```

正式运行全部组合：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml
```

如果中断后继续运行，直接重复正式命令。脚本默认检查每个组合的 `benchmark_spec.json`、`run_metadata.json`、`summary.json`、`results.json` 和 `eval_reports/rows.json`，完整组合会自动跳过。

强制重跑：

```bash
python scripts/run_5090_full_benchmark.py --config configs/server_benchmark_full.local.yaml --rerun
```

只跑一个子集用于排查：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny \
  --modes box
```

## 输出位置

假设 `paths.artifacts.root` 是 `/root/autodl-tmp/irsam2_artifacts`，正式结果会写到：

```text
/root/autodl-tmp/irsam2_artifacts/paper_5090/
├── runs/
│   ├── mask/{tiny,small,base_plus,large}/T5090_mask_prompt_modes/{dataset}/{method}/
│   ├── auto_mask/{tiny,small,base_plus,large}/T5090_no_prompt_auto_mask/{dataset}/sam2_no_prompt_auto_mask/
│   ├── prompt_box_protocol/base_plus/T5090_prompt_box_protocol_ablation/{dataset}/{method}/
│   └── rbgt_box/{tiny,small,base_plus,large}/T5090_rbgt_box_prompt_modes/rbgt_tiny_ir_box/{method}/
├── analysis/
│   ├── checkpoint_sweep_summary.csv
│   ├── checkpoint_sweep_summary.json
│   └── mask/{tiny,small,base_plus,large}/
├── generated/
│   ├── run_configs/
│   ├── matrices/
│   └── analysis_configs/
├── benchmark_manifest_latest.json
├── run_manifest_latest.json
└── run_failures_latest.json
```

每个 run 目录包含：

- `benchmark_spec.json`
- `run_metadata.json`
- `summary.json`
- `results.json`
- `eval_reports/rows.json`
- `eval_reports/error_log.jsonl`：仅当存在坏图或单图异常时生成
- `visuals/`

每个 mask checkpoint 的分析目录包含：

- `analysis-report.md`
- `stats-appendix.md`
- `figure-catalog.md`
- `tables/main_baseline_table.csv`
- `tables/significance_tests.csv`

跨 checkpoint 汇总表：

- `analysis/checkpoint_sweep_summary.csv`
- `analysis/checkpoint_sweep_summary.json`

## 注意事项

- 单卡 5090 上不要并行启动多个 full benchmark 进程。
- prompted 模式使用跨图像 batch；`LatencyMs` 是 batch 总耗时除以 batch 内样本数的摊销单图耗时。
- `no_prompt` 使用 SAM2 官方自动网格点批次 `auto_mask_points_per_batch`，不是跨图像重写。
- `no_prompt` 通常最慢，且容易产生大量 false alarm。
- 如果某张图在预测、mask 对齐、指标计算或可视化阶段报错，当前 run 不会退出；错误会写入 `eval_reports/error_log.jsonl`，字段包含 `sample_id`、`frame_id`、`image_path`、`mask_path`、`stage`、`error_type`、`error_message` 和 `traceback`，后续可按这些字段单独重跑。
- 默认 `seeds: [42]` 是有意设置；当前是 zero-shot 推理基准，不是训练稳定性实验。
- `RBGT-Tiny` 的结果只能作为弱标注补充证据，不应写进 mask 主表。
- 论文主文应把 `box` 写成 mask-derived loose-box oracle，不应写成数据集原生 box 标注。
