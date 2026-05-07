# 第三方模型与 SAM2-IR-QD 对比评估矩阵

日期：2026-05-08

## 目的

在继续 M8 实验前，先把当前 SAM2-IR-QD 结果和已导入的第三方预测 mask 放到同一张评估表中，判断论文方向是否有可发表优势。

本轮评估表由 `scripts/build_comparison_evaluation_matrix.py` 生成，默认输出到：

```bash
artifacts/comparison_evaluation_matrix_latest/
```

主要输出：

- `comparison-report.md`
- `tables/external_public_dataset.csv`
- `tables/external_public_macro.csv`
- `tables/ours_public_dataset.csv`
- `tables/ours_public_macro.csv`
- `tables/comparison_public_macro.csv`

## 当前口径

公开三数据集：

- `NUAA-SIRST`：426 个有效样本，跳过 1 个图像/标注尺寸不匹配样本。
- `NUDT-SIRST`：1327 个有效样本。
- `IRSTD-1K`：1001 个有效样本。

第三方模型直接读取 `artifacts/external_predictions/*` 下的二值 mask。其 `WholeMaskIoU25Rate/50Rate` 是快速代理召回，不等同于 benchmark 的目标级 `TargetRecallIoU25/50`。

SAM2-IR-QD 行读取 M5/M6 的 benchmark analysis CSV，因此其中的 `TargetRecallIoU25/50` 是 benchmark 目标级指标。

## 关键结果

公开三数据集宏平均的当前前列结果：

| Rank | Family | Model | mIoU | Dice | FApx/MP |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | IR supervised | SeRankDet | 0.8360 | 0.8965 | 36.2 |
| 2 | IR supervised | BGM | 0.8222 | 0.8847 | 36.5 |
| 3 | IR supervised | MSHNet | 0.7361 | 0.8280 | 60.3 |
| 4 | IR supervised | DRPCANet | 0.7319 | 0.8237 | 56.1 |
| 5 | IR supervised | RPCANet++ | 0.7252 | 0.8147 | 63.2 |
| 11 | Ours | M6 `sam2_ir_fa_rerank` | 0.4581 | 0.5964 | 4661.6 |
| 15 | Generic SAM | HQ-SAM-ViT-B | 0.4340 | 0.5654 | 22266.3 |

## 论文方向判断

当前结果不支持把 SAM2-IR-QD 写成“超过监督式红外小目标分割 SOTA”的论文。

更稳的论文定位是：

1. 面向红外小目标的 SAM2 自动提示迁移框架。
2. 在无人工 prompt 场景下，相比通用 SAM / 轻量 SAM 系列更适合红外小目标。
3. 相比监督式 IR 小目标模型仍存在明显差距，差距主要来自 proposal/localization。
4. M8 应优先解决 MultiModal-small 的 proposal recall，而不是继续堆普通 reranker。

## 下一步使用方式

重新生成评估表：

```bash
python scripts/build_comparison_evaluation_matrix.py
```

如果后续 M8 生成新的 analysis CSV，可以追加输入：

```bash
python scripts/build_comparison_evaluation_matrix.py \
  --ours-analysis M6=artifacts/sam2_ir_qd_m6_promptnet_v2_v1/analysis/checkpoint_sweep_summary.csv \
  --ours-analysis M8=artifacts/<m8-artifact>/analysis/checkpoint_sweep_summary.csv
```
