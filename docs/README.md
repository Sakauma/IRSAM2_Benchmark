# 文档索引

本目录只保留当前 benchmark 主流程仍在使用的说明。

## 运行文档

- `server_5090_benchmark.md`：完整 benchmark 运行方案。
- `server_benchmark_runbook.md`：服务器运行、验收和 artifact 归档清单。

## 研究复盘

- `research/2026-05-06--sam2-ir-qd--r04--experiment-review-and-paper-direction.md`：M2/M3/M3.1/M4 当前证据、论文方向和下一步实验计划。
- `research/2026-05-08--comparison-evaluation-matrix.md`：第三方模型、通用 SAM、SAM2-IR-QD 当前结果的统一对比矩阵。
- `research/2026-05-08--sam2-ir-qd--m8-multimodal-proposal-plan.md`：M8 MultiModal 小目标 proposal 训练计划、运行命令和判定标准。

## 规范文档

- `artifact_schema_spec.md`：输出文件与报告 schema。
- `prompt_policy_spec.md`：prompt 预算与 prompt 来源规则。
- `metric_cards.md`：指标定义与解释。
- `dataset_cards.md`：数据集 adapter 假设与数据侧说明。

建议阅读顺序：

1. 根目录 `README.md`
2. `server_5090_benchmark.md`
3. `server_benchmark_runbook.md`
4. `metric_cards.md`
5. `artifact_schema_spec.md`
