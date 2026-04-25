# 文档索引

本目录存放仓库中所有面向 benchmark 的说明文档。

## 核心规范

- `artifact_schema_spec.md`：输出文件与报告 schema
- `prompt_policy_spec.md`：prompt 预算与 prompt 来源规则
- `track_definitions.md`：Track A/B/C/D 的语义定义
- `metric_cards.md`：指标定义与解释说明
- `dataset_cards.md`：数据集 adapter 假设与数据侧说明
- `paper_experiment_matrix.md`：论文实验矩阵与启动入口
- `server_5090_benchmark.md`：单张 RTX 5090 上的完整基准运行方案
- `server_benchmark_runbook.md`：服务器运行、验收和 artifact 归档清单
- `eval_v2_roadmap.md`：后续评估分析模块改进计划
- `benchmark_improvement_plan.md`：当前 SAM2 baseline 评估风险与改进计划

## 实现说明

- `implementation_notes.md`：当前实现边界与运行时说明

## 历史说明

- `remote_session_handoff.md`：早期远程工作流遗留的交接说明

如果你第一次接触这个仓库，建议先阅读根目录 `README.md`，然后按下面顺序继续：

1. `track_definitions.md`
2. `metric_cards.md`
3. `dataset_cards.md`
4. `artifact_schema_spec.md`
