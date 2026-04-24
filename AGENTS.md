# 仓库指南

## 项目结构与模块组织
`src/irsam2_benchmark/` 是运行时代码包。主要模块包括：`data/` 用于数据集适配与提示生成，`models/` 和 `baselines/` 用于推理入口，`pipeline/` 用于流程编排，`evaluation/` 用于指标计算与报告输出。新增运行时代码应放在 `src/` 下，而不是直接放在 `main.py` 同级目录。

`configs/` 存放基准测试 YAML 配置，例如 `benchmark_smoke.yaml` 和 `benchmark_v1.yaml`。`tests/` 存放单元测试。`docs/` 存放模式定义与基准说明。`artifacts/` 是默认生成输出目录，`reference_results/` 存放用于回归比对的冻结参考结果。

## 构建、测试与开发命令
开始本地开发前，先以可编辑模式安装：

```powershell
python -m pip install -e .[yaml]
```

运行轻量级 smoke baseline：

```powershell
python main.py run baseline --config configs/benchmark_smoke.yaml --baseline bbox_rect
```

运行完整 baseline 或评估流程：

```powershell
python main.py run baseline --config configs/benchmark_v1.yaml --baseline sam2_zero_shot
python main.py run evaluate --config configs/benchmark_v1.yaml
```

Windows 环境也可以直接使用 `scripts/run_smoke.ps1`。

## 代码风格与命名约定
目标 Python 版本为 3.10+，统一使用 4 空格缩进，并保持显式类型标注。遵循现有命名模式：模块与函数使用 `snake_case`，类使用 `PascalCase`，配置文件使用具备描述性的 `snake_case` 文件名。保持 baseline ID 和 schema 字段名稳定，因为它们会流入产物目录和参考快照。仓库未提交格式化器或 linter 配置，因此按清晰、可读的 PEP 8 风格编写。

## 测试指南
测试使用标准 `unittest` 结构，文件命名为 `tests/test_*.py`，测试类类似 `TemporalMetricTests`，测试方法命名为 `test_*`。运行时建议使用可编辑安装，或显式设置 `PYTHONPATH=src`：

```powershell
$env:PYTHONPATH = "src"; python -m unittest discover -s tests
```

修改指标、提示生成、数据集适配器或任何影响 schema 输出的逻辑时，都应补充或更新测试。只有在确认回归结果需要变更时，才更新 `reference_results/`。

## 提交与 Pull Request 规范
当前工作区快照不包含 `.git`，因此这里无法读取本仓库的历史提交风格。建议使用简短的祈使句提交标题，并可选用 `feat:`、`fix:`、`docs:` 等前缀。提交 Pull Request 时，请概述行为变化，列出所需环境变量 `DATASET_ROOT`、`SAM2_REPO`、`ARTIFACT_ROOT`，并明确说明是否修改了 `configs/`、`docs/`、`reference_results/` 或生成产物。

## 配置建议
机器相关的数据集路径和模型检查点路径应通过环境变量配置，不要硬编码绝对路径。除非是有意保留的参考快照，否则不要提交临时性的 `artifacts/` 输出。
