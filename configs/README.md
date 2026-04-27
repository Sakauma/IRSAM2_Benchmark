# 配置索引

当前工程只保留完整 benchmark YAML 模板。

## 保留文件

- `server_benchmark_full.example.yaml`：完整配置模板。它同时定义路径、模型、方法、数据集、seed、batch、suite 和分析参数。

## 本地文件

复制模板后创建本地配置：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
```

`configs/*.local.yaml` 是机器私有配置，被 `.gitignore` 排除，不应提交。

## 配置原则

- 路径写在 `paths`。
- 模型 checkpoint 写在 `models`。
- 方法和实际 baseline 映射写在 `methods`。
- 运行组合写在 `suites`。
- batch、seed、可视化、resume 行为写在 `runtime` 或 `smoke_test_runtime`。

不要再新增拆分式路径 YAML 或单独 quick/smoke YAML。轻量测试应通过主脚本参数实现，例如：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny \
  --modes box \
  --smoke-test
```
