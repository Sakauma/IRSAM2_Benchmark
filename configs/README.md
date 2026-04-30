# 配置索引

当前工程只保留完整 benchmark YAML 模板。

## 保留文件

- `server_benchmark_full.example.yaml`：完整配置模板。它同时定义路径、模型、方法、数据集、seed、batch、suite 和分析参数。
- `server_auto_prompt_4090x4.example.yaml`：SAM2-IR-QD M1 自动 prompt 训练和 E2 评估模板，默认使用 GPU `0,1,6,7`。当前默认是 no-RBGT 安全配置，只用 `NUAA-SIRST`、`NUDT-SIRST` 和 `IRSTD-1K` 训练 learned auto prompt，不扫描 `RBGT-Tiny`。
- `server_auto_prompt_4090x4_smoke.yaml`：服务器可直接运行的 M1 smoke 配置。它将训练压缩到 1 epoch/64 samples，并只评估 2 个数据集和 5 个代表性模式。

## 本地文件

复制模板后创建本地配置：

```bash
cp configs/server_benchmark_full.example.yaml configs/server_benchmark_full.local.yaml
cp configs/server_auto_prompt_4090x4.example.yaml configs/server_auto_prompt_4090x4.local.yaml
```

`configs/*.local.yaml` 是机器私有配置，被 `.gitignore` 排除，不应提交。

## 配置原则

- 路径写在 `paths`。
- 模型 checkpoint 写在 `models`。
- 方法和实际 baseline 映射写在 `methods`。
- 运行组合写在 `suites`。
- batch、seed、可视化、resume 行为写在 `runtime` 或 `smoke_test_runtime`。

不要再新增拆分式路径 YAML。常规轻量测试应通过主脚本参数实现，例如：

```bash
python scripts/run_5090_full_benchmark.py \
  --config configs/server_benchmark_full.local.yaml \
  --suites mask \
  --checkpoints tiny \
  --modes box \
  --smoke-test
```

自动 prompt 服务器任务使用：

```bash
python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.local.yaml \
  --stop-on-error \
  --preflight-mode fast \
  --train-amp
```

如果直接使用提交版 example，也可以运行：

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4.example.yaml \
  --stop-on-error \
  --preflight-mode fast \
  --train-amp \
  --rerun-train
```

这个 example 不会把 `RBGT-Tiny` 加入训练。后续如果 no-RBGT 结果证明方向成立，再单独启用 RBGT-Tiny 弱监督配置。

需要先验证 auto-prompt 训练、MultiModal 数据读取和 learned prompt E2 链路时，可以直接运行提交版 smoke 配置：

```bash
PYTHONPATH=src python scripts/run_4090x4_auto_prompt.py \
  --config configs/server_auto_prompt_4090x4_smoke.yaml \
  --stop-on-error
```

这个文件本身已经使用 smoke 规模，不需要再加 `--smoke-test`。
