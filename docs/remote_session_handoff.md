# 远程会话交接说明

Author: Egor Izmaylov

## 用途

这个文件是给在远程 VS Code 窗口中开启的新 Codex 会话使用的交接说明。
如果远程 Codex 会话无法共享本地对话历史，请先阅读本文件，再根据下面记录的状态继续工作。

## 仓库状态

- 服务器上的仓库根目录：`/root/autodl-tmp/IRSAM2_Benchmark`
- 当时预期的最新提交：`2f05405 feat: add autodl bootstrap scripts`
- 旧仓库/旧工作流仅保留为历史参考；当前实际使用的是新的 package-based benchmark 平台
- 本地机器上的 benchmark 仓库当时位于：
  - `D:/workspace/sam_ir/ir_sam2_bench`

## 已实现内容

benchmark 平台本身已经可用。

已实现：

- 基于 package 的 benchmark 结构
- 数据集 adapter
- 通用 `images/ + masks/` 支持
- 面向 mask-only 数据集的 prompt synthesis
- baseline registry
- `bbox_rect`
- `sam2_pretrained_box_prompt`
- `sam2_pretrained_point_prompt`
- `sam2_pretrained_box_point_prompt`
- `sam2_no_prompt_auto_mask`
- `sam2_video_propagation` 接口
- 冻结的 artifact / report schema
- Linux 辅助脚本
- AutoDL 启动脚本

尚未完整实现为论文方法的部分：

- `adapt`
- `distill`
- `quantize`

这三个阶段目前只是稳定的 stage scaffold，还不是最终训练方法。

## AutoDL 服务器假设

当时租用服务器的目录约定为：

- `/root/autodl-tmp`：高速数据盘 / 运行盘
- `/root/autodl-fs`：持久化归档 / 存储盘

用户当时说明已经上传的数据集压缩包包括：

- `MultiModalCOCOClean.zip`
- `RBGT-Tiny.tar.gz`

预期路径为：

- `/root/autodl-fs/MultiModalCOCOClean.zip`
- `/root/autodl-fs/RBGT-Tiny.tar.gz`

用户当时已有：

- `/root/sam2`
- `/root/sam-hq`
- `/root/miniconda3`

## 推荐服务器目录布局

推荐使用以下运行路径：

- 仓库：`/root/autodl-tmp/IRSAM2_Benchmark`
- 数据集：`/root/autodl-tmp/datasets`
- 输出：`/root/autodl-tmp/runs`
- checkpoints：`/root/autodl-tmp/checkpoints`

不要把 `/` 作为主运行目录或输出目录，因为系统盘容量较小。

## 为 AutoDL 新增的辅助脚本

- `scripts/setup_autodl_server.sh`
- `scripts/run_autodl_smoke.sh`
- `configs/benchmark_smoke_rbgt_tiny.yaml`
- `configs/benchmark_v1_rbgt_tiny.yaml`

同时更新了：

- `scripts/run_baseline.sh`
- `scripts/run_tests.sh`
- `README.md`

## 服务器上的预期起步命令

如果仓库还没有 clone：

```bash
cd /root/autodl-tmp
git clone git@github.com:Sakauma/IRSAM2_Benchmark.git
cd IRSAM2_Benchmark
```

如果仓库已经存在：

```bash
cd /root/autodl-tmp/IRSAM2_Benchmark
git pull
```

然后初始化服务器工作区：

```bash
bash scripts/setup_autodl_server.sh
source .autodl_env.sh
```

接着运行 smoke baselines：

```bash
bash scripts/run_autodl_smoke.sh multimodal bbox_rect
bash scripts/run_autodl_smoke.sh multimodal sam2_pretrained_box_prompt
bash scripts/run_autodl_smoke.sh rbgt sam2_pretrained_box_prompt
```

如果需要指定 Python 解释器：

```bash
export PYTHON_BIN=/root/miniconda3/bin/python
```

也可以改成目标环境中的 `python` 路径。

## 接下来做什么

在服务器 bootstrap 能正常工作之后，建议按下面顺序继续：

1. 检查 `nvidia-smi`
2. 检查 Python 环境导入是否正常
3. 运行上面的 smoke baselines
4. 检查生成的 `artifacts/` 和 `reference_results/`
5. 只有这些都正常后，再进入正式 benchmark 运行

## 给下一次 Codex 会话的重要上下文

- 用户希望得到完整、论文级的 benchmark 平台
- `SAM2` baseline 支持是硬要求
- 通用 mask-only 数据集支持是硬要求
- 图像与序列/视频评估都必须支持
- `adapt / distill / quantize` 是下一阶段真正要补齐的方法实现
- 当时最紧迫的任务是把服务器环境准备好并跑通 smoke tests

## 给下一次会话的指令

如果你是新的远程 Codex 会话，请先做下面几件事：

1. 阅读本文件
2. 检查仓库是否位于提交 `2f05405`
3. 检查 `/root/autodl-fs/MultiModalCOCOClean.zip` 与 `/root/autodl-fs/RBGT-Tiny.tar.gz` 是否存在
4. 运行 AutoDL bootstrap 流程
5. 精确报告所有服务器侧错误
