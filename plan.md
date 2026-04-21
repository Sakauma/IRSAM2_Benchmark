# 红外 SAM2 迁移论文讨论计划

## 文档目标
本文件用于整理当前对话和 [research_experiment_code_summary.md](D:\workspace\sam_ir\pre_exp\deliverables\research_experiment_code_summary.md) 中已经明确的论文方向，作为后续讨论的统一底稿。

它的定位不是最终论文，也不是代码实现说明，而是一个围绕“论文目的、实验设计、实验蓝图、当前缺口和待决策问题”的讨论框架。后续讨论应优先在这个文件基础上增补，而不是分散到新的临时文档中。

## 论文目的
论文要解决的问题，不是单独证明 SAM2、蒸馏或量化中的某一个模块有效，而是围绕红外图像分割，建立一条从弱标注到可部署模型的完整链路：

`少量 box 标注 -> SAM2 生成/修正伪掩码 -> 红外域适配 -> 蒸馏到轻量学生模型 -> INT8 量化 -> 端侧部署验证`

当前论文目的可以概括为四个核心主张：

1. 少量 bounding box 标注可以显著降低 mask 标注成本，并作为红外分割的可行弱监督起点。
2. SAM2 在红外域不能简单依赖 zero-shot 使用，需要专门的红外域适配。
3. 迁移后的 teacher 不只是一个高精度模型，还应能为轻量 student 提供有效监督。
4. 经过蒸馏和量化后的 student，仍应保持可接受的分割质量、延迟和部署价值。

从当前材料判断，项目最强的部分是问题定义和实验设计，最弱的部分是证据链的完整性与实现强度。

## 研究问题与假设

### 核心研究问题
- `RQ1`：少量 box 标注能否在红外域产生足够高质量的 pseudo-mask。
- `RQ2`：红外域适配后的 SAM2 是否显著优于 `ZeroShotSAM2BoxPromptIR` 和常规轻量分割基线。
- `RQ3`：迁移后的 SAM2 能否被有效蒸馏到轻量学生模型。
- `RQ4`：量化后的 student 是否能在真实部署条件下保持可接受性能。
- `RQ5`：这套端到端框架是否具备独立创新点，而不是单纯模块拼装。
- `RQ6`：是否需要单独的 benchmark/protocol 设计，才能让研究问题本身成立。

### 当前主假设
#### H1. 红外分割的主瓶颈更可能是表征适配，而不是单纯 prompt 精修
对应比较：
- `ZeroShotSAM2BoxPromptIR`
- `CleanBoxPEFTSAM2Adapter`
- `NoisyBoxPromptRobustSAM2Adapter`

预期含义：
- 只做 prompt 工程而不做红外表征适配，收益有限。
- 先做红外 PEFT/LoRA 适配，再做 prompt robustness，效果应更稳定。

#### H2. box 标注是可行起点，但不一定是红外场景下最高性价比监督
预期含义：
- box-only 适合建立低门槛起点。
- 在相同标注成本下，混合少量更高信息密度监督可能更优。

注：这一假设在实验设计层已经提出，但在当前代码中尚未完全落地。

#### H3. SAM2 在红外中的最佳角色更可能是离线 teacher 和标注放大器，而不是被完整继承
对应比较：
- `FinalMaskDistilledIRStudent`
- `CorrectionTrajectoryDistilledIRStudent`
- 直接训练的 `DirectSupervisedIRSegFormerB0`
- 直接训练的 `DirectSupervisedIRPIDNetS`

预期含义：
- teacher 的价值可能主要体现在伪标签和 trajectory 信号。
- student 不一定需要完整继承 teacher 内部表示，也可能只需要继承最有部署价值的信息。

## 实验设计

### 数据与协议
实验协议围绕红外弱监督分割展开，设计包含四个部分：

- 少量 box 标注训练集
- 大量未标注图像池，用于 teacher/student 训练与 pseudo-label 生成
- 带 full-mask 的验证集和测试集
- 按设备来源、目标尺度、边界模糊程度进行分层分析

当前默认数据目录为：

- `/home/sakauma/dataset/MultiModal/img`
- `/home/sakauma/dataset/MultiModal/label`

当前代码从 JSON 标注中读取：

- `bbox`
- `mask polygon`
- `category`
- `device_source`

当前切分逻辑采用基于 `device_source` 的确定性划分，而不是随机图像级划分。这一点应继续保留，因为它更接近“跨设备泛化”这一真实问题设定。

### 指标
当前已围绕以下主指标组织：

- `mIoU`
- `BoundaryF1`
- `LatencyMs`

计划中还希望纳入以下指标，但当前未完整落地：

- `Dice`
- `ModelSize`
- `INT8AccuracyDrop`
- per-device / per-regime 结果分析

### 方法组
#### 基线
- `ZeroShotSAM2BoxPromptIR`
- `CleanBoxPEFTSAM2Adapter`
- `DirectSupervisedIRSegFormerB0`
- `DirectSupervisedIRPIDNetS`

#### 提出方法
- `NoisyBoxPromptRobustSAM2Adapter`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
- `FinalMaskDistilledIRStudent`
- `CorrectionTrajectoryDistilledIRStudent`
- `CorrectionTrajectoryDistilledINT8SegFormerB0`
- `FinalMaskDistilledINT8SegFormerB0`
- `CorrectionTrajectoryDistilledINT8PIDNetS`

#### 消融
- 去掉 noisy-box augmentation
- 只保留 jitter，不保留 offset / truncation
- 去掉 IR-specific pseudo-mask quality filter
- trajectory distillation 只保留 final state
- 打乱 trajectory 顺序
- 去掉 auxiliary trajectory head
- 只做 PTQ，不做量化恢复

当前实验矩阵的优点在于，它已经是“假设驱动的对照设计”，而不是简单堆模型试验。

## 实验蓝图
如果后续实现与证据链能够补齐，论文叙事应按如下顺序组织：

1. 证明 `ZeroShotSAM2BoxPromptIR` 在红外场景下存在明显域差距。
2. 证明红外适配和 prompt robustness 都有收益，但前者更关键。
3. 证明伪标签质量控制能够提高自训练稳定性，而不是单纯扩大伪标签规模。
4. 证明 `CorrectionTrajectoryDistilledIRStudent` 比单纯 `FinalMaskDistilledIRStudent` 更适合红外 student。
5. 证明 INT8 量化后的 student 仍具有真实部署价值，并给出精度、延迟、内存和模型体积之间的权衡。

对应到论文贡献表达，当前最可行的蓝图有两种：

- 路线 A：强调“少量 box 标注 + 红外适配 + 蒸馏 + 量化部署”的完整弱监督链路。
- 路线 B：强调“correction trajectory distillation”作为方法创新点，完整链路作为应用背景。

后续讨论需要明确主贡献到底押在哪一条路线，避免实验做得很散但叙事不聚焦。

## 当前实现与蓝图的差距
当前代码已经提供了研究骨架，但还不足以形成可直接支撑发表的证据链。主要差距有五项：

1. 当前所谓的 `CleanBoxPEFTSAM2Adapter` 更像对 `teacher.predict()` 输出结果做轻量修正，而不是真正的 SAM2 内部 LoRA/PEFT。
2. `QualityFilteredPseudoMaskSelfTrainingSAM2` 在概念上成立，但 pseudo-label 没有完整并入训练闭环，分支未真正跑通。
3. 当前量化更接近 fake quant 占位实现，不足以支撑强部署结论。
4. 当前训练规模偏小，更接近 smoke-test 或原型验证，而不是科研级训练。
5. checkpoint 选择和验证协议未完整实现，当前结果不足以作为严格论文证据。

因此，现阶段的关键不是继续扩展想法，而是补齐最核心的证据链。

## 后续讨论主轴
后续讨论优先围绕以下三个问题展开：

1. 论文主贡献应押在“完整弱监督部署链路”，还是押在“trajectory distillation”这一方法点上。
2. 当前蓝图里哪些实验必须补齐，哪些可以删减，以形成最小可发表版本。
3. protocol / benchmark 是否需要上升为论文贡献的一部分，而不是仅作为实验设置说明。

在这三个问题明确之前，不建议继续扩展新的方法分支。

## 当前结论
一句话总结：当前项目最强的是研究问题和实验蓝图，不是最终结果。

更准确地说，项目已经具备论文应有的问题定义、假设结构和实验矩阵，但现有代码仍更像“研究原型”，而不是“可直接支撑发表的证据实现”。后续工作的重点应是把关键方法链做实、做严、做可复现。

## 讨论记录

### 记录 01：现有实验代码能否解决问题
结论：目前不能说“已经解决问题”，只能说“已经形成了可用于方向筛选的研究原型”。

判断依据：

- 现有 `CleanBoxPEFTSAM2Adapter` 不是严格意义上的 SAM2 内部 LoRA/PEFT，更像 teacher 输出后的轻量修正。
- `QualityFilteredPseudoMaskSelfTrainingSAM2` 没有形成完整的 pseudo-label 训练闭环。
- 量化实现仍偏 fake quant 占位，不足以支撑强部署结论。
- 训练规模偏 prototype/smoke-test，而不是科研级训练。
- 数据监督仍实际依赖 polygon mask，和“少量 box 标注”叙事之间还有落差。
- checkpoint 选择与验证协议未完整落地。

这一轮讨论的核心结论是：当前代码能支撑“方向值得继续”，但还不能支撑“论文问题已经被解决”。

### 记录 02：第一部分讨论，哪些分支已经足够说明方向是对的
当前最能说明“方向是对的”的，不是整条链路，而是其中少数几个局部方向。

#### 1. `ZeroShotSAM2BoxPromptIR` 作为起点是成立的
从已有结果看，`ZeroShotSAM2BoxPromptIR` 能在红外场景上给出非零且不低的分割表现，说明“用可提示式 foundation model 作为红外弱监督起点”这个大方向并不是空想。

它当前的意义主要有两点：

- 证明 SAM2 在红外数据上不是完全失效，具备成为 teacher 或伪标签生成器的潜力。
- 同时也暴露出明显缺点，尤其是延迟高、直接部署价值弱，因此后续的适配、蒸馏和量化路线是有必要的。

换句话说，`ZeroShotSAM2BoxPromptIR` 不是论文答案，但它证明了问题入口选对了。

#### 2. `CleanBoxPEFTSAM2Adapter` 说明“围绕 SAM2 做适配”是值得继续的
在当前结果中，`CleanBoxPEFTSAM2Adapter` 是少数表现稳定、数值合理的分支之一。它比单纯 zero-shot 更像是在说明：围绕 SAM2 输出做针对红外场景的适配，确实可能带来收益。

但这里必须保留边界：

- 这只能支持“红外域适配有价值”这一较弱结论。
- 它还不能支持“真正的 SAM2 内部 PEFT 已被验证有效”，因为当前实现不是那个层级。

因此，这个分支说明的不是“方法已经做对了”，而是“论文主线里关于红外适配的方向是对的”。

#### 3. `device_source` 分层切分思路是对的
虽然这不是单个模型分支，但它是当前实验协议中最值得保留的设计之一。

它的价值在于：

- 避免随机图像切分带来的乐观偏差。
- 更接近真实跨设备泛化问题。
- 能让后续论文把“部署泛化”讲得更可信。

这部分我认为不是临时工程技巧，而是应保留到最终论文里的 protocol 资产。

#### 4. 哪些方向目前还不能算“已经说明是对的”
以下几个方向目前还没有足够正证据：

- `NoisyBoxPromptRobustSAM2Adapter`：当前结果不稳定，不能证明 prompt robustness 路线已经成立。
- `QualityFilteredPseudoMaskSelfTrainingSAM2`：概念有吸引力，但训练闭环没打通，当前结果不能作为强证据。
- `FinalMaskDistilledIRStudent` / `CorrectionTrajectoryDistilledIRStudent`：多个结果掉到 `mIoU = 0.0`，说明蒸馏链路还远未稳定。
- 各类 INT8 分支：目前更像流程占位验证，不能说明真实部署路线已经跑通。

#### 当前阶段的明确判断
如果只回答“哪些部分已经足够说明方向是对的”，我会保留三点：

1. 用 `ZeroShotSAM2BoxPromptIR` 作为红外弱监督入口是对的。
2. 围绕 SAM2 做红外适配是对的，但当前实现还只是弱版本验证。
3. 用 `device_source` 做分层协议是对的，而且值得上升为论文协议设计资产。

而蒸馏、伪标签闭环、量化部署这三条方法链，目前还处在“值得做，但尚未被现有代码证明”的阶段。

### 记录 03：当前讨论重心切换
当前讨论重心已经明确调整为：

- 先讨论实验如何设计。
- 先讨论实验代码应该如何写。
- 先把实验链路跑通，并得到有效结果。
- 论文主贡献、论文叙事和最终写法暂时后置。

这意味着后续的优先级排序应改为：

1. 先保证实验问题定义能被当前代码真实实现。
2. 先保证最小实验闭环可运行、可复现、可比较。
3. 先拿到可信的正反结果，再决定哪些内容值得进入论文主线。

对应地，后续讨论不再优先围绕“论文应该怎么讲”，而应优先围绕：

- 哪些实验必须先做
- 哪些方法链必须先跑通
- 代码结构如何改才能支撑这些实验

### 记录 04：开始执行后的重构策略
当前已经开始进入代码执行阶段，并明确采用如下重构策略：

- 允许对现有实验代码进行完全重构。
- 第一轮重构不再试图同时维护完整链路，而是聚焦最小有效实验闭环。
- 本轮优先实现的方向是：
  - teacher 基线与 teacher adaptation
  - pseudo-label 生成与自训练闭环
  - 统一的训练 / 验证 / 测试 runner
  - best validation checkpoint 选择
- 本轮显式后置的方向是：
  - distillation
  - INT8 quantization
  - 更完整的部署链路

当前执行原则变为：

1. 先把第一批实验跑通。
2. 先保证第一批实验结果可信。
3. 再在稳定 teacher 基础上扩到蒸馏与量化。

### 记录 05：WSL / conda / 数据集约束
当前实验运行环境的额外约束已经明确如下：

- 数据集和 conda 虚拟环境都位于 WSL2 的 Ubuntu 中，即 `/home/sakauma`。
- Windows 工作区 `D:\workspace\sam_ir` 在 WSL 中对应路径是 `/mnt/d/workspace/sam_ir`。
- 如果冒烟测试涉及 SAM2 相关代码，推荐使用 `sam_hq2` 环境运行。
- 当前已确认 `sam_hq2` 环境可用，Python 版本为 3.10。
- 当前推荐优先使用的两个数据集是：
  - `MultiModal`
  - `RBGT-Tiny`

这些约束直接影响实验代码设计：

- 实验代码必须支持从 WSL 路径读取数据，而不是只假设 Windows 本地 Python 环境。
- loader 必须支持按数据集名称分派，而不是把所有数据集都假定为同一种目录结构。
- 后续 smoke test 与 SAM2 调试应默认走 `sam_hq2` 环境。

### 记录 06：多目标数据集对 loader 的影响
当前又明确了一条会直接影响代码正确性的事实：

- `MultiModal` 是单张图像包含多个目标实例的数据集。
- `RBGT-Tiny` 也是多目标数据集，并且目录结构与 `MultiModal` 不同。

这意味着旧代码中的以下行为是错误的：

- 读取一张图后只保留第一个 instance。
- 假定所有数据都来自 `img/ + label/*.json` 结构。

因此，重构后的 loader 需要满足：

1. 对 `MultiModal`，每个 instance 都要成为一个独立样本，而不能在读到第一个实例后 `break`。
2. 对 `RBGT-Tiny`，需要按其 COCO / 图像目录结构单独实现读取逻辑。
3. 多目标图像应在 sample 级别展开为“同图多实例样本”，以便继续支持 box-prompt teacher 和实例级训练目标。

### 记录 07：WSL 后端调用已验证
当前已经完成以下运行环境验证：

- WSL 发行版：`ubuntu2004`
- Windows 工作区映射路径：`/mnt/d/workspace/sam_ir`
- SAM2 相关 smoke test 推荐环境：`sam_hq2`
- `sam_hq2` 已确认可用，Python 版本为 3.10

已经确认可以通过如下方式在 WSL 中执行实验代码：

- `wsl.exe -d ubuntu2004 bash -lc "..."`
- 并在内部使用 `/home/sakauma/data/miniconda3/envs/sam_hq2/bin/python` 或 `conda run -p /home/sakauma/data/miniconda3/envs/sam_hq2 ...`

这意味着后续实验执行策略已经明确：

1. Windows 下编辑代码
2. WSL / Ubuntu 下运行实验
3. SAM2 相关 smoke test 统一走 `sam_hq2`

### 记录 08：第一轮运行级验证结果
当前已经完成以下最小验证：

- `MultiModal` loader 已确认能读取多实例样本
- `RBGT-Tiny` loader 已确认能读取 COCO 风格多实例样本
- `ZeroShotSAM2BoxPromptIR` 已在 `MultiModal + sam_hq2 + WSL` 条件下完成一次 smoke test

当前这轮 smoke test 至少说明三件事：

1. 新的 runner 主流程是可执行的
2. WSL + `sam_hq2` + SAM2 的调用链已打通
3. 当前重构后的最小 teacher 基线可以开始进入真实调试阶段

### 记录 09：路径与环境配置抽象化
在进入下一轮实验前，已经专门检查并修正了“路径是否仍绑定本机环境”的问题。

当前结论：

- 运行时代码中的 `/home/sakauma` 等机器相关默认路径已经从实验代码中移除。
- 实验代码现在采用“环境变量优先 + 自动校验 + 清晰报错”的方式解析路径。
- 服务器部署时，不再需要依赖本机目录结构，只需要提供必要环境变量。

当前代码层面的运行配置原则为：

- `DATASET_ROOT`：指向包含 `MultiModal`、`RBGT-Tiny` 等数据集的根目录
- `DATASET_NAME`：指定当前运行的数据集
- `SAM2_REPO`：指向本地 SAM2 仓库
- `SAM2_CKPT`：指向具体 checkpoint 文件
- `SAM2_CFG`：指定 SAM2 配置文件

补充说明：

- `EXPERIMENT_PLAN.yaml` 和讨论文档中仍可能保留 `/home/sakauma` 路径作为历史记录或计划上下文。
- 但运行逻辑本身已经不再依赖这些硬编码默认值。

### 记录 10：MultiModal 全量 smoke 重跑后定位到 teacher 分支塌缩根因
在 `sam_hq2 + WSL + MultiModal` 条件下，已重新跑完一轮完整 smoke（`outputs_smoke_multimodal_full_teacher_pseudo_v3`）。

这一轮结果表明：

- `ZeroShotSAM2BoxPromptIR` 仍保持高测试表现，说明 SAM2 入口本身没有问题。
- 几乎所有 trainable teacher 分支都出现 `mIoU = 0.0 / BoundaryF1 = 0.0`，可视化显示预测塌成全背景。
- `pseudo_accept_count` 已经不是 0，说明 split 和 unlabeled pool 已经打通，问题不再是数据切分。

由此确认，当前主瓶颈不在 runner / split，而在 trainable teacher 方法内部的监督与前向设计。

### 记录 11：为避免全背景塌缩，对 teacher adaptation 与 pseudo-label 逻辑做了两处关键修正
为解决上述问题，已完成以下重构：

1. `PromptConditionedMaskAdapter` 改为 residual refinement 结构。
   - 适配器不再从随机 mask logits 直接起步。
   - 当前输出形式为 `teacher_logits + residual`。
   - residual head 以零初始化，使初始行为退化为 zero-shot teacher，而不是随机输出。

2. `QualityFilteredPseudoMaskSelfTrainingSAM2.generate_pseudo_samples()` 改为使用 warmup 后的 adapted predictor 生成 pseudo mask。
   - 旧逻辑错误地继续调用原始 `self.teacher.predict()` 生成伪标签。
   - 新逻辑先用当前 adapted method 的 `predict()` 生成 `pseudo_prob / pseudo_mask`。
   - 质量打分增加了 adapted mask 与 base teacher mask 的 agreement，以及 mask 内部平均置信度。

附带修正：

- 所有 trainable segmentation loss 的 BCE 改为 class-balanced BCE，以减轻小目标前景在极小样本下被背景压制的问题。

### 记录 12：修正后重新 smoke，teacher 主线已经恢复稳定
在完成上述修改后，已重新跑完一轮新的完整 smoke（`outputs_smoke_multimodal_full_teacher_pseudo_v4`）。

这轮结果出现了明显改善：

- `CleanBoxPEFTSAM2Adapter`
  - `best_val_mIoU = 0.9627`
  - `test mIoU = 0.9873`
  - `BoundaryF1 = 0.7687`
- `NoisyBoxPromptRobustSAM2Adapter`
  - `best_val_mIoU = 0.9627`
  - `test mIoU = 0.9873`
  - `BoundaryF1 = 0.7687`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
  - `pseudo_accept_count = 8`
  - `test mIoU = 0.9873`
  - `BoundaryF1 = 0.7687`
- `DirectSupervisedIRSegFormerB0`
  - `test mIoU = 0.2986`

当前阶段性判断：

- teacher adaptation 主线已经从“不可用的塌缩状态”恢复为“可运行且优于 direct student baseline”的状态。
- 当前 smoke 里，prompt robustness 和 pseudo-label quality filter 还没有真正拉开差距，因为多个 teacher 变体结果几乎一致。
- 这说明下一阶段重点不再是“先救活 trainable teacher”，而是“让不同 teacher 条件在更真实设置下产生可解释差异”。

### 记录 13：结果记录层面的一个补充修正
当前又补了一处 bookkeeping 修正：

- `ZeroShotSAM2BoxPromptIR` 之前在 `results.json` 中的 `best_val_mIoU = 0.0` 不是模型真实性能，而是 non-trainable 分支没有走统一的验证入口。
- 已在 runner 中统一改为：无论 trainable 与否，都走 `train_method()` 内的验证记录逻辑。

这保证后续导出的 `results.json / summary.json` 不会再对 zero-shot condition 记录错误的验证指标。

### 记录 14：`RBGT-Tiny` 跨数据集 smoke 已验证通过
在完成上述修正后，已进一步在第二个推荐数据集 `RBGT-Tiny` 上跑完同一套 teacher + pseudo smoke（`outputs_smoke_rbgt_tiny_full_teacher_pseudo_v1`）。

这一轮的关键信息如下：

- `ZeroShotSAM2BoxPromptIR`
  - `best_val_mIoU = 0.6540`
  - `test mIoU = 0.6053`
  - `BoundaryF1 = 0.5630`
- `CleanBoxPEFTSAM2Adapter`
  - `best_val_mIoU = 0.6549`
  - `test mIoU = 0.6581`
  - `BoundaryF1 = 0.6410`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
  - `pseudo_accept_count = 11`
  - `test mIoU = 0.6581`
  - `BoundaryF1 = 0.6410`
- `DirectSupervisedIRSegFormerB0`
  - `test mIoU = 0.0002`

当前结论：

- 重构后的实验骨架并不只在 `MultiModal` 单点成立，已在 `RBGT-Tiny` 上完成第二次运行级验证。
- SAM2 teacher 路线在 `RBGT-Tiny` 上仍显著优于 direct supervised lightweight baseline。
- 与 `MultiModal` 相比，`RBGT-Tiny` 上的 zero-shot 起点更弱，因此 teacher adaptation 的增益更有可解释性。
- 但当前 smoke 仍未把 `Clean / Noisy / Jitter / Pseudo-filtered` 这些 teacher 变体真正区分开，多个条件仍收敛到几乎相同的测试指标。

因此，当前阶段的重点已经进一步收敛为：

1. 不是继续证明“代码能否运行”，因为这件事已经在两个数据集上被验证。
2. 而是要设计下一轮更真实的训练与评估设置，让不同 teacher 条件能够拉开差异。

### 记录 15：运行配置层已加固到适合服务器批量实验
在跨数据集 smoke 通过后，又继续完善了运行配置层，使后续服务器实验不需要再改源码切规模。

本轮新增并验证的点包括：

- 修复 `config.py` 中路径校验写在 `return` 之后、实际上未执行的问题。
- 新增环境变量覆盖能力：
  - `EXPERIMENT_SEEDS`
  - `SUPERVISION_BUDGETS`
  - `MAX_SAMPLES`
  - `TRAIN_EPOCHS`
  - `PSEUDO_FINETUNE_EPOCHS`
  - `BATCH_SIZE`
  - `EVAL_LIMIT`
  - `NUM_WORKERS`
  - `PSEUDO_QUALITY_THRESHOLD`
  - `PSEUDO_SCORE_THRESHOLD`
  - `LR_TEACHER_ADAPTER`
  - `LR_SEGFORMER`
- `MAX_SAMPLES <= 0` 现已表示不限制样本数。
- `EVAL_LIMIT <= 0` 现已表示不截断验证 / 测试集。

已额外完成一次最小配置验证：

- 条件：`MultiModal + ZeroShotSAM2BoxPromptIR`
- 启动方式：完全通过环境变量指定 `EXPERIMENT_CONDITIONS / EXPERIMENT_SEEDS / SUPERVISION_BUDGETS / MAX_SAMPLES / EVAL_LIMIT`
- 结果：运行成功，`best_val_mIoU = 0.9639`

当前含义非常明确：

- 代码现在不仅能在本机 smoke test 中跑通，也已经具备“服务器上通过环境变量切实验规模和条件”的基础能力。
- 后续如果要开始正式实验，重点可以转向“怎么设 full run protocol”，而不是继续修改基础设施。

### 记录 16：已补充一份“实验代码概览”文档用于正式实验前对齐
在开始正式实验前，又新增了一份独立说明文档：

- `pre_exp/stage-11/experiment/EXPERIMENT_CODE_OVERVIEW.md`

这份文档的目的不是记录结果，而是帮助在开跑前快速对齐：

- 当前实验代码的目标是什么
- 代码目录如何拆分
- 每个核心模块分别负责什么
- 当前支持哪些实验条件
- runner 的基本执行流程是什么
- 现阶段代码的能力边界在哪里

当前意义：

- 后续无论是在本机还是服务器上启动正式实验，都可以先看这份文档，再决定具体运行配置。
- 这份文档也可以作为后续继续重构时的“当前系统说明”，避免代码结构变化后只剩结果、不剩结构说明。

### 记录 17：已新增“正式实验启动清单”文档
为了从 smoke 阶段进入服务器 full run，又新增了一份执行导向文档：

- `pre_exp/stage-11/experiment/FORMAL_EXPERIMENT_RUNBOOK.md`

这份文档的作用是把当前最合理的正式实验顺序固定下来，而不是每次重新讨论一遍。当前 runbook 明确规定：

1. 第一轮先跑 `RBGT-Tiny` 的 teacher-only full run
2. 第二轮再跑 `MultiModal` 的 teacher-only full run
3. 第三轮再进入 pseudo-label 对照

文档中已经写清：

- 服务器运行前的必要环境变量
- 推荐的 `EXPERIMENT_CONDITIONS`
- 推荐的 `EXPERIMENT_SEEDS`、`SUPERVISION_BUDGETS`
- `MAX_SAMPLES=0` 和 `EVAL_LIMIT=0` 的全量运行方式
- 每轮实验后的检查项与验收标准

当前意义：

- 后续如果开始正式跑服务器实验，默认就以这份 runbook 为直接执行基线。
- 后续讨论重点不再是“先跑哪条线”，而是“第一轮 full run 的结果是否足以区分 teacher 条件”。

### 记录 18：已落地服务器正式实验启动脚本
在 runbook 之外，当前又把推荐实验顺序进一步落地成了实际可执行的 Bash 脚本，位置如下：

- `pre_exp/stage-11/experiment/scripts/common.sh`
- `pre_exp/stage-11/experiment/scripts/run_rbgt_teacher.sh`
- `pre_exp/stage-11/experiment/scripts/run_multimodal_teacher.sh`
- `pre_exp/stage-11/experiment/scripts/run_rbgt_pseudo.sh`

这些脚本的作用是把当前 runbook 中最常用的正式实验命令固定下来，避免服务器上反复手写长命令。

当前脚本设计原则：

- 外部必须提供 `DATASET_ROOT` 和 `SAM2_REPO`
- 运行解释器默认使用 `PYTHON_BIN=python`，也允许外部覆盖
- 训练相关参数如 `EXPERIMENT_SEEDS`、`SUPERVISION_BUDGETS`、`TRAIN_EPOCHS`、`MAX_SAMPLES` 等都允许通过环境变量覆盖
- 默认输出落在 `formal_runs/` 下，但也允许外部覆盖 `OUTPUT_DIR` 或 `OUTPUT_ROOT`

已完成的验证：

- 通过 WSL `bash -n` 对上述脚本做了静态语法检查，当前语法通过

同时，`FORMAL_EXPERIMENT_RUNBOOK.md` 与 `EXPERIMENT_CODE_OVERVIEW.md` 也已同步补充脚本启动方式，后续可以直接：

- 手写完整环境变量命令运行
- 或者仅提供关键路径后直接调用现成脚本运行

### 记录 19：已在仓库根目录建立集中打包目录
为便于后续统一查看、迁移和拷贝，当前又把实验相关代码与文档集中复制到仓库根目录的新目录：

- `ir_sam2_exp/`

当前目录中包含：

- `main.py`
- `experiment_core/`
- `scripts/`
- `EXPERIMENT_PLAN.yaml`
- `EXPERIMENT_CODE_OVERVIEW.md`
- `FORMAL_EXPERIMENT_RUNBOOK.md`
- `plan.md`
- 所有现有 `outputs*` 实验输出目录

本次操作是“复制集中”，不是“移动替换”：

- 原始文件仍保留在 `pre_exp/stage-11/experiment/`
- 根目录下的新目录用于集中存放和后续迁移，不影响当前已有代码路径

补充说明：

- 打包目录中已移除 `__pycache__`
- 这样后续如果要整体复制到服务器或单独打包归档，直接围绕 `ir_sam2_exp/` 操作即可

### 记录 20：新目录已完成运行级迁移验证
当前又完成了一步关键确认：`ir_sam2_exp/` 已经不是单纯的拷贝目录，而是可直接运行的主实验目录。

已完成的验证：

- 从 `ir_sam2_exp/` 目录本身启动了一次 WSL + `sam_hq2` 的 smoke run
- 条件为 `ZeroShotSAM2BoxPromptIR`
- 输出写入：
  - `ir_sam2_exp/outputs_smoke_bundle_migration_check_v1`
- 运行成功，结果摘要：
  - `best_val_mIoU = 0.9639`

由此可确认：

- `main.py`、`experiment_core/`、`scripts/`、默认相对路径和输出目录都已经能在新目录下自洽运行
- 后续应默认把 `ir_sam2_exp/` 视为主运行目录
- `pre_exp/stage-11/experiment/` 可以保留为历史快照，但不再应作为主要实验入口

配套补充：

- 已在 `ir_sam2_exp/` 下新增 `README.md`

### 记录 21：根目录实验主目录重命名为更短路径
为避免路径过长，根目录实验主目录已从较长名称进一步收短为：

- `ir_sam2_exp/`

重命名原则：

- 全英文
- 不含中文
- 尽可能短
- 仍保留实际语义，表示 “infrared + sam2 + experiment”

后续默认统一使用 `ir_sam2_exp/` 作为主运行目录。

### 记录 22：已开始在笔记本上使用 `MultiModal` 做真实实验
当前已明确：用户提到的“小数据集”就是 `MultiModal`，实际图片数为 `665`。

基于当前硬件约束：

- 设备：8GB 显存的 4060M
- 环境：Ubuntu + conda + `sam_hq2`

已决定先在这台笔记本上跑“小而真”的实验，而不是直接上完整 full matrix。

当前策略：

- 优先用 `MultiModal`
- 先跑 teacher 相关主线
- 再补 pseudo-label 对照
- 所有运行统一从 `ir_sam2_exp/` 启动

### 记录 23：`MultiModal` 笔记本 teacher 线第一轮正式实验结果
已完成第一轮笔记本友好型正式实验，输出目录：

- `ir_sam2_exp/formal_runs/multimodal_laptop_teacher_v1`

运行配置：

- dataset: `MultiModal`
- 图片规模：全量 `665` 张
- conditions:
  - `ZeroShotSAM2BoxPromptIR`
  - `CleanBoxPEFTSAM2Adapter`
  - `NoisyBoxPromptRobustSAM2Adapter`
  - `DirectSupervisedIRSegFormerB0`
- `seed = 42`
- `budget = 0.1`
- `TRAIN_EPOCHS = 4`
- `BATCH_SIZE = 1`

关键结果：

- `ZeroShotSAM2BoxPromptIR`
  - `best_val_mIoU = 0.6717`
  - `test mIoU = 0.6631`
  - `BoundaryF1 = 0.3184`
- `CleanBoxPEFTSAM2Adapter`
  - `best_val_mIoU = 0.6869`
  - `test mIoU = 0.6714`
  - `BoundaryF1 = 0.3147`
- `NoisyBoxPromptRobustSAM2Adapter`
  - `best_val_mIoU = 0.6846`
  - `test mIoU = 0.6699`
  - `BoundaryF1 = 0.3176`
- `DirectSupervisedIRSegFormerB0`
  - `best_val_mIoU = 0.0419`
  - `test mIoU = 0.0314`

阶段性判断：

- 在 `MultiModal` 上，SAM2 teacher 路线明显优于 direct supervised lightweight baseline。
- `CleanBoxPEFT` 相比 zero-shot 有小幅但真实的提升。
- `Noisy` 与 `Clean` 非常接近，说明在当前设置下 prompt robustness 的增益尚未被明显拉开。
- 这一轮已经证明：8GB 4060M 可以完成小数据集上的单 seed teacher 正式实验。

### 记录 24：`MultiModal` 笔记本 pseudo-label 对照结果
在 teacher 线之后，又完成了 pseudo-label 对照，输出目录：

- `ir_sam2_exp/formal_runs/multimodal_laptop_pseudo_v1`

运行配置：

- dataset: `MultiModal`
- conditions:
  - `NoisyBoxPromptRobustSAM2Adapter`
  - `QualityFilteredPseudoMaskSelfTrainingSAM2`
  - `PseudoMaskSelfTrainingWithoutIRQualityFilter`
- `seed = 42`
- `budget = 0.1`
- `TRAIN_EPOCHS = 4`
- `PSEUDO_FINETUNE_EPOCHS = 4`

关键结果：

- `NoisyBoxPromptRobustSAM2Adapter`
  - `best_val_mIoU = 0.6845`
  - `test mIoU = 0.6698`
  - `BoundaryF1 = 0.3174`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
  - `pseudo_accept_count = 1307`
  - `best_val_mIoU = 0.6725`
  - `test mIoU = 0.6568`
  - `BoundaryF1 = 0.2865`
- `PseudoMaskSelfTrainingWithoutIRQualityFilter`
  - `pseudo_accept_count = 1464`
  - `best_val_mIoU = 0.6740`
  - `test mIoU = 0.6615`
  - `BoundaryF1 = 0.2991`

阶段性判断：

- 在当前 `MultiModal + budget 0.1 + seed 42` 设置下，pseudo-label 并没有优于单纯的 `NoisyBoxPromptRobustSAM2Adapter`。
- `quality filter` 的确减少了接收的伪标签数量（`1307` vs `1464`），但当前没有转化为更好的测试表现。
- 无过滤版本比过滤版本略好，但两者都低于 `Noisy` 本身。

工程判断：

- pseudo 分支在笔记本上是能跑通的，但耗时明显更长，不适合作为第一优先级的交互式调参对象。
- 当前更值得继续在笔记本上做的，是 teacher 线的 budget / seed 扩展，而不是继续深挖 pseudo。

### 记录 25：`MultiModal` 与 `RBGT-Tiny` 的标注协议不同，不能混成同一类结论
当前需要显式澄清一个关键问题：

- `MultiModal` 在当前使用方式下是 mask 监督数据
- `RBGT-Tiny` 在当前研究讨论中应视为 box-only 协议数据

这意味着：

- 两个数据集不能直接混在一起当成“同一种训练监督”来解释
- 也不能直接把跨数据集结果拼起来，写成统一的弱监督结论

当前代码层面的事实是：

- `MultiModal` loader 会读取 polygon mask
- trainable teacher 方法训练时显式使用 `batch["masks"]` 做 BCE + Dice

因此，当前已经跑出的 `MultiModal` teacher 实验，更准确地说是在回答：

- “在 `MultiModal` 的 mask 监督条件下，SAM2 adaptation 相比 zero-shot 和 direct baseline 是否有效”

而不是在回答：

- “只用 box supervision 是否已经足以支撑整条方法链”

这条记录会直接影响后续实验分层：

1. `MultiModal` 结果应作为 mask-supervised / fully-supervised adaptation 证据
2. box-only 问题需要单独定义协议，不能默认沿用 `MultiModal` 当前训练方式
3. 后续如果要坚持论文主线中的弱监督叙事，就必须明确哪些实验是 mask-supervised，哪些实验是 box-only
- 已在新目录内的 `EXPERIMENT_CODE_OVERVIEW.md` 与 `FORMAL_EXPERIMENT_RUNBOOK.md` 中显式注明“新目录为主运行目录”

### 记录 26：已导出 `MultiModal` 的 COCO 对齐版本
围绕“让 `MultiModal` 与 `RBGT-Tiny` 对齐格式”这一点，当前已经完成两项工作：

1. 在代码中新增了 `scripts/export_multimodal_coco.py`
2. 已经实际导出一份对齐后的数据到：
   - `ir_sam2_exp/data_prep/MultiModalCOCO`

当前导出结果：

- images: `665`
- annotations: `9263`
- categories: `324`
- annotation json:
  - `ir_sam2_exp/data_prep/MultiModalCOCO/annotations_coco/instances_multimodal_train2017.json`

当前导出策略：

- 保留原始类别信息
- 使用 polygon 生成轴对齐紧致 bbox
- 同时保留 `segmentation` 字段，使导出结果在格式上更接近 `RBGT-Tiny` 当前的 COCO 结构

当前工程判断：

- 对于现有 pipeline，轴对齐 bbox 比旋转最小外接矩形更合适，因为 SAM2 box prompt 和 COCO `bbox=[x,y,w,h]` 都是轴对齐框
- 这一步已经完成了“数据格式对齐”，但还没有自动完成“监督协议对齐”

### 记录 27：`MultiModal` 原始类别体系很脏，后续需要类别归一化
在导出 `MultiModalCOCO` 后，又发现一个新问题：

- 虽然导出时保留了原始类别信息，但当前类别总数高达 `324`
- 高频类别中出现了大量不适合直接用于稳定目标类别实验的标签，例如：
  - `The building`
  - `trees`
  - `road`
  - `The sky`
  - `The clouds`
  - `position`
  - `The date`

这说明：

- 当前 `MultiModal` 的 `category` 字段混入了大量场景元素、叙述性文本和非目标实体
- 如果后续要把“保留种类信息”真正用于目标检测 / 分割协议对齐，必须先做一次类别归一化或目标类别筛选

因此，当前对齐问题已经拆成两层：

1. 格式对齐：已完成
2. 类别语义对齐：尚未完成，后续需要单独处理

### 记录 28：已完成第一版类别清洗，生成 `MultiModalCOCOClean`
在格式对齐之后，当前又完成了第一版类别语义清洗：

- 新增脚本：
  - `ir_sam2_exp/scripts/clean_multimodal_coco_categories.py`
- 输出目录：
  - `ir_sam2_exp/data_prep/MultiModalCOCOClean`

清洗策略：

- 目标类别体系直接对齐 `RBGT-Tiny`：
  - `ship`
  - `car`
  - `cyclist`
  - `pedestrian`
  - `bus`
  - `drone`
  - `plane`
- 仅保留能高置信映射到上述类别的原始标签
- 其余场景类、背景类、描述性文本类和不可靠类别全部过滤

清洗后结果：

- kept_images: `477`
- kept_annotations: `1654`
- mapped counts:
  - `car: 937`
  - `drone: 283`
  - `pedestrian: 189`
  - `ship: 128`
  - `plane: 104`
  - `bus: 13`
  - `cyclist: 0`

关键判断：

- 这份 `MultiModalCOCOClean` 已经明显比原始 `MultiModalCOCO` 更适合做与 `RBGT-Tiny` 的类别对齐实验
- 当前清洗是“高精度优先”，宁可过滤掉不确定标签，也不盲目扩大映射范围
- `cyclist` 目前没有可靠样本，说明这类在 `MultiModal` 里暂时缺失或原始标签不足以稳定映射

配套文件：

- 清洗后的注释文件：
  - `ir_sam2_exp/data_prep/MultiModalCOCOClean/annotations_coco/instances_multimodal_clean_train2017.json`
- 清洗报告：
  - `ir_sam2_exp/data_prep/MultiModalCOCOClean/annotations_coco/category_cleaning_report.json`

### 记录 29：当前改动还停留在数据准备层，核心实验协议尚未切换
需要明确澄清：

- 当前已完成的是：
  - `MultiModal -> COCO` 格式导出
  - `MultiModalCOCO -> MultiModalCOCOClean` 类别清洗
  - data loader 对 COCO 结构的识别支持
- 当前尚未完成的是：
  - 让实验主流程真正按 “box-only 协议” 训练
  - 让 `MultiModalCOCOClean` 在训练时不再依赖原始 mask supervision
  - 让实验结果按 `mask-supervised` 与 `box-only` 两类协议显式区分

代码层面的关键事实仍然是：

- trainable teacher 方法训练时依旧使用 `batch["masks"]` 做 BCE + Dice
- 因此，当前核心实验代码还没有完成协议层面的修改

后续真正需要进入的，是“实验代码修改阶段”，而不是继续只做数据准备。

### 记录 30：已完成 `mask_supervised / box_only` 协议切换的核心代码修改
当前已经完成第一版实验协议切换，核心改动落在 `ir_sam2_exp/experiment_core/`：

- `config.py`
  - 新增 `SUPERVISION_PROTOCOL`
  - 当前支持：
    - `mask_supervised`
    - `box_only`
  - 新增 `LAMBDA_BOX_PROJECTION` 和 `LAMBDA_BOX_OUTSIDE`，用于 box-only 弱监督损失
- `runner.py`
  - 训练集 dataloader 不再默认要求 GT mask
  - 在 `box_only` 协议下，训练集会显式屏蔽 `gt_mask`
  - 评估集仍保留 mask，用于 `mIoU / BoundaryF1`
- `data.py`
  - `InfraredDataset` 新增 `allow_gt_masks`
  - `collate_fn` 改成支持 mixed batch：
    - 同一 batch 中可同时存在“只有 box 的样本”和“带 pseudo mask 的样本”
  - 修复了 COCO 样本按图像分组时只识别 `__inst_`、不识别 `__ann_` 的问题
- `methods.py`
  - trainable 方法不再默认强依赖 `batch["masks"]`
  - 当样本有 mask 时，仍然用 `BCE + Dice`
  - 当样本没有 mask 时，切到 box-only 弱监督损失

当前 box-only 弱监督采用的不是“把 bbox 直接填充成矩形 GT mask”，而是两部分：

- `box projection loss`
  - 约束预测在行投影和列投影上覆盖 bbox 范围
- `outside suppression loss`
  - 约束 bbox 外区域尽量不激活

这版设计的目的，是先让协议真正切开，并避免把模型显式训练成“只会画矩形框”。

### 记录 31：`box_only` 主干和 `box_only + pseudo` 已完成最小冒烟验证
当前已经在 `ubuntu2004 + sam_hq2` 环境里完成两轮新的 smoke test，数据使用：

- `DATASET_ROOT=/mnt/d/workspace/sam_ir/ir_sam2_exp/data_prep`
- `DATASET_NAME=MultiModalCOCOClean`
- `SUPERVISION_PROTOCOL=box_only`

第一轮输出目录：

- `ir_sam2_exp/outputs_smoke_box_only_protocol_v1`

运行条件：

- `ZeroShotSAM2BoxPromptIR`
- `CleanBoxPEFTSAM2Adapter`
- `DirectSupervisedIRSegFormerB0`

结果说明：

- `ZeroShotSAM2BoxPromptIR` 正常完成
- `CleanBoxPEFTSAM2Adapter` 正常完成
- `DirectSupervisedIRSegFormerB0` 也能在新的 box-only 协议下跑通
- 说明当前协议切换不是停留在静态代码层面，而是已经打通了 `box_only` 训练主链

第二轮输出目录：

- `ir_sam2_exp/outputs_smoke_box_only_pseudo_v1`

运行条件：

- `NoisyBoxPromptRobustSAM2Adapter`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
- `PseudoMaskSelfTrainingWithoutIRQualityFilter`

结果说明：

- `box_only + pseudo` 分支也已跑通
- `pseudo_accept_count` 重新变为非零
- 说明 warmup 阶段屏蔽 GT mask、finetune 阶段重新接入 pseudo mask 的 mixed supervision 链路已经成立

当前阶段性判断：

- 数据准备层和实验协议层现在都已经开始闭合
- 下一步不再是“有没有切协议”，而是“box-only 协议下哪些条件真正有效，哪些只是能跑”

### 记录 32：已重写代码说明文档，使其和当前协议切换后的实现一致
当前已经更新两份代码说明文档：

- `ir_sam2_exp/README.md`
- `ir_sam2_exp/EXPERIMENT_CODE_OVERVIEW.md`

这次更新的目标不是补充零散说明，而是把文档改成和当前代码状态一致的版本。重点补充了：

- `ir_sam2_exp/` 作为主运行目录的定位
- 当前代码主线仍是 teacher / pseudo，而不是完整论文全链路
- `mask_supervised` 和 `box_only` 两种监督协议的区别
- `MultiModal`、`RBGT-Tiny`、`MultiModalCOCOClean` 三种数据入口的角色
- 当前训练数据流
- 当前 box-only 弱监督损失的设计意图
- 当前 pseudo-label 在 `box_only` 协议下如何和 pseudo mask 混合工作
- 当前代码能回答什么问题、还不能回答什么问题

当前文档状态已经可以支撑后续讨论，不需要再靠口头补丁去解释“现在代码到底在做什么”。

### 记录 33：已补齐 baseline 代码，并把 eval 从“均值”扩成“可追溯报告”
当前已完成两类重要补充：

#### 1. baseline 代码补齐
新增了两个 baseline：

- `BBoxRectMaskBaseline`
  - 直接把 bbox rasterize 成矩形 mask
  - 用途不是发表，而是作为最小几何 sanity baseline
  - 任何 box-only 方法如果连它都打不过，都不能说明模型真的学到了超越“画框”的分割能力
- `DirectSupervisedIRPIDNetS`
  - 计划里原本已有，但之前代码未实现
  - 现在已经补到 `methods.py` 和 `models.py`

同时：

- `DirectSupervisedIRPIDNetS` 已从 deferred condition 转入可运行 condition
- `BBoxRectMaskBaseline` 已写入 `EXPERIMENT_PLAN.yaml`
- 新增脚本：
  - `ir_sam2_exp/scripts/run_box_only_baselines.sh`

#### 2. eval 补强
当前 eval 不再只输出单个 condition 的均值，而是会为每个 test 条件写出：

- `results.json`
  - 聚合指标
- `summary.json`
  - 汇总视图
- `eval_reports/<prefix>_eval.json`
  - per-sample 明细
  - 按 `device_source` 分组
  - 按 `target_scale` 分组
  - 按 `category_name` 分组
  - 按 `annotation_protocol_flag` 分组

当前额外加入的评估字段包括：

- `Dice`
- `BBoxIoU`
- `PredAreaRatio`
- `GTAreaRatio`
- `BBoxAreaRatio`

### 记录 34：baseline smoke 已跑通，并暴露出当前最关键的现实问题
当前已经在：

- `ir_sam2_exp/outputs_smoke_baselines_eval_v1`

完成一轮 baseline smoke，配置为：

- `DATASET_NAME=MultiModalCOCOClean`
- `SUPERVISION_PROTOCOL=box_only`
- 条件：
  - `BBoxRectMaskBaseline`
  - `ZeroShotSAM2BoxPromptIR`
  - `DirectSupervisedIRSegFormerB0`
  - `DirectSupervisedIRPIDNetS`

结果如下：

- `BBoxRectMaskBaseline`
  - `mIoU=0.7302`
  - `Dice=0.8375`
- `ZeroShotSAM2BoxPromptIR`
  - `mIoU=0.5625`
  - `Dice=0.7193`
- `DirectSupervisedIRSegFormerB0`
  - `mIoU=0.0011`
- `DirectSupervisedIRPIDNetS`
  - `mIoU=0.0000`

这个结果当前最重要的含义不是“哪个深度模型更好”，而是：

- 在当前 `MultiModalCOCOClean + box_only` 设定下，`BBoxRectMaskBaseline` 暂时是最强 baseline
- 这说明当前 cleaned 数据里的相当一部分 mask 与紧致 bbox 形状接近
- 因此后续所有方法的最低要求，不再是“优于 zero-shot”，而是“至少优于直接画框”

这会直接影响后续讨论方向：

- 如果 teacher adaptation 连矩形框 baseline 都打不过，说明现阶段 pipeline 还没进入“真正学会分割”的区间
- 如果 direct baseline 全部崩掉，不能立刻得出“teacher 路线更好”，更可能说明当前 box-only 损失还不足以支撑 direct baseline 学习

### 记录 35：当前应把 `tight box` 改成更接近真实弱监督的 `loose box`
当前讨论已经明确达成一个关键判断：

- `box` 标注不应继续使用“由 polygon 直接取 min/max 得到的 tight box”
- 当前 `BBoxRectMaskBaseline` 之所以过强，一个直接原因就是导出脚本现在使用的是过紧的框

当前代码事实是：

- `ir_sam2_exp/scripts/export_multimodal_coco.py`
  - 现在的 `polygon_bbox()` 直接返回 polygon 的最小外接轴对齐框

这会带来两个问题：

1. 不符合真实 box 标注习惯
   - 人工框标注通常会留一定上下文，不会贴着目标边界
2. 会把 `box_only` 任务做得过于容易
   - 当 GT mask 和 tight box 高度接近时，`BBoxRectMaskBaseline` 会被抬得不合理
   - 后续方法即使真的学到了一点分割能力，也可能在指标上看不出来

当前讨论倾向是：

- 应该把数据层的 canonical bbox 直接改成 `loose box`
- 不能只靠训练时的 jitter / offset augmentation 来弥补
- 因为“训练 augmentation”不等于“监督协议本身公平”

当前更合理的拆分应是：

1. canonical weak-supervision box
   - 一个固定、可复现、比 tight box 更宽松的标注框
2. train-time noisy prompt augmentation
   - 在 canonical box 基础上再做 jitter / offset / truncation
3. eval regimes
   - clean loose box
   - jittered box
   - offset box
   - truncated box

当前讨论还形成了一个重要方向：

- `loose box` 不应只靠纯比例放大
- 对小目标尤其要加最小像素 padding
- 否则像 `drone` 这种小目标，即使放大 10%，框也几乎还是 tight

### 记录 36：已把 `MultiModalCOCO` 的 canonical box 从 tight 改成 loose，并完成新的 baseline smoke
当前已经完成数据协议层的落地修改：

- `ir_sam2_exp/scripts/export_multimodal_coco.py`
  - 不再只输出 tight box
  - 现在会同时写出：
    - `bbox_tight`
    - `bbox_loose`
    - `bbox`
  - 其中当前 `bbox = bbox_loose`
- 默认 loose-box 规则为：
  - `pad_x = max(2 px, 0.15 * box_width)`
  - `pad_y = max(2 px, 0.15 * box_height)`
  - 扩框后最短边至少 `12 px`
  - 最后裁剪到图像边界
- `ir_sam2_exp/experiment_core/data.py`
  - 现在对 COCO 数据优先读取 `bbox_loose`
  - 同时保留 `bbox_tight` 和 `bbox_loose` 进入 sample metadata
- `ir_sam2_exp/experiment_core/runner.py`
  - eval report 现在会额外记录：
    - `bbox_tight`
    - `bbox_loose`
    - `TightBoxMaskIoU`
    - `LooseBoxMaskIoU`

当前已经重新导出：

- `ir_sam2_exp/data_prep/MultiModalCOCO`
- `ir_sam2_exp/data_prep/MultiModalCOCOClean`

并完成新的 baseline smoke：

- 输出目录：
  - `ir_sam2_exp/outputs_smoke_baselines_eval_loose_v1`

关键结果：

- 修改前：
  - `BBoxRectMaskBaseline mIoU = 0.7302`
- 修改后：
  - `BBoxRectMaskBaseline mIoU = 0.2361`
  - `ZeroShotSAM2BoxPromptIR mIoU = 0.3282`
  - `DirectSupervisedIRSegFormerB0 mIoU = 0.0011`
  - `DirectSupervisedIRPIDNetS mIoU = 0.0000`

这说明 loose-box 协议已经达到了最直接的目标：

- `BBoxRectMaskBaseline` 不再异常强势
- `ZeroShotSAM2BoxPromptIR` 重新超过“直接画框”
- 后续实验的最低要求，终于不再是“至少打平 tight-box 矩形框”

同时，新的 eval report 还给出了一个很重要的审计信号：

- 当前 smoke 样本上：
  - `TightBoxMaskIoU = 0.7302`
  - `LooseBoxMaskIoU = 0.2361`

这意味着当前 loose-box 改动不是表面字段替换，而是确实把监督协议难度拉开了。

### 记录 37：拟定以 SAM2 为中心的 benchmark 计划
当前建议把后续 benchmark 明确定位成：

- 不是“所有方法混在一起的总表”
- 而是“以 SAM2 为核心参照系的阶段化 benchmark”

也就是说，benchmark 的主问题应改写为：

- 在统一的红外 box-only 协议下，SAM2 从 zero-shot 到 adaptation 再到 pseudo，究竟能走到哪一步
- 其它 baseline 的角色，是作为参照和下界/对照，而不是抢主线

#### A. benchmark 目标
当前 benchmark 的目标建议固定为 4 个：

1. 建立几何下界
   - 先回答“直接画框”能做到什么程度
2. 建立 SAM2 zero-shot 基线
   - 回答“冻结 SAM2 + loose box prompt”到底有多强
3. 建立 SAM2 adaptation 基线
   - 回答 clean / noisy / ablation 之后是否稳定超过 zero-shot
4. 建立 SAM2 pseudo 扩展基线
   - 回答 pseudo-label 是否在 adaptation 基础上提供净收益

#### B. benchmark 协议
当前 benchmark 建议先只固定一套主协议：

- 数据：
  - `MultiModalCOCOClean`
- 监督：
  - `SUPERVISION_PROTOCOL=box_only`
- box：
  - canonical `bbox_loose`
- split：
  - `device_source` 确定性划分
- eval：
  - 主报告使用 clean loose box
  - robustness 作为单独 regime 扩展

同时保留两个辅助协议，但不作为第一轮主表：

- `MultiModal` + `mask_supervised`
  - 作为 upper bound / sanity check
- `RBGT-Tiny` + `box_only`
  - 作为跨数据集复验

#### C. benchmark 条件分层
当前建议 benchmark 分 4 层，而不是一次把所有条件都混进来：

第 0 层：几何 sanity baseline
- `BBoxRectMaskBaseline`

第 1 层：SAM2 zero-shot benchmark
- `ZeroShotSAM2BoxPromptIR`

第 2 层：SAM2 adaptation benchmark
- `CleanBoxPEFTSAM2Adapter`
- `NoisyBoxPromptRobustSAM2Adapter`
- `CleanPromptOnlyWithinPromptRobustAdapter`
- `JitterOnlyPromptRobustSAM2Adapter`

第 3 层：SAM2 pseudo benchmark
- `NoisyBoxPromptRobustSAM2Adapter`
- `QualityFilteredPseudoMaskSelfTrainingSAM2`
- `PseudoMaskSelfTrainingWithoutIRQualityFilter`

非主线对照层：
- `DirectSupervisedIRSegFormerB0`
- `DirectSupervisedIRPIDNetS`

当前建议是：

- benchmark 主图和主表先围绕第 0-3 层
- direct baseline 放在 supplementary / control 表
- distillation 和 INT8 先不进入 benchmark v1

#### D. benchmark 指标
当前 benchmark v1 建议强制报告这些指标：

主指标：
- `mIoU`
- `Dice`
- `BoundaryF1`

协议审计指标：
- `BBoxIoU`
- `TightBoxMaskIoU`
- `LooseBoxMaskIoU`
- `PredAreaRatio`
- `GTAreaRatio`

效率指标：
- `LatencyMs`

robustness 扩展指标：
- `gt_box`
- `jittered_box`
- `offset_box`
- `truncated_box`

当前建议：

- 第一轮正式 benchmark 先固定 clean loose box
- robustness regimes 作为第二轮扩展

#### E. benchmark 验收标准
当前建议每一层都有独立验收标准，而不是只看最终最高分：

第 0 层验收：
- `BBoxRectMaskBaseline` 跑通
- `TightBoxMaskIoU` 与 `LooseBoxMaskIoU` 差异合理

第 1 层验收：
- `ZeroShotSAM2BoxPromptIR` 稳定超过 `BBoxRectMaskBaseline`

第 2 层验收：
- 至少一个 adaptation 条件稳定超过 zero-shot
- `Noisy` 至少不明显弱于 `Clean`

第 3 层验收：
- `QualityFilteredPseudoMaskSelfTrainingSAM2` 至少在一个 budget/seed 组合上优于 `Noisy`
- 如果做不到，应接受“pseudo 当前不成立”这个结论

#### F. benchmark 执行顺序
当前最合理的 benchmark 执行顺序建议为：

1. baseline table
   - `BBoxRectMaskBaseline`
   - `ZeroShotSAM2BoxPromptIR`
   - `DirectSupervisedIRSegFormerB0`
   - `DirectSupervisedIRPIDNetS`
2. SAM2 adaptation table
   - `Clean`
   - `Noisy`
   - `CleanPromptOnly`
   - `JitterOnly`
3. SAM2 pseudo table
   - `Noisy`
   - `QualityFilteredPseudo`
   - `PseudoWithoutFilter`
4. robustness extension
   - clean loose box
   - jittered
   - offset
   - truncated

#### G. 当前建议的最终定位
当前 benchmark 的核心不是“谁分最高”，而是建立一条清晰的证据链：

- 直接画框是多强
- zero-shot SAM2 比它强多少
- adaptation 比 zero-shot 强多少
- pseudo 是否继续带来净收益

只有这条链成立，后面再谈 distillation 和 INT8 才有意义。

### 记录 38：已拆出独立 benchmark 工程 `ir_sam2_bench`
当前已经把 benchmark 从 `ir_sam2_exp/` 中拆成单独工程：

- 新目录：
  - `D:/workspace/sam_ir/ir_sam2_bench`

这个目录当前包含：

- `main.py`
- `experiment_core/`
- `scripts/`
- `EXPERIMENT_PLAN.yaml`
- `README.md`
- `BENCHMARK_OVERVIEW.md`
- `BENCHMARK_RUNBOOK.md`

当前设计原则是：

- benchmark 工程只保留能独立运行 benchmark 的最小代码和文档
- 不复制旧实验输出目录
- 数据仍通过 `DATASET_ROOT` 外部传入
- SAM2 仍通过 `SAM2_REPO` 外部传入

#### benchmark 工程的默认配置
当前已经把 benchmark 工程默认值调成 benchmark 语义：

- 默认 `EXPERIMENT_PHASE=benchmark_v1`
- 默认 `DATASET_NAME=MultiModalCOCOClean`
- 默认 `SUPERVISION_PROTOCOL=box_only`
- 默认输出目录根路径：
  - `benchmark_runs/`

#### benchmark 工程当前提供的脚本
- `scripts/run_baselines.sh`
- `scripts/run_adaptation.sh`
- `scripts/run_pseudo.sh`
- `scripts/run_full_benchmark.sh`

#### benchmark 工程当前提供的文档
- `README.md`
  - 解释这个工程是什么
- `BENCHMARK_OVERVIEW.md`
  - 解释 benchmark 的层次、协议和代码组成
- `BENCHMARK_RUNBOOK.md`
  - 解释怎么在服务器上逐层执行 benchmark

#### benchmark 工程已完成独立 smoke
当前已经直接从 `ir_sam2_bench/` 目录启动并完成最小 smoke：

- 条件：
  - `BBoxRectMaskBaseline`
  - `ZeroShotSAM2BoxPromptIR`
- 数据：
  - `MultiModalCOCOClean`
- 协议：
  - `box_only`

结果说明：

- benchmark 工程不是单纯复制的目录壳
- 当前已经能独立加载配置、运行 runner、落盘输出
- 中途还修复了一个工程级问题：
  - `output_dir.mkdir()` 缺少 `parents=True`

当前状态下，可以把 `ir_sam2_bench/` 视为后续 benchmark 讨论和执行的主目录，而把 `ir_sam2_exp/` 继续保留为更宽泛的实验工程。

### 记录 39：benchmark 的核心目标应从“跑一组实验”提升为“构建可复用的平台”
当前讨论明确了 benchmark 的更高优先级目标：

- benchmark 最重要的不是先把某一组方法跑完
- 而是先构建一个“以 baseline 为骨架、可自由切换数据集、可稳定输出 eval 结果”的平台

这意味着 benchmark 的定位需要进一步上移：

- 它首先应该是一个 benchmark platform
- 其次才是当前这篇工作所需的一组具体实验条件

当前平台化目标可拆成三层：

#### A. baseline-oriented platform
- 所有新方法都必须挂在 baseline 体系下比较
- baseline 不只是一个条件列表，而是 benchmark 的骨架
- 至少应固定：
  - geometry baseline
  - SAM2 zero-shot baseline
  - SAM2 adaptation baseline
  - direct-train control baseline

#### B. dataset-swappable platform
- benchmark 不应把 `MultiModalCOCOClean` 写死成唯一数据集
- 应该允许通过统一接口切换：
  - `MultiModal`
  - `MultiModalCOCOClean`
  - `RBGT-Tiny`
  - 后续新增的红外数据集
- 切换数据集时，训练与 eval 代码不应改动，只改配置和 dataset adapter

#### C. eval-first platform
- eval 不应只是最后顺手写一个 `results.json`
- eval 应是 benchmark 平台的核心接口之一
- 至少应稳定输出：
  - 聚合指标
  - per-sample 明细
  - 按数据集元信息分组统计
  - 可复查的协议审计指标

当前讨论导出的直接结论是：

- `ir_sam2_bench` 现在已经初步具备 platform 雏形
- 但它还更像“围绕当前项目整理出的独立 benchmark 工程”
- 还没有完全达到“可插拔数据集 + 可扩展 baseline + eval-first”的 benchmark platform 级别

因此后续 benchmark 工程建议进入一个新的设计目标：

- 不再只是补实验条件
- 而是把 benchmark 代码改造成 3 个稳定接口：
  1. dataset adapter interface
  2. baseline/method registry interface
  3. evaluation/report interface

## 2026-04-21 Benchmark Platform Status

### 已完成的平台化改造
- `ir_sam2_bench` 现在不再只是单一实验脚本集合，而是按 3 个稳定接口组织：
  - dataset adapter
  - method registry
  - evaluation/report
- `MultiModalCOCO` 的 tight / loose 标注可视化已经落地，输出在：
  - `ir_sam2_bench/visualizations/multimodal_coco_boxes_v1/tight`
  - `ir_sam2_bench/visualizations/multimodal_coco_boxes_v1/loose`
  - `ir_sam2_bench/visualizations/multimodal_coco_boxes_v1/comparison`

### 当前平台组成
- 配置与启动：
  - `ir_sam2_bench/experiment_core/config.py`
  - `ir_sam2_bench/scripts/common.sh`
  - `ir_sam2_bench/scripts/run_full_benchmark.sh`
- 数据接口：
  - `ir_sam2_bench/experiment_core/dataset_adapters.py`
  - 当前支持 `MultiModal`、COCO 风格数据集、`RBGT-Tiny` 的 IR-only 分支
- 方法接口：
  - `ir_sam2_bench/experiment_core/method_registry.py`
  - 当前统一注册 geometry baseline、SAM2 zero-shot、adaptation、pseudo、direct-train control
- 评估与报告接口：
  - `ir_sam2_bench/experiment_core/evaluation.py`
  - `ir_sam2_bench/experiment_core/reporting.py`

### 可迁移性判断
- 迁移到任意服务器：
  - 可以。当前 benchmark 运行时不依赖 `/home/sakauma` 这类硬编码路径，核心路径都通过环境变量传入。
- 迁移到任意数据集：
  - 不是“零改动任意数据集”，而是“零改动支持当前 adapter 覆盖的数据集；新增数据集时只需要加一个 dataset adapter，而不是改训练和 eval 主流程”。
- 上服务器最小前提：
  - 本地可用 Python 环境
  - 本地 SAM2 repo 与 checkpoint
  - 数据集目录满足当前 adapter，或补一个新 adapter

### 当前结论
- benchmark 平台已经具备独立运行能力。
- 对服务器迁移已经基本就绪。
- 对新数据集的扩展点已经明确收敛到 dataset adapter，而不是 benchmark 全局重写。

### 新增服务器操作模板
- 已补充后台运行模板：
  - `tmux`
  - `nohup`
- 已补充运行中检查命令：
  - `ps`
  - `tail -f`
  - `watch`
- 已补充跑完后的结果检查命令：
  - `summary.json`
  - `results.json`
  - `eval_reports/*.json`

### 新增正式运行档位
- 已补充 3 档 benchmark 运行配置：
  - `Light`
  - `Standard`
  - `Heavy`
- 并且按数据集分别给出：
  - `MultiModal`
  - `RBGT-Tiny`
- 当前建议：
  - 笔记本或单张中端 GPU 先从 `Light` 开始
  - 正式服务器第一次对比建议从 `Standard` 开始
  - `Heavy` 只在前两档已经证明稳定后再启动

### 新增“服务器首次正式运行建议”
- 已在 `BENCHMARK_RUNBOOK.md` 中补充独立章节：
  - `First Server Run Recommendation`
- 该章节不再给多个选项，而是固定推荐第一次正式服务器运行直接采用：
  - `MultiModal`: `Standard`
  - `RBGT-Tiny`: `Standard + MAX_IMAGES=1024`
- 并要求首次正式运行前先做：
  - `py_compile`
  - `SMOKE_TEST=1`
  - 再执行 full benchmark
## 2026-04-21 Benchmark Server Migration Checklist

### 1. 代码与目录
- 确认服务器上已有完整工程目录：
  - `ir_sam2_bench/`
- 确认关键文件存在：
  - `main.py`
  - `experiment_core/`
  - `scripts/common.sh`
  - `scripts/run_full_benchmark.sh`

### 2. Python 环境
- 确认服务器上有可用 Python 环境。
- 确认关键依赖可导入：
  - `torch`
  - `cv2`
  - `matplotlib`
  - `yaml`
  - `PIL`
  - `hydra`
  - `transformers`
  - `sam2`

### 3. SAM2 资源
- 准备本地 `SAM2_REPO`
- 准备本地 checkpoint 文件
- 确认以下环境变量能正确指向：
  - `SAM2_REPO`
  - 如有需要：`SAM2_CKPT`
  - 如有需要：`SAM2_CFG`

### 4. 数据集准备
- 确认数据集根目录存在，并可通过 `DATASET_ROOT` 访问。
- 当前零改动可运行的数据集：
  - `MultiModal`
  - `MultiModalCOCO`
  - `MultiModalCOCOClean`
  - `RBGT-Tiny`
- 若是新数据集：
  - 只需要新增一个 dataset adapter
  - 不需要改训练主流程和 eval 主流程

### 5. 数据格式检查
- `MultiModal`
  - 需要 `img/` 和 `label/`
- COCO 风格数据集
  - 需要 `annotations_coco/` 和 `image/`
- `RBGT-Tiny`
  - 当前默认只读 `01` 灰度分支

### 6. 运行前环境变量
- 最少需要设置：
  - `DATASET_ROOT`
  - `SAM2_REPO`
- 常用可选项：
  - `DATASET_NAME`
  - `PYTHON_BIN`
  - `OUTPUT_DIR`
  - `EXPERIMENT_SEEDS`
  - `SUPERVISION_BUDGETS`
  - `TRAIN_EPOCHS`
  - `PSEUDO_FINETUNE_EPOCHS`
  - `MAX_SAMPLES`
  - `MAX_IMAGES`
  - `EVAL_LIMIT`

### 7. 启动前验证
- 先跑语法检查：
  - `python -m py_compile main.py experiment_core/*.py scripts/*.py`
- 先跑 smoke：
  - `SMOKE_TEST=1`
- 再跑正式 benchmark

### 8. 输出检查
- 运行后应至少生成：
  - `results.json`
  - `summary.json`
  - `eval_reports/*.json`
- 需要重点检查：
  - `dataset_manifest`
  - `method_manifest`
  - `supervision_protocol`
  - 每个 condition 是否都有评估报告

### 9. 资源策略
- 显存或时间受限时优先收缩：
  - `EXPERIMENT_SEEDS`
  - `SUPERVISION_BUDGETS`
  - `TRAIN_EPOCHS`
  - `PSEUDO_FINETUNE_EPOCHS`
  - `MAX_IMAGES`
- 多目标数据集优先用 `MAX_IMAGES`，不要只用 `MAX_SAMPLES`

### 10. 迁移结论
- 换服务器：
  - 可以，当前 benchmark 已支持
- 换当前已支持的数据集：
  - 可以，零改动
- 换全新数据集：
  - 可以，但需要补一个 dataset adapter
