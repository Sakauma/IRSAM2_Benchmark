# IR SAM2 Benchmark

`ir_sam2_bench/` 是一个独立的红外分割 benchmark 工程，目标不是只复现一组实验，而是提供一个以 baseline 为中心、可切换数据集、可稳定产出评估结果的平台。

当前 benchmark v1 主线：

`BBox baseline -> SAM2 zero-shot -> SAM2 adaptation -> SAM2 pseudo`

## 工程结构
- `main.py`
  - benchmark 入口。
- `experiment_core/config.py`
  - 统一解析运行配置、数据路径、SAM2 路径和训练超参数。
- `experiment_core/dataset_adapters.py`
  - 数据集适配层。当前支持 `MultiModal` 原始格式、COCO 风格数据集、`RBGT-Tiny` 的 IR-only 分支。
- `experiment_core/method_registry.py`
  - 方法注册层。把 baseline、SAM2 主线和 control baselines 统一注册成可选择条件。
- `experiment_core/evaluation.py`
  - 执行逐样本评估并生成统一指标行。
- `experiment_core/reporting.py`
  - 写出 `results.json`、`summary.json` 和 `eval_reports/*.json`。
- `scripts/`
  - 数据准备、标注可视化和 benchmark 启动脚本。

## 数据集支持
- `MultiModal`
  - 原始 `img/ + label/` 布局。benchmark 会在线生成 `bbox_tight` 和 canonical `bbox_loose`。
- `MultiModalCOCO` / `MultiModalCOCOClean`
  - COCO 风格导出数据。
- `RBGT-Tiny`
  - 默认只读取灰度 `01` 分支，避免把彩色 `00` 分支混进红外 benchmark。

## 评估输出
每次运行至少会写出：
- `results.json`
- `summary.json`
- `eval_reports/*.json`

`eval_reports` 包含：
- 聚合指标：`mIoU`、`Dice`、`BoundaryF1`、`LatencyMs`
- 协议审计指标：`BBoxIoU`、`TightBoxMaskIoU`、`LooseBoxMaskIoU`、`PredAreaRatio`、`GTAreaRatio`
- 分组结果：`device_source`、`target_scale`、`category_name`、`annotation_protocol_flag`
- per-sample 明细

## 常用命令
```bash
DATASET_ROOT=/path/to/dataset \
SAM2_REPO=/path/to/sam2 \
bash scripts/run_full_benchmark.sh
```

服务器首次正式运行也可以直接用：

```bash
bash scripts/run_multimodal_server_first.sh
bash scripts/run_rbgt_server_first.sh
```

绘制 `MultiModalCOCO` 的 tight / loose 标注可视化：
```bash
python scripts/visualize_coco_boxes.py \
  --src-root /path/to/MultiModalCOCO \
  --output-root ./visualizations/multimodal_coco_boxes_v1
```
