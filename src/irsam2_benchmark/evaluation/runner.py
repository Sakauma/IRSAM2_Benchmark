from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np

from ..config import AppConfig
from ..core.interfaces import InferenceMode
from ..data.masks import sample_mask_array, sample_mask_or_zeros
from ..data.prompt_synthesis import mask_to_tight_box
from ..data.sample import Sample
from ..data.views import build_image_view, build_track_view
from .image_metrics import bbox_iou, boundary_f1, boundary_f1_tolerance, dice_score, mask_iou
from .instance_metrics import greedy_match_instances
from .prompt_metrics import prompt_metrics
from .small_target_metrics import small_target_metrics
from .temporal_metrics import compute_temporal_metrics


def _progress_bar(
    *,
    method,
    config: AppConfig | None,
    inference_mode: InferenceMode,
    error_context: Dict[str, Any],
    total: int,
    unit: str,
) -> Any | None:
    # 进度条只依赖 runtime.show_progress；测试环境或无 tqdm 环境会静默退化为无进度条。
    runtime = getattr(config, "runtime", None)
    if not bool(getattr(runtime, "show_progress", False)) or total <= 0:
        return None
    try:
        from tqdm import tqdm
    except Exception:
        return None
    dataset_id = getattr(getattr(config, "dataset", None), "dataset_id", "dataset")
    method_name = str(error_context.get("baseline_name") or getattr(method, "name", method.__class__.__name__))
    seed = error_context.get("seed")
    seed_label = f" seed={seed}" if seed is not None else ""
    return tqdm(
        total=total,
        unit=unit,
        desc=f"{dataset_id} {method_name}{seed_label} {inference_mode.value}",
        dynamic_ncols=True,
        leave=True,
        mininterval=float(getattr(runtime, "progress_update_interval_s", 1.0)),
        file=sys.stderr,
    )


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 每个 run 的 summary 是 sample-level 行的数值字段均值。
    # 多 seed 的均值/方差在 pipeline/reporting 层再处理。
    numeric: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    aggregate: Dict[str, Any] = {}
    for key, values in numeric.items():
        aggregate[key] = float(sum(values) / len(values)) if values else 0.0
    return aggregate


def _summarize_values(values: Iterable[str]) -> str:
    unique = sorted({value for value in values if value})
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "multiple"


def _chunks(items: List[Sample], batch_size: int) -> Iterable[List[Sample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and "cuda" in text and "out of memory" in text


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _error_log_path(config: AppConfig | None):
    output_dir = getattr(config, "output_dir", None)
    if output_dir is None:
        return None
    return output_dir / "eval_reports" / "error_log.jsonl"


def _reset_error_log(config: AppConfig | None) -> None:
    path = _error_log_path(config)
    if path is not None and path.exists():
        path.unlink()


def _write_sample_error(
    *,
    config: AppConfig | None,
    method,
    sample: Sample,
    exc: Exception,
    context: Dict[str, Any] | None = None,
) -> None:
    # 单样本失败不直接中断整个 benchmark；错误会进入 error_log.jsonl，方便后续定位坏样本。
    path = _error_log_path(config)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": getattr(getattr(config, "dataset", None), "dataset_id", ""),
        "model_id": getattr(getattr(config, "model", None), "model_id", ""),
        "config_path": str(getattr(config, "config_path", "")),
        "method_name": getattr(method, "name", method.__class__.__name__),
        "inference_mode": getattr(getattr(method, "inference_mode", None), "value", str(getattr(method, "inference_mode", ""))),
        "sample_id": sample.sample_id,
        "frame_id": sample.frame_id,
        "sequence_id": sample.sequence_id,
        "track_id": sample.track_id,
        "image_path": str(sample.image_path),
        "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **(context or {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_error_context(base: Dict[str, Any] | None, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if base:
        payload.update(base)
    payload.update(extra)
    return payload


def _mask_box(mask: np.ndarray) -> list[float] | None:
    if float((mask > 0.5).sum()) <= 0.0:
        return None
    return mask_to_tight_box(mask)


def _mask_to_2d(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}.")
    return arr


def _resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = _mask_to_2d(mask)
    if arr.shape == (height, width):
        return arr.astype(np.float32)
    src_h, src_w = arr.shape
    if src_h <= 0 or src_w <= 0:
        return np.zeros((height, width), dtype=np.float32)
    y_idx = np.minimum((np.arange(height) * (src_h / float(height))).astype(np.int64), src_h - 1)
    x_idx = np.minimum((np.arange(width) * (src_w / float(width))).astype(np.int64), src_w - 1)
    return arr[y_idx[:, None], x_idx[None, :]].astype(np.float32)


def align_mask_to_sample(mask: np.ndarray, item: Sample) -> tuple[np.ndarray, Dict[str, Any]]:
    # SAM2 返回的 mask 有时和原图尺寸不完全一致。这里用最近邻对齐，并记录是否发生 resize。
    raw = _mask_to_2d(mask)
    target_h = max(1, int(item.height))
    target_w = max(1, int(item.width))
    aligned = _resize_mask_nearest(raw, target_h, target_w)
    metadata = {
        "PredMaskOriginalHeight": int(raw.shape[0]),
        "PredMaskOriginalWidth": int(raw.shape[1]),
        "PredMaskAlignedHeight": target_h,
        "PredMaskAlignedWidth": target_w,
        "PredMaskWasResized": raw.shape != (target_h, target_w),
    }
    return aligned, metadata


def _predict_batch_with_fallback(
    method,
    batch: List[Sample],
    config: AppConfig,
    *,
    track_name: str,
    requested_batch_index: int,
    error_context: Dict[str, Any] | None = None,
) -> List[tuple[List[Sample], Dict[str, Dict[str, Any]], float]]:
    # prompted 模式支持批量推理。若批量预测 OOM，会递归二分批次，直到单样本仍失败才记录错误。
    start = time.perf_counter()
    try:
        predict_samples = getattr(method, "predict_samples", None)
        if callable(predict_samples):
            predictions = predict_samples(batch)
        else:
            predictions = {sample.sample_id: method.predict_sample(sample) for sample in batch}
        _validate_batch_predictions(batch, predictions)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [(batch, predictions, elapsed_ms)]
    except Exception as exc:
        if _is_cuda_oom(exc):
            _clear_cuda_cache()
        if len(batch) <= 1:
            _write_sample_error(
                config=config,
                method=method,
                sample=batch[0],
                exc=exc,
                context=_merge_error_context(
                    error_context,
                    {
                        "track_name": track_name,
                        "requested_batch_index": requested_batch_index,
                        "requested_batch_size": len(batch),
                        "stage": "prompted_batch_prediction",
                    },
                ),
            )
            return []
        midpoint = len(batch) // 2
        return _predict_batch_with_fallback(
            method,
            batch[:midpoint],
            config,
            track_name=track_name,
            requested_batch_index=requested_batch_index,
            error_context=error_context,
        ) + _predict_batch_with_fallback(
            method,
            batch[midpoint:],
            config,
            track_name=track_name,
            requested_batch_index=requested_batch_index,
            error_context=error_context,
        )


def _validate_batch_predictions(batch: List[Sample], predictions: Dict[str, Dict[str, Any]]) -> None:
    expected = [sample.sample_id for sample in batch]
    if len(set(expected)) != len(expected):
        raise RuntimeError(f"Batch contains duplicate sample_id values: {expected!r}.")
    expected_set = set(expected)
    actual_set = set(predictions)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise RuntimeError(f"Batch prediction/sample_id mismatch. missing={missing!r}, extra={extra!r}.")


def _image_level_row(
    items: List[Sample],
    elapsed_ms: float,
    instance_metrics: Dict[str, float],
    modality: str,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # no-prompt auto-mask 的 eval unit 是 image，不是 instance。
    # 一张图内多个 GT instance 会先做 greedy matching，再汇总成一行 image-level 指标。
    representative = items[0]
    row = {
        "sample_id": representative.frame_id,
        "frame_id": representative.frame_id,
        "sequence_id": representative.sequence_id,
        "eval_unit": "image",
        "modality": modality,
        "supervision_type": _summarize_values(item.supervision_type for item in items),
        "category_name": _summarize_values(item.category for item in items),
        "target_scale": _summarize_values(item.target_scale for item in items),
        "device_source": _summarize_values(item.device_source for item in items),
        "annotation_protocol_flag": _summarize_values(item.annotation_protocol_flag for item in items),
        "LatencyMs": elapsed_ms,
        **instance_metrics,
        **(extra_metadata or {}),
    }
    track_value = _summarize_values(item.track_id for item in items if item.track_id is not None)
    if track_value != "unknown":
        row["track_id"] = track_value
    return row


def evaluate_method(
    *,
    method,
    samples: List[Sample],
    config: AppConfig,
    track_name: str,
    inference_mode: InferenceMode,
    error_context: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # evaluate_method 是三类推理协议的统一入口：
    # 1. video_propagation：按 track/sequence 评估时序传播；
    # 2. no_prompt_auto_mask：按 image 评估自动 mask generator；
    # 3. 其它 prompted 模式：按 instance/sample 评估 box、point、box+point。
    modality = getattr(getattr(config, "dataset", None), "modality", "ir")
    run_error_context = dict(error_context or {})

    rows: List[Dict[str, Any]] = []
    if inference_mode == InferenceMode.VIDEO_PROPAGATION:
        # video 模式要求同一 track 内每帧只有一个 sample，否则无法明确传播目标。
        try:
            track_view = build_track_view(samples)
        except Exception as exc:
            for item in samples:
                _write_sample_error(
                    config=config,
                    method=method,
                    sample=item,
                    exc=exc,
                    context=_merge_error_context(run_error_context, {"track_name": track_name, "stage": "video_track_view"}),
                )
            return {}, []
        progress = _progress_bar(
            method=method,
            config=config,
            inference_mode=inference_mode,
            error_context=run_error_context,
            total=len(samples),
            unit="frames",
        )
        sequence_metrics = []
        try:
            for (sequence_id, track_id), items in track_view.items():
                group_sample_ids = [item.sample_id for item in items]
                frame_ids = {item.frame_id for item in items}
                if len(frame_ids) != len(items):
                    exc = RuntimeError(
                        f"Video propagation expects one sample per frame within a track, got duplicates for sequence_id={sequence_id!r}, track_id={track_id!r}."
                    )
                    for item in items:
                        _write_sample_error(
                            config=config,
                            method=method,
                            sample=item,
                            exc=exc,
                            context=_merge_error_context(
                                run_error_context,
                                {
                                    "track_name": track_name,
                                    "stage": "video_track_validation",
                                    "group_sample_ids": group_sample_ids,
                                },
                            ),
                        )
                    if progress is not None:
                        progress.update(len(items))
                    continue
                start = time.perf_counter()
                try:
                    predictions = method.predict_sequence(items)
                except Exception as exc:
                    for item in items:
                        _write_sample_error(
                            config=config,
                            method=method,
                            sample=item,
                            exc=exc,
                            context=_merge_error_context(
                                run_error_context,
                                {
                                    "track_name": track_name,
                                    "stage": "video_sequence_prediction",
                                    "group_sample_ids": group_sample_ids,
                                },
                            ),
                        )
                    if progress is not None:
                        progress.update(len(items))
                    continue
                elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(items))
                sequence_rows = []
                for item in items:
                    try:
                        pred_mask, mask_alignment = align_mask_to_sample(predictions[item.sample_id], item)
                        gt_mask = sample_mask_or_zeros(item)
                        row = build_segmentation_row(item, pred_mask, gt_mask, elapsed_ms, modality=modality, mask_alignment=mask_alignment)
                        row["track"] = track_name
                        row["track_id"] = track_id
                        sequence_rows.append({**row, "pred_mask": pred_mask})
                        rows.append(row)
                    except Exception as exc:
                        _write_sample_error(
                            config=config,
                            method=method,
                            sample=item,
                            exc=exc,
                            context=_merge_error_context(
                                run_error_context,
                                {
                                    "track_name": track_name,
                                    "stage": "video_row_build",
                                    "group_sample_ids": group_sample_ids,
                                },
                            ),
                        )
                if sequence_rows:
                    temporal = compute_temporal_metrics(sequence_rows)
                    temporal["sequence_id"] = sequence_id
                    temporal["track_id"] = track_id
                    sequence_metrics.append(temporal)
                if progress is not None:
                    progress.update(len(items))
        finally:
            if progress is not None:
                progress.close()
        summary = _aggregate(rows)
        if sequence_metrics:
            summary.update(_aggregate(sequence_metrics))
        return summary, rows

    if inference_mode == InferenceMode.NO_PROMPT_AUTO_MASK:
        # 自动掩码模式不使用外部 prompt，因此必须按图片聚合 GT instance，而不是逐 instance 调用。
        image_groups = list(build_image_view(samples).items())
        progress = _progress_bar(
            method=method,
            config=config,
            inference_mode=inference_mode,
            error_context=run_error_context,
            total=len(image_groups),
            unit="images",
        )
        try:
            for _, items in image_groups:
                representative = items[0]
                start = time.perf_counter()
                group_sample_ids = [item.sample_id for item in items]
                try:
                    pred = method.predict_sample(representative)
                except Exception as exc:
                    _write_sample_error(
                        config=config,
                        method=method,
                        sample=representative,
                        exc=exc,
                        context=_merge_error_context(
                            run_error_context,
                            {
                                "track_name": track_name,
                                "stage": "auto_mask_prediction",
                                "group_sample_ids": group_sample_ids,
                            },
                        ),
                    )
                    if progress is not None:
                        progress.update(1)
                    continue
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                try:
                    gt_instances = []
                    for item in items:
                        gt_mask = sample_mask_array(item)
                        if gt_mask is not None:
                            gt_instances.append({"mask": gt_mask})
                    pred_instances = []
                    for instance in pred.get("instances", []):
                        aligned_mask, _ = align_mask_to_sample(instance["mask"], representative)
                        pred_instances.append({**instance, "mask": aligned_mask})
                    if gt_instances:
                        instance_metrics = greedy_match_instances(pred_instances, gt_instances)
                    else:
                        instance_metrics = {"num_pred_instances": float(len(pred_instances)), "num_gt_instances": 0.0}
                    rows.append(
                        _image_level_row(
                            items,
                            elapsed_ms,
                            instance_metrics,
                            modality=modality,
                            extra_metadata={"AutoMaskPointsPerBatch": pred.get("auto_mask_points_per_batch")},
                        )
                    )
                except Exception as exc:
                    _write_sample_error(
                        config=config,
                        method=method,
                        sample=representative,
                        exc=exc,
                        context=_merge_error_context(
                            run_error_context,
                            {
                                "track_name": track_name,
                                "stage": "auto_mask_row_build",
                                "group_sample_ids": group_sample_ids,
                            },
                        ),
                    )
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        return _aggregate(rows), rows

    # prompted 模式是当前论文主表使用的路径。batch_size 写入每行，便于复现实验吞吐差异。
    configured_batch_size = max(1, int(getattr(getattr(config, "runtime", None), "image_batch_size", 1)))
    batch_index = 0
    progress = _progress_bar(
        method=method,
        config=config,
        inference_mode=inference_mode,
        error_context=run_error_context,
        total=len(samples),
        unit="samples",
    )
    try:
        for requested_batch_index, batch in enumerate(_chunks(samples, configured_batch_size)):
            batch_results = _predict_batch_with_fallback(
                method,
                batch,
                config,
                track_name=track_name,
                requested_batch_index=requested_batch_index,
                error_context=run_error_context,
            )
            for batch_split_index, (actual_batch, predictions, batch_elapsed_ms) in enumerate(batch_results):
                per_sample_elapsed_ms = batch_elapsed_ms / max(1, len(actual_batch))
                for batch_item_index, item in enumerate(actual_batch):
                    try:
                        pred = predictions[item.sample_id]
                        pred_mask, mask_alignment = align_mask_to_sample(pred["mask"], item)
                        gt_mask = sample_mask_or_zeros(item)
                        row = build_segmentation_row(item, pred_mask, gt_mask, per_sample_elapsed_ms, modality=modality, prompt=pred.get("prompt"), mask_alignment=mask_alignment)
                        row["BatchSize"] = len(actual_batch)
                        row["BatchIndex"] = batch_index
                        row["BatchRequestedIndex"] = requested_batch_index
                        row["BatchSplitIndex"] = batch_split_index
                        row["BatchItemIndex"] = batch_item_index
                        row["BatchLatencyMs"] = batch_elapsed_ms
                        rows.append(row)
                    except Exception as exc:
                        _write_sample_error(
                            config=config,
                            method=method,
                            sample=item,
                            exc=exc,
                            context=_merge_error_context(
                                run_error_context,
                                {
                                    "track_name": track_name,
                                    "stage": "prompted_row_build",
                                    "requested_batch_index": requested_batch_index,
                                    "batch_index": batch_index,
                                    "batch_split_index": batch_split_index,
                                    "batch_item_index": batch_item_index,
                                },
                            ),
                        )
                batch_index += 1
            if progress is not None:
                progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()
    return _aggregate(rows), rows


def build_segmentation_row(
    item: Sample,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    elapsed_ms: float,
    modality: str = "ir",
    prompt: Dict[str, object] | None = None,
    mask_alignment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # 构造单个 instance 的评估行。这里既记录预测 mask 指标，也记录 prompt 派生协议元数据。
    pred_area = float((pred_mask > 0.5).sum())
    gt_area = float((gt_mask > 0.5).sum())
    pred_box = _mask_box(pred_mask)
    gt_box = item.bbox_tight
    height = max(1, item.height)
    width = max(1, item.width)
    has_mask_gt = item.has_mask() and item.supervision_type == "mask"
    prompt_metadata: Dict[str, Any] = {}
    if prompt:
        prompt_metadata = {
            "PromptProtocol": prompt.get("protocol", "unknown"),
            "PromptSource": prompt.get("source", "unknown"),
            "PromptBoxVariant": prompt.get("box_variant", "none"),
            "PromptTightBoxRule": prompt.get("tight_box_rule", "unknown"),
            "PromptLooseBoxRule": prompt.get("loose_box_rule", "unknown"),
            "PromptLooseBoxPadRatio": prompt.get("loose_box_pad_ratio"),
            "PromptLooseBoxMinPad": prompt.get("loose_box_min_pad"),
            "PromptLooseBoxMinSide": prompt.get("loose_box_min_side"),
            "PromptLooseBoxMaxSideMultiplier": prompt.get("loose_box_max_side_multiplier"),
            "PromptPointRule": prompt.get("point_rule", "unknown"),
        }
    elif item.metadata.get("prompt_generation"):
        prompt_generation = item.metadata["prompt_generation"]
        prompt_metadata = {
            "PromptProtocol": prompt_generation.get("protocol", "unknown"),
            "PromptSource": prompt_generation.get("source", "unknown"),
            "PromptBoxVariant": "unused",
            "PromptTightBoxRule": prompt_generation.get("tight_box_rule", "unknown"),
            "PromptLooseBoxRule": prompt_generation.get("loose_box_rule", "unknown"),
            "PromptLooseBoxPadRatio": prompt_generation.get("loose_box_pad_ratio"),
            "PromptLooseBoxMinPad": prompt_generation.get("loose_box_min_pad"),
            "PromptLooseBoxMinSide": prompt_generation.get("loose_box_min_side"),
            "PromptLooseBoxMaxSideMultiplier": prompt_generation.get("loose_box_max_side_multiplier"),
            "PromptPointRule": prompt_generation.get("point_rule", "unknown"),
        }
    row = {
        "sample_id": item.sample_id,
        "frame_id": item.frame_id,
        "sequence_id": item.sequence_id,
        "eval_unit": "instance",
        "modality": modality,
        "supervision_type": item.supervision_type,
        "category_name": item.category,
        "target_scale": item.target_scale,
        "device_source": item.device_source,
        "annotation_protocol_flag": item.annotation_protocol_flag,
        "LatencyMs": elapsed_ms,
        "BBoxIoU": bbox_iou(pred_box, gt_box),
        "PredAreaRatio": pred_area / float(height * width),
        "PredAreaPixels": pred_area,
        **(mask_alignment or {}),
        **prompt_metadata,
    }
    if has_mask_gt:
        # mask-supervised 数据集输出完整分割指标；bbox-only 数据集只输出 bbox/prompt 可比指标。
        exact_boundary_f1 = boundary_f1(pred_mask, gt_mask)
        row.update(
            {
                "mIoU": mask_iou(pred_mask, gt_mask),
                "Dice": dice_score(pred_mask, gt_mask),
                "BoundaryF1": exact_boundary_f1,
                "BoundaryF1Exact": exact_boundary_f1,
                "BoundaryF1Tol1": boundary_f1_tolerance(pred_mask, gt_mask, radius=1),
                "TightBoxMaskIoU": mask_iou(box_to_area(item.bbox_tight, item.height, item.width), gt_mask) if item.bbox_tight else 0.0,
                "LooseBoxMaskIoU": mask_iou(box_to_area(item.bbox_loose, item.height, item.width), gt_mask) if item.bbox_loose else 0.0,
                "GTAreaPixels": gt_area,
                "GTAreaRatio": gt_area / float(height * width),
                **small_target_metrics(pred_mask, gt_mask),
                **prompt_metrics(prompt, gt_mask),
            }
        )
    elif prompt:
        prompt_box = prompt.get("box")
        prompt_point = prompt.get("point")
        if isinstance(prompt_box, list):
            row["PromptBoxBBoxIoU"] = bbox_iou([float(value) for value in prompt_box[:4]], item.bbox_tight)
        if isinstance(prompt_point, list) and len(prompt_point) >= 2 and item.bbox_tight is not None:
            x, y = float(prompt_point[0]), float(prompt_point[1])
            x1, y1, x2, y2 = item.bbox_tight
            row["PromptPointInBBox"] = 1.0 if x1 <= x <= x2 and y1 <= y <= y2 else 0.0
    if item.track_id is not None:
        row["track_id"] = item.track_id
    return row


def box_to_area(box: list[float] | None, height: int, width: int) -> np.ndarray:
    if box is None:
        return np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1.0
    return mask
