from __future__ import annotations

import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable, Literal


@dataclass(frozen=True)
class RunContext:
    out_root: Path
    conf: float
    iou: float


ProgressCb = Callable[[dict], None]
StageName = Literal[
    "tiling",
    "stage1_normal",
    "stage2_goal_only",
    "stage2_goal_from_removed",
    "stage2_goal_from_only",
    "stage2_pairs",
    "stage2_goal_base",
    "summary",
]


def append_summary_row(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def yolo_boxes_to_yolo_txt(box_xyxy, cls_id: int, w: int, h: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def parse_tile_xy(tile_path: Path) -> tuple[int, int]:
    m = re.match(r"x(\d+)_y(\d+)\.(?:tif{1,2}|png)$", tile_path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"tile文件名不符合 x<数字>_y<数字>.(tif/tiff/png): {tile_path.name}")
    return int(m.group(1)), int(m.group(2))


def _maybe_report(
    cb: ProgressCb | None,
    *,
    stage: str,
    i: int,
    total: int,
    t0: float,
    message: str = "",
    force: bool = False,
    last_emit: list[float] | None = None,
) -> None:
    if cb is None:
        return

    now = time.perf_counter()
    if (not force) and last_emit is not None:
        if now - last_emit[0] < 0.3 and i != total:
            return

    elapsed = now - t0
    rate = (i / elapsed) if elapsed > 0 else 0.0
    eta = ((total - i) / rate) if rate > 0 else None
    progress = (i * 100.0 / total) if total else 100.0

    cb(
        {
            "status": "running",
            "stage": stage,
            "progress": float(progress),
            "elapsed": float(elapsed),
            "eta": float(eta) if eta is not None else None,
            "message": message,
            "current": i,
            "total": total,
            "rate": float(rate),
        }
    )

    if last_emit is not None:
        last_emit[0] = now


# 兼容旧切片实现：code/main.py 会从 code.pipeline import tile_worker

def tile_worker(job: tuple[str, int, int, int, int, str]) -> str:
    """Top-level worker for Windows spawn multiprocessing (must be pickleable)."""
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    in_path_s, x, y, w, h, out_path_s = job

    import rasterio  # type: ignore
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(in_path_s) as ds:
                win = Window(x, y, w, h)
                data = ds.read(window=win, boundless=False)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    arr = np.transpose(data, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        amin = float(arr.min())
        amax = float(arr.max())
        if amax > amin:
            arr8 = ((arr.astype("float32") - amin) / (amax - amin) * 255.0).clip(0, 255).astype("uint8")
        else:
            arr8 = (arr.astype("float32") * 0).astype("uint8")
    else:
        arr8 = arr

    Image.fromarray(arr8, mode="RGB").save(out_path_s, compress_level=1)
    return out_path_s


def _compute_background_u8(img_u8, mask):
    import numpy as np  # type: ignore

    if img_u8.ndim == 2:
        img3 = np.expand_dims(img_u8, axis=-1)
    else:
        img3 = img_u8
    if img3.shape[-1] == 1:
        img3 = np.repeat(img3, 3, axis=-1)

    try:
        non = img3[~mask]
        if non.size > 0:
            bg = np.median(non.reshape(-1, img3.shape[-1]), axis=0).astype(np.uint8)
            return bg
    except Exception:
        pass

    bg = np.median(img3.reshape(-1, img3.shape[-1]), axis=0).astype(np.uint8)
    return bg


def _make_mask_from_boxes(h: int, w: int, boxes_xyxy):
    """用 OpenCV 批量填充矩形，比 Python for 循环快很多"""
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    mask = np.zeros((h, w), dtype=np.uint8)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return mask.astype(bool)

    boxes_int = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            boxes_int.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    if boxes_int:
        cv2.fillPoly(mask, np.array(boxes_int, dtype=np.int32), 255)

    return mask.astype(bool)


def _goal_class_color(cls_id: int) -> tuple[int, int, int]:
    return (255, 0, 0) if int(cls_id) == 0 else (0, 255, 0)


def _nms_xyxy(xyxy, scores, iou_thr: float):
    """纯 numpy NMS，输入 xyxy(N,4), scores(N,) -> keep indices"""
    import numpy as np  # type: ignore

    if xyxy is None or len(xyxy) == 0:
        return np.array([], dtype=np.int64)

    boxes = np.asarray(xyxy, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = sc.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= float(iou_thr))[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


# ===== 新增：normal_only 模式（直接预测 normal，输出左右 pair） =====

def run_normal_only_pairs(
    *,
    tiles_dir: Path,
    ctx: RunContext,
    weights_normal: Path,
    pairs_subdir: str = "normal_only_pairs",
    gap: int = 60,
    png_compression: int = 0,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    """仅预测 normal：输出左右 pair 图

    - 左图：从原图中“去掉 normal 框区域”（用背景填充）
    - 右图：只展示 normal 框区域（其余用背景填充）

    normal 的 conf/iou 来自 ctx.conf/ctx.iou（可选）。
    """

    import numpy as np  # type: ignore
    import cv2  # type: ignore
    import tifffile as tiff  # type: ignore
    from PIL import Image  # type: ignore
    from ultralytics import YOLO  # type: ignore

    pairs_dir = ctx.out_root / "stage2" / pairs_subdir
    labels_dir = ctx.out_root / "labels" / "normal_only"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.tif")) + list(tiles_dir.glob("*.tiff")))
    total = len(tile_paths)
    if total == 0:
        raise RuntimeError(f"No tiles found in: {tiles_dir}")

    model = YOLO(str(weights_normal))

    t0 = time.perf_counter()
    last_emit = [t0]

    for idx, tile_path in enumerate(tile_paths, start=1):
        if cancel_cb and cancel_cb():
            break

        if tile_path.suffix.lower() == ".png":
            img = np.array(Image.open(str(tile_path)).convert("RGB"))
        else:
            img = tiff.imread(str(tile_path))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        h, w = int(img.shape[0]), int(img.shape[1])

        r = model.predict(source=img, conf=float(ctx.conf), iou=float(ctx.iou), verbose=False)[0]
        boxes = r.boxes

        # 取 normal boxes
        n_xyxy = np.zeros((0, 4), dtype=np.float32)
        n_cls = np.zeros((0,), dtype=np.int32)
        if boxes is not None and len(boxes) > 0:
            n_xyxy = boxes.xyxy.cpu().numpy()
            n_cls = boxes.cls.cpu().numpy().astype(int)

        # 写 labels（便于排查）
        lines: list[str] = []
        if len(n_xyxy) > 0:
            for b, c in zip(n_xyxy, n_cls, strict=False):
                lines.append(yolo_boxes_to_yolo_txt(b, int(c), w, h))
        (labels_dir / f"{tile_path.stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        normal_mask = _make_mask_from_boxes(h, w, n_xyxy)

        # 背景（取非 mask 中位数）
        bg = _compute_background_u8(img.astype(np.uint8) if img.dtype == np.uint8 else img.astype(np.uint8), normal_mask)

        img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 左图：去掉 normal 区域
        left_bgr = img_bgr.copy()
        if normal_mask.any():
            left_bgr[normal_mask] = bg[::-1]

        # 右图：只保留 normal 区域
        right_bgr = np.full((h, w, 3), bg[::-1], dtype=np.uint8)
        if normal_mask.any():
            right_bgr[normal_mask] = img_bgr[normal_mask]

        gap_px = max(0, int(gap))
        canvas_w = w + gap_px + w
        canvas = np.full((h, canvas_w, 3), 255, dtype=np.uint8)
        canvas[:, 0:w, :] = left_bgr
        canvas[:, w + gap_px : w + gap_px + w, :] = right_bgr

        out_path = pairs_dir / f"{tile_path.stem}.png"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])

        _maybe_report(
            progress_cb,
            stage="stage2_pairs",
            i=idx,
            total=total,
            t0=t0,
            message=f"normal_only pair: {out_path.name}",
            last_emit=last_emit,
        )

    _maybe_report(progress_cb, stage="stage2_pairs", i=total, total=total, t0=t0, message="normal_only done", force=True)
    return pairs_dir


# ===== 原 two-stage streaming 实现（保持不动） =====

def run_two_stage_streaming_pairs(
    *,
    in_path: Path,
    out_dir: Path,
    tile_size: int = 1280,
    overlap: float = 0.0,
    tile_workers: int = 0,
    weights_normal: Path,
    weights_goal: Path,
    stage1_conf: float,
    stage1_iou: float,
    stage2_conf: float,
    stage2_iou: float,
    batch_size: int = 64,
    device: str = "0",
    half: bool = True,
    pairs_gap: int = 60,
    max_batch: int = 128,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    """两阶段流水线 v4：

    阶段1（目标5分钟）：只做推理，快速得到 labels，最大化 GPU 利用率
    阶段2（目标5分钟）：基于 labels 并行渲染图片，最大化 CPU 利用率

    - 不落盘 tiles
    - 阶段1：read_window + normal/goal GPU 推理 + 写 labels（无渲染）
    - 阶段2：多线程并行读取 tiles + 渲染 pairs
    - 去重：对同一 tile 的 goal 框按 cls 做额外 NMS

    约束：仅 normal 无框才跳过 goal。
    """

    import numpy as np  # type: ignore
    import torch
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from .main import compute_tiles, get_tiff_info  # type: ignore

    labels_normal_dir = out_dir / "labels" / "stage1_normal"
    labels_goal_dir = out_dir / "labels" / "stage2_goal_base"
    pairs_dir = out_dir / "stage2" / "pairs"
    for d in [labels_normal_dir, labels_goal_dir, pairs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and device not in {"cpu", "-1"}
    half = bool(half and use_cuda)
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "N/A"

    try_batches = [int(max_batch), 96, 64, 32]
    try_batches = [b for b in try_batches if b > 0]

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "config",
            "message": f"mem_pipeline_v3 CUDA={use_cuda} GPU={gpu_name} FP16={half} batch={batch_size} max_batch={max_batch} overlap={overlap}",
        })

    info = get_tiff_info(in_path)
    shape = info.get("shape")
    if not shape or len(shape) < 2:
        raise RuntimeError(f"不支持的tiff shape: {shape}")
    height = int(shape[0])
    width = int(shape[1])

    tiles = compute_tiles(width=width, height=height, tile=tile_size, overlap=overlap)
    total_tiles = len(tiles)

    model_n = YOLO(str(weights_normal))
    model_g = YOLO(str(weights_goal))

    try:
        dummy = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        _ = model_n.predict(source=[dummy] * 2, batch=2, conf=float(stage1_conf), iou=float(stage1_iou), device=device, half=half, verbose=False)
        _ = model_g.predict(source=[dummy] * 2, batch=2, conf=float(stage2_conf), iou=float(stage2_iou), device=device, half=half, verbose=False)
    except Exception:
        pass

    import rasterio  # type: ignore
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    import warnings

    t_all0 = time.perf_counter()

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "stage1_normal",
            "message": "阶段1：开始推理（只生成 labels，不渲染）",
        })

    t_infer0 = time.perf_counter()
    t_read = 0.0
    t_n = 0.0
    t_g = 0.0

    processed = 0
    goal_count = 0
    last_emit = [t_infer0]

    # 注意：stage1_normal 使用 effective_batch 控制每次送入 GPU 的 tile 数量。
    # 这里必须优先使用 batch_size（来自 UI/参数），否则会被 max_batch(默认128)直接顶满，极易 OOM。
    effective_batch = int(max(1, min(int(batch_size), int(max_batch))))
    tile_meta_list: list[tuple[str, int, int, int, int]] = []

    def _read_tile(ds, x: int, y: int, w: int, h: int) -> np.ndarray:
        win = Window(x, y, w, h)
        data = ds.read(window=win, boundless=False)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        arr = np.transpose(data, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            amin = float(arr.min())
            amax = float(arr.max())
            if amax > amin:
                arr8 = ((arr.astype("float32") - amin) / (amax - amin) * 255.0).clip(0, 255).astype("uint8")
            else:
                arr8 = (arr.astype("float32") * 0).astype("uint8")
            return arr8
        return arr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(str(in_path)) as ds:
                i = 0
                while i < total_tiles:
                    if cancel_cb and cancel_cb():
                        break

                    t0 = time.perf_counter()
                    batch_tiles = []
                    batch_meta = []
                    for _ in range(effective_batch):
                        if i >= total_tiles:
                            break
                        x, y, w, h = tiles[i]
                        arr = _read_tile(ds, x, y, w, h)
                        stem = f"x{x}_y{y}"
                        batch_tiles.append(arr)
                        batch_meta.append((stem, x, y, w, h))
                        i += 1
                    t_read += time.perf_counter() - t0

                    if not batch_tiles:
                        break

                    # normal - OOM 自动回退 batch
                    t0 = time.perf_counter()
                    while True:
                        try:
                            with torch.inference_mode():
                                res_n = model_n.predict(
                                    source=batch_tiles,
                                    batch=min(int(effective_batch), len(batch_tiles)),
                                    conf=float(stage1_conf),
                                    iou=float(stage1_iou),
                                    device=device,
                                    half=half,
                                    verbose=False,
                                )
                            if not isinstance(res_n, list):
                                res_n = [res_n] if res_n is not None else []
                            else:
                                res_n = list(res_n)
                            break
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            msg = str(e).lower()
                            if "out of memory" in msg or "cuda" in msg:
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                # 降 batch 并重试
                                next_b = 96 if effective_batch > 96 else 64 if effective_batch > 64 else 32 if effective_batch > 32 else 16
                                if next_b >= effective_batch:
                                    raise
                                effective_batch = int(next_b)
                                if progress_cb:
                                    progress_cb({
                                        "status": "running",
                                        "stage": "stage1_normal",
                                        "message": f"[OOM] normal 降 batch 重试: effective_batch={effective_batch}",
                                    })
                                continue
                            raise

                    if len(res_n) != len(batch_tiles):
                        raise RuntimeError(f"normal 推理失败：期望 {len(batch_tiles)} 个结果，得到 {len(res_n)}")
                    t_n += time.perf_counter() - t0

                    goal_tiles = []
                    goal_stems = []
                    for meta, rn, arr in zip(batch_meta, res_n, batch_tiles, strict=False):
                        stem = meta[0]
                        has_n = rn.boxes is not None and len(rn.boxes) > 0
                        if has_n:
                            goal_tiles.append(arr)
                            goal_stems.append(stem)

                        hh, ww = int(rn.orig_shape[0]), int(rn.orig_shape[1])
                        lines: list[str] = []
                        if has_n:
                            xyxy = rn.boxes.xyxy.cpu().numpy()
                            cls = rn.boxes.cls.cpu().numpy().astype(int)
                            for b, c in zip(xyxy, cls, strict=False):
                                lines.append(yolo_boxes_to_yolo_txt(b, int(c), ww, hh))
                        (labels_normal_dir / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

                    res_g_map: dict[str, object] = {}
                    if goal_tiles:
                        goal_count += len(goal_tiles)
                        t0 = time.perf_counter()
                        try:
                            with torch.inference_mode():
                                res_g = model_g.predict(
                                    source=goal_tiles,
                                    batch=min(int(effective_batch), len(goal_tiles)),
                                    conf=float(stage2_conf),
                                    iou=float(stage2_iou),
                                    device=device,
                                    half=half,
                                    verbose=False,
                                )
                            if not isinstance(res_g, list):
                                res_g = [res_g] if res_g is not None else []
                            else:
                                res_g = list(res_g)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            msg = str(e).lower()
                            if "out of memory" in msg or "cuda" in msg:
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                # OOM：分批处理当前 goal_tiles
                                fallback_batch = 32
                                if progress_cb:
                                    progress_cb({
                                        "status": "running",
                                        "stage": "stage1_normal",
                                        "message": f"[OOM] goal OOM，回退到 batch={fallback_batch} 分批处理",
                                    })
                                res_g = []
                                for chunk_start in range(0, len(goal_tiles), fallback_batch):
                                    chunk = goal_tiles[chunk_start : chunk_start + fallback_batch]
                                    with torch.inference_mode():
                                        chunk_res = model_g.predict(
                                            source=chunk,
                                            batch=len(chunk),
                                            conf=float(stage2_conf),
                                            iou=float(stage2_iou),
                                            device=device,
                                            half=half,
                                            verbose=False,
                                        )
                                    if not isinstance(chunk_res, list):
                                        chunk_res = [chunk_res] if chunk_res is not None else []
                                    res_g.extend(chunk_res)
                            else:
                                raise

                        if len(res_g) != len(goal_tiles):
                            raise RuntimeError(f"goal 推理失败：期望 {len(goal_tiles)} 个结果，得到 {len(res_g)}")
                        t_g += time.perf_counter() - t0
                        for stem, rg in zip(goal_stems, res_g, strict=False):
                            res_g_map[stem] = rg

                    for meta, rn, arr in zip(batch_meta, res_n, batch_tiles, strict=False):
                        stem = meta[0]
                        rg = res_g_map.get(stem)

                        g_xyxy = np.zeros((0, 4), dtype=np.float32)
                        g_cls = np.zeros((0,), dtype=np.int32)
                        g_conf = np.zeros((0,), dtype=np.float32)
                        if rg is not None and getattr(rg, "boxes", None) is not None and len(rg.boxes) > 0:
                            g_xyxy = rg.boxes.xyxy.cpu().numpy()
                            g_cls = rg.boxes.cls.cpu().numpy().astype(int)
                            conf = getattr(rg.boxes, "conf", None)
                            if conf is not None:
                                g_conf = conf.cpu().numpy().astype(np.float32)
                            else:
                                g_conf = np.ones((len(g_cls),), dtype=np.float32)

                        if len(g_xyxy) > 1:
                            keep_all = []
                            for c in sorted(set(int(x) for x in g_cls.tolist())):
                                idxs = np.where(g_cls == c)[0]
                                if idxs.size == 0:
                                    continue
                                keep = _nms_xyxy(g_xyxy[idxs], g_conf[idxs], iou_thr=0.6)
                                keep_all.extend(idxs[keep].tolist())
                            keep_all = sorted(set(int(x) for x in keep_all))
                            g_xyxy = g_xyxy[keep_all] if keep_all else np.zeros((0, 4), dtype=np.float32)
                            g_cls = g_cls[keep_all] if keep_all else np.zeros((0,), dtype=np.int32)
                            g_conf = g_conf[keep_all] if keep_all else np.zeros((0,), dtype=np.float32)

                        hh, ww = int(rn.orig_shape[0]), int(rn.orig_shape[1])
                        glines: list[str] = []
                        if len(g_xyxy) > 0:
                            for b, c in zip(g_xyxy, g_cls, strict=False):
                                glines.append(yolo_boxes_to_yolo_txt(b, int(c), ww, hh))
                        (labels_goal_dir / f"{stem}.txt").write_text(("\n".join(glines) + "\n") if glines else "", encoding="utf-8")

                        tile_meta_list.append((stem, meta[1], meta[2], meta[3], meta[4]))

                        processed += 1
                        _maybe_report(progress_cb, stage="stage1_normal", i=processed, total=total_tiles, t0=t_infer0, message=f"推理 x={meta[1]} y={meta[2]}", last_emit=last_emit)

    t_infer1 = time.perf_counter()
    infer_time = t_infer1 - t_infer0
    print(f"[阶段1完成] 推理耗时={infer_time:.1f}s (目标300s) | t_read={t_read:.1f}s t_normal={t_n:.1f}s t_goal={t_g:.1f}s")
    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "stage2_pairs",
            "message": f"阶段1完成：推理耗时 {infer_time:.1f}s，开始阶段2渲染",
        })

    t_render0 = time.perf_counter()

    def _read_tile_for_render(ds, x: int, y: int, w: int, h: int) -> np.ndarray:
        win = Window(x, y, w, h)
        data = ds.read(window=win, boundless=False)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        arr = np.transpose(data, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            amin = float(arr.min())
            amax = float(arr.max())
            if amax > amin:
                arr8 = ((arr.astype("float32") - amin) / (amax - amin) * 255.0).clip(0, 255).astype("uint8")
            else:
                arr8 = (arr.astype("float32") * 0).astype("uint8")
            return arr8
        return arr

    def render_single_tile(stem: str, x: int, y: int, w: int, h: int, in_path_str: str) -> tuple[str, bool, float]:
        try:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
                    with rasterio.open(in_path_str) as ds_local:
                        arr = _read_tile_for_render(ds_local, x, y, w, h)

            h_img, w_img = int(arr.shape[0]), int(arr.shape[1])

            normal_label_path = labels_normal_dir / f"{stem}.txt"
            goal_label_path = labels_goal_dir / f"{stem}.txt"

            n_xyxy = np.zeros((0, 4), dtype=np.float32)
            if normal_label_path.exists():
                lines = normal_label_path.read_text(encoding="utf-8").strip().split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (xc - bw / 2) * w_img
                        y1 = (yc - bh / 2) * h_img
                        x2 = (xc + bw / 2) * w_img
                        y2 = (yc + bh / 2) * h_img
                        n_xyxy = np.vstack([n_xyxy, [x1, y1, x2, y2]])

            g_xyxy = np.zeros((0, 4), dtype=np.float32)
            g_cls = np.zeros((0,), dtype=np.int32)
            if goal_label_path.exists():
                lines = goal_label_path.read_text(encoding="utf-8").strip().split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, xc, yc, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (xc - bw / 2) * w_img
                        y1 = (yc - bh / 2) * h_img
                        x2 = (xc + bw / 2) * w_img
                        y2 = (yc + bh / 2) * h_img
                        g_xyxy = np.vstack([g_xyxy, [x1, y1, x2, y2]])
                        g_cls = np.append(g_cls, int(cls_id))

            # 规则（框级别）：
            # - 若某个 normal 框与任意 goal 框的相交面积 / normal 框面积 >= 10%，则该 normal 框整块不去除（保留）
            # - 其余 normal 框整块去除：左图抹掉，右图保留（R1：normal_only）

            def _keep_normal_by_overlap(n_boxes: np.ndarray, g_boxes: np.ndarray, thr: float = 0.10) -> np.ndarray:
                if n_boxes is None or len(n_boxes) == 0:
                    return np.zeros((0,), dtype=bool)
                if g_boxes is None or len(g_boxes) == 0:
                    return np.zeros((len(n_boxes),), dtype=bool)

                n = np.asarray(n_boxes, dtype=np.float32)
                g = np.asarray(g_boxes, dtype=np.float32)

                nx1, ny1, nx2, ny2 = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
                na = ((nx2 - nx1).clip(min=0) * (ny2 - ny1).clip(min=0))
                keep = np.zeros((len(n),), dtype=bool)

                for i in range(len(n)):
                    if na[i] <= 0:
                        continue
                    ix1 = np.maximum(nx1[i], g[:, 0])
                    iy1 = np.maximum(ny1[i], g[:, 1])
                    ix2 = np.minimum(nx2[i], g[:, 2])
                    iy2 = np.minimum(ny2[i], g[:, 3])
                    iw = (ix2 - ix1).clip(min=0)
                    ih = (iy2 - iy1).clip(min=0)
                    inter = iw * ih
                    if inter.size > 0 and float(inter.max() / (na[i] + 1e-9)) >= float(thr):
                        keep[i] = True
                return keep

            keep_mask_box = _keep_normal_by_overlap(n_xyxy, g_xyxy, thr=0.10)
            n_keep = n_xyxy[keep_mask_box] if len(n_xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)
            n_remove = n_xyxy[~keep_mask_box] if len(n_xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)

            remove_mask = _make_mask_from_boxes(h_img, w_img, n_remove)

            # 左图：从原图中“去掉可去除 normal 框区域”（用背景填充），然后画 goal 框（CTC红/CEC绿）
            bg_left = _compute_background_u8(arr, remove_mask)
            left_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if remove_mask.any():
                left_bgr[remove_mask] = bg_left[::-1]

            if len(g_xyxy) > 0:
                for b, c in zip(g_xyxy, g_cls, strict=False):
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    color = (0, 0, 255) if int(c) == 0 else (0, 255, 0)
                    cv2.rectangle(left_bgr, (x1, y1), (x2, y2), color, 3)

            # 右图（R1：normal_only）：仅保留“可去除 normal 框区域”，其余用背景填充
            bg_right = _compute_background_u8(arr, remove_mask)
            right_bgr = np.full((h_img, w_img, 3), bg_right[::-1], dtype=np.uint8)
            if remove_mask.any():
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                right_bgr[remove_mask] = arr_bgr[remove_mask]

            gap = max(0, int(pairs_gap))
            canvas_w = w_img + gap + w_img
            canvas = np.full((h_img, canvas_w, 3), 255, dtype=np.uint8)
            canvas[:, 0:w_img, :] = left_bgr
            canvas[:, w_img + gap : w_img + gap + w_img, :] = right_bgr

            out_path = pairs_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            dt = time.perf_counter() - t0
            return stem, True, float(dt)
        except Exception as e:
            print(f"渲染 tile {stem} 失败: {e}")
            import traceback

            traceback.print_exc()
            return stem, False, 0.0

    render_workers = min(32, max(4, int((os.cpu_count() or 8) * 0.75)))
    render_ok = 0
    render_fail = 0

    in_path_str = str(in_path)
    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = []
        for stem, x, y, w, h in tile_meta_list:
            if cancel_cb and cancel_cb():
                break
            future = executor.submit(render_single_tile, stem, x, y, w, h, in_path_str)
            futures.append(future)

        t_render = 0.0
        for idx, future in enumerate(futures, 1):
            if cancel_cb and cancel_cb():
                break
            stem, ok, dt = future.result()
            t_render += float(dt)
            if ok:
                render_ok += 1
            else:
                render_fail += 1
            if progress_cb and idx % 100 == 0:
                progress_cb({
                    "status": "running",
                    "stage": "stage2_pairs",
                    "progress": (idx / len(futures)) * 100,
                    "message": f"渲染进度: {idx}/{len(futures)}",
                })

    t_render1 = time.perf_counter()
    render_time = t_render1 - t_render0

    t_all1 = time.perf_counter()
    total_time = t_all1 - t_all0
    summary_msg = (
        f"done tiles={processed}/{total_tiles} goal_ran={goal_count} total={total_time:.1f}s "
        f"rate={processed / total_time:.2f} tile/s | "
        f"[阶段1推理] t_read={t_read:.1f}s t_normal={t_n:.1f}s t_goal={t_g:.1f}s 耗时={infer_time:.1f}s | "
        f"[阶段2渲染] t_render={t_render:.1f}s 耗时={render_time:.1f}s ok={render_ok} fail={render_fail}"
    )
    print("[SUMMARY]", summary_msg)
    if progress_cb:
        progress_cb({"status": "running", "stage": "summary", "message": summary_msg})

    return pairs_dir



def create_two_stage_pairs(
    *,
    out_root: Path,
    removed_dir: Path,
    only_dir: Path,
    pairs_subdir: str = "pairs",
    gap: int = 10,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    from PIL import Image

    pairs_dir = out_root / "stage2" / pairs_subdir
    pairs_dir.mkdir(parents=True, exist_ok=True)

    removed_map: dict[str, Path] = {p.stem: p for p in sorted(removed_dir.glob("*.png"))}
    only_map: dict[str, Path] = {p.stem: p for p in sorted(only_dir.glob("*.png"))}

    keys = sorted(set(removed_map.keys()) | set(only_map.keys()))
    total = len(keys)
    if total == 0:
        return pairs_dir

    t0 = time.perf_counter()
    last_emit = [t0]

    for idx, stem in enumerate(keys, start=1):
        if cancel_cb and cancel_cb():
            break

        removed_p = removed_map.get(stem)
        only_p = only_map.get(stem)

        img_removed = Image.open(removed_p).convert("RGB") if removed_p else None
        img_only = Image.open(only_p).convert("RGB") if only_p else None

        h = 0
        w1 = 0
        w2 = 0
        if img_removed is not None:
            w1, h = img_removed.size
        if img_only is not None:
            w2, h2 = img_only.size
            if h == 0:
                h = h2
            elif h2 != h:
                img_only = img_only.resize((w2, h))
        if img_removed is None and img_only is None:
                    continue

        if img_removed is None:
            img_removed = Image.new("RGB", (w2, h), (0, 0, 0))
            w1 = w2
        if img_only is None:
            img_only = Image.new("RGB", (w1, h), (0, 0, 0))
            w2 = w1

        gap = max(0, int(gap))
        canvas = Image.new("RGB", (w1 + gap + w2, h), (0, 0, 0))
        canvas.paste(img_removed, (0, 0))
        canvas.paste(img_only, (w1 + gap, 0))

        out_path = pairs_dir / f"{stem}.png"
        canvas.save(str(out_path))

        _maybe_report(
            progress_cb,
            stage="stage2_pairs",
            i=idx,
            total=total,
            t0=t0,
            message=f"拼接结果: {out_path.name}",
            last_emit=last_emit,
        )

    _maybe_report(progress_cb, stage="stage2_pairs", i=total, total=total, t0=t0, message="拼接完成", force=True)
    return pairs_dir
