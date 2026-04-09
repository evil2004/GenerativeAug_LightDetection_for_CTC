from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


def run_normal_only_fast(
    *,
    tiles_dir: Path,
    out_root: Path,
    weights_normal: Path,
    conf: float,
    iou: float,
    device: str = "0",
    batch: int = 64,
    half: bool = True,
    pairs_subdir: str = "normal_only_pairs",
    gap: int = 60,
    png_compression: int = 0,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    """高性能 normal_only（优化版）：

    优化点：
    - 推理仍 batch=64（GPU）
    - CPU 渲染+写盘改为线程池并行，减少等待
    - PNG 读取优先用 cv2.imread（比 PIL 更快），tif 仍用 tifffile
    - 默认不再写 labels（避免大量小文件写入拖慢）

    输出：out_root/stage2/<pairs_subdir>/*.png
    """

    import numpy as np  # type: ignore
    import cv2  # type: ignore
    import tifffile as tiff  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.pipeline import _compute_background_u8, _make_mask_from_boxes, _maybe_report  # type: ignore

    pairs_dir = out_root / "stage2" / pairs_subdir
    pairs_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.tif")) + list(tiles_dir.glob("*.tiff")))
    total = len(tile_paths)
    if total == 0:
        raise RuntimeError(f"No tiles found in: {tiles_dir}")

    model = YOLO(str(weights_normal))

    t0 = time.perf_counter()
    last_emit = [t0]

    processed = 0
    t_read = 0.0
    t_infer = 0.0
    t_post = 0.0

    # 写盘/渲染线程池
    io_workers = min(16, max(4, int((os.cpu_count() or 8) * 0.75)))
    ex = ThreadPoolExecutor(max_workers=io_workers)
    pending = []

    def _render_and_write(
        *,
        stem: str,
        img_rgb: np.ndarray,
        n_xyxy: np.ndarray,
    ) -> bool:
        try:
            h, w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
            normal_mask = _make_mask_from_boxes(h, w, n_xyxy)
            bg = _compute_background_u8(img_rgb, normal_mask)

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            left_bgr = img_bgr.copy()
            if normal_mask.any():
                left_bgr[normal_mask] = bg[::-1]

            right_bgr = np.full((h, w, 3), bg[::-1], dtype=np.uint8)
            if normal_mask.any():
                right_bgr[normal_mask] = img_bgr[normal_mask]

            gap_px = max(0, int(gap))
            canvas_w = w + gap_px + w
            canvas = np.full((h, canvas_w, 3), 255, dtype=np.uint8)
            canvas[:, 0:w, :] = left_bgr
            canvas[:, w + gap_px : w + gap_px + w, :] = right_bgr

            out_path = pairs_dir / f"{stem}.png"
            return bool(cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]))
        except Exception:
            return False

    for start in range(0, total, int(max(1, batch))):
        if cancel_cb and cancel_cb():
            break

        chunk = tile_paths[start : start + int(max(1, batch))]
        if not chunk:
            break

        # 读图
        t1 = time.perf_counter()
        imgs: list[np.ndarray] = []
        for p in chunk:
            if p.suffix.lower() == ".png":
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"读取失败: {p}")
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                img = tiff.imread(str(p))
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
            imgs.append(img)
        t_read += time.perf_counter() - t1

        # 推理
        t2 = time.perf_counter()
        results = model.predict(
            source=imgs,
            conf=float(conf),
            iou=float(iou),
            device=str(device),
            half=bool(half),
            batch=len(imgs),
            stream=False,
            verbose=False,
        )
        if not isinstance(results, list):
            results = list(results) if results is not None else []
        if len(results) != len(imgs):
            raise RuntimeError(f"normal_only_fast: 期望 {len(imgs)} 个结果，得到 {len(results)}")
        t_infer += time.perf_counter() - t2

        # 提交渲染写盘任务
        t3 = time.perf_counter()
        for p, img, r in zip(chunk, imgs, results, strict=False):
            boxes = getattr(r, "boxes", None)
            n_xyxy = np.zeros((0, 4), dtype=np.float32)
            if boxes is not None and len(boxes) > 0:
                n_xyxy = boxes.xyxy.cpu().numpy()

            pending.append(ex.submit(_render_and_write, stem=p.stem, img_rgb=img, n_xyxy=n_xyxy))

        # 控制队列长度，避免内存爆
        if len(pending) > io_workers * 8:
            done_future = pending.pop(0)
            _ = done_future.result()

        t_post += time.perf_counter() - t3

        # 更新进度（按已提交的任务数估算）
        processed = min(total, start + len(chunk))
        _maybe_report(
            progress_cb,
            stage="stage2_pairs",
            i=processed,
            total=total,
            t0=t0,
            message=f"normal_only_fast batch done: {processed}/{total}",
            last_emit=last_emit,
        )

    # 等待剩余写盘
    for fu in pending:
        if cancel_cb and cancel_cb():
            break
        _ = fu.result()

    ex.shutdown(wait=False, cancel_futures=False)

    total_time = time.perf_counter() - t0
    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "summary",
            "message": (
                f"normal_only_fast done total={total_time:.1f}s tiles={processed}/{total} "
                f"t_read={t_read:.1f}s t_infer={t_infer:.1f}s t_dispatch={t_post:.1f}s io_workers={io_workers}"
            ),
            "progress": 100.0,
            "elapsed": float(total_time),
            "eta": 0,
        })

    _maybe_report(progress_cb, stage="stage2_pairs", i=processed, total=total, t0=t0, message="normal_only done", force=True)
    return pairs_dir
