from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Callable


def get_tiff_info(in_path: Path) -> dict:
    import rasterio  # type: ignore

    with rasterio.open(str(in_path)) as ds:
        return {"shape": (ds.height, ds.width, ds.count), "width": ds.width, "height": ds.height, "count": ds.count}


def compute_tiles(*, width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile = int(max(64, tile))
    stride = max(1, int(round(tile * (1.0 - float(overlap)))))
    xs = list(range(0, max(1, width), stride))
    ys = list(range(0, max(1, height), stride))
    if xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    out = []
    seen = set()
    for y in ys:
        for x in xs:
            w = min(tile, width - x)
            h = min(tile, height - y)
            key = (x, y, w, h)
            if w > 0 and h > 0 and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _progress(cb, *, stage: str, i: int, total: int, t0: float, message: str) -> None:
    if cb:
        cb({"stage": stage, "current": i, "total": total, "progress": i / max(1, total) * 100.0, "elapsed": time.time() - t0, "message": message})


def tile_big_tiff(
    *,
    in_path: Path,
    tiles_dir: Path,
    tile: int = 1280,
    overlap: float = 0.0,
    workers: int = 0,
    skip_black_tiles: bool = True,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import rasterio  # type: ignore
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import warnings

    tiles_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    thread_state = threading.local()

    def auto_workers(requested: int) -> int:
        if int(requested or 0) > 0:
            return max(1, min(int(requested), 16))
        cpu = os.cpu_count() or 8
        # Leave a little room for the desktop and browser. TIFF window reads
        # and PNG writes are I/O heavy, so a few more workers than pure CPU
        # cores is usually useful, but cap it to avoid freezing 3060 machines.
        return max(4, min(12, int(cpu * 0.75)))

    def is_mostly_black_bgr(img_bgr, *, black_thr: int = 8, ratio_thr: float = 0.9995) -> bool:
        dark = img_bgr.max(axis=2) <= int(black_thr)
        return float(dark.mean()) >= float(ratio_thr)

    def read_convert_write(tile_item: tuple[int, int, int, int]) -> bool:
        x, y, w, h = tile_item
        ds = getattr(thread_state, "ds", None)
        if ds is None:
            thread_state.ds = rasterio.open(str(in_path))
            ds = thread_state.ds
        data = ds.read(window=Window(x, y, w, h), boundless=False)
        arr = np.transpose(data, (1, 2, 0))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            amin, amax = float(arr.min()), float(arr.max())
            arr = ((arr.astype("float32") - amin) / max(amax - amin, 1e-6) * 255.0).clip(0, 255).astype("uint8")
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        if bool(skip_black_tiles) and is_mostly_black_bgr(bgr):
            return False
        out_path = tiles_dir / f"x{x}_y{y}.png"
        ok = cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise RuntimeError(f"Failed to write tile: {out_path}")
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(str(in_path)) as ds:
                tiles = compute_tiles(width=ds.width, height=ds.height, tile=int(tile), overlap=float(overlap))
                total = len(tiles)
            max_workers = auto_workers(int(workers or 0))
            max_pending = max_workers * 6
            done = 0
            skipped = 0
            submitted = 0
            pending = set()
            last_report_time = 0.0
            last_report_done = 0

            _progress(progress_cb, stage="tiling", i=0, total=total, t0=t0, message=f"tiling start: {total} tiles, workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for tile_item in tiles:
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("cancelled")
                    pending.add(executor.submit(read_convert_write, tile_item))
                    submitted += 1

                    while len(pending) >= max_pending:
                        finished, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                        for fut in finished:
                            if not fut.result():
                                skipped += 1
                            done += 1
                        now = time.time()
                        if done == total or done - last_report_done >= 5 or now - last_report_time >= 0.5:
                            last_report_time = now
                            last_report_done = done
                            _progress(progress_cb, stage="tiling", i=done, total=total, t0=t0, message=f"tile {done}/{total} | skipped blank {skipped} | queued {submitted}/{total} | workers={max_workers}")

                while pending:
                    finished, pending = wait(pending, timeout=0.15, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        if not fut.result():
                            skipped += 1
                        done += 1
                    now = time.time()
                    if done == total or done - last_report_done >= 5 or now - last_report_time >= 0.5:
                        last_report_time = now
                        last_report_done = done
                        _progress(progress_cb, stage="tiling", i=done, total=total, t0=t0, message=f"tile {done}/{total} | skipped blank {skipped} | workers={max_workers}")
    return tiles_dir
