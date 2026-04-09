from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


ProgressCb = Callable[[dict], None]


def run_two_stage_goal_first_pairs(
    *,
    in_path: Path,
    out_dir: Path,
    weights_goal: Path,
    weights_normal: Path,
    tile_size: int = 1280,
    overlap: float = 0.10,
    batch_size: int = 64,
    device: str = "0",
    half: bool = True,
    pairs_gap: int = 60,
    goal_conf: float = 0.25,
    goal_iou: float = 0.45,
    normal_conf: float = 0.25,
    normal_iou: float = 0.45,
    normal_keep_overlap_ratio: float = 0.10,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    """两阶段（goal-first）流水线（旧实现）：

    当前实现从大图按 window 读取 tile（不落盘 tiles）。

    注意：为了让 two_stage 的 goal/normal 预测与 goal_only/normal_only 保持一致，
    Web 端已改为优先使用 :func:`run_two_stage_goal_first_pairs_from_tiles`。
    这里保留旧实现以兼容可能的 CLI/历史调用。
    """

    import numpy as np  # type: ignore
    import torch
    import cv2  # type: ignore
    import rasterio  # type: ignore
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    import warnings
    from ultralytics import YOLO  # type: ignore

    from code.main import compute_tiles, get_tiff_info  # type: ignore
    from code.pipeline import _compute_background_u8, _make_mask_from_boxes, _maybe_report  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = out_dir / "stage2" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and str(device) not in {"cpu", "-1"}
    half = bool(half and use_cuda)

    info = get_tiff_info(in_path)
    shape = info.get("shape")
    if not shape or len(shape) < 2:
        raise RuntimeError(f"不支持的tiff shape: {shape}")
    height = int(shape[0])
    width = int(shape[1])

    tiles = compute_tiles(width=width, height=height, tile=tile_size, overlap=overlap)
    total_tiles = len(tiles)

    model_g = YOLO(str(weights_goal))
    model_n = YOLO(str(weights_normal))

    # 预热
    try:
        dummy = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        _ = model_g.predict(source=[dummy] * 2, batch=2, conf=float(goal_conf), iou=float(goal_iou), device=device, half=half, verbose=False)
        _ = model_n.predict(source=[dummy] * 2, batch=2, conf=float(normal_conf), iou=float(normal_iou), device=device, half=half, verbose=False)
    except Exception:
        pass

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "config",
            "message": (
                f"two_stage_goal_first(legacy) CUDA={use_cuda} half={half} device={device} batch={batch_size} "
                f"tile={tile_size} overlap={overlap} goal(conf={goal_conf},iou={goal_iou}) "
                f"normal(conf={normal_conf},iou={normal_iou}) keep_overlap_ratio={normal_keep_overlap_ratio} tiles={total_tiles}"
            ),
        })

    t_infer0 = time.perf_counter()
    last_emit = [t_infer0]

    goal_boxes_cache: dict[str, tuple["np.ndarray", "np.ndarray"]] = {}
    normal_boxes_cache: dict[str, tuple["np.ndarray", "np.ndarray"]] = {}

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

    effective_batch = int(max(1, int(batch_size)))

    processed = 0
    t_read = 0.0
    t_goal = 0.0
    t_normal = 0.0

    if progress_cb:
        progress_cb({"status": "running", "stage": "stage1", "message": "阶段1：开始推理（legacy，window读取）"})

    tile_meta_list: list[tuple[str, int, int, int, int]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(str(in_path)) as ds:
                i = 0
                while i < total_tiles:
                    if cancel_cb and cancel_cb():
                        break

                    t0 = time.perf_counter()
                    batch_tiles: list[np.ndarray] = []
                    batch_meta: list[tuple[str, int, int, int, int]] = []
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

                    # goal batch predict
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        res_g = model_g.predict(
                            source=batch_tiles,
                            batch=len(batch_tiles),
                            conf=float(goal_conf),
                            iou=float(goal_iou),
                            device=device,
                            half=half,
                            verbose=False,
                            stream=False,
                        )
                    if not isinstance(res_g, list):
                        res_g = list(res_g) if res_g is not None else []
                    if len(res_g) != len(batch_tiles):
                        raise RuntimeError(f"goal 推理失败：期望 {len(batch_tiles)} 个结果，得到 {len(res_g)}")
                    t_goal += time.perf_counter() - t0

                    # normal batch predict
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        res_n = model_n.predict(
                            source=batch_tiles,
                            batch=len(batch_tiles),
                            conf=float(normal_conf),
                            iou=float(normal_iou),
                            device=device,
                            half=half,
                            verbose=False,
                            stream=False,
                        )
                    if not isinstance(res_n, list):
                        res_n = list(res_n) if res_n is not None else []
                    if len(res_n) != len(batch_tiles):
                        raise RuntimeError(f"normal 推理失败：期望 {len(batch_tiles)} 个结果，得到 {len(res_n)}")
                    t_normal += time.perf_counter() - t0

                    for (stem, x, y, w, h), rg, rn in zip(batch_meta, res_g, res_n, strict=False):
                        g_xyxy = np.zeros((0, 4), dtype=np.float32)
                        g_cls = np.zeros((0,), dtype=np.int32)
                        if getattr(rg, "boxes", None) is not None and len(rg.boxes) > 0:
                            g_xyxy = rg.boxes.xyxy.cpu().numpy()
                            g_cls = rg.boxes.cls.cpu().numpy().astype(int)
                        goal_boxes_cache[stem] = (g_xyxy, g_cls)

                        n_xyxy = np.zeros((0, 4), dtype=np.float32)
                        n_cls = np.zeros((0,), dtype=np.int32)
                        if getattr(rn, "boxes", None) is not None and len(rn.boxes) > 0:
                            n_xyxy = rn.boxes.xyxy.cpu().numpy()
                            n_cls = rn.boxes.cls.cpu().numpy().astype(int)
                        normal_boxes_cache[stem] = (n_xyxy, n_cls)

                        tile_meta_list.append((stem, x, y, w, h))
                        processed += 1
                        _maybe_report(progress_cb, stage="stage1", i=processed, total=total_tiles, t0=t_infer0, message=f"infer {stem}", last_emit=last_emit)

    infer_time = time.perf_counter() - t_infer0

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "stage2",
            "message": f"阶段1完成(legacy)：infer={infer_time:.1f}s t_read={t_read:.1f}s t_goal={t_goal:.1f}s t_normal={t_normal:.1f}s，开始阶段2渲染",
        })

    # legacy: 为避免复杂度，这里不再实现渲染；建议走 tiles 版本。
    raise RuntimeError("legacy two_stage 已弃用，请使用 run_two_stage_goal_first_pairs_from_tiles")


def run_two_stage_goal_first_pairs_from_tiles(
    *,
    tiles_dir: Path,
    out_dir: Path,
    weights_goal: Path,
    weights_normal: Path,
    batch_size: int = 64,
    device: str = "0",
    half: bool = True,
    pairs_gap: int = 60,
    goal_conf: float = 0.25,
    goal_iou: float = 0.45,
    normal_conf: float = 0.25,
    normal_iou: float = 0.45,
    normal_keep_overlap_ratio: float = 0.10,
    progress_cb: ProgressCb | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Path:
    """两阶段（goal-first）流水线（tiles 版，推荐）：

    目标：让 two_stage 的 goal/normal 推理输入与 goal_only_fast/normal_only_fast 一致，
    即直接使用切片后的 tiles 文件进行推理，从而避免原始大图（例如 0/1 二值）导致的分布差异。

    左图（L）：从原图 tile 开始 -> 去掉 normal -> 叠加 goal 框（CTC红，CEC绿）
    右图（R）：仅展示 normal 区域，但从 normal 中扣除 goal 区域

    输出：out_dir/stage2/pairs/*.png
    """

    import numpy as np  # type: ignore
    import torch
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.pipeline import _compute_background_u8, _make_mask_from_boxes, _maybe_report  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = out_dir / "stage2" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        raise RuntimeError(f"tiles_dir 不存在: {tiles_dir}")

    tile_paths = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.tif")) + list(tiles_dir.glob("*.tiff")))
    total = len(tile_paths)
    if total == 0:
        raise RuntimeError(f"tiles_dir 下没有找到切片: {tiles_dir}")

    use_cuda = torch.cuda.is_available() and str(device) not in {"cpu", "-1"}
    half = bool(half and use_cuda)

    model_g = YOLO(str(weights_goal))
    model_n = YOLO(str(weights_normal))

    # 预热
    try:
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = model_g.predict(source=[dummy] * 2, batch=2, conf=float(goal_conf), iou=float(goal_iou), device=device, half=half, verbose=False)
        _ = model_n.predict(source=[dummy] * 2, batch=2, conf=float(normal_conf), iou=float(normal_iou), device=device, half=half, verbose=False)
    except Exception:
        pass

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "config",
            "message": (
                f"two_stage_from_tiles CUDA={use_cuda} half={half} device={device} batch={batch_size} "
                f"goal(conf={goal_conf},iou={goal_iou}) normal(conf={normal_conf},iou={normal_iou}) "
                f"keep_overlap_ratio={normal_keep_overlap_ratio} tiles={total}"
            ),
        })

    t_infer0 = time.perf_counter()
    last_emit = [t_infer0]

    goal_boxes_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    normal_boxes_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    effective_batch = int(max(1, int(batch_size)))

    processed = 0
    if progress_cb:
        progress_cb({"status": "running", "stage": "stage1", "message": "阶段1：开始推理（tiles输入）"})

    for start in range(0, total, effective_batch):
        if cancel_cb and cancel_cb():
            break

        chunk = tile_paths[start : start + effective_batch]
        if not chunk:
            break

        # goal predict
        with torch.inference_mode():
            res_g = model_g.predict(
                source=[str(p) for p in chunk],
                batch=len(chunk),
                conf=float(goal_conf),
                iou=float(goal_iou),
                device=device,
                half=half,
                verbose=False,
                stream=False,
                save=False,
            )
        if not isinstance(res_g, list):
            res_g = list(res_g) if res_g is not None else []
        if len(res_g) != len(chunk):
            raise RuntimeError(f"goal 推理失败：期望 {len(chunk)} 个结果，得到 {len(res_g)}")

        # normal predict
        with torch.inference_mode():
            res_n = model_n.predict(
                source=[str(p) for p in chunk],
                batch=len(chunk),
                conf=float(normal_conf),
                iou=float(normal_iou),
                device=device,
                half=half,
                verbose=False,
                stream=False,
                save=False,
            )
        if not isinstance(res_n, list):
            res_n = list(res_n) if res_n is not None else []
        if len(res_n) != len(chunk):
            raise RuntimeError(f"normal 推理失败：期望 {len(chunk)} 个结果，得到 {len(res_n)}")

        for p, rg, rn in zip(chunk, res_g, res_n, strict=False):
            stem = p.stem

            g_xyxy = np.zeros((0, 4), dtype=np.float32)
            g_cls = np.zeros((0,), dtype=np.int32)
            if getattr(rg, "boxes", None) is not None and len(rg.boxes) > 0:
                g_xyxy = rg.boxes.xyxy.cpu().numpy()
                g_cls = rg.boxes.cls.cpu().numpy().astype(int)
            goal_boxes_cache[stem] = (g_xyxy, g_cls)

            n_xyxy = np.zeros((0, 4), dtype=np.float32)
            n_cls = np.zeros((0,), dtype=np.int32)
            if getattr(rn, "boxes", None) is not None and len(rn.boxes) > 0:
                n_xyxy = rn.boxes.xyxy.cpu().numpy()
                n_cls = rn.boxes.cls.cpu().numpy().astype(int)
            normal_boxes_cache[stem] = (n_xyxy, n_cls)

            processed += 1
            _maybe_report(progress_cb, stage="stage1", i=processed, total=total, t0=t_infer0, message=f"infer {p.name}", last_emit=last_emit)

    infer_time = time.perf_counter() - t_infer0

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "stage2",
            "message": f"阶段1完成：infer={infer_time:.1f}s，开始阶段2渲染",
        })

    def _keep_normal_by_overlap(n_boxes, g_boxes, thr: float):
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

    t_render0 = time.perf_counter()
    ok = 0
    fail = 0

    def render_single_tile(p: Path):
        try:
            if cancel_cb and cancel_cb():
                return False

            stem = p.stem

            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"读取tile失败: {p}")

            h_img, w_img = int(img_bgr.shape[0]), int(img_bgr.shape[1])

            n_xyxy, _ncls = normal_boxes_cache.get(stem, (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)))
            g_xyxy, g_cls = goal_boxes_cache.get(stem, (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)))

            keep_mask_box = _keep_normal_by_overlap(n_xyxy, g_xyxy, thr=float(normal_keep_overlap_ratio))
            n_remove = n_xyxy[~keep_mask_box] if len(n_xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)
            remove_mask = _make_mask_from_boxes(h_img, w_img, n_remove)

            # L: 去 normal + 画 goal 框
            arr_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            bg_left = _compute_background_u8(arr_rgb, remove_mask)
            left_bgr = img_bgr.copy()
            if remove_mask.any():
                left_bgr[remove_mask] = bg_left[::-1]

            if g_xyxy is not None and len(g_xyxy) > 0:
                for b, c in zip(g_xyxy, g_cls, strict=False):
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    color = (0, 0, 255) if int(c) == 0 else (0, 255, 0)
                    cv2.rectangle(left_bgr, (x1, y1), (x2, y2), color, 2)

            # R: 仅 normal（扣除 goal 区域）
            n_mask_all = _make_mask_from_boxes(h_img, w_img, n_xyxy)
            g_mask_all = _make_mask_from_boxes(h_img, w_img, g_xyxy)
            n_mask_only = (n_mask_all & (~g_mask_all)) if (n_mask_all.any() and g_mask_all.any()) else n_mask_all

            bg_right = _compute_background_u8(arr_rgb, n_mask_only)
            right_bgr = np.full((h_img, w_img, 3), bg_right[::-1], dtype=np.uint8)
            if n_mask_only.any():
                right_bgr[n_mask_only] = img_bgr[n_mask_only]

            gap = max(0, int(pairs_gap))
            canvas_w = w_img + gap + w_img
            canvas = np.full((h_img, canvas_w, 3), 255, dtype=np.uint8)
            canvas[:, 0:w_img, :] = left_bgr
            canvas[:, w_img + gap : w_img + gap + w_img, :] = right_bgr

            out_path = pairs_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return True
        except Exception:
            return False

    render_workers = min(32, max(4, int((os.cpu_count() or 8) * 0.75)))
    with ThreadPoolExecutor(max_workers=render_workers) as ex:
        futures = [ex.submit(render_single_tile, p) for p in tile_paths]
        for idx, fu in enumerate(futures, 1):
            if cancel_cb and cancel_cb():
                break
            ok1 = bool(fu.result())
            if ok1:
                ok += 1
            else:
                fail += 1
            if progress_cb and idx % 100 == 0:
                progress_cb({
                    "status": "running",
                    "stage": "stage2_pairs",
                    "progress": (idx / max(1, len(futures))) * 100.0,
                    "current": idx,
                    "total": len(futures),
                    "message": f"渲染进度: {idx}/{len(futures)}",
                })

    render_time = time.perf_counter() - t_render0

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "summary",
            "message": f"done tiles={ok}/{total} render={render_time:.1f}s fail={fail}",
        })

    return pairs_dir
