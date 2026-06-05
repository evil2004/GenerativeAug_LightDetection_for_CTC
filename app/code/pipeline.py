from __future__ import annotations

import time
from pathlib import Path


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
    batch_size: int = 24,
    device: str = "0",
    half: bool = True,
    pairs_gap: int = 40,
    max_batch: int = 24,
    strict_nms_iou: float = 0.25,
    strict_nms_overlap_min: float = 0.80,
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    """Two-stage pipeline for large TIFFs without pre-writing all tiles.

    The old path wrote every tile to disk and then read those PNGs back for
    normal/goal detection. This version reads a small batch of TIFF windows,
    runs normal, cleans selected normal regions, runs goal only on selected
    tiles, writes the before/after pair image, and saves original tile PNGs
    only when a final CTC/CEC box is kept for U-FISH cropping.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import rasterio  # type: ignore
    import warnings
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.main import compute_tiles
    from code.unified_detection import _class_name, _draw_box, _is_mostly_black_bgr, _strict_deduplicate_detections, _valid_cell_box

    pairs_dir = out_dir / "stage2" / "pairs"
    tiles_dir = out_dir / "tiles"
    label_dir = out_dir / "labels" / "stage2_goal_base"
    for d in (pairs_dir, tiles_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_normal = YOLO(str(weights_normal))
    model_goal = YOLO(str(weights_goal))
    goal_names = getattr(model_goal, "names", {})
    bs = max(1, min(int(batch_size), int(max_batch), 24))
    t0 = time.time()

    def report(stage: str, current: int, total: int, message: str) -> None:
        if progress_cb:
            progress_cb({
                "stage": stage,
                "current": current,
                "total": total,
                "progress": current / max(1, total) * 100.0,
                "elapsed": time.time() - t0,
                "message": message,
            })

    def bgr_from_window(ds, x: int, y: int, w: int, h: int):
        data = ds.read(window=Window(x, y, w, h), boundless=False)
        arr = np.transpose(data, (1, 2, 0))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            amin, amax = float(arr.min()), float(arr.max())
            arr = ((arr.astype("float32") - amin) / max(amax - amin, 1e-6) * 255.0).clip(0, 255).astype("uint8")
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

    def draw_normal_box(img_bgr, box, score: float) -> None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        color = (0, 180, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, f"normal {score:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def make_mask(h: int, w: int, boxes) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        for b in boxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
        return mask

    def clean_normal_regions(img_bgr, boxes):
        h, w = img_bgr.shape[:2]
        mask = make_mask(h, w, boxes)
        if not mask.any():
            return img_bgr.copy(), mask
        mask_u8 = (mask.astype("uint8") * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        mask_dilated = mask_u8 > 0
        # Inpaint smears dense red/blue fluorescence into large polygon-like
        # artifacts. Fill removed normal cells with a dark local background
        # instead; final goal boxes overlapping this mask are filtered below.
        bg_pixels = img_bgr[~mask_dilated]
        if bg_pixels.size:
            fill = np.percentile(bg_pixels.reshape(-1, 3), 5, axis=0).astype("uint8")
        else:
            fill = np.array([0, 0, 0], dtype="uint8")
        clean = img_bgr.copy()
        clean[mask_dilated] = fill
        return clean, mask_dilated

    def mostly_black_box(box, img_bgr: np.ndarray, *, black_thr: int = 18, ratio_thr: float = 0.90) -> bool:
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return True
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return True
        dark = (roi.max(axis=2) <= int(black_thr))
        return float(dark.mean()) >= float(ratio_thr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(str(in_path)) as ds:
                tiles = compute_tiles(width=ds.width, height=ds.height, tile=int(tile_size), overlap=float(overlap))
                total = len(tiles)
                report("tiling", 0, total, f"streaming tiles prepared: {total}, batch={bs}")
                done = 0

                for start in range(0, total, bs):
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("cancelled")

                    chunk_tiles = tiles[start : start + bs]
                    batch_items = []
                    for read_i, (x, y, w, h) in enumerate(chunk_tiles, start=1):
                        if cancel_cb and cancel_cb():
                            raise RuntimeError("cancelled")
                        img = bgr_from_window(ds, x, y, w, h)
                        if _is_mostly_black_bgr(img):
                            done += 1
                            if done == total or done % 8 == 0:
                                report("tiling", done, total, f"streaming skipped blank {done}/{total}")
                            continue
                        batch_items.append({"x": x, "y": y, "w": w, "h": h, "stem": f"x{x}_y{y}", "img": img})
                        if read_i == len(chunk_tiles) or read_i % 4 == 0:
                            report("tiling", min(total, start + read_i), total, f"streaming read {min(total, start + read_i)}/{total}")

                    if not batch_items:
                        continue

                    report("two_stage_normal", done, total, f"stage1 normal detecting {done}/{total}")
                    normal_results = list(model_normal.predict(
                        source=[item["img"] for item in batch_items],
                        conf=float(stage1_conf),
                        iou=float(stage1_iou),
                        batch=len(batch_items),
                        device=device,
                        half=half,
                        verbose=False,
                    ))

                    goal_inputs = []
                    goal_map = []
                    for idx, (item, rn) in enumerate(zip(batch_items, normal_results, strict=False)):
                        img = item["img"]
                        normal_img = img.copy()
                        n_boxes = []
                        if getattr(rn, "boxes", None) is not None and len(rn.boxes) > 0:
                            n_boxes = [b for b in rn.boxes.xyxy.cpu().numpy()]
                            for b, s in zip(n_boxes, rn.boxes.conf.cpu().numpy(), strict=False):
                                draw_normal_box(normal_img, b, float(s))
                        cleaned, removed_mask = clean_normal_regions(img, n_boxes)
                        item["normal_img"] = normal_img
                        item["cleaned"] = cleaned
                        item["removed_mask"] = removed_mask
                        if n_boxes:
                            goal_inputs.append(cleaned)
                            goal_map.append(idx)

                    goal_by_idx = {}
                    if goal_inputs:
                        report("two_stage_goal", done, total, f"stage2 goal detecting selected {len(goal_inputs)}/{len(batch_items)}")
                        goal_results = list(model_goal.predict(
                            source=goal_inputs,
                            conf=float(stage2_conf),
                            iou=float(stage2_iou),
                            batch=min(len(goal_inputs), bs),
                            device=device,
                            half=half,
                            verbose=False,
                        ))
                        goal_by_idx = {idx: rg for idx, rg in zip(goal_map, goal_results, strict=False)}

                    for idx, item in enumerate(batch_items):
                        if cancel_cb and cancel_cb():
                            raise RuntimeError("cancelled")
                        img = item["img"]
                        h, w = img.shape[:2]
                        goal_img = item["cleaned"].copy()
                        rg = goal_by_idx.get(idx)
                        dets = []
                        if rg is not None and getattr(rg, "boxes", None) is not None and len(rg.boxes) > 0:
                            for b, c, s in zip(rg.boxes.xyxy.cpu().numpy(), rg.boxes.cls.cpu().numpy(), rg.boxes.conf.cpu().numpy(), strict=False):
                                cls_name = _class_name(goal_names, int(c))
                                if cls_name in {"CTC", "CEC"}:
                                    box = [float(v) for v in b]
                                    if _valid_cell_box(box) and not mostly_black_box(box, goal_img):
                                        dets.append({"box": box, "cls_id": int(c), "class_name": cls_name, "score": float(s)})
                        dets = _strict_deduplicate_detections(dets, iou_thr=float(strict_nms_iou), overlap_min_thr=float(strict_nms_overlap_min))

                        lines = []
                        for det in dets:
                            _draw_box(goal_img, det["box"], det["class_name"], float(det["score"]))
                            x1, y1, x2, y2 = det["box"]
                            lines.append(f'{det["cls_id"]} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {float(det["score"]):.6f}')

                        (label_dir / f'{item["stem"]}.txt').write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                        if dets:
                            cv2.imwrite(str(tiles_dir / f'{item["stem"]}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                        gap = max(0, int(pairs_gap))
                        canvas = np.full((h, w + gap + w, 3), 255, dtype=np.uint8)
                        canvas[:, :w] = item["normal_img"]
                        canvas[:, w + gap : w + gap + w] = goal_img
                        cv2.putText(canvas, "stage1 normal", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
                        cv2.putText(canvas, "stage2 goal CTC/CEC", (w + gap + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 70, 235), 2, cv2.LINE_AA)
                        cv2.imwrite(str(pairs_dir / f'{item["stem"]}.png'), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        done += 1
                        if done == total or done % 4 == 0:
                            report("two_stage", done, total, f"saving pair images {done}/{total}")

                    report("two_stage", done, total, f"two-stage streaming done {done}/{total}")

    return pairs_dir
