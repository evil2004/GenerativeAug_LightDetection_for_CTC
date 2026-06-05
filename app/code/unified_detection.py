from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable


def _rel_to_outputs(path: Path, out_root: Path) -> str:
    try:
        resolved = path.resolve()
        for parent in [resolved.parent, *resolved.parents]:
            if parent.name.lower() == "outputs":
                return str(resolved.relative_to(parent)).replace("\\", "/")
        return str(resolved.relative_to(out_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _parse_tile_offset(tile_path: Path) -> tuple[int, int]:
    import re

    m = re.match(r"x(\d+)_y(\d+)$", tile_path.stem, re.I)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _class_name(names, cls_id: int) -> str:
    raw = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
    up = str(raw).upper()
    if "CEC" in up or int(cls_id) == 1:
        return "CEC"
    if "CTC" in up or int(cls_id) == 0:
        return "CTC"
    return up


def _expand_box(box, width: int, height: int, ratio: float):
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x = (x2 - x1) * max(0.0, float(ratio))
    pad_y = (y2 - y1) * max(0.0, float(ratio))
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def _overlap_min(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (min(aa, bb) + 1e-9)


def _strict_deduplicate_detections(detections: list[dict], *, iou_thr: float, overlap_min_thr: float) -> list[dict]:
    kept = []
    for det in sorted(detections, key=lambda d: float(d["score"]), reverse=True):
        dup = False
        for old in kept:
            if _box_iou(det["box"], old["box"]) >= iou_thr or _overlap_min(det["box"], old["box"]) >= overlap_min_thr:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


def _draw_box(img_bgr, box, cls_name: str, score: float) -> None:
    import cv2  # type: ignore

    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    color = (45, 70, 235) if cls_name == "CTC" else (72, 205, 88)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_bgr, f"{cls_name} {score:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _valid_cell_box(box, *, min_side: float = 20.0, max_aspect: float = 5.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < float(min_side) or h < float(min_side):
        return False
    aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
    return aspect <= float(max_aspect)


def _is_mostly_black_bgr(img_bgr, *, black_thr: int = 8, ratio_thr: float = 0.9995) -> bool:
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return True
    dark = img_bgr.max(axis=2) <= int(black_thr)
    return float(dark.mean()) >= float(ratio_thr)


def _mostly_black_box_bgr(box, img_bgr, *, black_thr: int = 18, ratio_thr: float = 0.90) -> bool:
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return True
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True
    roi = img_bgr[y1:y2, x1:x2]
    dark = roi.max(axis=2) <= int(black_thr)
    return float(dark.mean()) >= float(ratio_thr)


def run_unified_yolo_ufish(
    *,
    tiles_dir: Path,
    out_root: Path,
    weights_yolo: Path,
    conf: float,
    iou: float,
    device: str = "0",
    batch: int = 16,
    half: bool = True,
    enable_ufish: bool = True,
    ufish_onnx_path: Path | None = None,
    ufish_pth_path: Path | None = None,
    ufish_red_threshold: float = 0.5,
    ufish_green_threshold: float = 0.5,
    ufish_min_area: int = 2,
    ufish_nms_distance: int = 5,
    ufish_nucleus_min_size: int = 200,
    ufish_nucleus_dilation_radius: int = 3,
    ufish_batch_size: int = 32,
    ufish_input_size: int = 256,
    crop_expand_ratio: float = 0.15,
    strict_nms_iou: float = 0.25,
    strict_nms_overlap_min: float = 0.80,
    min_cell_box_side: float = 20.0,
    max_cell_box_aspect: float = 5.0,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.ufish import UFishCounter

    yolo_dir = out_root / "visualization" / "yolo_tiles"
    crop_dir = out_root / "cells" / "crops"
    ufish_dir = out_root / "cells" / "ufish"
    export_dir = out_root / "exports"
    label_dir = out_root / "labels" / "unified_yolo"
    for d in [yolo_dir, crop_dir, ufish_dir, export_dir, label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff") for p in Path(tiles_dir).glob(ext)])
    model = YOLO(str(weights_yolo))
    names = getattr(model, "names", {})
    counter = UFishCounter(ufish_onnx_path, ufish_pth_path, device=device) if enable_ufish else None
    records, yolo_images, pending = [], [], []
    cell_id = ctc = cec = red_total = green_total = 0
    t0 = time.time()
    run_batch = max(1, min(int(batch), 24))

    for start in range(0, len(tile_paths), run_batch):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        chunk = tile_paths[start : start + run_batch]
        active: list[tuple[Path, object]] = []
        for tile_path in chunk:
            img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
            if img is None or _is_mostly_black_bgr(img):
                continue
            active.append((tile_path, img))
        if not active:
            if progress_cb:
                progress_cb({"stage": "predict", "current": min(start + run_batch, len(tile_paths)), "total": len(tile_paths), "progress": min(start + run_batch, len(tile_paths)) / max(1, len(tile_paths)) * 100, "elapsed": time.time() - t0, "message": "YOLO detecting skipped blank tiles"})
            continue
        results = model.predict(source=[img for _, img in active], conf=float(conf), iou=float(iou), batch=len(active), device=device, half=half, verbose=False)
        results = list(results)
        for (tile_path, img), result in zip(active, results, strict=False):
            h, w = img.shape[:2]
            ox, oy = _parse_tile_offset(tile_path)
            detections = []
            if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                for b, c, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=False):
                    cls_name = _class_name(names, int(c))
                    box = [float(v) for v in b]
                    if cls_name in {"CTC", "CEC"} and _valid_cell_box(box, min_side=min_cell_box_side, max_aspect=max_cell_box_aspect) and not _mostly_black_box_bgr(box, img):
                        detections.append({"box": box, "cls_id": int(c), "class_name": cls_name, "score": float(s)})
            detections = _strict_deduplicate_detections(detections, iou_thr=float(strict_nms_iou), overlap_min_thr=float(strict_nms_overlap_min))
            annotated = img.copy()
            lines = []
            for det in detections:
                box, cls_name, score = det["box"], det["class_name"], float(det["score"])
                if cls_name == "CTC":
                    ctc += 1
                else:
                    cec += 1
                _draw_box(annotated, box, cls_name, score)
                x1, y1, x2, y2 = box
                lines.append(f'{det["cls_id"]} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {score:.6f}')
                cx1, cy1, cx2, cy2 = _expand_box(box, w, h, crop_expand_ratio)
                crop = img[cy1:cy2, cx1:cx2].copy()
                cell_id += 1
                stem = f"cell_{cell_id:04d}_{cls_name}"
                crop_path = crop_dir / f"{stem}.png"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                records.append({
                    "id": cell_id, "tile": tile_path.name, "class": cls_name, "yolo_conf": round(score, 6),
                    "bbox_tile": [round(float(v), 2) for v in box],
                    "bbox_global": [round(float(ox + x1), 2), round(float(oy + y1), 2), round(float(ox + x2), 2), round(float(oy + y2), 2)],
                    "crop_path": _rel_to_outputs(crop_path, out_root), "ufish_path": "",
                    "crop_width": int(max(0, cx2 - cx1)), "crop_height": int(max(0, cy2 - cy1)),
                    "red_probe_count": 0, "green_probe_count": 0, "red_components": [], "green_components": [], "ufish_backend": getattr(counter, "backend", "disabled"),
                })
                pending.append((len(records) - 1, crop_path, stem))
            (label_dir / f"{tile_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            out_img = yolo_dir / f"{tile_path.stem}.png"
            cv2.imwrite(str(out_img), annotated if detections else img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            yolo_images.append({"name": out_img.name, "path": _rel_to_outputs(out_img, out_root), "tile": tile_path.name, "has_detection": bool(detections)})
        if progress_cb:
            progress_cb({"stage": "predict", "current": min(start + run_batch, len(tile_paths)), "total": len(tile_paths), "progress": min(start + run_batch, len(tile_paths)) / max(1, len(tile_paths)) * 100, "elapsed": time.time() - t0, "message": "YOLO detecting"})

    if counter and pending:
        if progress_cb:
            progress_cb({"stage": "ufish", "current": 0, "total": len(pending), "progress": 0, "elapsed": time.time() - t0, "message": f"U-FISH counting 0/{len(pending)}"})
        for start in range(0, len(pending), max(1, min(int(ufish_batch_size), 32))):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            chunk = pending[start : start + max(1, min(int(ufish_batch_size), 32))]
            crops = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, p, _ in chunk]
            counted = counter.count_many(crops, red_threshold=ufish_red_threshold, green_threshold=ufish_green_threshold, min_area=ufish_min_area, nms_distance=ufish_nms_distance, nucleus_min_size=ufish_nucleus_min_size, nucleus_dilation_radius=ufish_nucleus_dilation_radius, cancel_cb=cancel_cb)
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            for (idx, _, stem), (res, overlay) in zip(chunk, counted, strict=False):
                op = ufish_dir / f"{stem}_ufish.png"
                cv2.imwrite(str(op), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                records[idx]["red_probe_count"] = int(res.red_count)
                records[idx]["green_probe_count"] = int(res.green_count)
                # Do not persist thousands of point objects in the web JSON.
                # The overlay PNG plus counts are enough for UI/export, and
                # keeping components made large runs tens of MB and froze the browser.
                records[idx]["red_components"] = []
                records[idx]["green_components"] = []
                records[idx]["ufish_path"] = _rel_to_outputs(op, out_root)
                red_total += int(res.red_count)
                green_total += int(res.green_count)
            if progress_cb:
                done = min(start + max(1, min(int(ufish_batch_size), 32)), len(pending))
                progress_cb({"stage": "ufish", "current": done, "total": len(pending), "progress": done / max(1, len(pending)) * 100, "elapsed": time.time() - t0, "message": f"U-FISH counting {done}/{len(pending)}"})
    if enable_ufish and records and any(not r.get("ufish_path") for r in records):
        missing = sum(1 for r in records if not r.get("ufish_path"))
        raise RuntimeError(f"U-FISH counting did not finish for {missing} detected cells")

    csv_path = export_dir / "single_cell_results.csv"
    json_path = export_dir / "results.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["id", "tile", "class", "yolo_conf", "red_probe_count", "green_probe_count", "bbox_tile", "bbox_global", "crop_path", "ufish_path"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["bbox_tile"] = json.dumps(row["bbox_tile"], ensure_ascii=False)
            row["bbox_global"] = json.dumps(row["bbox_global"], ensure_ascii=False)
            writer.writerow(row)
    manifest = {
        "schema_version": 2, "output": str(out_root),
        "summary": {"total_cells": len(records), "ctc_total": ctc, "cec_total": cec, "red_probe_total": red_total, "green_probe_total": green_total, "ufish_enabled": enable_ufish},
        "config": {"weights_yolo": str(weights_yolo), "conf": float(conf), "iou": float(iou), "ufish_red_threshold": float(ufish_red_threshold), "ufish_green_threshold": float(ufish_green_threshold)},
        "yolo_images": yolo_images, "cells": records,
        "exports": {"csv": _rel_to_outputs(csv_path, out_root), "json": _rel_to_outputs(json_path, out_root), "annotated_images_dir": _rel_to_outputs(yolo_dir, out_root)},
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_unified_yolo_ufish_streaming_tiff(
    *,
    in_path: Path,
    out_root: Path,
    weights_yolo: Path,
    tile_size: int = 1280,
    overlap: float = 0.0,
    conf: float,
    iou: float,
    device: str = "0",
    batch: int = 16,
    half: bool = True,
    enable_ufish: bool = True,
    ufish_onnx_path: Path | None = None,
    ufish_pth_path: Path | None = None,
    ufish_red_threshold: float = 0.5,
    ufish_green_threshold: float = 0.5,
    ufish_min_area: int = 2,
    ufish_nms_distance: int = 5,
    ufish_nucleus_min_size: int = 200,
    ufish_nucleus_dilation_radius: int = 3,
    ufish_batch_size: int = 32,
    crop_expand_ratio: float = 0.15,
    strict_nms_iou: float = 0.25,
    strict_nms_overlap_min: float = 0.80,
    min_cell_box_side: float = 20.0,
    max_cell_box_aspect: float = 5.0,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import rasterio  # type: ignore
    import warnings
    from rasterio.errors import NotGeoreferencedWarning  # type: ignore
    from rasterio.windows import Window  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.main import compute_tiles
    from code.ufish import UFishCounter

    yolo_dir = out_root / "visualization" / "yolo_tiles"
    crop_dir = out_root / "cells" / "crops"
    ufish_dir = out_root / "cells" / "ufish"
    export_dir = out_root / "exports"
    label_dir = out_root / "labels" / "unified_yolo"
    for d in [yolo_dir, crop_dir, ufish_dir, export_dir, label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_yolo))
    names = getattr(model, "names", {})
    counter = UFishCounter(ufish_onnx_path, ufish_pth_path, device=device) if enable_ufish else None
    records, yolo_images, pending = [], [], []
    cell_id = ctc = cec = red_total = green_total = skipped_blank = 0
    t0 = time.time()
    run_batch = max(1, min(int(batch), 24))

    def report(stage: str, current: int, total: int, message: str) -> None:
        if progress_cb:
            progress_cb({"stage": stage, "current": current, "total": total, "progress": current / max(1, total) * 100, "elapsed": time.time() - t0, "message": message})

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
            with rasterio.open(str(in_path)) as ds:
                tiles = compute_tiles(width=ds.width, height=ds.height, tile=int(tile_size), overlap=float(overlap))
                total = len(tiles)
                report("tiling", 0, total, f"single-stage streaming tiles: {total}, batch={run_batch}")

                for start in range(0, total, run_batch):
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("cancelled")
                    active = []
                    for x, y, w, h in tiles[start : start + run_batch]:
                        img = bgr_from_window(ds, x, y, w, h)
                        if _is_mostly_black_bgr(img):
                            skipped_blank += 1
                            continue
                        active.append({"x": x, "y": y, "w": w, "h": h, "stem": f"x{x}_y{y}", "img": img})
                    if not active:
                        report("predict", min(start + run_batch, total), total, f"YOLO skipped blank tiles {skipped_blank}")
                        continue

                    results = list(model.predict(
                        source=[item["img"] for item in active],
                        conf=float(conf),
                        iou=float(iou),
                        batch=len(active),
                        device=device,
                        half=half,
                        verbose=False,
                    ))

                    for item, result in zip(active, results, strict=False):
                        img = item["img"]
                        h, w = img.shape[:2]
                        ox, oy = int(item["x"]), int(item["y"])
                        detections = []
                        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                            for b, c, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=False):
                                cls_name = _class_name(names, int(c))
                                box = [float(v) for v in b]
                                if cls_name in {"CTC", "CEC"} and _valid_cell_box(box, min_side=min_cell_box_side, max_aspect=max_cell_box_aspect) and not _mostly_black_box_bgr(box, img):
                                    detections.append({"box": box, "cls_id": int(c), "class_name": cls_name, "score": float(s)})
                        detections = _strict_deduplicate_detections(detections, iou_thr=float(strict_nms_iou), overlap_min_thr=float(strict_nms_overlap_min))
                        annotated = img.copy()
                        lines = []
                        for det in detections:
                            box, cls_name, score = det["box"], det["class_name"], float(det["score"])
                            if cls_name == "CTC":
                                ctc += 1
                            else:
                                cec += 1
                            _draw_box(annotated, box, cls_name, score)
                            x1, y1, x2, y2 = box
                            lines.append(f'{det["cls_id"]} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {score:.6f}')
                            cx1, cy1, cx2, cy2 = _expand_box(box, w, h, crop_expand_ratio)
                            crop = img[cy1:cy2, cx1:cx2].copy()
                            cell_id += 1
                            stem = f"cell_{cell_id:04d}_{cls_name}"
                            crop_path = crop_dir / f"{stem}.png"
                            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                            records.append({
                                "id": cell_id, "tile": f'{item["stem"]}.png', "class": cls_name, "yolo_conf": round(score, 6),
                                "bbox_tile": [round(float(v), 2) for v in box],
                                "bbox_global": [round(float(ox + x1), 2), round(float(oy + y1), 2), round(float(ox + x2), 2), round(float(oy + y2), 2)],
                                "crop_path": _rel_to_outputs(crop_path, out_root), "ufish_path": "",
                                "crop_width": int(max(0, cx2 - cx1)), "crop_height": int(max(0, cy2 - cy1)),
                                "red_probe_count": 0, "green_probe_count": 0, "red_components": [], "green_components": [], "ufish_backend": getattr(counter, "backend", "disabled"),
                            })
                            pending.append((len(records) - 1, crop_path, stem))

                        (label_dir / f'{item["stem"]}.txt').write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                        out_img = yolo_dir / f'{item["stem"]}.png'
                        cv2.imwrite(str(out_img), annotated if detections else img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        yolo_images.append({"name": out_img.name, "path": _rel_to_outputs(out_img, out_root), "tile": out_img.name, "has_detection": bool(detections)})

                    report("predict", min(start + run_batch, total), total, f"single-stage YOLO streaming {min(start + run_batch, total)}/{total} | skipped blank {skipped_blank}")

    if counter and pending:
        report("ufish", 0, len(pending), f"U-FISH counting 0/{len(pending)}")
        batch_n = max(1, min(int(ufish_batch_size), 32))
        for start in range(0, len(pending), batch_n):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            chunk = pending[start : start + batch_n]
            crops = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, p, _ in chunk]
            counted = counter.count_many(crops, red_threshold=ufish_red_threshold, green_threshold=ufish_green_threshold, min_area=ufish_min_area, nms_distance=ufish_nms_distance, nucleus_min_size=ufish_nucleus_min_size, nucleus_dilation_radius=ufish_nucleus_dilation_radius, cancel_cb=cancel_cb)
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            for (idx, _, stem), (res, overlay) in zip(chunk, counted, strict=False):
                op = ufish_dir / f"{stem}_ufish.png"
                cv2.imwrite(str(op), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                records[idx]["red_probe_count"] = int(res.red_count)
                records[idx]["green_probe_count"] = int(res.green_count)
                records[idx]["red_components"] = []
                records[idx]["green_components"] = []
                records[idx]["ufish_path"] = _rel_to_outputs(op, out_root)
                red_total += int(res.red_count)
                green_total += int(res.green_count)
            report("ufish", min(start + batch_n, len(pending)), len(pending), f"U-FISH counting {min(start + batch_n, len(pending))}/{len(pending)}")

    if enable_ufish and records and any(not r.get("ufish_path") for r in records):
        missing = sum(1 for r in records if not r.get("ufish_path"))
        raise RuntimeError(f"U-FISH counting did not finish for {missing} detected cells")

    csv_path = export_dir / "single_cell_results.csv"
    json_path = export_dir / "results.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["id", "tile", "class", "yolo_conf", "red_probe_count", "green_probe_count", "bbox_tile", "bbox_global", "crop_path", "ufish_path"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["bbox_tile"] = json.dumps(row["bbox_tile"], ensure_ascii=False)
            row["bbox_global"] = json.dumps(row["bbox_global"], ensure_ascii=False)
            writer.writerow(row)

    manifest = {
        "schema_version": 2, "output": str(out_root),
        "summary": {"total_cells": len(records), "ctc_total": ctc, "cec_total": cec, "red_probe_total": red_total, "green_probe_total": green_total, "ufish_enabled": enable_ufish, "skipped_blank": skipped_blank},
        "config": {"weights_yolo": str(weights_yolo), "conf": float(conf), "iou": float(iou), "ufish_red_threshold": float(ufish_red_threshold), "ufish_green_threshold": float(ufish_green_threshold), "streaming": True},
        "yolo_images": yolo_images, "cells": records,
        "exports": {"csv": _rel_to_outputs(csv_path, out_root), "json": _rel_to_outputs(json_path, out_root), "annotated_images_dir": _rel_to_outputs(yolo_dir, out_root)},
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
