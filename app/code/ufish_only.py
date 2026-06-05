from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def run_ufish_only(
    *,
    input_path: Path,
    out_root: Path,
    ufish_onnx_path: Path | None = None,
    ufish_pth_path: Path | None = None,
    ufish_red_threshold: float = 0.5,
    ufish_green_threshold: float = 0.5,
    ufish_min_area: int = 2,
    ufish_nms_distance: int = 5,
    ufish_nucleus_min_size: int = 200,
    ufish_nucleus_dilation_radius: int = 3,
    ufish_batch_size: int = 32,
    device: str = "0",
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    import csv
    import json
    import shutil

    import cv2  # type: ignore

    from code.ufish import UFishCounter
    from code.unified_detection import _is_mostly_black_bgr, _rel_to_outputs

    crop_dir = out_root / "cells" / "crops"
    ufish_dir = out_root / "cells" / "ufish"
    export_dir = out_root / "exports"
    for d in (crop_dir, ufish_dir, export_dir):
        d.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    elif input_path.suffix.lower() in IMAGE_SUFFIXES:
        image_paths = [input_path]
    else:
        raise RuntimeError(f"U-FISH only mode expects a cell image or image folder: {input_path}")

    counter = UFishCounter(ufish_onnx_path, ufish_pth_path, device=device)
    records: list[dict] = []
    pending: list[tuple[int, Path, str]] = []
    red_total = green_total = skipped_blank = 0

    for idx, src in enumerate(image_paths, start=1):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None or _is_mostly_black_bgr(img):
            skipped_blank += 1
            continue
        stem = f"cell_{len(records) + 1:04d}_{src.stem}"
        crop_path = crop_dir / f"{stem}.png"
        cv2.imwrite(str(crop_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        h, w = img.shape[:2]
        records.append({
            "id": len(records) + 1,
            "tile": src.name,
            "class": "CELL",
            "yolo_conf": "",
            "bbox_tile": [0, 0, w, h],
            "bbox_global": [0, 0, w, h],
            "crop_path": _rel_to_outputs(crop_path, out_root),
            "ufish_path": "",
            "crop_width": int(w),
            "crop_height": int(h),
            "red_probe_count": 0,
            "green_probe_count": 0,
            "red_components": [],
            "green_components": [],
            "ufish_backend": getattr(counter, "backend", "disabled"),
        })
        pending.append((len(records) - 1, crop_path, stem))
        if progress_cb and (idx == len(image_paths) or idx % 50 == 0):
            progress_cb({"stage": "ufish", "current": idx, "total": len(image_paths), "progress": idx / max(1, len(image_paths)) * 100, "message": f"U-FISH loading cells {idx}/{len(image_paths)}"})

    batch = max(1, min(int(ufish_batch_size), 32))
    for start in range(0, len(pending), batch):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        chunk = pending[start : start + batch]
        crops = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, p, _ in chunk]
        counted = counter.count_many(
            crops,
            red_threshold=ufish_red_threshold,
            green_threshold=ufish_green_threshold,
            min_area=ufish_min_area,
            nms_distance=ufish_nms_distance,
            nucleus_min_size=ufish_nucleus_min_size,
            nucleus_dilation_radius=ufish_nucleus_dilation_radius,
            cancel_cb=cancel_cb,
        )
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        for (record_idx, _, stem), (res, overlay) in zip(chunk, counted, strict=False):
            op = ufish_dir / f"{stem}_ufish.png"
            cv2.imwrite(str(op), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            records[record_idx]["red_probe_count"] = int(res.red_count)
            records[record_idx]["green_probe_count"] = int(res.green_count)
            records[record_idx]["ufish_path"] = _rel_to_outputs(op, out_root)
            red_total += int(res.red_count)
            green_total += int(res.green_count)
        if progress_cb:
            done = min(start + batch, len(pending))
            progress_cb({"stage": "ufish", "current": done, "total": len(pending), "progress": done / max(1, len(pending)) * 100, "message": f"U-FISH counting {done}/{len(pending)}"})

    csv_path = export_dir / "single_cell_results.csv"
    json_path = export_dir / "results.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["id", "tile", "class", "red_probe_count", "green_probe_count", "bbox_tile", "crop_path", "ufish_path"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["bbox_tile"] = json.dumps(row["bbox_tile"], ensure_ascii=False)
            writer.writerow(row)

    manifest = {
        "schema_version": 2,
        "output": str(out_root),
        "summary": {
            "total_cells": len(records),
            "ctc_total": 0,
            "cec_total": 0,
            "red_probe_total": red_total,
            "green_probe_total": green_total,
            "ufish_enabled": True,
            "skipped_blank": skipped_blank,
        },
        "config": {"detection_mode": "ufish_only", "ufish_red_threshold": float(ufish_red_threshold), "ufish_green_threshold": float(ufish_green_threshold)},
        "yolo_images": [],
        "cells": records,
        "exports": {"csv": _rel_to_outputs(csv_path, out_root), "json": _rel_to_outputs(json_path, out_root), "annotated_images_dir": _rel_to_outputs(ufish_dir, out_root)},
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
