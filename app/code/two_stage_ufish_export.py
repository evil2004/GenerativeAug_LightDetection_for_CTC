from __future__ import annotations

from pathlib import Path


def _label_to_box(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
    score = float(parts[5]) if len(parts) > 5 else 1.0
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls_id, score, [x1, y1, x2, y2]


def build_two_stage_ufish_manifest(
    *,
    out_root: Path,
    pairs_dir: Path,
    labels_goal_dir: Path,
    config: dict,
    tiles_dir: Path | None = None,
    source_image: Path | None = None,
    tile_size: int = 1280,
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
    device: str = "0",
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    import csv
    import json
    import cv2  # type: ignore

    from code.ufish import UFishCounter
    from code.unified_detection import _expand_box, _parse_tile_offset, _rel_to_outputs

    crop_dir = out_root / "cells" / "crops"
    ufish_dir = out_root / "cells" / "ufish"
    export_dir = out_root / "exports"
    for d in [crop_dir, ufish_dir, export_dir]:
        d.mkdir(parents=True, exist_ok=True)

    counter = UFishCounter(ufish_onnx_path, ufish_pth_path, device=device) if enable_ufish else None
    records, pending = [], []
    ctc = cec = red_total = green_total = 0
    cell_id = 0
    label_paths = sorted(Path(labels_goal_dir).glob("*.txt"))
    for lp in label_paths:
        if tiles_dir is None:
            continue
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            p = Path(tiles_dir) / f"{lp.stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        ox, oy = _parse_tile_offset(img_path)
        for line in lp.read_text(encoding="utf-8").splitlines():
            parsed = _label_to_box(line, w, h)
            if parsed is None:
                continue
            cls_id, score, box = parsed
            cls_name = "CEC" if cls_id == 1 else "CTC"
            ctc += 1 if cls_name == "CTC" else 0
            cec += 1 if cls_name == "CEC" else 0
            x1, y1, x2, y2 = box
            cx1, cy1, cx2, cy2 = _expand_box(box, w, h, crop_expand_ratio)
            crop = img[cy1:cy2, cx1:cx2].copy()
            cell_id += 1
            stem = f"cell_{cell_id:04d}_{cls_name}"
            crop_path = crop_dir / f"{stem}.png"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            records.append({
                "id": cell_id, "tile": img_path.name, "class": cls_name, "yolo_conf": round(float(score), 6),
                "bbox_tile": [round(float(v), 2) for v in box],
                "bbox_global": [round(float(ox + x1), 2), round(float(oy + y1), 2), round(float(ox + x2), 2), round(float(oy + y2), 2)],
                "crop_path": _rel_to_outputs(crop_path, out_root), "ufish_path": "",
                "red_probe_count": 0, "green_probe_count": 0, "red_components": [], "green_components": [], "ufish_backend": getattr(counter, "backend", "disabled"),
            })
            pending.append((len(records) - 1, crop_path, stem))

    if counter and pending:
        batch = max(1, min(int(ufish_batch_size), 32))
        if progress_cb:
            progress_cb({"stage": "ufish", "current": 0, "total": len(pending), "progress": 0, "message": f"U-FISH counting 0/{len(pending)}"})
        for start in range(0, len(pending), batch):
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            chunk = pending[start:start + batch]
            crops = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, p, _ in chunk]
            counted = counter.count_many(crops, red_threshold=ufish_red_threshold, green_threshold=ufish_green_threshold, min_area=ufish_min_area, nms_distance=ufish_nms_distance, nucleus_min_size=ufish_nucleus_min_size, nucleus_dilation_radius=ufish_nucleus_dilation_radius, cancel_cb=cancel_cb)
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            for (idx, _, stem), (res, overlay) in zip(chunk, counted, strict=False):
                op = ufish_dir / f"{stem}_ufish.png"
                cv2.imwrite(str(op), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                records[idx]["red_probe_count"] = int(res.red_count)
                records[idx]["green_probe_count"] = int(res.green_count)
                records[idx]["ufish_path"] = _rel_to_outputs(op, out_root)
                red_total += int(res.red_count)
                green_total += int(res.green_count)
            if progress_cb:
                done = min(start + batch, len(pending))
                progress_cb({"stage": "ufish", "current": done, "total": len(pending), "progress": done / max(1, len(pending)) * 100, "message": f"U-FISH counting {done}/{len(pending)}"})
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
    yolo_images = [{"name": p.name, "path": _rel_to_outputs(p, out_root), "has_detection": True, "mode": "two_stage"} for p in sorted(Path(pairs_dir).glob("*.png"))]
    manifest = {
        "schema_version": 2, "output": str(out_root),
        "summary": {"total_cells": len(records), "ctc_total": ctc, "cec_total": cec, "red_probe_total": red_total, "green_probe_total": green_total, "ufish_enabled": enable_ufish},
        "config": config, "yolo_images": yolo_images, "cells": records,
        "exports": {"csv": _rel_to_outputs(csv_path, out_root), "json": _rel_to_outputs(json_path, out_root), "annotated_images_dir": _rel_to_outputs(Path(pairs_dir), out_root)},
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
