from __future__ import annotations

from pathlib import Path


def run_normal_only_fast(
    *,
    tiles_dir: Path,
    out_root: Path,
    weights_normal: Path,
    conf: float,
    iou: float,
    device: str = "0",
    batch: int = 24,
    half: bool = True,
    pairs_subdir: str = "normal_only",
    gap: int = 40,
    png_compression: int = 1,
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    out_dir = out_root / "stage2" / pairs_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff") for p in Path(tiles_dir).glob(ext)])
    model = YOLO(str(weights_normal))
    bs = max(1, min(int(batch), 24))
    done = 0
    for start in range(0, len(tile_paths), bs):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        chunk = tile_paths[start : start + bs]
        results = model.predict(source=[str(p) for p in chunk], conf=float(conf), iou=float(iou), batch=len(chunk), device=device, half=half, verbose=False)
        for p, r in zip(chunk, list(results), strict=False):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if getattr(r, "boxes", None) is not None:
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 180, 255), 2)
            cv2.imwrite(str(out_dir / f"{p.stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
            done += 1
        if progress_cb:
            progress_cb({"stage": "normal_only", "current": done, "total": len(tile_paths), "progress": done / max(1, len(tile_paths)) * 100, "message": "normal detecting"})
    return out_dir
