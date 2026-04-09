from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


def run_goal_only_fast(
    *,
    tiles_dir: Path,
    out_root: Path,
    weights_goal: Path,
    conf: float,
    iou: float,
    device: str = "0",
    batch: int = 64,
    half: bool = True,
    pred_subdir: str = "goal_only",
    labels_subdir: str = "goal_only",
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict:
    """高性能 goal_only：
    - 批量推理（batch=64 默认）
    - 显式 device/half
    - 直接用 Ultralytics 保存带框图
    - 同时写 YOLO txt labels

    关键：保存结果图时不绘制文字标签（CTC/CEC:conf），只画框，避免遮挡。
    """

    import numpy as np  # type: ignore
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.pipeline import _maybe_report, yolo_boxes_to_yolo_txt  # type: ignore

    tile_paths = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.tif")) + list(tiles_dir.glob("*.tiff")))
    total = len(tile_paths)
    if total == 0:
        raise RuntimeError(f"No tiles found in: {tiles_dir}")

    labels_dir = out_root / "labels" / labels_subdir
    labels_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = out_root / "stage2" / pred_subdir
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_goal))

    if progress_cb:
        progress_cb({
            "status": "running",
            "stage": "predict",
            "message": f"goal_only_fast predict params conf={float(conf)} iou={float(iou)} device={device} batch={batch} half={bool(half)} weights={weights_goal.name}",
        })

    t0 = time.perf_counter()
    last_emit = [t0]

    processed = 0
    boxes_total = 0
    ctc = 0
    cec = 0
    first_batch_reported = False

    def _draw_boxes_no_text(img_bgr: np.ndarray, xyxy: np.ndarray, cls: np.ndarray) -> np.ndarray:
        out = img_bgr
        for b, c in zip(xyxy, cls, strict=False):
            x1, y1, x2, y2 = [int(round(float(v))) for v in b]
            color = (0, 0, 255) if int(c) == 0 else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return out

    for start in range(0, total, int(max(1, batch))):
        if cancel_cb and cancel_cb():
            break

        chunk = tile_paths[start : start + int(max(1, batch))]
        if not chunk:
            break

        # 注意：这里不使用 save=True，因为 Ultralytics 默认会画文字标签
        results = model.predict(
            source=[str(p) for p in chunk],
            conf=float(conf),
            iou=float(iou),
            device=str(device),
            half=bool(half),
            batch=len(chunk),
            stream=False,
            verbose=False,
            save=False,
        )
        if not isinstance(results, list):
            results = list(results) if results is not None else []
        if len(results) != len(chunk):
            raise RuntimeError(f"goal_only_fast: 期望 {len(chunk)} 个结果，得到 {len(results)}")

        # 首批统计：boxes数与conf分布
        if (not first_batch_reported) and progress_cb:
            try:
                confs: list[float] = []
                total_boxes_batch = 0
                for r in results:
                    b = getattr(r, "boxes", None)
                    if b is None or len(b) == 0:
                        continue
                    total_boxes_batch += int(len(b))
                    bc = getattr(b, "conf", None)
                    if bc is not None:
                        confs.extend(bc.detach().cpu().numpy().astype(float).tolist())
                msg = f"goal_only_fast first_batch boxes={total_boxes_batch}"
                if confs:
                    arr = np.asarray(confs, dtype=float)
                    msg += f" conf(min/mean/max)={arr.min():.4f}/{arr.mean():.4f}/{arr.max():.4f}"
                progress_cb({"status": "running", "stage": "predict", "message": msg})
            except Exception:
                pass
            first_batch_reported = True

        for p, r in zip(chunk, results, strict=False):
            stem = p.stem
            h, w = int(r.orig_shape[0]), int(r.orig_shape[1])

            xyxy = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int32)
            if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)

            # 写 labels
            lines: list[str] = []
            for b, c in zip(xyxy, cls, strict=False):
                c = int(c)
                lines.append(yolo_boxes_to_yolo_txt(b, c, w, h))
                boxes_total += 1
                if c == 0:
                    ctc += 1
                elif c == 1:
                    cec += 1
            (labels_dir / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            # 保存无文字框图
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                base = getattr(r, "orig_img", None)
                if base is None:
                    base = np.zeros((h, w, 3), dtype=np.uint8)
                img_bgr = base.copy() if hasattr(base, "copy") else base

            out_bgr = _draw_boxes_no_text(img_bgr, xyxy, cls)
            cv2.imwrite(str(pred_dir / f"{stem}.png"), out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            processed += 1
            _maybe_report(
                progress_cb,
                stage="stage2_goal_only",
                i=processed,
                total=total,
                t0=t0,
                message=f"goal_only_fast: {p.name}",
                last_emit=last_emit,
            )

    _maybe_report(progress_cb, stage="stage2_goal_only", i=processed, total=total, t0=t0, message="Stage 2 Done", force=True)

    return {
        "tiles": processed,
        "boxes": boxes_total,
        "ctc": ctc,
        "cec": cec,
        "pred_dir": str(pred_dir),
        "labels_dir": str(labels_dir),
    }
