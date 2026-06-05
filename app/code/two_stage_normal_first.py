from __future__ import annotations

from pathlib import Path


def run_two_stage_normal_first_from_tiles(
    *,
    tiles_dir: Path,
    out_dir: Path,
    weights_normal: Path,
    weights_goal: Path,
    normal_conf: float,
    normal_iou: float,
    goal_conf: float,
    goal_iou: float,
    batch_size: int = 24,
    device: str = "0",
    half: bool = True,
    pairs_gap: int = 40,
    labels_subdir: str = "stage2_goal_base",
    strict_nms_iou: float = 0.25,
    strict_nms_overlap_min: float = 0.80,
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from ultralytics import YOLO  # type: ignore

    from code.unified_detection import _class_name, _draw_box, _strict_deduplicate_detections, _valid_cell_box

    pairs_dir = out_dir / "stage2" / "pairs"
    label_dir = out_dir / "labels" / labels_subdir
    pairs_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff") for p in Path(tiles_dir).glob(ext)])
    model_normal = YOLO(str(weights_normal))
    model_goal = YOLO(str(weights_goal))
    names = getattr(model_goal, "names", {})
    bs = max(1, min(int(batch_size), 24))
    done = 0

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

    def clean_normal_regions(img_bgr, boxes) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        mask = make_mask(h, w, boxes)
        if not mask.any():
            return img_bgr.copy(), mask
        mask_u8 = (mask.astype("uint8") * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        mask_dilated = mask_u8 > 0
        # Dense fluorescence cells create severe smearing artifacts with
        # cv2.inpaint. A low-percentile local background fill is visually
        # calmer and keeps the removed-normal mask available for final filtering.
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

    for start in range(0, len(tile_paths), bs):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled")
        chunk = tile_paths[start : start + bs]
        if progress_cb:
            progress_cb({
                "stage": "two_stage_normal",
                "current": start,
                "total": len(tile_paths),
                "progress": start / max(1, len(tile_paths)) * 100,
                "message": f"stage1 normal detecting {start}/{len(tile_paths)}",
            })
        normal_results = model_normal.predict(
            source=[str(p) for p in chunk],
            conf=float(normal_conf),
            iou=float(normal_iou),
            batch=len(chunk),
            device=device,
            half=half,
            verbose=False,
        )
        normal_results = list(normal_results)

        prepared = []
        goal_inputs = []
        goal_map = []
        for idx, (p, rn) in enumerate(zip(chunk, normal_results, strict=False)):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                prepared.append(None)
                continue
            h, w = img.shape[:2]
            normal_img = img.copy()
            n_boxes = []
            if getattr(rn, "boxes", None) is not None and len(rn.boxes) > 0:
                n_boxes = [b for b in rn.boxes.xyxy.cpu().numpy()]
                for b, s in zip(n_boxes, rn.boxes.conf.cpu().numpy(), strict=False):
                    draw_normal_box(normal_img, b, float(s))
            cleaned, removed_mask = clean_normal_regions(img, n_boxes)
            prepared.append({"path": p, "img": img, "normal_img": normal_img, "cleaned": cleaned, "removed_mask": removed_mask, "shape": (h, w)})

            # Speed rule: goal is only useful on tiles selected by normal.
            # Empty-normal tiles get a comparison image but skip the expensive goal pass.
            if n_boxes:
                goal_inputs.append(cleaned)
                goal_map.append(idx)

        goal_by_idx = {}
        if goal_inputs:
            if progress_cb:
                progress_cb({
                    "stage": "two_stage_goal",
                    "current": done,
                    "total": len(tile_paths),
                    "progress": done / max(1, len(tile_paths)) * 100,
                    "message": f"stage2 goal detecting selected {len(goal_inputs)}/{len(chunk)}",
                })
            goal_results = model_goal.predict(
                source=goal_inputs,
                conf=float(goal_conf),
                iou=float(goal_iou),
                batch=min(len(goal_inputs), bs),
                device=device,
                half=half,
                verbose=False,
            )
            goal_by_idx = {idx: rg for idx, rg in zip(goal_map, list(goal_results), strict=False)}

        for idx, item in enumerate(prepared):
            if item is None:
                continue
            p = item["path"]
            h, w = item["shape"]
            normal_img = item["normal_img"]
            goal_img = item["cleaned"].copy()
            removed_mask = item["removed_mask"]
            rg = goal_by_idx.get(idx)

            dets = []
            if rg is not None and getattr(rg, "boxes", None) is not None and len(rg.boxes) > 0:
                for b, c, s in zip(rg.boxes.xyxy.cpu().numpy(), rg.boxes.cls.cpu().numpy(), rg.boxes.conf.cpu().numpy(), strict=False):
                    cls_name = _class_name(names, int(c))
                    if cls_name in {"CTC", "CEC"}:
                        box = [float(v) for v in b]
                        if _valid_cell_box(box) and not mostly_black_box(box, goal_img):
                            dets.append({"box": box, "cls_id": int(c), "class_name": cls_name, "score": float(s)})
            dets = _strict_deduplicate_detections(dets, iou_thr=float(strict_nms_iou), overlap_min_thr=float(strict_nms_overlap_min))
            lines = []
            for d in dets:
                _draw_box(goal_img, d["box"], d["class_name"], d["score"])
                x1, y1, x2, y2 = d["box"]
                lines.append(f'{d["cls_id"]} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {d["score"]:.6f}')
            (label_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            gap = max(0, int(pairs_gap))
            canvas = np.full((h, w + gap + w, 3), 255, dtype=np.uint8)
            canvas[:, :w] = normal_img
            canvas[:, w + gap : w + gap + w] = goal_img
            cv2.putText(canvas, "stage1 normal", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "stage2 goal CTC/CEC", (w + gap + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 70, 235), 2, cv2.LINE_AA)
            cv2.imwrite(str(pairs_dir / f"{p.stem}.png"), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            done += 1
            if progress_cb and (done == len(tile_paths) or done % 4 == 0):
                progress_cb({
                    "stage": "two_stage",
                    "current": done,
                    "total": len(tile_paths),
                    "progress": done / max(1, len(tile_paths)) * 100,
                    "message": f"saving pair images {done}/{len(tile_paths)}",
                })
        if progress_cb:
            progress_cb({
                "stage": "two_stage",
                "current": done,
                "total": len(tile_paths),
                "progress": done / max(1, len(tile_paths)) * 100,
                "message": f"two-stage done {done}/{len(tile_paths)}",
            })
    return pairs_dir
