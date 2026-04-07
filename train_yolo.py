import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image


def _get_metrics(results):
    box = getattr(results, "box", None)
    if box is None:
        return None, None, None, None
    return (
        getattr(box, "mp", None),
        getattr(box, "mr", None),
        getattr(box, "map50", None),
        getattr(box, "map", None),
    )


def _print_metrics(tag, results):
    mp, mr, map50, map5095 = _get_metrics(results)
    if None in (mp, mr, map50, map5095):
        print(f"[{tag}] Metrics parsed with missing fields.")
        return
    print(f"[{tag}] P={mp:.4f}  R={mr:.4f}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")


def _eval_split(model, data_yaml, split, imgsz=1280, batch=16):
    return model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=0.01,
        iou=0.45,
        max_det=1000,
        augment=True,
        plots=True,
    )


def _extract_target_only_labels(src_labels_dir: Path, dst_labels_dir: Path, target_cls: int):
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    for txt in src_labels_dir.glob("*.txt"):
        out_lines = []
        content = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in content:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
            except Exception:
                continue
            if cls_id == target_cls:
                parts[0] = "0"  # single-class training label remap
                out_lines.append(" ".join(parts))
        (dst_labels_dir / txt.name).write_text(
            ("\n".join(out_lines) + "\n") if out_lines else "",
            encoding="utf-8",
        )


def _prepare_target_only_dataset(repo_root: Path, cell_type: str, target_cls: int):
    base = repo_root / "yolo" / "Dataset" / "splits"
    out_root = repo_root / "yolo" / "Dataset" / f"splits_{cell_type.lower()}_only"

    for split in ("train", "val", "test"):
        src_img = base / split / "images"
        src_lbl = base / split / "labels"
        dst_img = out_root / split / "images"
        dst_lbl = out_root / split / "labels"

        if not src_img.exists() or not src_lbl.exists():
            raise FileNotFoundError(f"Missing split directory: {src_img} or {src_lbl}")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() or dst_img.is_symlink():
            if dst_img.is_symlink() or dst_img.is_file():
                dst_img.unlink()
        if not dst_img.exists():
            os.symlink(src_img, dst_img)

        _extract_target_only_labels(src_lbl, dst_lbl, target_cls)

    return out_root


def _yolo_to_xyxy(xc, yc, w, h, iw, ih):
    x1 = (xc - w / 2.0) * iw
    y1 = (yc - h / 2.0) * ih
    x2 = (xc + w / 2.0) * iw
    y2 = (yc + h / 2.0) * ih
    return [x1, y1, x2, y2]


def _iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _find_latest_best_pt(repo_root: Path) -> Optional[Path]:
    runs_root = repo_root / "yolo" / "runs"
    candidates = sorted(runs_root.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _mine_hard_cases(ctc_splits_root: Path, model_pt: Optional[Path]) -> tuple[set[str], set[str]]:
    """Mine hard positives(FN-like). Keep hard backgrounds disabled for stability."""
    if model_pt is None or not model_pt.exists():
        print("--- No previous best.pt found. Skip hard mining ---")
        return set(), set()

    from ultralytics import YOLO

    print(f"--- Mining hard positives from: {model_pt} ---")
    model = YOLO(str(model_pt))
    train_images_dir = ctc_splits_root / "train" / "images"
    train_labels_dir = ctc_splits_root / "train" / "labels"

    hard_pos = set()
    hard_bg = set()
    checked_pos = 0
    checked_bg = 0

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        image_paths.extend(train_images_dir.glob(ext))
    image_paths = sorted(image_paths)

    for img in image_paths:
        lbl = train_labels_dir / f"{img.stem}.txt"
        if not lbl.exists() or lbl.stat().st_size == 0:
            continue

        with Image.open(img) as im:
            iw, ih = im.size

        gt_boxes = []
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                cls_id = int(float(p[0]))
                if cls_id != 0:
                    continue
                xc, yc, w, h = map(float, p[1:5])
            except Exception:
                continue
            gt_boxes.append(_yolo_to_xyxy(xc, yc, w, h, iw, ih))

        pred = model.predict(source=str(img), imgsz=1280, conf=0.15, iou=0.5, max_det=500, verbose=False, device="cuda:0")
        pboxes = pred[0].boxes.xyxy.cpu().numpy().tolist() if len(pred) else []

        if gt_boxes:
            checked_pos += 1
            if not pboxes:
                hard_pos.add(img.stem)
                continue

            missed = False
            for g in gt_boxes:
                best_iou = max((_iou_xyxy(g, pb) for pb in pboxes), default=0.0)
                if best_iou < 0.45:
                    missed = True
                    break
            if missed:
                hard_pos.add(img.stem)
        else:
            checked_bg += 1

    print(
        f"--- Hard mining done: checked_pos={checked_pos}, hard_pos={len(hard_pos)}, "
        f"checked_bg={checked_bg}, hard_bg={len(hard_bg)} ---"
    )
    return hard_pos, hard_bg


def _build_weighted_train_list_ctc(
    repo_root: Path,
    ctc_splits_root: Path,
    hard_pos_stems: set[str],
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    train_images_dir = ctc_splits_root / "train" / "images"
    train_labels_dir = ctc_splits_root / "train" / "labels"

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        image_paths.extend(train_images_dir.glob(ext))
    image_paths = sorted(image_paths)

    # Use original split labels to determine positive/negative status for stability across runs
    orig_train_labels_dir = repo_root / "yolo" / "Dataset" / "splits" / "train" / "labels"

    weighted = []
    pos_count, bg_kept, hard_boosted = 0, 0, 0
    for img in image_paths:
        orig_lbl = orig_train_labels_dir / f"{img.stem}.txt"
        aug_lbl = train_labels_dir / f"{img.stem}.txt"
        has_orig_ctc = orig_lbl.exists() and orig_lbl.stat().st_size > 0
        has_aug_ctc = aug_lbl.exists() and aug_lbl.stat().st_size > 0
        has_ctc = has_orig_ctc or has_aug_ctc

        if has_ctc:
            rep = 7
            if img.stem in hard_pos_stems:
                rep = 14
                hard_boosted += 1
            weighted.extend([img] * rep)
            pos_count += 1
        else:
            if rng.random() < 0.35:
                weighted.append(img)
                bg_kept += 1

    if not weighted:
        raise RuntimeError("Weighted CTC train list is empty.")

    out_txt = repo_root / "yolo" / "Dataset" / "splits_ctc_only" / "train_weighted_ctc_only.txt"
    out_txt.write_text("\n".join(str(p) for p in weighted) + "\n", encoding="utf-8")

    print("--- CTC-only weighted train list created ---")
    print(
        f"base_images={len(image_paths)} weighted_images={len(weighted)} "
        f"ctc_positive={pos_count} hard_boosted={hard_boosted} bg_kept={bg_kept}"
    )
    return out_txt


def _build_target_only_yaml(repo_root: Path, train_txt: Path, cell_type: str) -> Path:
    split_name = f"splits_{cell_type.lower()}_only"
    yaml_path = repo_root / "yolo" / "Dataset" / f"{cell_type.lower()}_only_runtime.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {repo_root / 'yolo' / 'Dataset'}",
                f"train: {train_txt}",
                f"val: {split_name}/val",
                f"test: {split_name}/test",
                "",
                "names:",
                f"  0: {cell_type}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO for CTC/CEC single-class target.")
    parser.add_argument(
        "--cell",
        choices=["CTC", "CEC", "ctc", "cec"],
        default="CTC",
        help="Target cell type to train. Default: CTC",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cell_type = args.cell.upper()
    target_cls = 0 if cell_type == "CTC" else 1

    repo_root = Path(__file__).resolve().parent
    ul_root = repo_root / "ultralytics"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(ul_root) not in sys.path:
        sys.path.insert(0, str(ul_root))

    from ultralytics import YOLO

    print("--- Starting CTC Copy-Paste Data Augmentation ---")
    augmentation_script = repo_root / "augment_ctc.py"
    if augmentation_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(augmentation_script)], check=True)
        print("--- CTC Copy-Paste Data Augmentation Finished ---")

    print(f"--- Preparing {cell_type}-only dataset view ---")
    target_splits_root = _prepare_target_only_dataset(repo_root, cell_type, target_cls)

    prev_best = _find_latest_best_pt(repo_root)
    hard_pos_stems, _ = _mine_hard_cases(target_splits_root, prev_best)

    train_txt = _build_weighted_train_list_ctc(
        repo_root,
        target_splits_root,
        hard_pos_stems,
        seed=42,
    )
    data_yaml = _build_target_only_yaml(repo_root, train_txt, cell_type)

    # best-known parameters (single run)
    train_args = dict(
        data=str(data_yaml),
        task="detect",
        epochs=420,
        patience=180,
        imgsz=1280,
        batch=12,
        device="cuda:0",
        workers=6,
        pretrained="yolo11n.pt",
        seed=0,
        deterministic=True,
        project=str(repo_root / "yolo" / "runs"),
        name="y11_ctc_only",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=-1,
        cache=False,
        val=True,
        optimizer="AdamW",
        lr0=1.8e-4,
        lrf=0.03,
        momentum=0.92,
        weight_decay=5e-4,
        cos_lr=True,
        warmup_epochs=3.0,
        warmup_momentum=0.75,
        warmup_bias_lr=0.08,
        box=9.2,
        cls=1.4,
        dfl=1.45,
        nbs=64,
        mosaic=0.12,
        close_mosaic=45,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        translate=0.02,
        scale=0.14,
        fliplr=0.5,
        flipud=0.0,
        degrees=1.2,
        shear=0.15,
        perspective=0.0,
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.2,
        bgr=0.0,
        erasing=0.0,
        multi_scale=False,
        rect=False,
        max_det=500,
    )

    print(f"\n--- Training {cell_type}-only detector ---")
    model = YOLO(str(repo_root / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml"))
    model.train(**train_args, end2end=False)

    save_dir = None
    if hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Could not find trainer save_dir after training.")

    best_pt = Path(save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError("best.pt not found after training.")

    model_best = YOLO(str(best_pt))

    print(f"\n--- Final validation on VAL split ({cell_type} only) ---")
    val_results = _eval_split(model_best, data_yaml, split="val", imgsz=1280, batch=16)
    _print_metrics(f"VAL-{cell_type}", val_results)

    print(f"\n--- Final evaluation on TEST split ({cell_type} only) ---")
    test_results = _eval_split(model_best, data_yaml, split="test", imgsz=1280, batch=16)
    _print_metrics(f"TEST-{cell_type}", test_results)

    print(f"\n--- Exporting model: {best_pt} ---")
    model_best.export(format="onnx", end2end=False, opset=20)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
