import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# 兼容本地源码安装 detectron2
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_D2_SRC = PROJECT_ROOT / "detectron2"
if LOCAL_D2_SRC.exists() and (LOCAL_D2_SRC / "detectron2").exists():
    sys.path.insert(0, str(LOCAL_D2_SRC))

DETECTRON2_IMPORT_ERROR = ""
try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.structures import BoxMode

    HAS_DETECTRON2 = True
except Exception as e:
    HAS_DETECTRON2 = False
    DETECTRON2_IMPORT_ERROR = repr(e)

    # 让脚本在未安装 detectron2 时也能导入成功（用于 resnet101_custom 分支）
    model_zoo = None
    DetectionCheckpointer = None
    get_cfg = None
    DatasetCatalog = None
    MetadataCatalog = None
    build_detection_test_loader = None
    DefaultPredictor = None
    COCOEvaluator = None
    inference_on_dataset = None
    BoxMode = None
    DefaultTrainer = object
    HookBase = object


CLASS_NAME_TO_ID = {"CTC": 0, "CEC": 1}

# fast_rcnn 这里按目标检测常用配置映射为 faster r-cnn
MODEL_ZOO_MAP = {
    "retinanet": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    "fast_rcnn": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "resnet101_custom": "custom_torchvision",
}


@dataclass
class RecordBuildResult:
    records: List[Dict]
    skipped_empty: int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(images_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def maybe_run_copy_paste(repo_root: Path, enable: bool):
    if not enable:
        print("[INFO] 跳过粘贴增强。")
        return

    aug_script = repo_root / "augment_ctc.py"
    if not aug_script.exists():
        print(f"[WARN] 未找到增强脚本: {aug_script}，跳过。")
        return

    print(f"[INFO] 运行粘贴增强: {aug_script}")
    subprocess.run([sys.executable, str(aug_script)], check=True)


def collect_dataset_stems(images_dir: Path, labels_dir: Path):
    stems = []
    for txt in sorted(labels_dir.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        stem = txt.stem
        if find_image_path(images_dir, stem) is not None:
            stems.append(stem)
    return stems


def yolo_line_to_xyxy(line: str, w: int, h: int):
    p = line.strip().split()
    if len(p) < 5:
        return None
    cls_id = int(float(p[0]))
    xc, yc, bw, bh = map(float, p[1:5])
    x1 = max(0.0, (xc - bw / 2) * w)
    y1 = max(0.0, (yc - bh / 2) * h)
    x2 = min(float(w - 1), (xc + bw / 2) * w)
    y2 = min(float(h - 1), (yc + bh / 2) * h)
    if x2 <= x1 or y2 <= y1:
        return None
    return cls_id, [x1, y1, x2, y2]


def collect_gt_boxes_for_stem(images_dir: Path, labels_dir: Path, stem: str, target_class_id: int):
    img_path = find_image_path(images_dir, stem)
    label_path = labels_dir / f"{stem}.txt"
    if img_path is None or not label_path.exists():
        return None, []

    img = cv2.imread(str(img_path))
    if img is None:
        return None, []
    h, w = img.shape[:2]

    gt_boxes = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        out = yolo_line_to_xyxy(line, w, h)
        if out is None:
            continue
        cls_id, box = out
        if cls_id == target_class_id:
            gt_boxes.append(box)
    return img_path, gt_boxes


def binary_metrics_from_counts(tp: int, fp: int, fn: int, tn: int):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_image_level_metrics_detectron2(predictor, images_dir: Path, labels_dir: Path, stems: List[str], target_class_id: int, score_thresh: float):
    tp = fp = fn = tn = 0
    for stem in stems:
        img_path, gt_boxes = collect_gt_boxes_for_stem(images_dir, labels_dir, stem, target_class_id)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        out = predictor(img)
        inst = out["instances"].to("cpu")
        scores = inst.scores.numpy().tolist() if hasattr(inst, "scores") else []
        pred_positive = any(s >= score_thresh for s in scores)
        gt_positive = len(gt_boxes) > 0

        if gt_positive and pred_positive:
            tp += 1
        elif (not gt_positive) and pred_positive:
            fp += 1
        elif gt_positive and (not pred_positive):
            fn += 1
        else:
            tn += 1

    return binary_metrics_from_counts(tp, fp, fn, tn)


def evaluate_image_level_metrics_torchvision(model, device, images_dir: Path, labels_dir: Path, stems: List[str], target_class_id: int, score_thresh: float):
    tp = fp = fn = tn = 0
    model.eval()

    for stem in stems:
        img_path, gt_boxes = collect_gt_boxes_for_stem(images_dir, labels_dir, stem, target_class_id)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        with torch.no_grad():
            pred = model([img_t.to(device)])[0]

        scores = pred.get("scores", torch.empty((0,))).detach().cpu().numpy().tolist()
        pred_positive = any(s >= score_thresh for s in scores)
        gt_positive = len(gt_boxes) > 0

        if gt_positive and pred_positive:
            tp += 1
        elif (not gt_positive) and pred_positive:
            fp += 1
        elif gt_positive and (not pred_positive):
            fn += 1
        else:
            tn += 1

    return binary_metrics_from_counts(tp, fp, fn, tn)


# ---------- Detectron2 分支 ----------
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, output_dir: Path):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._output_dir = output_dir
        self.best_metric = float("inf")
        self.history = []

    def _get_loss(self):
        total = 0.0
        count = 0
        for inputs in self._data_loader:
            with torch.no_grad():
                loss_dict = self._model(inputs)
                losses = sum(loss_dict.values())
                total += float(losses.cpu().item())
                count += 1
        return total / max(count, 1)

    def after_step(self):
        nxt = self.trainer.iter + 1
        is_final = nxt == self.trainer.max_iter
        if (self._period > 0 and nxt % self._period == 0) or is_final:
            val_loss = self._get_loss()
            self.history.append((nxt, val_loss))
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                save_path = self._output_dir / "best_model.pth"
                DetectionCheckpointer(self._model).save(str(save_path.with_suffix("")))
            self.trainer.storage.put_scalar("validation_loss", val_loss)


class TrainerWithValLoss(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def build_detectron_records(images_dir: Path, labels_dir: Path, stems: List[str], target_class_id: int, dataset_name: str):
    records = []
    skipped_empty = 0
    for i, stem in enumerate(stems):
        img_path = find_image_path(images_dir, stem)
        label_path = labels_dir / f"{stem}.txt"
        if img_path is None or not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        annos = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            out = yolo_line_to_xyxy(line, w, h)
            if out is None:
                continue
            cls_id, box = out
            if cls_id != target_class_id:
                continue
            annos.append(
                {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                    "iscrowd": 0,
                }
            )

        if not annos:
            skipped_empty += 1
            continue

        records.append(
            {
                "file_name": str(img_path),
                "image_id": f"{dataset_name}_{i}_{stem}",
                "height": h,
                "width": w,
                "annotations": annos,
            }
        )
    return RecordBuildResult(records=records, skipped_empty=skipped_empty)


def register_dataset(name: str, records: List[Dict], class_name: str):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, lambda r=records: r)
    MetadataCatalog.get(name).set(thing_classes=[class_name])


def create_cfg(model_key: str, train_name: str, val_name: str, out_dir: Path, max_iter: int, workers: int, lr: float, ims_per_batch: int, score_thresh: float):
    cfg = get_cfg()
    model_yaml = MODEL_ZOO_MAP[model_key]

    # 避免 detectron2.model_zoo 依赖 pkg_resources 导致导入失败
    # 直接用本地 yaml + 手动权重 URL 映射
    local_cfg_path = PROJECT_ROOT / "detectron2" / "configs" / model_yaml
    if not local_cfg_path.exists():
        raise FileNotFoundError(f"未找到 Detectron2 配置文件: {local_cfg_path}")
    cfg.merge_from_file(str(local_cfg_path))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = workers

    weight_map = {
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml": "detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml": "detectron2://ImageNetPretrained/MSRA/R-101.pkl",
    }
    cfg.MODEL.WEIGHTS = weight_map.get(model_yaml, "")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = max(100, max_iter // 10)

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 800, 960)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.TEST.EVAL_PERIOD = max(100, max_iter // 10)
    cfg.OUTPUT_DIR = str(out_dir)
    return cfg


def read_metrics_json(metrics_file: Path):
    rows = []
    if not metrics_file.exists():
        return rows
    for line in metrics_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


# ---------- 自定义 ResNet-101 分支（torchvision） ----------
class YoloOneClassDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, stems: List[str], target_class_id: int):
        self.samples = []
        for stem in stems:
            img_path = find_image_path(images_dir, stem)
            label_path = labels_dir / f"{stem}.txt"
            if img_path is None or not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            boxes = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                out = yolo_line_to_xyxy(line, w, h)
                if out is None:
                    continue
                cls_id, box = out
                if cls_id == target_class_id:
                    boxes.append(box)

            if boxes:
                self.samples.append((img_path, boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)  # 单类 id=1
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }
        return img, target, str(img_path)


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def build_custom_resnet101_detector(num_classes: int = 2):
    backbone = resnet_fpn_backbone("resnet101", pretrained=True)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model


def evaluate_val_loss(model, loader, device):
    model.train()
    vals = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            vals.append(float(sum(loss_dict.values()).detach().cpu().item()))
    return float(np.mean(vals)) if vals else 9999.0


def draw_predictions_torchvision(model, dataset: YoloOneClassDataset, output_dir: Path, device, score_thresh=0.25, n=20):
    vis_dir = output_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    total = min(n, len(dataset))
    for i in range(total):
        img_t, _, img_path = dataset[i]
        with torch.no_grad():
            pred = model([img_t.to(device)])[0]

        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = pred.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
        scores = pred.get("scores", torch.empty((0,))).detach().cpu().numpy()
        for box, score in zip(boxes, scores):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(vis_dir / f"pred_{i:03d}.jpg"), img)


def plot_loss_curves(output_dir: Path, train_curve: List[Tuple[int, float]], val_curve: List[Tuple[int, float]]):
    out_png = output_dir / "loss_curve.png"
    plt.figure(figsize=(10, 6))
    if train_curve:
        xs, ys = zip(*train_curve)
        plt.plot(xs, ys, label="train_loss")
    if val_curve:
        xs, ys = zip(*val_curve)
        plt.plot(xs, ys, label="val_loss")
    plt.xlabel("Step/Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def rename_best_to_pt(output_dir: Path):
    pth = output_dir / "best_model.pth"
    pt = output_dir / "best.pt"
    if pth.exists():
        shutil.copy2(pth, pt)
    return pt if pt.exists() else None


def run_single_fold_detectron2(
    fold_idx: int,
    model_key: str,
    class_name: str,
    train_stems: List[str],
    val_stems: List[str],
    images_dir: Path,
    labels_dir: Path,
    run_root: Path,
    max_iter: int,
    ims_per_batch: int,
    num_workers: int,
    lr: float,
    score_thresh: float,
):
    if not HAS_DETECTRON2:
        hint = ""
        if "pkg_resources" in DETECTRON2_IMPORT_ERROR:
            hint = "\n[HINT] 缺少 pkg_resources，请执行: python -m pip install -U setuptools"
        raise RuntimeError(
            "detectron2 不可用，请使用 --model resnet101_custom 或安装 detectron2。"
            f"\n[DEBUG] detectron2 import error: {DETECTRON2_IMPORT_ERROR}"
            f"\n[DEBUG] python={sys.executable}"
            f"\n[DEBUG] cwd={os.getcwd()}"
            f"\n[DEBUG] sys.path[0:5]={sys.path[:5]}"
            f"{hint}"
        )

    target_class_id = CLASS_NAME_TO_ID[class_name]
    fold_dir = run_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_name = f"ctccec_train_{class_name.lower()}_f{fold_idx}"
    val_name = f"ctccec_val_{class_name.lower()}_f{fold_idx}"

    train_result = build_detectron_records(images_dir, labels_dir, train_stems, target_class_id, train_name)
    val_result = build_detectron_records(images_dir, labels_dir, val_stems, target_class_id, val_name)

    if not train_result.records or not val_result.records:
        raise RuntimeError(f"Fold {fold_idx} 数据为空 train={len(train_result.records)} val={len(val_result.records)}")

    register_dataset(train_name, train_result.records, class_name)
    register_dataset(val_name, val_result.records, class_name)

    cfg = create_cfg(
        model_key=model_key,
        train_name=train_name,
        val_name=val_name,
        out_dir=fold_dir,
        max_iter=max_iter,
        workers=num_workers,
        lr=lr,
        ims_per_batch=ims_per_batch,
        score_thresh=score_thresh,
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithValLoss(cfg)
    trainer.resume_or_load(resume=False)

    # 关闭自定义 val-loss hook（test loader 无 gt_instances，会触发 RPN requires gt_instances）
    trainer.train()

    # 训练结束后保存当前模型为 best_model（避免无 best 权重）
    DetectionCheckpointer(trainer.model).save(str((fold_dir / "best_model").resolve()))

    evaluator = COCOEvaluator(val_name, cfg, False, output_dir=str(fold_dir / "inference"))
    val_loader = build_detection_test_loader(cfg, val_name)
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)

    metrics_rows = read_metrics_json(fold_dir / "metrics.json")
    train_curve = [(x.get("iteration", 0), x.get("total_loss", 0.0)) for x in metrics_rows if "total_loss" in x]
    plot_loss_curves(fold_dir, train_curve, [])

    best_path = fold_dir / "best_model.pth"
    if best_path.exists():
        DetectionCheckpointer(trainer.model).load(str(best_path))

    predictor = DefaultPredictor(cfg)
    vis_dir = fold_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(val_result.records[:20]):
        img = cv2.imread(rec["file_name"])
        if img is None:
            continue
        out = predictor(img)
        inst = out["instances"].to("cpu")
        if hasattr(inst, "pred_boxes"):
            boxes_np = inst.pred_boxes.tensor.numpy() if hasattr(inst.pred_boxes, "tensor") else np.array(inst.pred_boxes)
            for box in boxes_np:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(vis_dir / f"pred_{i:03d}.jpg"), img)

    test_metrics = evaluate_image_level_metrics_detectron2(
        predictor,
        images_dir,
        labels_dir,
        val_stems,
        target_class_id,
        score_thresh,
    )

    best_pt = rename_best_to_pt(fold_dir)
    summary = {
        "fold": fold_idx,
        "backend": "detectron2",
        "model": model_key,
        "train_samples": len(train_result.records),
        "val_samples": len(val_result.records),
        "metrics": metrics,
        "test_metrics": test_metrics,
        "best_model_pt": str(best_pt) if best_pt else "",
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_single_fold_resnet101_custom(
    fold_idx: int,
    class_name: str,
    train_stems: List[str],
    val_stems: List[str],
    images_dir: Path,
    labels_dir: Path,
    run_root: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    score_thresh: float,
):
    target_class_id = CLASS_NAME_TO_ID[class_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dir = run_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = YoloOneClassDataset(images_dir, labels_dir, train_stems, target_class_id)
    val_ds = YoloOneClassDataset(images_dir, labels_dir, val_stems, target_class_id)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Fold {fold_idx} 数据为空 train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = build_custom_resnet101_detector(num_classes=2).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    train_curve = []
    val_curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for images, targets, _ in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 9999.0
        val_loss = evaluate_val_loss(model, val_loader, device)

        train_curve.append((epoch, train_loss))
        val_curve.append((epoch, val_loss))

        print(f"[Fold {fold_idx}] Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

    best_path = fold_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    draw_predictions_torchvision(model, val_ds, fold_dir, device, score_thresh=score_thresh, n=20)
    plot_loss_curves(fold_dir, train_curve, val_curve)

    test_metrics = evaluate_image_level_metrics_torchvision(
        model,
        device,
        images_dir,
        labels_dir,
        val_stems,
        target_class_id,
        score_thresh,
    )

    best_pt = rename_best_to_pt(fold_dir)

    summary = {
        "fold": fold_idx,
        "backend": "torchvision",
        "model": "resnet101_custom",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_loss": best_loss,
        "test_metrics": test_metrics,
        "best_model_pt": str(best_pt) if best_pt else "",
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="CTC/CEC 检测训练（Detectron2 + 自定义ResNet101）")
    parser.add_argument("--augment", choices=["yes", "no"], default="no", help="是否先做粘贴增强")
    parser.add_argument(
        "--model",
        choices=["retinanet", "fast_rcnn", "resnet101_custom"],
        default="retinanet",
        help="模型选择；resnet101_custom 为纯 torchvision 复现",
    )
    parser.add_argument("--cell", choices=["CTC", "CEC"], default="CTC", help="单次训练细胞类别")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # detectron2 分支
    parser.add_argument("--max-iter", type=int, default=3500)

    # custom 分支
    parser.add_argument("--epochs", type=int, default=40)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def interactive_override(args):
    print("\n=== 交互式配置 ===")
    aug = input("1) 是否进行粘贴增强？(yes/no, 默认 no): ").strip().lower()
    if aug in {"yes", "no"}:
        args.augment = aug

    model = input("2) 模型选择？(retinanet/fast_rcnn/resnet101_custom, 默认 retinanet): ").strip().lower()
    if model in MODEL_ZOO_MAP:
        args.model = model

    cell = input("3) 单次训练细胞类别？(CTC/CEC, 默认 CTC): ").strip().upper()
    if cell in CLASS_NAME_TO_ID:
        args.cell = cell

    k = input("4) k折数量？(默认 5): ").strip()
    if k.isdigit() and int(k) >= 2:
        args.kfolds = int(k)

    print("=== 配置完成 ===\n")
    return args


def main():
    args = parse_args()
    if args.interactive:
        args = interactive_override(args)

    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parent
    images_dir = repo_root / "Dataset" / "images"
    labels_dir = repo_root / "Dataset" / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"数据集路径不存在: {images_dir} | {labels_dir}。"
            "请确认你使用的是 Dataset/images 和 Dataset/labels"
        )

    maybe_run_copy_paste(repo_root, args.augment == "yes")

    stems = collect_dataset_stems(images_dir, labels_dir)
    if len(stems) < args.kfolds:
        raise RuntimeError(f"样本数({len(stems)}) < kfolds({args.kfolds})")

    run_root = repo_root / "runs_detectron2" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}_{args.cell.lower()}"
    run_root.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
    all_stems = np.array(stems)

    summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_stems), start=1):
        print(f"\n[INFO] ===== Fold {fold_idx}/{args.kfolds} =====")
        train_stems = all_stems[train_idx].tolist()
        val_stems = all_stems[val_idx].tolist()

        if args.model == "resnet101_custom":
            summary = run_single_fold_resnet101_custom(
                fold_idx=fold_idx,
                class_name=args.cell,
                train_stems=train_stems,
                val_stems=val_stems,
                images_dir=images_dir,
                labels_dir=labels_dir,
                run_root=run_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lr=args.lr,
                score_thresh=args.score_thresh,
            )
        else:
            summary = run_single_fold_detectron2(
                fold_idx=fold_idx,
                model_key=args.model,
                class_name=args.cell,
                train_stems=train_stems,
                val_stems=val_stems,
                images_dir=images_dir,
                labels_dir=labels_dir,
                run_root=run_root,
                max_iter=args.max_iter,
                ims_per_batch=args.batch_size,
                num_workers=args.num_workers,
                lr=args.lr,
                score_thresh=args.score_thresh,
            )
        summaries.append(summary)

    (run_root / "kfold_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_avg = {}
    for k in metric_keys:
        vals = [s.get("test_metrics", {}).get(k) for s in summaries if s.get("test_metrics", {}).get(k) is not None]
        metric_avg[k] = float(np.mean(vals)) if vals else 0.0

    (run_root / "kfold_test_metrics.json").write_text(
        json.dumps({"mean": metric_avg, "folds": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[INFO] 训练完成，结果目录:", run_root)
    print("[INFO] 每折输出包含: best_model.pth / best.pt / loss_curve.png / predictions/*.jpg / summary.json")
    print(
        "[INFO] 最终测试指标(折均值): "
        f"Accuracy={metric_avg['accuracy']:.4f}, "
        f"Precision={metric_avg['precision']:.4f}, "
        f"Recall={metric_avg['recall']:.4f}, "
        f"F1={metric_avg['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
