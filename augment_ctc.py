import os
import cv2
import numpy as np
import random
import glob
import shutil
from pathlib import Path
from tqdm import tqdm


def _find_image_by_stem(base_img_dir: Path, stem: str):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
    for ext in exts:
        p = base_img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def rebuild_splits_if_missing(dataset_root: Path, seed: int = 0, ratios=(0.6, 0.2, 0.2)):
    """
    若 Dataset/splits/{train,val,test} 结构缺失，则基于 Dataset/labels 中的样本重新划分并复制生成：
    - 不使用软链接，全部复制真实文件；
    - 复制到 splits/{train,val,test}/{images,labels}；
    - 以 labels 为准（没有 label 的图不参与训练/验证/测试）。
    """
    splits_root = dataset_root / "splits"
    base_img_dir = dataset_root / "images"
    base_label_dir = dataset_root / "labels"

    train_dir = splits_root / "train"
    val_dir = splits_root / "val"
    test_dir = splits_root / "test"

    # 若 val/test 不存在，则重建全部 splits（避免结构不完整导致训练报错）
    if (val_dir / "images").exists() and (test_dir / "images").exists():
        return

    print("Splits missing (val/test). Rebuilding splits by copying from Dataset/images & Dataset/labels ...")

    # 收集所有 base labels
    label_files = sorted(base_label_dir.glob("*.txt"))
    stems = []
    for lf in label_files:
        if lf.name == "classes.txt":
            continue
        stem = lf.stem
        img = _find_image_by_stem(base_img_dir, stem)
        if img is None:
            continue
        stems.append(stem)

    if not stems:
        raise RuntimeError(f"No valid samples found in {base_label_dir} that have matching images in {base_img_dir}")

    rng = random.Random(seed)
    rng.shuffle(stems)

    r_train, r_val, r_test = ratios
    n = len(stems)
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    # 剩余全部给 test，保证总数一致
    n_test = n - n_train - n_val

    train_stems = stems[:n_train]
    val_stems = stems[n_train:n_train + n_val]
    test_stems = stems[n_train + n_val:]

    def _prepare_dir(split_dir: Path):
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    for d in [train_dir, val_dir, test_dir]:
        _prepare_dir(d)

    def _copy_split(stem_list, split_dir: Path):
        for stem in stem_list:
            src_img = _find_image_by_stem(base_img_dir, stem)
            src_lbl = base_label_dir / f"{stem}.txt"
            if src_img is None or not src_lbl.exists():
                continue
            shutil.copy2(src_img, split_dir / "images" / src_img.name)
            shutil.copy2(src_lbl, split_dir / "labels" / src_lbl.name)

    _copy_split(train_stems, train_dir)
    _copy_split(val_stems, val_dir)
    _copy_split(test_stems, test_dir)

    print(f"Rebuilt splits: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)} (seed={seed})")


def load_sub_images(sub_images_dir):
    """
    加载所有 CTC 子图，并统计平均尺寸（用于后续缩放到接近统一大小，误差不超过约 5%）。
    支持常见格式：png/jpg/jpeg/bmp/tif。
    """
    sub_images = []
    heights = []
    widths = []

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    for pat in patterns:
        for img_path in glob.glob(os.path.join(sub_images_dir, pat)):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                continue
            sub_images.append(img)
            heights.append(h)
            widths.append(w)

    if not sub_images:
        return [], 0, 0

    avg_h = int(np.mean(heights))
    avg_w = int(np.mean(widths))
    return sub_images, avg_h, avg_w


def get_existing_bboxes(label_path, img_width, img_height):
    """Load existing bounding boxes from a label file."""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, w, h = map(float, parts)
                x1 = int((x_center - w / 2) * img_width)
                y1 = int((y_center - h / 2) * img_height)
                x2 = int((x_center + w / 2) * img_width)
                y2 = int((y_center + h / 2) * img_height)
                bboxes.append((x1, y1, x2, y2))
    return bboxes


def is_overlapping(new_box, existing_boxes):
    """Check if the new box overlaps with any of the existing boxes."""
    nx1, ny1, nx2, ny2 = new_box
    for ex1, ey1, ex2, ey2 in existing_boxes:
        if not (nx2 < ex1 or nx1 > ex2 or ny2 < ey1 or ny1 > ey2):
            return True
    return False


def find_paste_location(image, sub_img_h, sub_img_w, existing_boxes):
    """Find a valid (black, non-overlapping) location to paste the sub-image."""
    img_h, img_w = image.shape[:2]
    attempts = 100  # Max attempts to find a valid spot
    for _ in range(attempts):
        x1 = random.randint(0, img_w - sub_img_w)
        y1 = random.randint(0, img_h - sub_img_h)
        x2 = x1 + sub_img_w
        y2 = y1 + sub_img_h

        # 1. Check for overlap with existing bboxes
        if is_overlapping((x1, y1, x2, y2), existing_boxes):
            continue

        # 2. Check if the area is mostly black
        paste_area = image[y1:y2, x1:x2]
        if np.mean(paste_area) < 12:  # relaxed threshold for near-black background
            return (x1, y1)

    return None


def augment_image(image_path, label_path, ctc_sub_images, avg_h, avg_w, paste_prob=1.0):
    """
    对单张图像进行 Copy-Paste 增强：
    - 按 paste_prob 决定是否对该图执行粘贴；
    - 执行时随机粘贴 5–10 个 CTC 子图；
    - 子图在粘贴前进行轻微旋转 + 缩放（大小接近全体 CTC 子图平均尺寸，上下浮动 <5%）；
    - 只粘贴在黑色区域，且不遮挡原有 CTC/CEC 或已粘贴实例。
    """
    if random.random() > paste_prob:
        return 0

    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    img_h, img_w = image.shape[:2]
    existing_boxes = get_existing_bboxes(label_path, img_w, img_h)
    new_labels = []

    num_to_paste = random.randint(2, 5)
    pasted_count = 0

    for _ in range(num_to_paste):
        if not ctc_sub_images:
            break

        # 1) 从素材池中随机选择一张 CTC 子图，并复制一份做变换
        base_sub_img = random.choice(ctc_sub_images)
        sub_img = base_sub_img.copy()
        sub_h, sub_w = sub_img.shape[:2]

        # 2) 按照「平均尺寸 ±5%」进行缩放，避免粘贴得过大/过小
        if avg_h > 0 and sub_h > 0:
            # 目标高度在 [0.95, 1.05] * avg_h 区间内随机
            target_h = avg_h * random.uniform(0.95, 1.05)
            scale = target_h / float(sub_h)
            # 避免数值异常
            if scale > 0:
                new_w = max(1, int(round(sub_w * scale)))
                new_h = max(1, int(round(sub_h * scale)))
                sub_img = cv2.resize(sub_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                sub_h, sub_w = sub_img.shape[:2]

        # 3) 对子图做轻微旋转等「温和」增强，不改变颜色分布和整体形状
        angle = random.uniform(-15, 15)  # 小角度旋转
        center = (sub_w / 2.0, sub_h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 旋转后空白区域填充为黑色，避免引入奇怪颜色
        sub_img = cv2.warpAffine(
            sub_img,
            rot_mat,
            (sub_w, sub_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # 4) 在原图上查找一个不遮挡原有目标、且区域为黑色的粘贴位置
        location = find_paste_location(image, sub_h, sub_w, existing_boxes)

        if location:
            x1, y1 = location
            x2, y2 = x1 + sub_w, y1 + sub_h

            # 粘贴子图（这里假定子图为 3 通道；若有透明通道，可后续再加 alpha 融合逻辑）
            image[y1:y2, x1:x2] = sub_img

            # Add to list of boxes to avoid self-overlap in the same image
            existing_boxes.append((x1, y1, x2, y2))

            # Create new label in YOLO format
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = sub_w / img_w
            height = sub_h / img_h
            new_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            pasted_count += 1

    if pasted_count > 0:
        # Overwrite the image
        cv2.imwrite(str(image_path), image)
        # Append new labels to the label file
        with open(label_path, 'a') as f:
            for label in new_labels:
                f.write(f"\n{label}")
    
    return pasted_count


def restore_clean_train_split(train_dir: Path, base_img_dir: Path, base_label_dir: Path):
    """
    每次训练前，将 train split 还原为「原图 + 原标签」的干净状态：
    - 不再依赖软链接，直接从 Dataset/images & Dataset/labels 复制真实文件到 splits/train；
    - 这样可以避免多次运行粘贴增强导致同一张图上细胞越堆越多。
    """
    images_subdir = train_dir / "images"
    labels_subdir = train_dir / "labels"
    images_subdir.mkdir(parents=True, exist_ok=True)
    labels_subdir.mkdir(parents=True, exist_ok=True)

    # 先根据现有的 train/labels 文件名，确定 train 集合的样本列表
    label_files = list(labels_subdir.glob("*.txt"))
    if not label_files:
        raise RuntimeError(
            f"Train split labels are empty: {labels_subdir}. "
            f"Please ensure splits are built before augmentation."
        )

    for lf in label_files:
        stem = lf.stem

        # 1) 还原标签：从 Dataset/labels 复制到 splits/train/labels
        src_label = base_label_dir / lf.name
        if src_label.exists():
            shutil.copy2(src_label, lf)
        else:
            # 如果原始 labels 中不存在，说明这个样本本来就不在原始集合中，跳过
            print(f"Warning: base label not found for {lf.name} in {base_label_dir}")
            continue

        # 2) 还原图像：在 Dataset/images 中按多种后缀查找同名文件
        src_img = _find_image_by_stem(base_img_dir, stem)

        if src_img is None:
            print(f"Warning: base image not found for {stem} in {base_img_dir}")
            continue

        dst_img = images_subdir / src_img.name
        shutil.copy2(src_img, dst_img)


def main():
    """Main function to run the augmentation process."""
    print("Starting Copy-Paste augmentation for the training set...")
    
    repo_root = Path(__file__).resolve().parent
    dataset_root = repo_root / "Dataset"
    train_dir = dataset_root / "splits" / "train"
    ctc_dir = dataset_root / "CTC"
    base_img_dir = dataset_root / "images"
    base_label_dir = dataset_root / "labels"

    # 0) 若 splits 结构缺失（比如 val/test 不存在），先重建 splits 目录，保证训练可读取数据集
    rebuild_splits_if_missing(dataset_root, seed=0, ratios=(0.6, 0.2, 0.2))

    # 关键修正：
    # 每次运行前，先用原始 Dataset/images & Dataset/labels 还原 train split，
    # 不再直接在原始目录上做修改，也避免粘贴效果在多次训练之间不断累加。
    restore_clean_train_split(train_dir, base_img_dir, base_label_dir)

    # 训练集图片支持多种格式
    image_files = []
    for pat in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]:
        image_files.extend((train_dir / "images").glob(pat))

    ctc_sub_images, avg_h, avg_w = load_sub_images(ctc_dir)

    if not ctc_sub_images:
        print("Error: No CTC sub-images found in", ctc_dir)
        return

    # Reduce synthetic bias: only augment part of images each run
    paste_prob = 0.25
    total_pasted = 0
    for img_path in tqdm(image_files, desc="Augmenting training images"):
        label_path = (train_dir / "labels" / (img_path.stem + ".txt"))
        pasted_count = augment_image(img_path, label_path, ctc_sub_images, avg_h, avg_w, paste_prob=paste_prob)
        if pasted_count:
            total_pasted += pasted_count

    print(f"Augmentation complete. Total of {total_pasted} CTC instances pasted.")


if __name__ == "__main__":
    main()
