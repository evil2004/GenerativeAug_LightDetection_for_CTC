"""
CTC 无条件生成数据集 + 10折交叉验证
=====================================
无条件 GAN 只需要真实图像（无需配对、无需标注）。
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def get_image_paths(root: str) -> list:
    root  = Path(root)
    paths = []
    for ext in IMG_EXTS:
        paths.extend(root.rglob(f'*{ext}'))
        paths.extend(root.rglob(f'*{ext.upper()}'))
    return sorted(set(str(p) for p in paths))


def build_transform(img_size: int, is_train: bool) -> transforms.Compose:
    base = [
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]
    if is_train:
        aug = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.03,
            ),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


class CTCDataset(Dataset):
    def __init__(self, paths: list, img_size: int = 128, is_train: bool = True):
        self.paths     = paths
        self.transform = build_transform(img_size, is_train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.paths[idx]


class KFoldDataModule:
    """
    10折数据模块。
    Train 折: 训练 G/D
    Val   折: 计算 FID（作为"真实分布"参照）
    """
    def __init__(self, dataset_dir: str, cfg: dict):
        tc, mc = cfg['training'], cfg['model']
        self.img_size    = mc['img_size']
        self.n_folds     = tc['n_folds']
        self.batch_size  = tc['batch_size']
        self.num_workers = tc['num_workers']
        self.seed        = tc['seed']

        all_paths = get_image_paths(dataset_dir)
        if not all_paths:
            raise RuntimeError(
                f"未找到图像: {dataset_dir}\n支持格式: {IMG_EXTS}")
        self.all_paths = np.array(all_paths)
        print(f"[Dataset] 共找到 {len(all_paths)} 张图像  ← {dataset_dir}")

        self.kf = KFold(n_splits=self.n_folds, shuffle=True,
                        random_state=self.seed)

    def folds(self):
        for fold, (tr_idx, va_idx) in enumerate(self.kf.split(self.all_paths)):
            tr_ds = CTCDataset(
                self.all_paths[tr_idx].tolist(), self.img_size, is_train=True)
            va_ds = CTCDataset(
                self.all_paths[va_idx].tolist(), self.img_size, is_train=False)

            tr_loader = DataLoader(tr_ds, batch_size=self.batch_size,
                                   shuffle=True,  num_workers=self.num_workers,
                                   pin_memory=True, drop_last=True)
            va_loader = DataLoader(va_ds, batch_size=self.batch_size,
                                   shuffle=False, num_workers=2,
                                   pin_memory=True)
            yield fold, tr_loader, va_loader

    def full_loader(self, is_train=False):
        ds = CTCDataset(self.all_paths.tolist(), self.img_size, is_train)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=is_train, num_workers=2, pin_memory=True)
