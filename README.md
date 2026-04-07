# GenerativeAug_LightDetection for CTC

Official code for the paper: **"A Generative Data Augmentation and Lightweight Detection Framework for Rare Circulating Cells in Liquid Biopsy"**

This repository implements a **Generative Data Augmentation** pipeline using a Diffusion Score-Guided GAN (DSG-GAN) to generate synthetic CTC images, combined with **Copy-Paste augmentation** to improve lightweight object detection (YOLO11 / Faster R-CNN) on microscopy blood smear images for Circulating Tumor Cells (CTC) and Circulating Endothelial Cells (CEC).

This project uses a **Diffusion Score-Guided GAN (DSG-GAN)** to generate synthetic CTC images, and applies **Copy-Paste augmentation** to improve lightweight object detection (YOLO11 / Faster R-CNN) on microscopy blood smear images.

---

## Project Structure

```
├── augment_ctc.py           # Copy-Paste data augmentation
├── train_yolo.py            # YOLO11 single-class detection training
├── train_detectron2.py      # Detectron2 / torchvision Faster R-CNN training
├── train.py                 # DSG-GAN generative training (2-stage)
├── configs/
│   └── config.yaml          # GAN hyperparameters
├── models/                  # GAN model definitions
│   ├── ddpm.py              # DDPM (U-Net, diffusion teacher)
│   ├── generator.py         # StyleGAN-like generator
│   ├── discriminator.py     # Discriminator
│   └── losses.py            # C1-MGSM / C2-DFA / C3-ANL losses
├── data/
│   └── dataset.py           # CTC dataset loader with K-fold splitting
├── utils/
│   ├── metrics.py           # FID, IS, Precision, Recall, Diversity
│   └── visualization.py     # Training curves, grids, radar plots
├── requirements.txt
├── LICENSE                  # Apache 2.0
└── README.md
```

---

## Requirements

- Python >= 3.10
- CUDA >= 12.x
- PyTorch >= 2.0

### Installation

```bash
# 1. Install PyTorch (match your CUDA version)
#    See https://pytorch.org/get-started/locally/
pip install torch torchvision

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. (Optional) Install Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## Dataset Preparation

Organize your dataset as follows:

```
Dataset/
├── images/          # Full-size microscopy images (.png/.jpg/.tif)
├── labels/          # YOLO-format labels (.txt), class 0=CTC, class 1=CEC
├── CTC/             # Cropped CTC cell sub-images (for Copy-Paste augmentation)
└── splits/          # Auto-generated train/val/test splits (by the scripts)
    ├── train/{images,labels}
    ├── val/{images,labels}
    └── test/{images,labels}
```

**YOLO label format** (one line per object):
```
class_id x_center y_center width height
```
All coordinates are normalized to [0, 1] relative to image dimensions.

For the GAN module, place cropped CTC cell images in a separate directory (default: `CTC/` at project root, configurable in `configs/config.yaml`).

The train/val/test split is **automatically created** (60/20/20) on first run if `splits/` is missing.

---

## Usage

### 1. Copy-Paste Data Augmentation

Paste CTC sub-images onto black background regions of training images. Run this standalone or it will be called automatically by the training scripts.

```bash
python augment_ctc.py
```

**What it does:**
- Restores the training split to its clean (unaugmented) state from `Dataset/images` and `Dataset/labels`
- For 25% of training images, randomly pastes 2-5 CTC sub-images
- Sub-images are scaled to average size (±5%) and lightly rotated (±15°)
- Only pastes onto black (background) regions, avoiding overlap with existing objects
- Appends new YOLO-format bounding box labels

### 2. YOLO11 Detection Training

```bash
# Train CTC detector (default)
python train_yolo.py --cell CTC

# Train CEC detector
python train_yolo.py --cell CEC
```

**Pipeline:**
1. Runs Copy-Paste augmentation automatically
2. Extracts single-class labels (CTC=0 or CEC=1 → remapped to class 0)
3. Performs hard positive mining using the previous best checkpoint (if exists)
4. Builds a weighted training list: positive samples ×7, hard positives ×14, background ×0.35
5. Trains YOLO11n (`imgsz=1280`, 420 epochs, AdamW, cosine LR)
6. Evaluates on val/test splits
7. Exports the best model to ONNX

**Outputs:** saved to `yolo/runs/y11_ctc_only/` (weights, plots, ONNX)

> **Note:** This script expects a local `ultralytics/` directory with the YOLO11 config at `ultralytics/cfg/models/11/yolo11.yaml`. Clone or install [Ultralytics](https://github.com/ultralytics/ultralytics) and place it at the project root.

### 3. Detectron2 / Faster R-CNN Training

```bash
# RetinaNet (Detectron2 backend)
python train_detectron2.py --model retinanet --cell CTC

# Faster R-CNN (Detectron2 backend)
python train_detectron2.py --model fast_rcnn --cell CTC

# ResNet-101 + FPN (pure torchvision, no Detectron2 required)
python train_detectron2.py --model resnet101_custom --cell CTC --augment yes

# Interactive mode (prompts for all options)
python train_detectron2.py --interactive
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `retinanet` | `retinanet`, `fast_rcnn`, or `resnet101_custom` |
| `--cell` | `CTC` | Target cell type: `CTC` or `CEC` |
| `--augment` | `no` | Run Copy-Paste augmentation before training (`yes`/`no`) |
| `--kfolds` | `5` | Number of K-fold cross-validation folds |
| `--epochs` | `40` | Training epochs (for `resnet101_custom` only) |
| `--max-iter` | `3500` | Max iterations (for Detectron2 models) |
| `--batch-size` | `4` | Batch size |
| `--lr` | `2e-4` | Learning rate |
| `--seed` | `42` | Random seed |

**Outputs:** saved to `runs_detectron2/<timestamp>_<model>_<cell>/`

### 4. DSG-GAN Generative Training

```bash
# Full training (DDPM pretraining + 10-fold GAN)
python train.py

# Options
python train.py --skip_ddpm          # Skip DDPM stage (requires existing checkpoint)
python train.py --fold 3             # Train only fold #3
python train.py --epochs 100         # Override GAN epochs
python train.py --ddpm_epochs 200    # Override DDPM epochs
python train.py --gpu 1              # Use specific GPU
```

**Two-stage pipeline:**

| Stage | Description | Default Epochs |
|-------|-------------|----------------|
| 1. DDPM Teacher | Pretrain a lightweight U-Net (~4M params) on all CTC images | 300 |
| 2. DSG-GAN | 10-fold cross-validation; frozen DDPM guides GAN training | 300 per fold |

**Three novel loss components:**
- **C1-MGSM** (Multi-Granularity Score Matching): Score matching at low/mid/high noise levels for fine textures, structure, and global morphology
- **C2-DFA-G** (Diffusion Feature Alignment): Multi-resolution feature consistency between generated and real images
- **C3-ANL-D** (Adaptive Noise-Level Discriminator): Curriculum-based noise level scheduling for the discriminator

**Config:** All hyperparameters are in `configs/config.yaml`. Key fields:

```yaml
paths:
  dataset_dir: "CTC"           # CTC source images directory
  output_dir:  "output"        # Training outputs

ddpm:                          # Stage 1 settings
  epochs: 300
  img_size: 128
  T: 1000

training:                      # Stage 2 settings
  n_folds: 10
  epochs: 300
  batch_size: 16
```

**Evaluation metrics:** FID, IS, Precision, Recall, Diversity — computed every `val_interval` epochs with early stopping (patience = 30 on FID).

**Outputs:**
```
output/
├── images/
│   ├── fold_XX/              # Generated images per fold
│   └── grids/fold_XX/        # Training process grids
├── images/plots/             # Loss curves, PSD, radar plots, metric charts
├── txt/
│   ├── fold_XX.txt           # Best metrics per fold
│   ├── fold_XX_val_log.csv   # Validation history (CSV)
│   └── overall.txt           # 10-fold summary
└── checkpoints/fold_XX/      # Best generator weights
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@software{generativeaug_ctc,
  title = {GenerativeAug: Diffusion Score-Guided GAN for CTC Detection},
  author = {Chream},
  license = {Apache-2.0},
  url = {https://github.com/<your-repo>/GenerativeAug_LightDetection_for_CTC}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
