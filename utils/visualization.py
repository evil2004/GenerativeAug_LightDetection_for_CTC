"""可视化：生成网格 / 训练曲线 / 动态权重 / PSD / 雷达图"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from PIL import Image as PILImage
from pathlib import Path


def _denorm(x): return (x.clamp(-1,1)+1)/2


def save_grid(imgs, path, nrow=4):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    save_image(_denorm(imgs), path, nrow=nrow, padding=2)


def save_individual(imgs, out_dir, fold, start=0):
    d = Path(out_dir)/f'fold_{fold:02d}'; d.mkdir(parents=True, exist_ok=True)
    arr = (_denorm(imgs)*255).byte().permute(0,2,3,1).cpu().numpy()
    for i, img in enumerate(arr):
        PILImage.fromarray(img).save(str(d/f'gen_{start+i:04d}.png'))


def save_shape_debug(real_imgs, fake_imgs, path, nrow=4):
    """保存 real/fake 的蓝色主导软掩膜对比图。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def blue_soft_mask(x):
        x01 = _denorm(x)
        r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
        score = b - 0.5 * (r + g)
        return torch.sigmoid((score - 0.03) * 20.0)

    n = min(real_imgs.size(0), fake_imgs.size(0), nrow * nrow)
    real = real_imgs[:n].detach().cpu()
    fake = fake_imgs[:n].detach().cpu()

    real_m = blue_soft_mask(real).repeat(1, 3, 1, 1)
    fake_m = blue_soft_mask(fake).repeat(1, 3, 1, 1)

    # 上两行: real图 + real掩膜；下两行: fake图 + fake掩膜
    vis = torch.cat([_denorm(real), real_m, _denorm(fake), fake_m], dim=0)
    save_image(vis, path, nrow=n, padding=2)


def plot_loss_curves(history, path, fold):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    g_keys = [k for k in history if k.startswith('G')]
    d_keys = [k for k in history if k.startswith('D')]
    eps = range(1, len(next(iter(history.values())))+1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'Training Curves — Fold {fold+1}', fontweight='bold')
    for k in g_keys:
        axes[0].plot(eps, history[k], label=k, lw=1.5)
    axes[0].set_title('Generator'); axes[0].legend(fontsize=7)
    axes[0].set_xlabel('Epoch'); axes[0].grid(alpha=0.3)
    for k in d_keys:
        axes[1].plot(eps, history[k], label=k, lw=1.5)
    axes[1].set_title('Discriminator'); axes[1].legend(fontsize=7)
    axes[1].set_xlabel('Epoch'); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mgsm_weights(weight_log, path, fold):
    """绘制 MGSM 动态权重随训练进度变化的曲线。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    steps = range(len(weight_log))
    for i, label in enumerate(['w_low (精细纹理)', 'w_mid (结构+纹理)', 'w_high (全局形态)']):
        ax.plot(steps, [w[i] for w in weight_log], label=label, lw=1.8)
    ax.set_xlabel('Validation Step'); ax.set_ylabel('Weight')
    ax.set_title(f'Fold {fold+1}: MGSM Dynamic Weights (C1) — 课程式从高频→低频')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_anl_schedule(t_levels_log, path, fold):
    """绘制 ANL-D 噪声水平随训练自适应降低的曲线。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    steps = range(len(t_levels_log))
    for i in range(len(t_levels_log[0])):
        ax.plot(steps, [tl[i] for tl in t_levels_log],
                label=f'Level {i+1}', lw=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Noise Step t')
    ax.set_title(f'Fold {fold+1}: ANL-D Adaptive Noise Schedule (C3)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_psd(real_imgs, fake_imgs, path, fold):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    def rpsd(imgs):
        imgs = imgs.float().mean(1)
        xf   = torch.fft.fft2(imgs, norm='ortho')
        mag  = torch.fft.fftshift(xf.abs().pow(2), dim=(-2,-1)).mean(0).cpu().numpy()
        H,W  = mag.shape; cy,cx = H//2, W//2
        y,x  = np.ogrid[:H,:W]
        r    = np.sqrt((y-cy)**2+(x-cx)**2).astype(int)
        rmax = min(cy,cx)
        return np.array([mag[r==i].mean() if (r==i).any() else 0
                         for i in range(rmax)])
    r_psd=rpsd(real_imgs); f_psd=rpsd(fake_imgs)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(r_psd, label='Real CTC',       color='steelblue', lw=2)
    ax.semilogy(f_psd, label='DSG-GAN (Ours)', color='tomato',    lw=2, ls='--')
    ax.set_xlabel('Radial Frequency'); ax.set_ylabel('PSD (log)')
    ax.set_title(f'Fold {fold+1}: Power Spectral Density')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_radar(all_metrics, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lower_better = {'FID'}
    names = [k for k in all_metrics[0] if k not in ('IS_std',)]
    if not names: return
    fv = {m: np.array([f.get(m,0) for f in all_metrics]) for m in names}
    norm = {}
    for m, v in fv.items():
        mn,mx = v.min(), v.max()
        arr   = (v-mn)/(mx-mn+1e-8)
        norm[m] = 1-arr if m in lower_better else arr
    angles = np.linspace(0,2*np.pi,len(names),endpoint=False).tolist()+[0]
    cmap   = plt.colormaps['tab10']
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for i, fm in enumerate(all_metrics):
        vals = [norm[m][i] for m in names]+[norm[names[0]][i]]
        ax.plot(angles, vals, lw=1.3, color=cmap(i/len(all_metrics)),
                alpha=0.8, label=f'Fold {i+1}')
        ax.fill(angles, vals, alpha=0.05, color=cmap(i/len(all_metrics)))
    ax.set_thetagrids(np.degrees(angles[:-1]), names, fontsize=9)
    ax.set_ylim(0,1); ax.set_title('Metric Radar (↑ = better)', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), fontsize=8, ncol=2)
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def plot_bar(all_metrics, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    names = [k for k in all_metrics[0] if k not in ('IS_std',)]
    means = [np.mean([f.get(m,0) for f in all_metrics]) for m in names]
    stds  = [np.std ([f.get(m,0) for f in all_metrics]) for m in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10,len(names)*1.5), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_title('Cross-Fold Summary (mean ± std)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for b, mu in zip(bars, means):
        ax.annotate(f'{mu:.3f}', xy=(b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0,4), textcoords='offset points', ha='center', fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
