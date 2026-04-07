"""
DSG-GAN 一键训练脚本
=====================
运行: python train.py

═══════════════════════════════════════════════════════
两阶段自动执行：

  阶段一  DDPM 教师预训练
          · 使用全部 CTC 数据
          · 轻量 U-Net (~4M)，约 300 epoch
          · 完成后保存 ddpm_teacher.pth，后续自动跳过

  阶段二  DSG-GAN 10折交叉验证训练
          · 每折独立训练 G + D
          · DDPM 冻结作为教师
          · 同时优化 C1-MGSM + C2-DFA + C3-ANL 三个创新点
          · 验证指标: FID / IS / Precision / Recall / Diversity
          · 早停: FID 连续 30 轮无改善则停止当前折
═══════════════════════════════════════════════════════

输出:
  output/images/fold_XX/          生成图
  output/images/grids/fold_XX/    训练过程网格
  output/images/plots/            曲线图、PSD、动态权重图
                                  + val_metrics_fold_XX.png  验证指标曲线
                                  + final_eval_fold_XX.png   最终指标柱状图
  output/txt/fold_XX.txt          每折最优指标
  output/txt/fold_XX_val_log.csv  每折验证历史（可导入Excel）
  output/txt/overall.txt          10折汇总
  output/checkpoints/fold_XX/     最优权重
"""

import os, sys, json, time, copy, random, argparse, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from models import DDPM, Generator, Discriminator, DSGLoss
from data   import KFoldDataModule, get_image_paths, CTCDataset
from utils  import (
    MetricEvaluator,
    save_grid, save_individual, save_shape_debug,
    plot_loss_curves, plot_mgsm_weights, plot_anl_schedule,
    plot_psd, plot_radar, plot_bar,
)
from torch.utils.data import DataLoader


# ────────────────────────────────────────────────────────────
#  早停配置
# ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 30     # FID 连续 30 个验证点无改善则停止

# 打印宽度常量
W  = 62          # 主分隔线宽度
SEP_THICK = '=' * W
SEP_THIN  = '-' * W
SEP_DOT   = '- ' * (W // 2)

LOWER_BETTER = {'FID'}


# ────────────────────────────────────────────────────────────
#  打印工具函数
# ────────────────────────────────────────────────────────────

def _fv(v, decimals=4):
    """格式化指标值。"""
    if v is None:             return '-'
    if isinstance(v, int):    return str(v)
    if isinstance(v, float):  return f'{v:.{decimals}f}'
    return str(v)


def _arrow(name):
    return '↓ lower is better' if name in LOWER_BETTER else '↑ higher is better'


def print_val_header(fold: int):
    """每折开始时打印一次表头说明（简洁模式）。"""
    tqdm.write(f'\n  {SEP_THICK}')
    tqdm.write(f'  Fold {fold+1}  验证指标  (每 val_interval 轮打印一次)')
    tqdm.write(f'  {SEP_THICK}')


def print_val_row(epoch: int, total_ep: int,
                  metrics: dict,
                  es_counter: int,
                  is_best: bool):
    """每轮验证打印两列表格（横向布局）。"""
    best_tag = '  *** NEW BEST ***' if is_best else ''
    es_tag   = f'ES: {es_counter}/{EARLY_STOP_PATIENCE}'

    tqdm.write(f'\n  {SEP_THIN}')
    tqdm.write(f'  Ep {epoch:03d}/{total_ep}{best_tag:<20}  {es_tag}')
    tqdm.write(f'  {SEP_THIN}')

    keys = ['FID', 'IS_mean', 'Precision', 'Recall', 'Diversity', 'PSNR', 'SSIM']
    max_k = max(len(k) for k in keys)
    col1 = '  ' + '  '.join(f'{k:<{max_k}}' for k in keys)
    col2 = '  ' + '  '.join(f'{_fv(metrics.get(k)):<{max_k}}' for k in keys)
    tqdm.write(col1)
    tqdm.write(col2)

    tqdm.write(f'  {SEP_THIN}')


def print_val_footer():
    """折训练结束时的分隔线（保持接口一致）。"""
    tqdm.write(f'  {SEP_THICK}')


def print_final_table(metrics: dict, fold: int, title: str = '最终评估'):
    """
    每折训练结束后打印完整的最终评估结果块：

    ==============================================================
    Final Evaluation  Fold 1   (best_ep=147)
    ==============================================================
      Metric              Value         Direction
    --------------------------------------------------------------
      FID               45.2310        ↓ lower is better
      IS_mean            2.1840        ↑ higher is better
      ...
    ==============================================================
    """
    skip = {'IS_std', 'best_epoch', 'stopped_epoch'}

    tqdm.write(f'\n  {SEP_THICK}')
    best_ep = metrics.get('best_epoch', '?')
    stopped = metrics.get('stopped_epoch', '?')
    tqdm.write(f'  {title}  --  Fold {fold+1}  '
               f'(best_ep={best_ep}, stopped_ep={stopped})')
    tqdm.write(f'  {SEP_THICK}')
    tqdm.write(f'  {"Metric":<20}  {"Value":>12}    {"Direction"}')
    tqdm.write(f'  {SEP_THIN}')

    for k, v in metrics.items():
        if k in skip:
            continue
        val_str = _fv(v) if isinstance(v, float) else str(v)
        arrow   = f'  {_arrow(k)}' if isinstance(v, float) else ''
        tqdm.write(f'  {k:<20}  {val_str:>12}{arrow}')

    tqdm.write(f'  {SEP_THICK}\n')


def print_overall_table(all_metrics: list):
    """
    10折全部结束后打印汇总表：

    ==============================================================
    10-Fold Cross-Validation  Final Summary
    ==============================================================
      Metric              Mean          Std         Direction
    --------------------------------------------------------------
      FID               45.2310       3.1200       ↓ lower is better
      IS_mean            2.1840       0.2100       ↑ higher is better
      ...
    ==============================================================
    """
    skip  = {'IS_std', 'best_epoch', 'stopped_epoch'}
    names = [k for k in all_metrics[0] if k not in skip]

    tqdm.write(f'\n  {SEP_THICK}')
    tqdm.write('  10-Fold Cross-Validation  Final Summary')
    tqdm.write(f'  {SEP_THICK}')
    tqdm.write(f'  {"Metric":<20}  {"Mean":>10}  {"Std":>10}    {"Direction"}')
    tqdm.write(f'  {SEP_THIN}')

    for m in names:
        vals  = [f.get(m, float('nan')) for f in all_metrics]
        mu    = np.nanmean(vals)
        sg    = np.nanstd(vals)
        arrow = f'  {_arrow(m)}'
        tqdm.write(f'  {m:<20}  {mu:>10.4f}  {sg:>10.4f}{arrow}')

    tqdm.write(f'  {SEP_THICK}\n')


# ────────────────────────────────────────────────────────────
#  指标曲线图（每折验证历史）
# ────────────────────────────────────────────────────────────

def plot_val_metrics(val_log: list[dict], save_path: str, fold: int):
    """
    绘制验证指标随 epoch 变化的曲线图。
    val_log: [{'epoch':10, 'FID':82.3, 'IS_mean':1.8, ...}, ...]
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if not val_log:
        return

    metric_keys = ['FID', 'IS_mean', 'Precision', 'Recall', 'Diversity', 'PSNR', 'SSIM']
    available   = [k for k in metric_keys if k in val_log[0]]
    if not available:
        return

    epochs = [d['epoch'] for d in val_log]
    n      = len(available)
    cols   = min(n, 3)
    rows   = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()

    lower = {'FID'}
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#f39c12']

    for i, key in enumerate(available):
        vals = [d.get(key, None) for d in val_log]
        ax   = axes[i]
        ax.plot(epochs, vals, color=colors[i % len(colors)],
                lw=2, marker='o', markersize=4, label=key)

        # 标出最优点
        arr = np.array(vals, dtype=float)
        best_idx = int(np.nanargmin(arr) if key in lower else np.nanargmax(arr))
        ax.axvline(epochs[best_idx], ls='--', color='gray', alpha=0.5)
        ax.scatter([epochs[best_idx]], [vals[best_idx]],
                   s=80, color='gold', zorder=5, label=f'best={vals[best_idx]:.3f}')

        direction = '↓' if key in lower else '↑'
        ax.set_title(f'{key} {direction}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 隐藏多余子图
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'DSG-GAN  Fold {fold+1}  Validation Metrics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_eval(metrics: dict, save_path: str, fold: int):
    """绘制最终评估的柱状图（单折）。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    skip  = {'IS_std', 'best_epoch'}
    lower = {'FID'}
    names = [k for k in metrics if k not in skip]
    vals  = [metrics[k] for k in names if isinstance(metrics[k], float)]
    names = [k for k in names if isinstance(metrics[k], float)]
    if not names:
        return

    colors = ['#e74c3c' if n in lower else '#2980b9' for n in names]
    x      = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 4))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_title(f'DSG-GAN  Fold {fold+1}  Final Evaluation Metrics',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, v in zip(bars, vals):
        ax.annotate(f'{v:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9)

    # 图例说明颜色含义
    from matplotlib.patches import Patch
    legend_els = [Patch(fc='#e74c3c', label='↓ lower is better'),
                  Patch(fc='#2980b9', label='↑ higher is better')]
    ax.legend(handles=legend_els, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_val_csv(val_log: list[dict], save_path: str):
    """将每折验证历史保存为 CSV，方便导入 Excel 分析。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if not val_log:
        return
    keys = list(val_log[0].keys())
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(val_log)


# ────────────────────────────────────────────────────────────
#  其他工具函数
# ────────────────────────────────────────────────────────────

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device(gpu_id: int | None = None):
    if torch.cuda.is_available():
        if gpu_id is None:
            gpu_id = int(os.environ.get('CTC_GPU', 1))
        d = torch.device(f'cuda:{gpu_id}')
        print(f"  [设备] GPU{gpu_id} ▶ {torch.cuda.get_device_name(gpu_id)}")
    else:
        d = torch.device('cpu')
        print("  [设备] CPU")
    return d

def load_cfg(p):
    with open(p, encoding='utf-8') as f: return yaml.safe_load(f)

def pcount(m): return f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M"

def write_txt(d, path, title=''):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sep   = '─' * 58
    lines = [sep, f'  {title}', sep] if title else [sep]
    for k, v in d.items():
        lines.append(f"  {k:<22s}: {v:.6f}" if isinstance(v, float)
                     else f"  {k:<22s}: {v}")
    lines.append(sep)
    txt = '\n'.join(lines) + '\n'
    with open(path, 'w', encoding='utf-8') as f: f.write(txt)
    return txt

def write_overall(all_m, path, timing):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sep   = '═' * 62
    lines = [sep, '  DSG-GAN · 10-Fold Cross-Validation · Overall Results', sep, '']
    for m in [k for k in all_m[0] if k not in ('IS_std', 'best_epoch')]:
        vals = [f.get(m, float('nan')) for f in all_m]
        mu = np.nanmean(vals); sg = np.nanstd(vals)
        arrow = '↓ better' if m == 'FID' else '↑ better'
        lines.append(f"  {m:<20s}  mean={mu:.4f}  std={sg:.4f}  [{arrow}]")
    lines += ['',
              f"  Folds:          {len(all_m)}",
              f"  Total time:     {timing['total']:.1f} s",
              f"  Avg / fold:     {timing['per_fold']:.1f} s", sep]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1-self.decay)

    def sample(self, n, device):
        return self.shadow.sample(n, device)


# ════════════════════════════════════════════════════════════
#  阶段一：DDPM 预训练
# ════════════════════════════════════════════════════════════

def pretrain_ddpm(cfg, device, out_dir):
    dc   = cfg['ddpm']
    ckpt = cfg['paths']['ddpm_ckpt']
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    if os.path.exists(ckpt):
        print(f"\n  [阶段一] DDPM 检查点已存在，跳过预训练 ▶ {ckpt}")
        ddpm = DDPM(cfg).to(device)
        ddpm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        ddpm.freeze()
        return ddpm

    print(f"\n{'═'*62}")
    print("  阶段一：DDPM 教师预训练")
    print(f"{'═'*62}")

    ddpm = DDPM(cfg).to(device)
    print(f"  DDPM 参数量: {ddpm.param_count()}")

    all_paths = get_image_paths(cfg['paths']['dataset_dir'])
    ds     = CTCDataset(all_paths, dc['img_size'], is_train=True)
    loader = DataLoader(ds, batch_size=dc['batch_size'],
                        shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    opt    = optim.AdamW(ddpm.parameters(), lr=dc['lr'], weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, dc['epochs'], eta_min=1e-6)
    use_amp = False
    best_loss = float('inf')

    print(f'\n  {SEP_THICK}')
    print('  DDPM 教师预训练进度')
    print(f'  {SEP_THICK}')
    print(f'  {"Epoch":<14}  {"Loss":>10}  {"Best":>10}  {"LR":>12}')
    print(f'  {SEP_THIN}')

    for epoch in range(1, dc['epochs'] + 1):
        ddpm.train()
        ep_loss = 0.0; n = 0

        pbar = tqdm(loader,
                    desc=f"  [DDPM] Ep {epoch:03d}/{dc['epochs']}",
                    ncols=100, leave=False)

        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            loss = ddpm.training_loss(imgs)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                opt.step()
            else:
                tqdm.write(
                    f'  [警告] DDPM step NaN loss={loss.item():.3f} — 跳过')
            ep_loss += loss.item(); n += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        sched.step()
        avg = ep_loss / n

        if avg < best_loss:
            best_loss = avg
            torch.save(ddpm.state_dict(), ckpt)

        if epoch % 50 == 0 or epoch == dc['epochs']:
            lr_str = f"{opt.param_groups[0]['lr']:.2e}"
            tqdm.write(f'  {epoch:03d}/{dc["epochs"]:<10}  '
                       f'{avg:>10.4f}  {best_loss:>10.4f}  {lr_str:>12}')

    tqdm.write(f'  {SEP_THIN}')
    tqdm.write(f"\n  [DDPM] 预训练完成，最优 loss={best_loss:.4f}")
    tqdm.write(f"  [DDPM] 权重已保存 → {ckpt}\n")

    ddpm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    ddpm.freeze()
    return ddpm


# ════════════════════════════════════════════════════════════
#  阶段二：单折 GAN 训练
# ════════════════════════════════════════════════════════════

def train_fold(fold, train_loader, val_loader,
               ddpm, cfg, device, out_dir):

    tc  = cfg['training']
    mc  = cfg['model']
    ac  = cfg['anl']
    epochs   = tc['epochs']
    z_dim    = mc['z_dim']
    use_amp  = False
    r1_every = tc['r1_every']

    # ── 模型 ────────────────────────────────────────────────
    G = Generator(z_dim=mc['z_dim'], w_dim=mc['w_dim'],
                  ngf=mc['ngf'], map_layers=4,
                  img_size=mc['img_size']).to(device)
    D = Discriminator(in_c=3, ndf=mc['ndf'],
                      n_levels=ac['n_noise_levels']).to(device)
    ema       = EMA(G, tc['ema_decay'])
    criterion = DSGLoss(cfg).to(device)

    print(f"\n  G 参数量: {pcount(G)}  │  D 参数量: {pcount(D)}")

    # ── 优化器 ───────────────────────────────────────────────
    mgsm_params    = list(criterion.mgsm_loss.weight_mlp.parameters())
    other_g_params = [p for p in G.parameters()
                      if not any(p is mp for mp in mgsm_params)]

    opt_G = optim.Adam(
        other_g_params + list(criterion.align_loss.parameters()),
        lr=tc['lr_g'], betas=(tc['beta1'], tc['beta2']))
    opt_W = optim.Adam(mgsm_params,
                        lr=cfg['mgsm']['weight_lr'],
                        betas=(0.9, 0.99))
    opt_D = optim.Adam(D.parameters(),
                        lr=tc['lr_d'], betas=(tc['beta1'], tc['beta2']))

    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, epochs, eta_min=1e-6)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, epochs, eta_min=1e-6)

    # ── 路径 ─────────────────────────────────────────────────
    ckpt_dir = Path(out_dir) / 'checkpoints' / f'fold_{fold:02d}'
    grid_dir = Path(out_dir) / 'images' / 'grids' / f'fold_{fold:02d}'
    plot_dir = Path(out_dir) / 'images' / 'plots'
    txt_dir  = Path(out_dir) / 'txt'
    for d in [ckpt_dir, grid_dir, plot_dir, txt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fixed_z = torch.randn(tc['n_vis'], z_dim, device=device)

    # ── 历史 & 早停 ──────────────────────────────────────────
    history      = defaultdict(list)
    val_log      = []          # 用于绘图和CSV
    mgsm_w_log   = []
    anl_t_log    = []
    best_fid     = float('inf')
    best_metrics = {}
    es_counter   = 0           # 早停计数器
    stopped_ep   = epochs      # 实际停止轮次
    global_step  = 0
    evaluator    = MetricEvaluator(str(device), cfg['metrics']['compute_fid'])

    # 打印验证表头（每折只打印一次）
    print_val_header(fold)

    # ══════════════════ 训练主循环 ════════════════════════════
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        ep_loss  = defaultdict(float)
        n_batch  = 0
        progress = (epoch - 1) / max(epochs - 1, 1)

        # ANL-D 噪声水平 & DFA 对齐水平
        t_levels = D.get_t_levels(epoch-1, epochs,
                                   ac['t_max_start'], ac['t_max_end'])
        anl_t_log.append(t_levels)

        dc2     = cfg['dfa']
        t_align = int(dc2['t_align_start'] +
                      (dc2['t_align_end'] - dc2['t_align_start']) * progress)

        pbar = tqdm(
            train_loader,
            desc  = (f"  Fold {fold+1}/{tc['n_folds']}  "
                     f"Ep {epoch:03d}/{epochs}  "
                     f"t_align={t_align}"),
            ncols = 120,
            leave = False,
        )

        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device, non_blocking=True)
            B         = real_imgs.shape[0]
            global_step += 1
            do_r1 = (global_step % r1_every == 0)

            # ════ 判别器 ════════════════════════════════════
            # 对抗损失（float32）
            opt_D.zero_grad(set_to_none=True)
            z         = torch.randn(B, z_dim, device=device)
            fake_imgs = G(z).detach()
            real_logits = D(real_imgs, ddpm.scheduler,
                             t_levels, return_feats=False)
            fake_logits    = D(fake_imgs, ddpm.scheduler,
                               t_levels, return_feats=False)
            d_adv = criterion.hinge.d_loss(real_logits, fake_logits)
            if torch.isfinite(d_adv):
                d_adv.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()
            else:
                tqdm.write(
                    f'  [警告] step{global_step} D NaN loss={d_adv.item():.3f} — 跳过')

            # R1惩罚单独做，全程float32与AMP完全隔离
            d_r1_val = 0.0
            if do_r1:
                opt_D.zero_grad(set_to_none=True)
                real_f32 = real_imgs.detach().float().requires_grad_(True)
                rl_f32   = D(real_f32, ddpm.scheduler,
                              t_levels, return_feats=False).float()
                grad = torch.autograd.grad(
                    outputs=rl_f32.sum(), inputs=real_f32,
                    create_graph=True, only_inputs=True,
                )[0]
                r1_loss = criterion.lambda_r1 * 0.5 * grad.pow(2).reshape(B, -1).sum(1).mean()
                r1_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()
                d_r1_val = r1_loss.item()

            d_log = {
                'D_adv':   d_adv.item(),
                'D_r1':    d_r1_val,
                'D_total': d_adv.item() + d_r1_val,
            }

            # ════ 生成器 ════════════════════════════════════
            opt_G.zero_grad(set_to_none=True)
            opt_W.zero_grad(set_to_none=True)

            # 对抗 + 特征匹配（float32）
            z_g = torch.randn(B, z_dim, device=device)
            fake_imgs_g, gen_feats = G.forward_with_features(z_g)
            fake_logits_g, fake_feats = D(fake_imgs_g, ddpm.scheduler,
                                           t_levels, return_feats=True)
            _,             real_feats = D(real_imgs,   ddpm.scheduler,
                                           t_levels, return_feats=True)
            g_adv = criterion.hinge.g_loss(fake_logits_g)
            g_fm  = criterion.fm_loss(real_feats, fake_feats)

            # DFA特征对齐（float32）
            ddpm_feats    = ddpm.get_features(real_imgs, t_align)
            gen_feats_f32 = {k: v.float() if v is not None else None
                             for k, v in gen_feats.items()}
            g_align = criterion.align_loss(gen_feats_f32, ddpm_feats)

            # MGSM分数匹配（float32，梯度必须流过fake_imgs_g，禁止no_grad）
            g_mgsm, mgsm_log = criterion.mgsm_loss(
                fake_imgs_g,   # 保留梯度图
                real_imgs,
                ddpm, progress,
            )

            g_color = criterion.color_prior_loss(fake_imgs_g, real_imgs)
            g_shape = criterion.shape_prior_loss(fake_imgs_g, real_imgs)

            g_loss = (g_adv
                      + criterion.lambda_fm    * g_fm
                      + criterion.lambda_align * g_align
                      + criterion.lambda_mgsm  * g_mgsm
                      + criterion.lambda_color * g_color
                      + criterion.lambda_shape * g_shape)

            g_log = {
                'G_adv':   g_adv.item(),
                'G_fm':    g_fm.item(),
                'G_align': g_align.item(),
                'G_color': g_color.item(),
                'G_shape': g_shape.item(),
                'G_total': g_loss.item(),
                **mgsm_log,
            }

            if torch.isfinite(g_loss):
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()
                opt_W.step()
                ema.update(G)
            else:
                tqdm.write(
                    f'  [警告] step{global_step} G NaN loss={g_loss.item():.3f} — 跳过')

            # NaN守卫：仅统计有效批次
            if torch.isfinite(g_loss) and torch.isfinite(d_adv):
                for k, v in {**g_log, **d_log}.items():
                    ep_loss[k] += v
                n_batch += 1
            else:
                tqdm.write(
                    f'  [警告] step{global_step} NaN '
                    f'G={g_log["G_total"]:.3f} D={d_log["D_total"]:.3f} — 跳过')

            pbar.set_postfix({
                'G':     f"{g_log['G_total']:.3f}",
                'D':     f"{d_log['D_total']:.3f}",
                'align': f"{g_log['G_align']:.4f}",
                'mgsm':  f"{mgsm_log.get('mgsm_total', 0):.4f}",
                'lr_G':  f"{opt_G.param_groups[0]['lr']:.1e}",
            }, refresh=True)

        # n_batch守卫：防止全部NaN时除零崩溃
        if n_batch > 0:
            for k, v in ep_loss.items():
                history[k].append(v / n_batch)
        else:
            tqdm.write(f'  [警告] Ep{epoch} 全部批次NaN，跳过历史记录')
            for k in list(ep_loss.keys()) or ['G_total', 'D_total']:
                history[k].append(float('nan'))

        sched_G.step(); sched_D.step()

        # ════ 验证 ═══════════════════════════════════════════
        if epoch % tc['val_interval'] == 0 or epoch == epochs:
            G.eval(); ema.shadow.eval(); evaluator.reset()
            real_batch_vis = None

            for real_imgs, _ in val_loader:
                if real_batch_vis is None:
                    real_batch_vis = real_imgs
                evaluator.update_real(real_imgs.to(device))

            n_gen_v  = max(len(val_loader.dataset), tc['n_vis'])
            fake_buf = []
            with torch.no_grad():
                for i in range(0, n_gen_v, tc['batch_size']):
                    bsz = min(tc['batch_size'], n_gen_v - i)
                    gen = ema.shadow(torch.randn(bsz, z_dim, device=device))
                    evaluator.update_fake(gen)
                    fake_buf.append(gen.cpu())

            metrics = evaluator.compute()

            # 记录 MGSM 权重
            mgsm_w = [g_log.get('w_low', 0),
                      g_log.get('w_mid', 0),
                      g_log.get('w_high', 0)]
            mgsm_w_log.append(mgsm_w)

            # val_log 追加（只记录评估指标，不记录训练损失）
            log_entry = {'epoch': epoch, **metrics}
            val_log.append(log_entry)

            # ── 判断是否最优 & 早停 ───────────────────────
            fid_now = metrics.get('FID', float('inf'))
            is_best = fid_now < best_fid

            if is_best:
                best_fid     = fid_now
                best_metrics = metrics.copy()
                best_metrics['best_epoch'] = epoch
                es_counter   = 0
                torch.save({
                    'epoch':  epoch,
                    'G':      G.state_dict(),
                    'G_ema':  ema.shadow.state_dict(),
                    'D':      D.state_dict(),
                    'metrics': best_metrics,
                }, str(ckpt_dir / 'best.pth'))
            else:
                es_counter += 1

            # ── 验证指标多行打印 ──────────────────────────
            print_val_row(epoch, epochs, metrics, es_counter, is_best)

            # 固定噪声可视化
            with torch.no_grad():
                vis = ema.shadow(fixed_z)
            save_grid(vis, str(grid_dir / f'ep_{epoch:03d}.png'), nrow=4)

            # 形态调试图（real/fake + 蓝色软掩膜）
            if fake_buf and real_batch_vis is not None:
                save_shape_debug(
                    real_batch_vis,
                    fake_buf[0],
                    str(plot_dir / f'fold_{fold:02d}_shape_debug_ep_{epoch:03d}.png'),
                    nrow=4,
                )

            # PSD（最后验证时）
            if (epoch == epochs or es_counter >= EARLY_STOP_PATIENCE) and fake_buf:
                real_batch = next(iter(val_loader))[0]
                plot_psd(real_batch, fake_buf[0],
                         str(plot_dir / f'fold_{fold:02d}_psd.png'), fold)

            # ── 早停检查 ──────────────────────────────────
            if es_counter >= EARLY_STOP_PATIENCE:
                tqdm.write(
                    f"\n  ⚠ 早停触发  FID 连续 {EARLY_STOP_PATIENCE} 次验证"
                    f"无改善（当前 FID={fid_now:.2f}，最优 FID={best_fid:.2f}）"
                    f"  实际训练至 Ep {epoch}")
                stopped_ep = epoch
                break

        # 定期网格
        if epoch % tc['save_interval'] == 0:
            with torch.no_grad():
                vis = ema.shadow(torch.randn(16, z_dim, device=device))
            save_grid(vis, str(grid_dir / f'ep_{epoch:03d}_extra.png'))

    # 关闭验证表格
    print_val_footer()

    # ════ 最终评估 ════════════════════════════════════════════
    tqdm.write(f"\n  [Fold {fold+1}] 最终评估（加载最优权重 Ep={best_metrics.get('best_epoch','?')}）")
    if (ckpt_dir / 'best.pth').exists():
        ckpt_data = torch.load(str(ckpt_dir / 'best.pth'),
                               map_location=device, weights_only=False)
        if 'G_ema' in ckpt_data:
            ema.shadow.load_state_dict(ckpt_data['G_ema'])
        else:
            ema.shadow.load_state_dict(ckpt_data['G'])
    ema.shadow.eval()

    # 在完整 val 集上重新计算最终指标
    final_evaluator = MetricEvaluator(str(device), cfg['metrics']['compute_fid'])
    for real_imgs, _ in val_loader:
        final_evaluator.update_real(real_imgs.to(device))

    final_fake_buf = []
    n_final = max(len(val_loader.dataset) * 2, tc['n_vis'])   # 生成2×验证集大小
    with torch.no_grad():
        for i in range(0, n_final, tc['batch_size']):
            bsz = min(tc['batch_size'], n_final - i)
            gen = ema.shadow(torch.randn(bsz, z_dim, device=device))
            final_evaluator.update_fake(gen)
            final_fake_buf.append(gen.cpu())

    final_metrics = final_evaluator.compute()
    if best_metrics:
        final_metrics['best_epoch'] = best_metrics.get('best_epoch', stopped_ep)
    final_metrics['stopped_epoch'] = stopped_ep

    # ── 最终指标：表格打印 ────────────────────────────────
    print_final_table(final_metrics, fold,
                      title=f'Final Evaluation  (best_ep={final_metrics.get("best_epoch","?")})')

    # ════ 折结束：保存图像 & 图表 ════════════════════════════

    # 1. 生成并保存全部图像
    tqdm.write(f"  [Fold {fold+1}] 保存生成图像 …")
    n_save = len(train_loader.dataset) + len(val_loader.dataset)
    idx    = 0
    with torch.no_grad():
        for i in range(0, n_save, tc['batch_size']):
            bsz = min(tc['batch_size'], n_save - i)
            gen = ema.shadow(torch.randn(bsz, z_dim, device=device))
            save_individual(gen, str(Path(out_dir) / 'images'), fold, idx)
            idx += bsz

    # 2. 训练损失曲线
    plot_loss_curves(dict(history),
                     str(plot_dir / f'fold_{fold:02d}_curves.png'), fold)

    # 3. MGSM 动态权重曲线
    if mgsm_w_log:
        plot_mgsm_weights(mgsm_w_log,
                          str(plot_dir / f'fold_{fold:02d}_mgsm_w.png'), fold)

    # 4. ANL-D 噪声调度曲线
    plot_anl_schedule(anl_t_log,
                      str(plot_dir / f'fold_{fold:02d}_anl.png'), fold)

    # 5. 验证指标曲线（新增）
    plot_val_metrics(val_log,
                     str(plot_dir / f'val_metrics_fold_{fold:02d}.png'), fold)

    # 6. 最终评估柱状图（新增）
    plot_final_eval(final_metrics,
                    str(plot_dir / f'final_eval_fold_{fold:02d}.png'), fold)

    # 7. 验证历史 CSV（新增）
    save_val_csv(val_log,
                 str(txt_dir / f'fold_{fold:02d}_val_log.csv'))

    # 8. 最优指标 TXT
    write_txt(
        final_metrics,
        str(txt_dir / f'fold_{fold:02d}.txt'),
        title=f'DSG-GAN · Fold {fold+1} · Final Metrics',
    )

    tqdm.write(f"  [Fold {fold+1}] 所有图表已保存至 {plot_dir}")

    return final_metrics


# ════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('DSG-GAN Training')
    parser.add_argument('--config',      default='configs/config.yaml')
    parser.add_argument('--fold',        type=int, default=0,
                        help='只跑某折（默认=0；-1=全部）')
    parser.add_argument('--epochs',      type=int, default=-1,
                        help='覆盖 GAN 训练轮数')
    parser.add_argument('--ddpm_epochs', type=int, default=-1,
                        help='覆盖 DDPM 预训练轮数')
    parser.add_argument('--skip_ddpm',   action='store_true',
                        help='跳过 DDPM 预训练（需检查点已存在）')
    parser.add_argument('--gpu',         type=int, default=None,
                        help='指定 GPU 编号，如 1 或 2（也可用环境变量 CTC_GPU）')
    args = parser.parse_args()

    cfg_path = str(Path(__file__).parent / args.config)
    cfg      = load_cfg(cfg_path)
    if args.epochs      > 0: cfg['training']['epochs'] = args.epochs
    if args.ddpm_epochs > 0: cfg['ddpm']['epochs']      = args.ddpm_epochs
    if args.skip_ddpm:        cfg['_skip_ddpm']          = True

    set_seed(cfg['training']['seed'])
    device  = get_device(args.gpu)
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("""
╔══════════════════════════════════════════════════════════════╗
║   DSG-GAN: Diffusion Score-Guided GAN                       ║
║                                                              ║
║   C1  MGSM  — 多粒度分数匹配损失（动态权重课程学习）        ║
║   C2  DFA-G — 扩散特征对齐生成器（流形锚定）                ║
║   C3  ANL-D — 自适应噪声水平判别器（防小数据饱和）          ║
║                                                              ║
║   早停策略  patience = 30 次验证无改善 → 停止当前折         ║
╚══════════════════════════════════════════════════════════════╝""")

    print(f"\n  Config   : {cfg_path}")
    print(f"  数据集   : {cfg['paths']['dataset_dir']}")
    print(f"  输出     : {out_dir}")
    if args.fold >= 0:
        print(f"  GAN轮数  : {cfg['training']['epochs']}  运行: 单折 fold={args.fold+1}/{cfg['training']['n_folds']}")
    else:
        print(f"  GAN轮数  : {cfg['training']['epochs']}  折数: {cfg['training']['n_folds']}")
    print(f"  DDPM轮数 : {cfg['ddpm']['epochs']}")
    print(f"  早停     : patience = {EARLY_STOP_PATIENCE} 次验证")
    print(f"  分辨率   : {cfg['model']['img_size']}×{cfg['model']['img_size']}\n")

    # ── 阶段一：DDPM ──────────────────────────────────────────
    if cfg.get('_skip_ddpm') and os.path.exists(cfg['paths']['ddpm_ckpt']):
        print(f"  [阶段一] --skip_ddpm，直接加载 {cfg['paths']['ddpm_ckpt']}")
        ddpm = DDPM(cfg).to(device)
        ddpm.load_state_dict(
            torch.load(cfg['paths']['ddpm_ckpt'],
                       map_location=device, weights_only=True))
        ddpm.freeze()
    else:
        ddpm = pretrain_ddpm(cfg, device, out_dir)

    # ── 阶段二：GAN ───────────────────────────────────────────
    print(f"\n{'═'*62}")
    if args.fold >= 0:
        print(f"  阶段二：DSG-GAN 单折训练（fold={args.fold+1}/{cfg['training']['n_folds']}）")
    else:
        print("  阶段二：DSG-GAN 10折交叉验证训练")
    print(f"{'═'*62}")

    dm          = KFoldDataModule(cfg['paths']['dataset_dir'], cfg)
    all_metrics = []
    fold_times  = []
    t0          = time.time()

    for fold, tr_loader, va_loader in dm.folds():
        if args.fold >= 0 and fold != args.fold:
            continue

        # 保存一份真实样本网格，确认模型看到的真实分布
        if fold == 0:
            real_batch = next(iter(tr_loader))[0][:16].to(device)
            save_grid(real_batch,
                      str(Path(out_dir) / 'images' / 'debug_real.png'),
                      nrow=4)

        print(f"\n{'━'*62}")
        print(f"  FOLD {fold+1}/{cfg['training']['n_folds']}  "
              f"Train={len(tr_loader.dataset)}  "
              f"Val={len(va_loader.dataset)}")
        print(f"{'━'*62}")

        t_fold = time.time()
        result = train_fold(fold, tr_loader, va_loader,
                             ddpm, cfg, device, out_dir)
        fold_times.append(time.time() - t_fold)
        all_metrics.append(result)

        print(f"\n  Fold {fold+1} 完成，用时 {fold_times[-1]/60:.1f} min")

    # ── 10折汇总 ─────────────────────────────────────────────
    t_total = time.time() - t0
    timing  = {'total': t_total, 'per_fold': float(np.mean(fold_times))}

    # 汇总表格（控制台）
    print_overall_table(all_metrics)
    print(f"  总用时: {t_total/60:.1f} min  │  "
          f"均值: {np.mean(fold_times)/60:.1f} min/fold\n")

    # 写 TXT + JSON
    txt_dir = Path(out_dir) / 'txt'
    write_overall(all_metrics, str(txt_dir / 'overall.txt'), timing)
    with open(str(txt_dir / 'all_fold_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # 跨折汇总图
    plot_dir = str(Path(out_dir) / 'images' / 'plots')
    if len(all_metrics) > 1:
        plot_radar(all_metrics, os.path.join(plot_dir, 'radar.png'))
        plot_bar  (all_metrics, os.path.join(plot_dir, 'summary_bar.png'))

    print(f"  所有输出 → {out_dir}")
    print(f"  图表目录 → {plot_dir}")
    print("  训练完成 ✓\n")


if __name__ == '__main__':
    main()
