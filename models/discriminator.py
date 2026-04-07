"""
ANL-D: Adaptive Noise-Level Discriminator  [C3 核心创新]
=========================================================
标准 GAN 判别器只看干净图像，在小数据集（400张）上
判别器会迅速饱和（对真实图100%正确），
梯度消失导致生成器停止更新。

ANL-D 的解决方案：
  在多个噪声水平 {t_1, t_2, t_3} 上同时判别图像，
  并随着训练进度自适应降低噪声水平：

  早期训练: 噪声大 → 图像模糊，判别器看全局形态（容易），
            梯度信号充足
  后期训练: 噪声小 → 接近干净图，判别器看精细纹理（难），
            推动生成器提升细节

自适应调度：
  t_max(epoch) = t_max_start → t_max_end  线性递减

设计：
  共享骨干网络 + 3个独立预测头（每个噪声水平一个）
  使用 DDPM 前向加噪（和 DDPM 教师共享调度器）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── 基础块 ─────────────────────────────────────

def sn_conv(in_c, out_c, k=4, s=2, p=1, bias=False):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, k, s, p, bias=bias))


class DiscBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_norm=True):
        super().__init__()
        layers = [sn_conv(in_c, out_c, s=stride)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────── 共享骨干（不含输出头）───────────────────────────

class SharedBackbone(nn.Module):
    """
    输入任意噪声水平的图像，提取共享特征。
    返回中间特征（用于特征匹配损失）+ 最终特征图。
    """
    def __init__(self, in_c: int = 3, ndf: int = 64):
        super().__init__()
        self.l1 = DiscBlock(in_c,   ndf,    use_norm=False)  # 128→64
        self.l2 = DiscBlock(ndf,    ndf*2)                   # 64→32
        self.l3 = DiscBlock(ndf*2,  ndf*4)                   # 32→16
        self.l4 = DiscBlock(ndf*4,  ndf*8, stride=1)         # 16→16

    def forward(self, x, return_feats=False):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        if return_feats:
            return f4, [f1, f2, f3, f4]
        return f4


# ─────────── ANL-D 主体 ─────────────────────────────────────

class Discriminator(nn.Module):
    """
    ANL-D: Adaptive Noise-Level Discriminator

    forward(x, scheduler, t_levels) → fused_logits, feats

    训练时传入不同 epoch 对应的噪声水平列表 t_levels，
    判别器自动在这些水平上对图像加噪并多头判别。

    最终分数 = 各噪声水平 logits 的可学习加权均值。
    """
    def __init__(self, in_c: int = 3, ndf: int = 64, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        self.backbone = SharedBackbone(in_c, ndf)

        # 每个噪声水平的独立输出头
        self.heads = nn.ModuleList([
            nn.utils.spectral_norm(
                nn.Conv2d(ndf*8, 1, 4, 1, 1, bias=False))
            for _ in range(n_levels)
        ])

        # 可学习融合权重（初始化为均等权重）
        self.level_weights = nn.Parameter(torch.zeros(n_levels))

    def _add_noise(self,
                   x:         torch.Tensor,
                   t_val:     int,
                   scheduler) -> torch.Tensor:
        """利用 DDPM 调度器对图像加噪到时间步 t_val。"""
        if t_val == 0:
            return x
        B      = x.shape[0]
        device = x.device
        t      = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_noisy, _ = scheduler.q_sample(x, t)
        return x_noisy

    def forward(self,
                x:          torch.Tensor,
                scheduler,
                t_levels:   list[int],
                return_feats: bool = False):
        """
        x:        (B, 3, H, W) 干净图像（真实或生成）
        scheduler: DDPM 调度器（用于加噪）
        t_levels: 当前要使用的噪声水平列表，长度 = n_levels
        """
        weights = F.softmax(self.level_weights, dim=0)  # (n_levels,)
        all_logits = []
        all_feats  = []

        for i, t_val in enumerate(t_levels):
            x_noisy = self._add_noise(x, t_val, scheduler)
            feat, feats = self.backbone(x_noisy, return_feats=True)
            logits = self.heads[i](feat)
            all_logits.append(logits * weights[i])
            all_feats.append(feats)

        fused = torch.stack(all_logits, dim=0).sum(0)  # (B,1,H',W')

        if return_feats:
            # 返回 t=0（最低噪声）对应的特征（最稳定）
            return fused, all_feats[0]
        return fused

    def get_t_levels(self,
                     epoch:     int,
                     total_eps: int,
                     t_max_start: int = 600,
                     t_max_end:   int = 100) -> list[int]:
        """
        根据训练进度自适应计算噪声水平列表。

        epoch=0:   t_levels ≈ [600, 400, 200]  （高噪声，看全局）
        epoch=末:  t_levels ≈ [100,  66,  33]  （低噪声，看细节）
        """
        progress = min(epoch / max(total_eps - 1, 1), 1.0)
        t_max    = int(t_max_start + (t_max_end - t_max_start) * progress)
        step     = t_max // self.n_levels
        return [max(t_max - i * step, 0) for i in range(self.n_levels)]
