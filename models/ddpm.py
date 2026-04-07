"""
轻量 DDPM 教师网络（第一阶段，离线预训练）
==========================================
架构：小型 U-Net，约 4M 参数
功能：
  1. 标准 DDPM 去噪训练（预测噪声 ε）
  2. 暴露编码器中间特征 → 供 DFA-G 特征对齐使用
  3. 暴露分数函数 s_θ(x_t, t) → 供 MGSM 损失使用

分数函数定义：
  s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────── 时间步嵌入 ─────────────────────────────────

class SinusoidalPE(nn.Module):
    """正弦位置编码，将标量时间步 t 映射到向量。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  →  (B, dim)
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=1)    # (B, dim)


class TimeEmbedding(nn.Module):
    def __init__(self, t_dim: int, out_dim: int):
        super().__init__()
        self.pe  = SinusoidalPE(t_dim)
        self.mlp = nn.Sequential(
            nn.Linear(t_dim,  out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pe(t))


# ──────────────── 基础卷积块（含时间注入）────────────────────

class ResBlock(nn.Module):
    """ResBlock with time-conditioning via addition."""
    def __init__(self, in_c: int, out_c: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_c), in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
        )
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_c))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h  = self.conv1(x)
        h  = h + self.t_proj(t_emb).view(-1, h.shape[1], 1, 1)
        h  = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.res  = ResBlock(in_c,  out_c, t_dim)
        self.down = nn.Conv2d(out_c, out_c, 4, 2, 1)   # stride-2 conv

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.down(h), h          # 返回下采样结果 + skip feature


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, t_dim):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_c,   in_c,  4, 2, 1)
        self.res = ResBlock(in_c + skip_c, out_c, t_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        return self.res(torch.cat([x, skip], dim=1), t_emb)


# ──────────────── 轻量 U-Net ──────────────────────────────────

class LightUNet(nn.Module):
    """
    轻量 U-Net，用作 DDPM 的噪声预测网络 ε_θ。
    参数量约 4M，适合 400 张小数据集快速预训练。

    通道配置（img_size=128，base_ch=64）：
      enc0: 3  → 64,  128×128  (无下采样，仅卷积)
      enc1: 64 → 128,  64×64
      enc2: 128→ 256,  32×32   ← 对齐点 A（供 DFA-G 使用）
      enc3: 256→ 256,  16×16   ← 对齐点 B（供 DFA-G 使用）
      bot:  256→ 256,  16×16   (bottleneck)
      dec3: 256→ 128,  32×32
      dec2: 128→  64,  64×64
      dec1:  64→  64, 128×128
      out:   64→   3, 128×128  (预测噪声)
    """
    def __init__(self, base_ch: int = 64, t_dim: int = 256):
        super().__init__()
        ch = base_ch
        self.t_emb = TimeEmbedding(128, t_dim)

        # ── 编码器 ──────────────────────────────────────────
        self.enc0  = ResBlock(3,    ch,    t_dim)         # 128×128
        self.enc1  = DownBlock(ch,   ch*2,  t_dim)        # 64×64
        self.enc2  = DownBlock(ch*2, ch*4,  t_dim)        # 32×32  ← 对齐A
        self.enc3  = DownBlock(ch*4, ch*4,  t_dim)        # 16×16  ← 对齐B

        # ── 瓶颈 ────────────────────────────────────────────
        self.bot   = ResBlock(ch*4, ch*4, t_dim)

        # ── 解码器 ──────────────────────────────────────────
        self.dec3  = UpBlock(ch*4, ch*4, ch*4, t_dim)    # 32×32
        self.dec2  = UpBlock(ch*4, ch*4, ch*2, t_dim)    # 64×64
        self.dec1  = UpBlock(ch*2, ch*2, ch,   t_dim)    # 128×128

        self.out   = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 1),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                x:    torch.Tensor,
                t:    torch.Tensor,
                return_features: bool = False):
        """
        x: (B, 3, H, W) 加噪图像
        t: (B,)          整数时间步
        return_features: 是否返回对齐用的中间特征

        返回:
          pred_noise: (B, 3, H, W)  预测的噪声 ε
          features (可选): dict {'feat_32': (B,256,32,32),
                                  'feat_16': (B,256,16,16)}
        """
        t_emb = self.t_emb(t)                             # (B, t_dim)

        # 编码
        h0          = self.enc0(x, t_emb)                 # 128×128
        h1, skip1   = self.enc1(h0, t_emb)                # 64×64
        h2, skip2   = self.enc2(h1, t_emb)                # 32×32
        h3, skip3   = self.enc3(h2, t_emb)                # 16×16

        # 瓶颈
        h = self.bot(h3, t_emb)

        # 解码
        h  = self.dec3(h,  skip3, t_emb)                  # 32×32
        h  = self.dec2(h,  skip2, t_emb)                  # 64×64
        h  = self.dec1(h,  skip1, t_emb)                  # 128×128

        pred_noise = self.out(h)

        if return_features:
            return pred_noise, {
                'feat_32': skip2.detach(),   # 32×32, ch*4
                'feat_16': skip3.detach(),   # 16×16, ch*4
            }
        return pred_noise


# ──────────────── DDPM 调度器 ────────────────────────────────

class DDPMScheduler:
    """
    线性 β 调度，预计算所有扩散系数。
    提供前向加噪、分数函数提取接口。
    """
    def __init__(self,
                 T:          int   = 1000,
                 beta_start: float = 0.0001,
                 beta_end:   float = 0.02,
                 device:     str   = 'cpu'):
        self.T      = T
        self.device = device

        betas          = torch.linspace(beta_start, beta_end, T, device=device)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register('betas',          betas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('sqrt_alphas_cp', alphas_cumprod.sqrt())
        self.register('sqrt_one_minus', (1 - alphas_cumprod).sqrt())

    def register(self, name, tensor):
        setattr(self, name, tensor)

    def to(self, device):
        self.device = device
        for attr in ['betas', 'alphas_cumprod',
                     'sqrt_alphas_cp', 'sqrt_one_minus']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple:
        """
        前向扩散：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        返回 (x_t, ε)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a  = self.sqrt_alphas_cp[t].view(-1, 1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1a * noise, noise

    def score(self,
              eps_pred:  torch.Tensor,
              t:         torch.Tensor) -> torch.Tensor:
        """
        分数函数：s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
        """
        sqrt_1a = self.sqrt_one_minus[t].view(-1, 1, 1, 1)
        return -eps_pred / (sqrt_1a + 1e-8)


# ──────────────── 完整 DDPM 模型 ─────────────────────────────

class DDPM(nn.Module):
    """
    轻量 DDPM 教师。

    预训练后冻结（requires_grad=False），
    在 GAN 训练中提供：
      - 分数函数（MGSM 损失）
      - 编码器特征（DFA-G 对齐）
    """
    def __init__(self, cfg: dict):
        super().__init__()
        dc = cfg['ddpm']
        self.unet      = LightUNet(
            base_ch=dc['base_ch'],
            t_dim  =dc['time_emb_dim'] * 2,
        )
        self.scheduler = DDPMScheduler(
            T          = dc['T'],
            beta_start = dc['beta_start'],
            beta_end   = dc['beta_end'],
        )
        self.T = dc['T']

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def training_loss(self,
                      x0: torch.Tensor) -> torch.Tensor:
        """标准 DDPM 训练损失：||ε - ε_θ(x_t, t)||²"""
        B      = x0.shape[0]
        device = x0.device
        t      = torch.randint(0, self.T, (B,), device=device)
        x_t, noise = self.scheduler.q_sample(x0, t)
        pred        = self.unet(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def get_score(self,
                  x0:   torch.Tensor,
                  t_val: int) -> torch.Tensor:
        """
        计算分数函数 s_θ(x_t, t) 用于 MGSM 损失。
        x0: 干净图像 (B,3,H,W)  →  先加噪到 t，再用 DDPM 估计分数
        """
        B      = x0.shape[0]
        device = x0.device
        t      = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_t, _ = self.scheduler.q_sample(x0, t)
        eps    = self.unet(x_t, t)
        return self.scheduler.score(eps, t)

    @torch.no_grad()
    def get_features(self,
                     x0:   torch.Tensor,
                     t_val: int) -> dict:
        """
        提取 DDPM 编码器的中间特征，用于 DFA-G 对齐。
        """
        B      = x0.shape[0]
        device = x0.device
        t      = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_t, _ = self.scheduler.q_sample(x0, t)
        _, feats = self.unet(x_t, t, return_features=True)
        return feats

    def freeze(self):
        """冻结所有参数，进入教师模式。"""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.2f}M"
