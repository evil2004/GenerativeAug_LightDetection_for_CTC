"""
DFA-G: Diffusion Feature-Aligned Generator  [C2 核心创新]
=========================================================
创新核心：
  生成器的中间特征在对应分辨率上，被强制向
  冻结 DDPM 编码器的特征「锚定」。

  DDPM 编码器特征（真实图加噪后提取）编码了
  「真实数据流形的切线空间」，对齐操作把生成器
  的每一层都约束在这个流形附近，解决小数据集
  生成器特征漂移和模式崩溃问题。

架构：
  z → MappingNet → w
  Const(4×4) → SynthBlock×5 → RGB(128×128)

  对齐点：
    SynthBlock @ 16×16  ←→  DDPM feat_16 (256ch)
    SynthBlock @ 32×32  ←→  DDPM feat_32 (256ch)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────── Mapping Network ────────────────────────

class MappingNetwork(nn.Module):
    """z → w：将高斯噪声映射到解耦的风格空间。"""
    def __init__(self, z_dim: int = 128, w_dim: int = 256, n_layers: int = 4):
        super().__init__()
        layers, in_d = [], z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, w_dim), nn.LeakyReLU(0.2, inplace=True)]
            in_d = w_dim
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * (z.shape[1] ** 0.5)
        return self.net(z)


# ─────────────────── AdaIN ──────────────────────────────────

class AdaIN(nn.Module):
    """标准自适应实例归一化，w 控制 γ,β。"""
    def __init__(self, channels: int, w_dim: int):
        super().__init__()
        self.gamma = nn.Linear(w_dim, channels)
        self.beta  = nn.Linear(w_dim, channels)
        nn.init.ones_ (self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        γ    = self.gamma(w).view(-1, x.shape[1], 1, 1) + 1.0
        β    = self.beta (w).view(-1, x.shape[1], 1, 1)
        mean = x.mean([2,3], keepdim=True)
        std  = x.std ([2,3], keepdim=True) + 1e-8
        return γ * (x - mean) / std + β


# ─────────────────── 合成块 ─────────────────────────────────

class SynthBlock(nn.Module):
    """
    上采样合成块：AdaIN 风格调制 + 双卷积 + 残差。
    """
    def __init__(self, in_ch: int, out_ch: int, w_dim: int):
        super().__init__()
        self.norm1 = AdaIN(in_ch,  w_dim)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False)
        self.norm2 = AdaIN(out_ch, w_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
                     if in_ch != out_ch else nn.Identity()
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, a=0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h    = F.interpolate(x, scale_factor=2,
                             mode='bilinear', align_corners=False)
        skip = self.skip(h)
        h    = F.leaky_relu(self.norm1(h, w), 0.2)
        h    = self.conv1(h)
        h    = F.leaky_relu(self.norm2(h, w), 0.2)
        h    = self.conv2(h)
        return (h + skip) * (2 ** -0.5)


# ─────────────────── DFA-G 生成器 ───────────────────────────

class Generator(nn.Module):
    """
    DFA-G Generator.

    z(128) → MappingNet → w(256)
    Const(4×4×512)
      → SynthBlock 4→8   (ch 512→512)
      → SynthBlock 8→16  (ch 512→256)  ← 对齐点 B  feat_16
      → SynthBlock 16→32 (ch 256→128)  ← 对齐点 A  feat_32
      → SynthBlock 32→64 (ch 128→64)
      → SynthBlock 64→128(ch  64→64)
      → to_rgb → Tanh

    forward() 返回生成图；
    forward_with_features() 额外返回对齐点特征（用于 L_align）。
    """
    def __init__(self,
                 z_dim: int = 128,
                 w_dim: int = 256,
                 ngf:   int = 64,
                 map_layers: int = 4,
                 img_size:   int = 128):
        super().__init__()
        self.z_dim = z_dim

        # 通道序列: 4→8→16→32→64→128
        n_up = int(math.log2(img_size)) - 2       # 128 → 5
        ch_list = []
        for i in range(n_up + 1):
            ch_list.append(min(ngf * (2 ** max(0, n_up - 1 - i)), 512))
        # e.g. [512, 512, 256, 128, 64, 64] for ngf=64, img_size=128

        self.mapping = MappingNetwork(z_dim, w_dim, map_layers)
        self.const   = nn.Parameter(torch.randn(1, ch_list[0], 4, 4))

        self.blocks  = nn.ModuleList([
            SynthBlock(ch_list[i], ch_list[i+1], w_dim)
            for i in range(n_up)
        ])
        # 合成层分辨率: 8, 16, 32, 64, 128  →  对应 blocks[0..4]
        # blocks[1] 输出 16×16, ch=256  → 对齐点 B
        # blocks[2] 输出 32×32, ch=128  → 对齐点 A

        # 对齐投影头（把 G 的 ch 对齐到 DDPM 的 ch*4=256）
        ddpm_ch4 = 256   # DDPM base_ch=64, ch*4=256
        self.proj_16 = nn.Conv2d(ch_list[2], ddpm_ch4, 1, bias=False)
        self.proj_32 = nn.Conv2d(ch_list[3], ddpm_ch4, 1, bias=False)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch_list[-1], 3, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.to_rgb[0].weight)
        nn.init.zeros_(self.to_rgb[0].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w  = self.mapping(z)
        x  = self.const.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return self.to_rgb(x)

    def forward_with_features(self, z: torch.Tensor) -> tuple:
        """
        额外返回 16×16 和 32×32 处的投影特征，
        供 DFA-G 对齐损失 L_align 使用。
        """
        w  = self.mapping(z)
        x  = self.const.expand(z.shape[0], -1, -1, -1)
        feat_16, feat_32 = None, None

        for i, block in enumerate(self.blocks):
            x = block(x, w)
            if i == 1:                         # 输出分辨率 16×16
                feat_16 = self.proj_16(x)
            elif i == 2:                       # 输出分辨率 32×32
                feat_32 = self.proj_32(x)

        img = self.to_rgb(x)
        return img, {'feat_16': feat_16, 'feat_32': feat_32}

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=device)
        return self.forward(z)
