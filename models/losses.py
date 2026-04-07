"""
DSG-GAN 损失函数
=================
C1  MGSM — 多粒度分数匹配损失（含动态权重 MLP）
     Hinge — 对抗损失
     R1    — 梯度惩罚
     FM    — 特征匹配损失
C2  Align — DFA-G 特征对齐损失（在损失模块中计算）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── Hinge 对抗损失 ─────────────────────────────

class HingeLoss(nn.Module):
    def d_loss(self, real_logits, fake_logits):
        return (F.relu(1.0 - real_logits).mean() +
                F.relu(1.0 + fake_logits).mean()) * 0.5

    def g_loss(self, fake_logits):
        return -fake_logits.mean()


# ─────────────── R1 梯度惩罚 ────────────────────────────────

def r1_penalty(real_logits: torch.Tensor,
               real_imgs:   torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs      = real_logits.sum(),
        inputs       = real_imgs,
        create_graph = True,
        retain_graph = True,
        only_inputs  = True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


# ─────────────── 特征匹配损失 ────────────────────────────────

class FeatureMatchingLoss(nn.Module):
    def forward(self,
                real_feats: list,
                fake_feats: list) -> torch.Tensor:
        loss = sum(F.l1_loss(ff, rf.detach())
                   for rf, ff in zip(real_feats, fake_feats))
        return loss / max(len(real_feats), 1)


# ─────────────── C2 特征对齐损失 ─────────────────────────────

class AlignmentLoss(nn.Module):
    """
    DFA-G 特征对齐损失。

    L_align = Σ_l ||f_l^G(z) - sg(f_l^DDPM(x_real, t*))||²

    sg = stop gradient（DDPM 特征不参与梯度）
    t* 随训练进度从大到小（课程学习）
    """
    def forward(self,
                gen_feats:  dict,
                ddpm_feats: dict) -> torch.Tensor:
        loss = 0.0
        n    = 0
        for key in gen_feats:
            if key in ddpm_feats and gen_feats[key] is not None:
                gf = gen_feats[key]
                df = ddpm_feats[key].detach()
                # 尺寸可能因 DDPM 与 G 的 padding 差1，插值对齐
                if gf.shape != df.shape:
                    df = F.interpolate(df, size=gf.shape[-2:],
                                       mode='bilinear', align_corners=False)
                loss = loss + F.mse_loss(gf, df)
                n   += 1
        return loss / max(n, 1)


# ─────────────── C1 MGSM 多粒度分数匹配 ─────────────────────

class DynamicWeightMLP(nn.Module):
    """
    小型 MLP，根据训练进度 p ∈ [0,1] 输出 3 个噪声水平的权重。

    设计逻辑：
      早期（p→0）: w_high 大 → 先在高噪声水平匹配全局形态
      后期（p→1）: w_low  大 → 再在低噪声水平匹配精细纹理
    """
    def __init__(self, n_levels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,  16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_levels),
        )
        # 初始化偏置让早期倾向高噪声（最后一个 level）
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor(
                [-1.0, 0.0, 1.0][:n_levels])

    def forward(self, progress: float) -> torch.Tensor:
        p = torch.tensor([[progress]], dtype=torch.float32,
                          device=next(self.parameters()).device)
        return F.softmax(self.net(p).squeeze(0), dim=0)   # (n_levels,)


class MGSMLoss(nn.Module):
    """
    Multi-Granularity Score Matching Loss  [C1]

    L_MGSM = Σ_{k} λ_k(p) · ||s_θ(G(z), t_k) - s_θ(x_real, t_k)||²

    λ_k(p) 由 DynamicWeightMLP 根据训练进度动态调整。
    分数函数通过冻结的 DDPM 教师计算。
    所有 DDPM 调用均为 no_grad。
    """
    def __init__(self, t_levels: list[int]):
        super().__init__()
        self.t_levels    = t_levels          # [t_low, t_mid, t_high]
        self.weight_mlp  = DynamicWeightMLP(len(t_levels))

    def forward(self,
                gen_imgs:   torch.Tensor,
                real_imgs:  torch.Tensor,
                ddpm:       nn.Module,
                progress:   float) -> tuple:
        """
        gen_imgs:  生成图 (B,3,H,W)  — 需要梯度
        real_imgs: 真实图 (B,3,H,W)
        ddpm:      冻结的 DDPM 教师
        progress:  epoch / total_epochs ∈ [0,1]
        """
        weights = self.weight_mlp(progress)   # (n_levels,) no grad through MLP
        loss    = torch.tensor(0.0, device=gen_imgs.device)
        log     = {}

        # MGSM 全程在 float32 下运算，避免 score 分母溢出
        gen_imgs_f32  = gen_imgs.float()
        real_imgs_f32 = real_imgs.float()

        for i, t_val in enumerate(self.t_levels):
            # 真实图分数（no_grad，DDPM 已冻结）
            with torch.no_grad():
                s_real = ddpm.get_score(real_imgs_f32, t_val)   # float32

            B      = gen_imgs_f32.shape[0]
            device = gen_imgs_f32.device
            t      = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise  = torch.randn_like(gen_imgs_f32)
            sqrt_a  = ddpm.scheduler.sqrt_alphas_cp[t].view(-1, 1, 1, 1)
            sqrt_1a = ddpm.scheduler.sqrt_one_minus[t].view(-1, 1, 1, 1)
            # 对生成图加噪（梯度仍能回传到 gen_imgs_f32 → gen_imgs）
            x_t_gen = sqrt_a * gen_imgs_f32 + sqrt_1a * noise   # float32

            # DDPM 前向，全程 float32，避免 score 分母极小时 float16 溢出 NaN
            eps_pred = ddpm.unet(x_t_gen, t)
            s_gen    = ddpm.scheduler.score(eps_pred, t)         # float32

            level_loss = F.mse_loss(s_gen, s_real.detach())
            loss       = loss + weights[i] * level_loss
            log[f'mgsm_t{t_val}'] = level_loss.item()

        log['mgsm_total'] = loss.item()
        log['w_low']  = weights[0].item()
        log['w_mid']  = weights[1].item()
        log['w_high'] = weights[2].item()
        return loss, log


# ─────────────── 汇总损失类 ─────────────────────────────────

class DSGLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg['loss']
        self.lambda_mgsm  = lc['lambda_mgsm']
        self.lambda_align = lc['lambda_align']
        self.lambda_r1    = lc['lambda_r1']
        self.lambda_fm    = lc['lambda_fm']
        self.lambda_color = lc.get('lambda_color', 1.0)
        self.lambda_shape = lc.get('lambda_shape', 1.0)
        self.r1_every     = cfg['training']['r1_every']

        mc = cfg['mgsm']
        self.mgsm_loss = MGSMLoss(
            t_levels=[mc['t_low'], mc['t_mid'], mc['t_high']])
        self.align_loss = AlignmentLoss()
        self.hinge      = HingeLoss()
        self.fm_loss    = FeatureMatchingLoss()

    @staticmethod
    def _to_01(imgs: torch.Tensor) -> torch.Tensor:
        return ((imgs.float().clamp(-1, 1) + 1.0) * 0.5)

    def _soft_cell_mask(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        根据蓝色主导性构建可微软掩膜（适配 CTC 蓝色椭圆先验）。
        """
        x = self._to_01(imgs)
        b = x[:, 2:3]
        rg = 0.5 * (x[:, 0:1] + x[:, 1:2])
        score = b - rg
        thr = score.mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid(18.0 * (score - thr))

    def color_prior_loss(self, gen_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
        """颜色先验：匹配 RGB 一阶统计 + 蓝色主导度。"""
        g = self._to_01(gen_imgs)
        r = self._to_01(real_imgs).detach()

        g_mean = g.mean(dim=(0, 2, 3))
        r_mean = r.mean(dim=(0, 2, 3))
        mean_loss = F.l1_loss(g_mean, r_mean)

        g_blue_dom = (g[:, 2] - 0.5 * (g[:, 0] + g[:, 1])).mean()
        r_blue_dom = (r[:, 2] - 0.5 * (r[:, 0] + r[:, 1])).mean()
        blue_loss = F.l1_loss(g_blue_dom, r_blue_dom)

        return mean_loss + blue_loss

    def _ellipse_stats(self, mask: torch.Tensor):
        """从软掩膜计算面积占比与椭圆偏心率（可微）。"""
        B, _, H, W = mask.shape
        device = mask.device
        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
        xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)

        m00 = mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        mx = (mask * xx).sum(dim=(2, 3), keepdim=True) / m00
        my = (mask * yy).sum(dim=(2, 3), keepdim=True) / m00

        dx = xx - mx
        dy = yy - my
        mu20 = (mask * dx * dx).sum(dim=(2, 3), keepdim=True) / m00
        mu02 = (mask * dy * dy).sum(dim=(2, 3), keepdim=True) / m00
        mu11 = (mask * dx * dy).sum(dim=(2, 3), keepdim=True) / m00

        tr = mu20 + mu02
        det = (mu20 * mu02 - mu11 * mu11).clamp_min(1e-8)
        disc = (tr * tr - 4.0 * det).clamp_min(0.0).sqrt()
        l1 = ((tr + disc) * 0.5).clamp_min(1e-8)
        l2 = ((tr - disc) * 0.5).clamp_min(1e-8)
        ecc = torch.sqrt((1.0 - l2 / l1).clamp(0.0, 1.0))

        area = mask.mean(dim=(1, 2, 3), keepdim=True)
        return area, ecc

    def shape_prior_loss(self, gen_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
        """形态先验：匹配细胞面积比例与椭圆偏心率统计。"""
        g_mask = self._soft_cell_mask(gen_imgs)
        r_mask = self._soft_cell_mask(real_imgs).detach()

        g_area, g_ecc = self._ellipse_stats(g_mask)
        r_area, r_ecc = self._ellipse_stats(r_mask)

        area_loss = F.l1_loss(g_area.mean(), r_area.mean())
        ecc_mean_loss = F.l1_loss(g_ecc.mean(), r_ecc.mean())
        ecc_std_loss = F.l1_loss(g_ecc.std(unbiased=False), r_ecc.std(unbiased=False))
        return area_loss + ecc_mean_loss + 0.5 * ecc_std_loss

    # ── 判别器损失 ────────────────────────────────────────
    def d_step(self,
               real_logits, fake_logits,
               real_imgs,   do_r1: bool) -> tuple:
        adv   = self.hinge.d_loss(real_logits, fake_logits)
        total = adv
        log   = {'D_adv': adv.item()}

        if do_r1:
            r1   = r1_penalty(real_logits, real_imgs)
            total = total + self.lambda_r1 * 0.5 * self.r1_every * r1
            log['D_r1'] = r1.item()

        log['D_total'] = total.item()
        return total, log

    # ── 生成器损失 ────────────────────────────────────────
    def g_step(self,
               fake_logits,
               real_feats,  fake_feats,
               gen_imgs,    real_imgs,
               gen_feats_aligned, ddpm_feats,
               ddpm,        progress: float) -> tuple:

        adv   = self.hinge.g_loss(fake_logits)
        fm    = self.fm_loss(real_feats, fake_feats)
        align = self.align_loss(gen_feats_aligned, ddpm_feats)
        mgsm, mgsm_log = self.mgsm_loss(
            gen_imgs, real_imgs, ddpm, progress)

        color = self.color_prior_loss(gen_imgs, real_imgs)
        shape = self.shape_prior_loss(gen_imgs, real_imgs)

        total = (adv +
                 self.lambda_fm    * fm    +
                 self.lambda_align * align +
                 self.lambda_mgsm  * mgsm +
                 self.lambda_color * color +
                 self.lambda_shape * shape)

        log = {
            'G_adv':   adv.item(),
            'G_fm':    fm.item(),
            'G_align': align.item(),
            'G_color': color.item(),
            'G_shape': shape.item(),
            **mgsm_log,
            'G_total': total.item(),
        }
        return total, log
