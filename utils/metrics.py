"""
评估指标: FID / IS / Precision / Recall / Diversity
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
import torchvision.models as tvm
from torchvision import transforms


class InceptionExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        inc = tvm.inception_v3(
            weights=tvm.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False, aux_logits=True)
        inc.eval()
        for p in inc.parameters():
            p.requires_grad_(False)
        self.inc    = inc.to(device)
        self.device = device
        self.pre    = transforms.Compose([
            transforms.Resize(299, antialias=True),
            transforms.CenterCrop(299),
        ])

    def _prep(self, imgs):
        return self.pre((imgs.clamp(-1,1)+1)*0.5).to(self.device)

    @torch.no_grad()
    def features(self, imgs):
        x = self._prep(imgs)
        x = self.inc.Conv2d_1a_3x3(x); x = self.inc.Conv2d_2a_3x3(x)
        x = self.inc.Conv2d_2b_3x3(x); x = F.max_pool2d(x,3,2)
        x = self.inc.Conv2d_3b_1x1(x); x = self.inc.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x,3,2)
        for blk in [self.inc.Mixed_5b, self.inc.Mixed_5c, self.inc.Mixed_5d,
                    self.inc.Mixed_6a, self.inc.Mixed_6b, self.inc.Mixed_6c,
                    self.inc.Mixed_6d, self.inc.Mixed_6e,
                    self.inc.Mixed_7a, self.inc.Mixed_7b, self.inc.Mixed_7c]:
            x = blk(x)
        return F.adaptive_avg_pool2d(x,(1,1)).squeeze(-1).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def logits(self, imgs):
        out = self.inc(self._prep(imgs))
        if isinstance(out, tuple): out = out[0]
        return F.softmax(out, dim=1).cpu().numpy()


def compute_fid(real_f, fake_f, eps=1e-6):
    mu_r, mu_f   = real_f.mean(0), fake_f.mean(0)
    cov_r, cov_f = np.cov(real_f, rowvar=False), np.cov(fake_f, rowvar=False)
    diff = mu_r - mu_f
    cm, _ = linalg.sqrtm(cov_r @ cov_f, disp=False)
    if not np.isfinite(cm).all():
        cm = linalg.sqrtm((cov_r+eps*np.eye(cov_r.shape[0])) @
                          (cov_f+eps*np.eye(cov_f.shape[0])))
    if np.iscomplexobj(cm): cm = cm.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2*cm))


def compute_is(probs, splits=5):
    n = probs.shape[0]; sz = n // splits; scores = []
    for i in range(splits):
        p = probs[i*sz:(i+1)*sz]
        py = p.mean(0, keepdims=True)
        scores.append(np.exp((p*(np.log(p+1e-10)-np.log(py+1e-10))).sum(1).mean()))
    return float(np.mean(scores)), float(np.std(scores))


def compute_prec_recall(real_f, fake_f, k=3):
    def pdist(A, B):
        A2=(A**2).sum(1,keepdims=True); B2=(B**2).sum(1,keepdims=True)
        return np.sqrt(np.maximum(A2+B2.T-2*A@B.T, 0))
    rr=pdist(real_f,real_f); ff=pdist(fake_f,fake_f); rf=pdist(real_f,fake_f)
    k = max(1, min(k, rr.shape[0]-1, ff.shape[0]-1))
    r_rad=np.sort(rr,1)[:,k]; f_rad=np.sort(ff,1)[:,k]
    precision = (rf.min(0) <= f_rad).mean()
    recall    = (rf.min(1) <= r_rad).mean()
    return float(precision), float(recall)


def compute_diversity(fake_f, n_pairs=500):
    n = len(fake_f)
    if n < 2: return 0.0
    idx = np.random.choice(n, size=(min(n_pairs, n*(n-1)//2), 2), replace=True)
    return float(np.linalg.norm(fake_f[idx[:,0]]-fake_f[idx[:,1]], axis=1).mean())


def compute_psnr(real_imgs_t: torch.Tensor, fake_imgs_t: torch.Tensor):
    n = min(real_imgs_t.size(0), fake_imgs_t.size(0))
    if n == 0:
        return float('nan')
    real = ((real_imgs_t[:n].detach().float().clamp(-1, 1) + 1) * 0.5)
    fake = ((fake_imgs_t[:n].detach().float().clamp(-1, 1) + 1) * 0.5)
    mse = F.mse_loss(fake, real).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(real_imgs_t: torch.Tensor, fake_imgs_t: torch.Tensor, C1=0.01**2, C2=0.03**2):
    n = min(real_imgs_t.size(0), fake_imgs_t.size(0))
    if n == 0:
        return float('nan')
    x = ((real_imgs_t[:n].detach().float().clamp(-1, 1) + 1) * 0.5)
    y = ((fake_imgs_t[:n].detach().float().clamp(-1, 1) + 1) * 0.5)

    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = y.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + 1e-12)
    return float(ssim_map.mean().item())


class MetricEvaluator:
    def __init__(self, device='cpu', active=True):
        self.active = active
        if active:
            self.ext = InceptionExtractor(device)
        self.reset()

    def reset(self):
        self._rf=[]; self._ff=[]; self._fp=[]
        self._real_imgs=[]; self._fake_imgs=[]

    @torch.no_grad()
    def update_real(self, imgs):
        if self.active:
            self._rf.append(self.ext.features(imgs))
        self._real_imgs.append(imgs.detach().cpu())

    @torch.no_grad()
    def update_fake(self, imgs):
        if self.active:
            self._ff.append(self.ext.features(imgs))
            self._fp.append(self.ext.logits(imgs))
        self._fake_imgs.append(imgs.detach().cpu())

    def compute(self):
        if not self._fake_imgs:
            return {}

        real_t = torch.cat(self._real_imgs, dim=0) if self._real_imgs else torch.empty(0)
        fake_t = torch.cat(self._fake_imgs, dim=0)
        res = {
            'PSNR': compute_psnr(real_t, fake_t),
            'SSIM': compute_ssim(real_t, fake_t),
        }

        if not self.active or not self._ff:
            return res

        rf = np.concatenate(self._rf); ff = np.concatenate(self._ff)
        fp = np.concatenate(self._fp)
        res['FID']     = compute_fid(rf, ff)
        mu, std        = compute_is(fp)
        res['IS_mean'] = mu; res['IS_std'] = std
        if len(rf)>=4 and len(ff)>=4:
            p, r = compute_prec_recall(rf, ff)
            res['Precision'] = p; res['Recall'] = r
        res['Diversity'] = compute_diversity(ff)
        return res
