"""Second-order readout ``so2_4096_pca`` for the GA fitness (v4, opt-in ``svm_fitness.readout: so2_4096``).

The SAME computation as ``research/spifil_b0b4/readout_x/eval_backbone_head.py`` (head
``so2_4096_pca``, the readout that is REPORTED): on the map X of the LAST encoder block,
``z = Norm(pad_4096([mu(PX), sigma(PX), triu(sqrtm Cov(PX))]))`` with P = PCA of the channels fitted
ONLY on train pixels (``n_px`` pixels per train image, seeded per image), k = 88
(88+88+88*89/2 = 4092 + 4 zeros), Norm = signed-sqrt + L2. fp32 forward, no autocast, fp64 moments.
Followed by the repo SVM (linear, C=100, NO scaler) -- the v3 fitness used cov_head + StandardScaler,
whose kappa had rho 0.09 with the reported number. Nothing here sees validation/test labels.
The pure functions are copied (not imported) so the GA does not depend on the research tree;
``tests/modehb/test_so2_readout.py`` checks they are bitwise equal to the reference ones.
"""
from __future__ import annotations

import torch

SO2_K = {1280: 48, 4096: 88}
PCA_SEED = 0


def so2_dim(k: int) -> int:
    return 2 * k + k * (k + 1) // 2


def pad_to(z, D: int):
    if z.shape[1] > D:
        raise ValueError(f"vetor {z.shape[1]} > {D}")
    return torch.cat([z, z.new_zeros(z.shape[0], D - z.shape[1])], 1)


def norm_ssqrt_l2(z):
    z = z.sign() * z.abs().sqrt()
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _sqrtm_psd(C):
    C = (C + C.transpose(1, 2)) / 2
    w, U = torch.linalg.eigh(C)
    return (U * w.clamp(min=0).sqrt()[:, None, :]) @ U.transpose(1, 2)


def _flat(x):
    return x.detach().double().cpu().permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def so2_vector(fmap, mu, V, D: int):
    Y = (_flat(fmap) - mu) @ V
    m = Y.mean(1)
    Yc = Y - m[:, None, :]
    cov = Yc.transpose(1, 2) @ Yc / Y.shape[1]
    sd = cov.diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
    S = _sqrtm_psd(cov)
    k = V.shape[1]
    iu = torch.triu_indices(k, k)
    z = torch.cat([m, sd, S[:, iu[0], iu[1]]], 1)
    return norm_ssqrt_l2(pad_to(z, D)).float()


class PixelMoments:
    def __init__(self, n_px: int = 64, seed: int = PCA_SEED):
        self.n_px, self.seed = int(n_px), int(seed)
        self.s1 = self.s2 = None
        self.n = 0

    def add(self, fmap, img_ids):
        f = fmap.detach().permute(0, 2, 3, 1).reshape(fmap.shape[0], -1, fmap.shape[1])
        rows = []
        for b, gid in enumerate(img_ids):
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(gid))
            hw = f.shape[1]
            idx = torch.randperm(hw, generator=g)[: min(self.n_px, hw)]
            rows.append(f[b, idx.to(f.device)].double().cpu())
        P = torch.cat(rows)
        if self.s1 is None:
            self.s1 = torch.zeros(P.shape[1], dtype=torch.float64)
            self.s2 = torch.zeros(P.shape[1], P.shape[1], dtype=torch.float64)
        self.s1 += P.sum(0)
        self.s2 += P.T @ P
        self.n += P.shape[0]

    def pca(self, k: int):
        mu = self.s1 / self.n
        cov = (self.s2 - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        cov = (cov + cov.T) / 2
        _, V = torch.linalg.eigh(cov)
        V = V.flip(1)[:, : min(k, V.shape[1])]
        sgn = V.gather(0, V.abs().argmax(0, keepdim=True)).sign()
        return mu, (V * torch.where(sgn == 0, torch.ones_like(sgn), sgn)).contiguous()


@torch.no_grad()
def _last_block(encoder, x):
    h = x
    for blk in encoder:
        h = blk(h)
    return h.float()


class So2Readout:
    """fit(encoder, x_train) on TRAIN pixels only, then __call__/embed -> [N, D] fp32."""

    def __init__(self, D: int = 4096, n_px: int = 64, seed: int = PCA_SEED):
        self.D, self.k, self.n_px, self.seed = int(D), SO2_K[int(D)], int(n_px), int(seed)
        self.mu = self.V = None

    @torch.no_grad()
    def fit(self, encoder, x_train, device, bs: int = 16):
        encoder.eval()
        mom = PixelMoments(self.n_px, self.seed)
        for i0 in range(0, x_train.shape[0], bs):
            xb = x_train[i0:i0 + bs].to(device)
            mom.add(_last_block(encoder, xb), list(range(i0, i0 + xb.shape[0])))
        self.mu, self.V = mom.pca(self.k)
        return self

    @torch.no_grad()
    def embed(self, encoder, x, device, bs: int = 16):
        assert self.V is not None, "So2Readout.fit (train) first"
        encoder.eval()
        out = [so2_vector(_last_block(encoder, x[i0:i0 + bs].to(device)), self.mu, self.V, self.D)
               for i0 in range(0, x.shape[0], bs)]
        return torch.cat(out)
