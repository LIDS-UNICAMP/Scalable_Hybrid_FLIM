# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""
Regularizador do UR-JEPA (arXiv 2606.01443): CGLT + ancora AD, sobre z [B, C], sem rotulo.

Forma do codigo oficial (SPATIOLYX/UR-JEPA, ur_jepa/cglt.py): escada log-uniforme com K+1
raios, de r_max = mediana ||z_i - media|| ate r_min = r_max * N^(-1/n).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def ur_jepa_reg(z: Tensor, n: int = 7, K: int = 5, lambda_ad: float = 0.1) -> Tensor:
    z = z.float()
    N = z.size(0)
    zc = z - z.mean(0)
    d2 = (zc[:, None] - zc[None]).pow(2).sum(-1)
    with torch.no_grad():
        r_max = zc.norm(dim=1).quantile(0.5).clamp_min(1e-6)
        log_range = math.log(N) / n
        r = r_max * torch.exp(-log_range * torch.arange(K + 1, device=z.device) / K)
    log_theta = torch.stack(
        [torch.logsumexp(-d2 / (2 * q * q), 1) - n * q.log() - math.log(N) for q in r], 1
    )
    delta = log_theta[:, 1:] - log_theta[:, :-1]
    cglt = (log_range / K) * delta.pow(2).mean(0).sum()
    return cglt + lambda_ad * log_theta.var(0, unbiased=False).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 32, 1280

    def mean_reg(make):
        return sum(ur_jepa_reg(make()).item() for _ in range(8)) / 8

    def eff_dim(z):
        e = torch.linalg.svdvals(z - z.mean(0)).pow(2)
        return (e.sum() ** 2 / e.pow(2).sum()).item()

    def low_rank(k, noise=0.0):
        return torch.randn(N, k) @ torch.randn(k, D) + noise * torch.randn(N, D)

    iso = mean_reg(lambda: torch.randn(N, D))
    rank2 = mean_reg(lambda: low_rank(2))
    rank1 = mean_reg(lambda: low_rank(1))
    point = mean_reg(lambda: torch.full((N, D), 2.7))
    tiny = mean_reg(lambda: 1e-3 * torch.randn(N, D))
    print(f"iso {iso:.4f} < rank2 {rank2:.4f} < rank1 {rank1:.4f} < ponto {point:.4f}")
    print(f"iso encolhido 1000x {tiny:.4f}: a escada acompanha a nuvem, encolher nao custa")
    assert iso < rank2 < rank1 < point and abs(tiny - iso) < 0.01

    z = low_rank(2, noise=0.05).requires_grad_()
    before = eff_dim(z.detach())
    opt = torch.optim.Adam([z], lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        ur_jepa_reg(z).backward()
        opt.step()
    print(f"descida em z: dim efetiva {before:.1f} -> {eff_dim(z.detach()):.1f}")
    assert eff_dim(z.detach()) > before

    z = torch.full((N, D), 2.7, requires_grad=True)
    ur_jepa_reg(z).backward()
    print(f"gradiente no ponto exato: {z.grad.norm().item():.1e}")
