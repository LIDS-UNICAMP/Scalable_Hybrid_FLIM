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

"""NT-Xent (SimCLR, Chen et al. 2020, eq. 1) com margem aditiva opcional no positivo:
CosFace sem rotulo, como no NT-Xent-AM (Lepage & Dehak, arXiv 2404.14913). s = 1/tau.
Com V > 2 vistas, cada par (i, j) de vistas da mesma imagem e um positivo; as outras
vistas da mesma imagem tambem ficam no denominador. margin=0 e V=2 da o SimCLR exato.
"""

import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, z, return_stats=False):
        """z: [V, B, D], V >= 2 vistas de B imagens."""
        V, B, _ = z.shape
        z = F.normalize(z.reshape(V * B, -1), dim=1)
        cos = z @ z.T
        ids = torch.arange(V * B, device=z.device) % B
        self_mask = torch.eye(V * B, dtype=torch.bool, device=z.device)
        pos = (ids[:, None] == ids[None, :]) & ~self_mask
        logits = (cos - self.margin * pos).masked_fill(self_mask, float("-inf")) / self.temperature
        loss = -logits.log_softmax(1)[pos].mean()
        if not return_stats:
            return loss
        neg = ~pos & ~self_mask
        return loss, {"pos_cos": cos[pos].mean().detach(), "neg_cos": cos[neg].mean().detach()}


def _reference_simclr(za, zb, tau):
    z = F.normalize(torch.cat([za, zb]), dim=1)
    n, B = z.size(0), za.size(0)
    total = 0.0
    for i in range(n):
        j = (i + B) % n
        den = sum(torch.exp(z[i] @ z[k] / tau) for k in range(n) if k != i)
        total = total - torch.log(torch.exp(z[i] @ z[j] / tau) / den)
    return total / n


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 16, 64
    za, zb = torch.randn(B, D), torch.randn(B, D)
    f = NTXentLoss(temperature=0.1)
    ours, ref = f(torch.stack([za, zb])).item(), _reference_simclr(za, zb, 0.1).item()
    same, rand = f(torch.stack([za, za])).item(), ours
    z3 = torch.randn(3, B, D, requires_grad=True)
    l0, stats = f(z3, return_stats=True)
    l1 = NTXentLoss(temperature=0.1, margin=0.1)(z3)
    l1.backward()
    print(f"margin0 {ours:.6f} ref {ref:.6f} | iguais {same:.4f} aleatorias {rand:.4f} | "
          f"V=3 m0 {l0.item():.4f} m0.1 {l1.item():.4f} | pos {stats['pos_cos']:.3f} neg {stats['neg_cos']:.3f}")
    assert abs(ours - ref) < 1e-5
    assert same < rand
    assert l1.item() > l0.item()
    assert torch.isfinite(z3.grad).all()
    print("ok")
