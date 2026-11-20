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
Barlow Twins (2103.03230) sem expansor e sem rotulo, normalizado por medias:
    R = mean_i (1 - C_ii)^2 + mean_{i != j} C_ij^2
O peso fica todo no lambda externo. Canal morto (std < 1e-5 no lote, em qualquer
vista) sai do gradiente, mas conta como C_ii = 0: matar canal nao baixa a perda.
"""

import torch


def _standardize(z, live):
    zc = z - z.mean(0)
    return torch.where(live, zc / zc.std(0, unbiased=False).clamp_min(1e-5), torch.zeros_like(zc))


def barlow_twins(za, zb):
    """za, zb: [B, D], as duas vistas das mesmas imagens, na mesma ordem."""
    live = (za.detach().std(0, unbiased=False) >= 1e-5) & (zb.detach().std(0, unbiased=False) >= 1e-5)
    c = _standardize(za, live).T @ _standardize(zb, live) / za.size(0)
    d = c.diagonal()
    n = c.size(0)
    return (1 - d).pow(2).mean() + (c.pow(2).sum() - d.pow(2).sum()) / (n * (n - 1))


def off_diagonal(x):
    """Copiado do main.py oficial: elementos fora da diagonal de uma matriz quadrada, achatados."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_meta(z1, z2, bn, lambd=0.0051):
    """BarlowTwins.forward oficial, 1 GPU: bn = BatchNorm1d(D, affine=False). Devolve (loss, on, off)."""
    c = bn(z1).T @ bn(z2) / z1.size(0)
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag, on_diag, off_diag


def barlow_twins_flip(embed, x):
    """embed: imagem [B, C, H, W] -> [B, D]. Segunda vista = flip horizontal."""
    return barlow_twins(embed(x), embed(x.flip(-1)))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 32, 1280
    iso = torch.randn(B, D)
    noise = lambda: 0.01 * torch.randn(B, D)
    rank1 = torch.randn(B, 1) * torch.randn(1, D)
    const = torch.ones(B, D)
    r_iso = barlow_twins(iso, iso + noise()).item()
    r_rank1 = barlow_twins(rank1, rank1 + 0.01 * noise()).item()
    r_const = barlow_twins(const, const).item()
    print(f"isotropico {r_iso:.4f} | posto 1 {r_rank1:.4f} | constante {r_const:.4f}")
    assert r_iso < r_rank1 and r_iso < r_const

    r_indep = barlow_twins(iso, torch.randn(B, D)).item()
    print(f"vistas independentes {r_indep:.4f}")
    assert r_iso < r_indep

    dead = iso.clone()
    dead[:, :111] = 2.73
    dead.requires_grad_()
    r_dead = barlow_twins(dead, dead + noise())
    r_dead.backward()
    print(f"111 canais mortos {r_dead.item():.4f} | grad morto {dead.grad[:, :111].abs().max().item():.1e}")
    assert r_dead.item() > r_iso and torch.isfinite(dead.grad).all() and dead.grad[:, :111].abs().max() == 0

    conv = torch.nn.Conv2d(3, 16, 3)
    r_flip = barlow_twins_flip(lambda x: conv(x).relu().mean((2, 3)), torch.randn(8, 3, 16, 16))
    r_flip.backward()
    print(f"flip {r_flip.item():.4f} | grad conv finito {torch.isfinite(conv.weight.grad).all().item()}")
    assert conv.weight.grad.abs().sum() > 0
