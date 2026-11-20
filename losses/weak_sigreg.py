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
Weak-SIGReg (Akbar, arXiv 2603.05924): empurra a covariancia de um sketch
aleatorio de z para a identidade. Uma vista, sem projetor, sem rotulo.

    S ~ N(0, 1/C) em R^{K x C}, novo a cada chamada (so se C > K; senao K = C)
    Y = (z - media(z)) S^T
    A = Y^T Y / (N - 1 + 1e-6)
    L = ||A - I_K||_F / sqrt(K)

Diferencas em relacao ao artigo:
- USAR EM SSL E ADAPTACAO EXPERIMENTAL. O artigo cola o termo em treino
  supervisionado; aqui ele acompanha a reconstrucao do autoencoder, sem rotulo.
  Nenhum resultado do artigo garante ganho neste cenario.
- A divisao por sqrt(K) nao e do artigo: deixa o valor comparavel entre K.
- K = 16, nao 64: com lote 32 a centragem limita o posto a 31, e K > N - 1 poe
  um piso em ||A - I||_F de sqrt(K - N + 1). Mesmo com K = 16, uma Gaussiana
  perfeita da ~0,7 em N = 32, por ruido de amostra: a perda nao chega a zero.
"""

import torch


def weak_sigreg(z: torch.Tensor, k: int = 16, eps: float = 1e-6) -> torch.Tensor:
    """z: [N, C]. Devolve um escalar."""
    z = z.float()
    y = z - z.mean(0)
    if y.shape[1] > k:
        y = y @ torch.randn(k, y.shape[1], device=y.device).div(y.shape[1] ** 0.5).T
    k = y.shape[1]
    a = y.T @ y / (y.shape[0] - 1 + eps)
    return torch.linalg.matrix_norm(a - torch.eye(k, device=y.device)) / k**0.5


if __name__ == "__main__":
    torch.manual_seed(0)
    n, c, reps = 32, 1280, 200
    mean = lambda f: torch.stack([weak_sigreg(f()) for _ in range(reps)]).mean().item()
    iso = mean(lambda: torch.randn(n, c))
    const = mean(lambda: torch.ones(n, c) * 2.73)
    rank1 = mean(lambda: torch.randn(n, 1) * torch.randn(1, c))
    print(f"isotropico {iso:.3f} | colapsado constante {const:.3f} | colapsado posto 1 {rank1:.3f}")
    assert iso < const and iso < rank1
    print(f"C=8 <= K, sem sketch: {weak_sigreg(torch.randn(n, 8)).item():.3f}")
    z = torch.randn(n, c, requires_grad=True)
    weak_sigreg(z).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    print("ok")
