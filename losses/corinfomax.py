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
"""CorInfoMax (Ozsoy et al., arXiv 2209.07999), sem rotulo, no vetor depois do GeM.

eps=1e-6 segue o codigo oficial (parsing_file.py:27); o texto do artigo diz 1e-8.
R sai so do lote: sem o R recursivo do oficial, porque o sketch muda a cada chamada.
"""
import torch
import torch.nn.functional as F


def _logdet(y, eps):
    y = y - y.mean(0)
    R = y.T @ y / y.size(0)
    return torch.logdet(R + eps * torch.eye(y.size(1), dtype=y.dtype, device=y.device))


def corinfomax(z, z2=None, k=16, eps=1e-6, alpha=500.0):
    """z: [N, C], com C > k e N > k. Com z2 (segunda vista) soma o log-det dela e alpha * MSE."""
    S = torch.randn(k, z.size(1), dtype=torch.float64, device=z.device) / z.size(1) ** 0.5

    def sketch(v):
        v = v - v.mean(0)
        return (v / v.pow(2).mean().sqrt().detach().clamp_min(1e-12)).double() @ S.T

    y = sketch(z)
    if z2 is None:
        return -_logdet(y, eps) / k
    y2 = sketch(z2)
    return -(_logdet(y, eps) + _logdet(y2, eps)) / k + alpha * F.mse_loss(y, y2)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C = 32, 1280
    iso = torch.randn(N, C)
    casos = {
        "constante": torch.ones(N, C),
        "posto 1": torch.randn(N, 1) @ torch.randn(1, C),
        "posto 3": torch.randn(N, 3) @ torch.randn(3, C),
        "isotropico": iso,
    }
    perdas = {n: corinfomax(v).item() for n, v in casos.items()}
    print({n: round(p, 3) for n, p in perdas.items()})
    assert all(perdas[n] > perdas["isotropico"] for n in casos if n != "isotropico")
    print("duas vistas:", round(corinfomax(iso, iso + 0.1 * torch.randn(N, C)).item(), 3))
    z = iso.clone().requires_grad_()
    corinfomax(z).backward()
    print("grad finito:", z.grad.isfinite().all().item(), "| d(perda)/d(escala):", round((z.grad * z).sum().item(), 3))
