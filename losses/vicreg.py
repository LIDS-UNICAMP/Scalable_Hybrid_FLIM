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
"""VICReg (Bardes et al., 2105.04906) sem expansor e sem rotulo, sobre z [B, D].

Diferenca para o artigo: cov e a media dos D(D-1) termos fora da diagonal,
e inv e a media tambem sobre D, como no codigo oficial.
"""
import torch
import torch.nn.functional as F


def invariance_term(z, z2):
    return F.mse_loss(z, z2)


def variance_term(z, gamma=1.0, eps=1e-4):
    return F.relu(gamma - torch.sqrt(z.var(dim=0) + eps)).mean()


def covariance_term(z):
    n, d = z.shape
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / (n - 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / (d * (d - 1))


def vicreg(z, z2=None, lam=0.0, mu=1.0, nu=1.0, gamma=1.0, eps=1e-4):
    """Com duas vistas, var e cov sao a media das vistas: o peso externo nao muda."""
    if lam and z2 is None:
        raise ValueError("lam > 0 exige a segunda vista z2")
    views = (z,) if z2 is None else (z, z2)
    inv = invariance_term(z, z2) if lam else z.new_zeros(())
    var = sum(variance_term(v, gamma, eps) for v in views) / len(views)
    cov = sum(covariance_term(v) for v in views) / len(views)
    return {"loss": lam * inv + mu * var + nu * cov, "inv": inv, "var": var, "cov": cov}


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 32
    for d in (32, 1280):
        iso = torch.randn(n, d)
        cases = {
            "isotropico": iso,
            "isotropico+2.73": iso + 2.73,
            "colapso_constante": torch.full((n, d), 2.73),
            "colapso_posto1": torch.randn(n, 1).expand(n, d).clone(),
        }
        out = {k: vicreg(v) for k, v in cases.items()}
        for k, r in out.items():
            print(f"D={d:5d} {k:18s} " + " ".join(f"{t}={r[t].item():.4f}" for t in ("loss", "var", "cov")))
        assert torch.isclose(out["isotropico"]["loss"], out["isotropico+2.73"]["loss"], atol=1e-4)
        assert out["colapso_constante"]["loss"] > out["isotropico"]["loss"]
        assert out["colapso_posto1"]["loss"] > out["isotropico"]["loss"]

        zc = iso - iso.mean(0)
        c = zc.T @ zc / (n - 1)
        paper = (c.pow(2).sum() - c.diagonal().pow(2).sum()) / d
        assert torch.isclose(covariance_term(iso) * (d - 1), paper, rtol=1e-4)

    z = torch.full((n, 64), 2.73, requires_grad=True)
    vicreg(z)["loss"].backward()
    assert torch.isfinite(z.grad).all()

    z = torch.randn(n, 64)
    z2 = z + 0.1 * torch.randn(n, 64)
    two = vicreg(z, z2, lam=1.0)
    print(f"duas vistas inv={two['inv'].item():.4f} (esperado ~0.01)")
    assert 0.005 < two["inv"] < 0.02
    print("OK")
