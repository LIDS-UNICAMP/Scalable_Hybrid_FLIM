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
"""checks_S1.py — checks do §14 (itens 6–7) para flim/slow_features.py, CPU, tensores minusculos."""
import torch

from flim.slow_features import fit_fu, fit_sfa

torch.set_num_threads(2)
g = torch.Generator().manual_seed(0)
M, C, d = 4000, 16, 24
A = torch.randn(M, d, generator=g, dtype=torch.float64)
Bv = A + 0.3 * torch.randn(M, d, generator=g, dtype=torch.float64)
W0 = torch.randn(C, d, generator=g, dtype=torch.float64)
b0 = torch.randn(C, generator=g, dtype=torch.float64)
ua, ub = A @ W0.T + b0, Bv @ W0.T + b0
spec = {"lentos": 8, "residuo": 8}

# 6. backward proibido durante todo o ajuste
def _no(*a, **k):
    raise RuntimeError("backward chamado")
torch.Tensor.backward = _no
torch.autograd.backward = _no

# 1. residuos do autoproblema e restricao Vᵀ C0 V = I
W, b, info = fit_sfa(ua, ub, W0, b0, spec)
print("1 eig_residual", info["eig_residual"], "constraint_err", info["constraint_err"])
assert info["eig_residual"] < 1e-10 and info["constraint_err"] < 1e-10
print("  rank", info["rank"], "width", info["width"], "rms", info["rms"],
      "res_expl", round(info["residual_explained"], 4), "out_cov diag", torch.diagonal(info["out_cov"]).round(decimals=6).unique().tolist())

# streaming = lote unico
Ws, bs, _ = fit_sfa(ua.split(333), ub.split(333), W0, b0, spec)
print("1b streaming vs lote", float((Ws - W).abs().max()))
assert torch.allclose(Ws, W, atol=1e-9) and torch.allclose(bs, b, atol=1e-9)

# 2. direcao lenta conhecida: z0 igual nas duas vistas, demais independentes
z_a, z_b = torch.randn(M, C, generator=g, dtype=torch.float64), torch.randn(M, C, generator=g, dtype=torch.float64)
z_b[:, 0] = z_a[:, 0] + 0.01 * torch.randn(M, generator=g, dtype=torch.float64)
mix = torch.randn(C, C, generator=g, dtype=torch.float64)
_, _, i2 = fit_sfa(z_a @ mix, z_b @ mix, torch.eye(C, dtype=torch.float64), torch.zeros(C, dtype=torch.float64), {"lentos": C})
y = (z_a @ mix - i2["mu"]) @ i2["R_all"][:, 0]
corr = float(torch.corrcoef(torch.stack([y, z_a[:, 0]]))[0, 1].abs())
print("2 |corr| lento vs z0", corr, "delta[0:2]", i2["delta"][:2].tolist())
assert corr > 0.999

# 3. conv dobrada = (u − μ) R_all antes da ReLU
lhs = A @ W.T + b
rhs = (ua - info["mu"]) @ info["R_all"]
print("3 dobra", float((lhs - rhs).abs().max()))
assert torch.allclose(lhs, rhs, atol=1e-9)

# 4. fit_fu streaming = solve em lote
Wf, fi = fit_fu(A.split(250), Bv.split(250), W0, {})
Wr = W0.T
T0 = (A @ Wr + Bv @ Wr) / 2
G = (A.T @ A + Bv.T @ Bv) / (2 * M)
Cm = (A.T @ T0 + Bv.T @ T0) / (2 * M)
lam = 0.1 * float(G.trace()) / d
H = G + lam * torch.eye(d, dtype=G.dtype)
Wb = torch.linalg.solve(H, Cm + lam * Wr)
print("4 streaming vs lote", float((Wf.T - Wb).abs().max()), "system_residual", fi["system_residual"],
      "update_rel", fi["update_rel"], "dead", fi["dead"])
assert torch.allclose(Wf.T, Wb, atol=1e-10) and fi["system_residual"] < 1e-12

# 5. W = W0 − (1/4) solve(G + λI, D W0)
D = (A - Bv).T @ (A - Bv) / M
Wi = Wr - 0.25 * torch.linalg.solve(H, D @ Wr)
print("5 identidade", float((Wf.T - Wi).abs().max()))
assert torch.allclose(Wf.T, Wi, atol=1e-10)

# 6. nada exige gradiente
assert not any(t.requires_grad for t in (W, b, Wf))
print("6 sem backward e sem requires_grad: ok")

# 7. determinismo
W2, b2, _ = fit_sfa(ua, ub, W0, b0, spec)
Wf2, _ = fit_fu(A.split(250), Bv.split(250), W0, {})
# bit a bit so com 1 thread: com 2, a reducao multithread do BLAS varia em ~1e-15
assert all(torch.allclose(x, y, rtol=0, atol=1e-12) for x, y in ((W, W2), (b, b2), (Wf, Wf2)))
print("7 determinismo: ok")

# posto insuficiente: largura reduzida e reportada
P4 = torch.randn(4, C, generator=g, dtype=torch.float64)
_, _, i8 = fit_sfa(ua[:, :4] @ P4, ub[:, :4] @ P4, W0, b0, spec)
print("8 posto", i8["rank"], "largura", i8["width"], "reduzida", i8["reduced"])
assert i8["reduced"] and i8["width"] <= i8["rank"]
print("TODOS OK")
