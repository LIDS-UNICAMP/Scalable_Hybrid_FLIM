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
"""slow_features.py — S (SFA por duas vistas) e F-U, por algebra linear float64, sem rotulo nem autograd.

Convencao de patch linha: U = A W0ᵀ + b0, com W0 no layout da conv (C, d). As funcoes recebem e
devolvem pesos nesse layout. Entradas sao tensores ou iteraveis de lotes (streaming dos momentos).

fit_sfa: CΔ v = δ C0 v, com C0 a covariancia das duas vistas juntas e Vᵀ C0 V = I. Lentos = menores δ.
Com `residuo` > 0, R_all = [R_slow, (I − R_slow P_rec) V_res]; cada bloco calibrado para RMS 1.
Dobra: W_novo = (W0ᵀ R_all)ᵀ, b_novo = (b0 − μ) R_all.

fit_fu: W = solve(G + λI, C + λW0), alvo congelado T0 = (A W0 + B W0)/2, λ = 0,1·tr(G)/d, Cholesky.
"""
import torch

TOL = 1e-6        # autovalor de C0 abaixo de TOL·max conta como direcao nula
PISO_RMS = 1e-3   # piso da calibracao RMS por bloco


def _pairs(a, b):
    return [(a, b)] if torch.is_tensor(a) else zip(a, b)


def _sign(V):
    """Componente de maior modulo de cada coluna fica positiva (reprodutivel entre runs)."""
    s = torch.sign(V.gather(0, V.abs().argmax(0, keepdim=True)))
    return V * torch.where(s == 0, torch.ones_like(s), s)


@torch.no_grad()
def fit_sfa(u_a, u_b, W0, b0, spec):
    tol, piso = spec.get("tol", TOL), spec.get("piso_rms", PISO_RMS)
    n = s = sd = sdd = shift = None
    for xa, xb in _pairs(u_a, u_b):
        xa, xb = xa.double(), xb.double()
        if shift is None:                       # desloca pelo 1o lote: soma de produtos estavel
            shift = torch.cat([xa, xb]).mean(0)
            C = xa.shape[1]
            n, s, sdd = 0, torch.zeros(C, dtype=xa.dtype), torch.zeros(C, C, dtype=xa.dtype)
            sd = torch.zeros(C, C, dtype=xa.dtype)
        x = torch.cat([xa, xb]) - shift
        dx = xa - xb
        n, s, sdd, sd = n + len(xa), s + x.sum(0), sdd + x.T @ x, sd + dx.T @ dx
    m = s / (2 * n)
    mu = shift + m
    C0 = sdd / (2 * n) - torch.outer(m, m)
    Cd = sd / n
    C0, Cd = (C0 + C0.T) / 2, (Cd + Cd.T) / 2

    lam, E = torch.linalg.eigh(C0)
    keep = lam > tol * lam.max()
    rank = int(keep.sum())
    P = E[:, keep] / lam[keep].sqrt()           # branqueia o subespaco de variancia positiva
    delta, Q = torch.linalg.eigh(P.T @ Cd @ P)  # δ crescente: primeiro o mais lento
    V = _sign(P @ Q)

    n_slow = min(spec.get("lentos", C), rank)
    R_slow = V[:, :n_slow]
    blocks, info = [R_slow], {}
    if spec.get("residuo", 0):
        S0 = R_slow.T @ C0
        P_rec = torch.linalg.lstsq(S0 @ R_slow, S0).solution
        M = torch.eye(C, dtype=C0.dtype) - R_slow @ P_rec
        Ce = M.T @ C0 @ M
        le, Ve = torch.linalg.eigh((Ce + Ce.T) / 2)
        le, Ve = le.flip(0), Ve.flip(1)
        n_res = min(spec["residuo"], int((le > tol * lam.max()).sum()))
        blocks.append(M @ _sign(Ve[:, :n_res]))
        info.update(n_res=n_res, P_rec=P_rec,
                    residual_explained=float(le[:n_res].sum() / le.clamp(min=0).sum()),
                    slow_explained=float(1 - Ce.trace() / C0.trace()))
    rms = [float(torch.diagonal(R.T @ C0 @ R).mean().sqrt()) for R in blocks]
    R_all = torch.cat([R / max(r, piso) for R, r in zip(blocks, rms)], 1)

    W = (W0.double().T @ R_all).T
    b = (b0.double() - mu) @ R_all
    Vk = V[:, :rank]
    width = R_all.shape[1]
    want = spec.get("lentos", C) + spec.get("residuo", 0)
    info.update(
        delta=delta, rank=rank, width=width, width_requested=want, reduced=width < want,
        n_slow=n_slow, rms=rms, mu=mu, R_all=R_all, out_cov=R_all.T @ C0 @ R_all,
        eig_residual=float(((Cd @ Vk - C0 @ Vk * delta).norm(dim=0) / Cd.norm()).max()),
        constraint_err=float((Vk.T @ C0 @ Vk - torch.eye(rank, dtype=C0.dtype)).abs().max()))
    return W.to(W0.dtype), b.to(b0.dtype), info


@torch.no_grad()
def fit_fu(A, B, W0, spec):
    Wr = W0.double().T                          # (d, C), patch linha
    n = G = Cm = s = None
    for a, bb in _pairs(A, B):
        a, bb = a.double(), bb.double()
        T0 = (a @ Wr + bb @ Wr) / 2             # alvo congelado, sem bias (b0 cancela)
        if G is None:
            d = a.shape[1]
            n, G, Cm, s = 0, torch.zeros(d, d, dtype=a.dtype), torch.zeros_like(Wr), torch.zeros(d, dtype=a.dtype)
        n, G, Cm = n + len(a), G + a.T @ a + bb.T @ bb, Cm + (a + bb).T @ T0
        s = s + a.sum(0) + bb.sum(0)
    G, Cm, m = G / (2 * n), Cm / (2 * n), s / (2 * n)
    lam = spec.get("lam_frac", 0.1) * float(G.trace()) / d
    H, rhs = G + lam * torch.eye(d, dtype=G.dtype), Cm + lam * Wr
    L, err = torch.linalg.cholesky_ex(H)
    W = torch.cholesky_solve(rhs, L) if not err else torch.linalg.lstsq(H, rhs).solution
    cov = G - torch.outer(m, m)
    var, var0 = torch.diagonal(W.T @ cov @ W), torch.diagonal(Wr.T @ cov @ Wr)
    info = dict(lam=lam, cholesky_failed=bool(err),
                system_residual=float((H @ W - rhs).norm() / rhs.norm()),
                update_rel=float((W - Wr).norm() / Wr.norm()), var=var, var0=var0,
                dead=int((var < spec.get("dead_frac", 1e-3) * var0).sum()), mean_out=m @ W)
    return W.T.to(W0.dtype), info
