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
"""local_predictive.py — CLAPP sem autograd para uma camada conv (prompt_encdeep §7.2/§7.3).

Par = duas vistas (q=+1, mesma imagem) ou duas imagens (q=-1); nunca classe. Para cada par:
a = X W^T + b, F = ReLU(a), c = vec(U F_a), z = vec(U F_b), score = z^T P c, R = P^T.
g = q 1[q score < 1]; m_b = U^T(P c) 1[a_b>0]; m_a = U^T(P^T z) 1[a_a>0];
dW = lr g sum_p(m_b x_b^T + m_a x_a^T), db = lr g sum_p(m_b + m_a), dP = lr_p g z c^T.
Todos os incrementos saem do mesmo snapshot e sao a media dos pares do lote. P e descartado.
C0: saida (1,1); C1: (2,2) com bins de area real (AdaptiveAvgPool). vec e canal-major (c*K + k).
"""

import math

import torch
import torch.nn.functional as F


def pool_op(grid, out_size):
    """U (K, H*W) = AdaptiveAvgPool explicito; Ut = U^T (bins sobrepostos somam)."""
    (H, W), (oh, ow) = grid, out_size
    U = torch.zeros(oh * ow, H * W, dtype=torch.float64)
    for i in range(oh):
        r0, r1 = (i * H) // oh, math.ceil((i + 1) * H / oh)
        for j in range(ow):
            c0, c1 = (j * W) // ow, math.ceil((j + 1) * W / ow)
            cell = torch.zeros(H, W, dtype=torch.float64)
            cell[r0:r1, c0:c1] = 1.0 / ((r1 - r0) * (c1 - c0))
            U[i * ow + j] = cell.flatten()
    return U, U.t().contiguous()


def rows(X, spec):
    """(B,P,d) ou (P,d) passam; mapa (B,Cin,H,W) vira patches 'same' com a dilatacao do spec."""
    if X.dim() == 4:
        k, dil = spec.get("kernel", 3), spec.get("dilation", 1)
        return F.unfold(X, k, dilation=dil, padding=dil * (k - 1) // 2).transpose(1, 2)
    return X if X.dim() == 3 else X[None]


@torch.no_grad()
def clapp_step(Xa, Xb, q, W, b, P, U, Ut):
    """Incrementos (sem lr) de um lote a partir do snapshot (W, b, P)."""
    B, C, K = Xa.shape[0], W.shape[0], U.shape[0]
    aa, ab = Xa @ W.t() + b, Xb @ W.t() + b                      # (B,P,C)
    c = torch.einsum("kp,bpc->bck", U, aa.clamp_min(0)).reshape(B, C * K)
    z = torch.einsum("kp,bpc->bck", U, ab.clamp_min(0)).reshape(B, C * K)
    score = (z @ P * c).sum(1)                                  # z^T P c
    g = q * (q * score < 1).to(W.dtype)
    m_b = torch.einsum("pk,bck->bpc", Ut, (c @ P.t()).reshape(B, C, K)) * (ab > 0)   # U^T(P c)
    m_a = torch.einsum("pk,bck->bpc", Ut, (z @ P).reshape(B, C, K)) * (aa > 0)       # U^T(P^T z)
    dW = (torch.einsum("b,bpc,bpd->cd", g, m_b, Xb) + torch.einsum("b,bpc,bpd->cd", g, m_a, Xa)) / B
    db = torch.einsum("b,bpc->c", g, m_b + m_a) / B
    dP = torch.einsum("b,bi,bj->ij", g, z, c) / B
    alive = ((aa > 0) | (ab > 0)).flatten(0, 1).any(0)
    return dW, db, dP, score, g, alive


@torch.no_grad()
def learn_layer_clapp(pairs, W0, b0, grid, spec):
    """pairs: iteravel de (Xa_rows, Xb_rows, q), um passo por lote. Devolve (W, b, info); P e descartado."""
    W, b = W0.clone(), b0.clone()
    U, Ut = (t.to(W) for t in pool_op(grid, tuple(spec.get("saida", (1, 1)))))
    P = spec.get("p_init", 0.0) * torch.eye(W.shape[0] * U.shape[0], dtype=W.dtype, device=W.device)
    lr, lr_p = spec["lr"], spec["lr_p"]
    act, n, pos, neg, rel, alive = 0.0, 0, [], [], [], torch.zeros(W.shape[0], dtype=torch.bool, device=W.device)
    for Xa, Xb, q in pairs:
        Xa, Xb = rows(Xa, spec).to(W), rows(Xb, spec).to(W)
        q = torch.as_tensor(q, dtype=W.dtype, device=W.device).expand(Xa.shape[0])
        dW, db, dP, score, g, live = clapp_step(Xa, Xb, q, W, b, P, U, Ut)
        rel.append((lr * dW).norm().item() / max(W.norm().item(), 1e-12))
        W += lr * dW
        b += lr * db
        P += lr_p * dP
        act, n = act + (g != 0).sum().item(), n + len(g)
        pos += score[q > 0].tolist()
        neg += score[q < 0].tolist()
        alive |= live
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    info = {"hinge_active": act / max(n, 1), "score_pos": mean(pos), "score_neg": mean(neg),
            "rel_update": mean(rel), "drift": ((W - W0).norm() / W0.norm()).item(),
            "dead": int((~alive).sum()), "predictor_params": P.numel(), "steps": len(rel)}
    return W, b, info
