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
"""checks_C1.py — checks do §14 item 5 para flim/local_predictive.py (CPU, tensores minusculos)."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from flim.local_predictive import clapp_step, learn_layer_clapp, pool_op, rows

torch.set_num_threads(2)
D = torch.float64


def toy(B=3, grid=(5, 5), C=4, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    P_ = grid[0] * grid[1]
    return (torch.randn(B, P_, d, generator=g, dtype=D), torch.randn(B, P_, d, generator=g, dtype=D),
            torch.randn(C, d, generator=g, dtype=D), 0.1 * torch.randn(C, generator=g, dtype=D))


def ops(grid, out):
    return pool_op(grid, out)


# 1. hinge inativo => update exatamente zero
Xa, Xb, W0, b0 = toy()
for out in [(1, 1), (2, 2)]:
    U, Ut = ops((5, 5), out)
    P = 1e3 * torch.eye(4 * U.shape[0], dtype=D)
    dW, db, dP, s, g, _ = clapp_step(Xa, Xa, torch.ones(3, dtype=D), W0, b0, P, U, Ut)
    assert (s >= 1).all() and (g == 0).all() and not dW.any() and not db.any() and not dP.any()
    W, b, info = learn_layer_clapp([(Xa, Xa, 1.0)], W0, b0, (5, 5), {"lr": 1, "lr_p": 1, "p_init": 1e3, "saida": out})
    assert torch.equal(W, W0) and torch.equal(b, b0) and info["hinge_active"] == 0
print("1 hinge inativo: zero exato  OK")


# 2. sinal: positivo com score baixo sobe, negativo com score alto desce (passo pequeno, W e P)
def score_after(Xa, Xb, q, P, U, Ut, lr=1e-4, lr_p=1e-4):
    dW, db, dP, s0, g, _ = clapp_step(Xa, Xb, q, W0, b0, P, U, Ut)
    s1 = clapp_step(Xa, Xb, q, W0 + lr * dW, b0 + lr * db, P + lr_p * dP, U, Ut)[3]
    s2 = clapp_step(Xa, Xb, q, W0 + lr * dW, b0 + lr * db, P, U, Ut)[3]       # so W/b
    return s0, s1, s2


for out in [(1, 1), (2, 2)]:
    U, Ut = ops((5, 5), out)
    Pp = 0.01 * torch.eye(4 * U.shape[0], dtype=D)
    s0, s1, s2 = score_after(Xa, Xb, torch.ones(3, dtype=D), Pp, U, Ut)
    assert (s0 < 1).all() and (s1 > s0).all() and (s2 > s0).all(), (s0, s1, s2)
    U_, _ = ops((5, 5), out)
    Pn = 10 * torch.eye(4 * U.shape[0], dtype=D)                               # score alto e > -1
    s0, s1, s2 = score_after(Xa, Xb, -torch.ones(3, dtype=D), Pn, U, Ut)
    assert (s0 > -1).all() and (s1 < s0).all() and (s2 < s0).all(), (s0, s1, s2)
print("2 sinais pos/neg (W+P e so W): OK")


# 3. um passo == numpy com lacos explicitos das equacoes do §7.2
def numpy_step(Xa, Xb, q, W, b, P, grid, out, lr, lr_p):
    Xa, Xb, q, W, b, P = (t.numpy() for t in (Xa, Xb, q, W, b, P))
    (H, Wd), (oh, ow) = grid, out
    B, Pn, d = Xa.shape
    C, K = W.shape[0], oh * ow
    bins = [(range((i * H) // oh, math.ceil((i + 1) * H / oh)), range((j * Wd) // ow, math.ceil((j + 1) * Wd / ow)))
            for i in range(oh) for j in range(ow)]
    dW, db, dP = np.zeros_like(W), np.zeros_like(b), np.zeros_like(P)
    for n in range(B):
        a = {v: np.array([[sum(W[j, e] * X[n, p, e] for e in range(d)) + b[j] for j in range(C)] for p in range(Pn)])
             for v, X in (("a", Xa), ("b", Xb))}
        vec = {}
        for v in "ab":
            Fm = np.maximum(a[v], 0)
            vec[v] = np.zeros(C * K)
            for j in range(C):
                for k, (rr, cc) in enumerate(bins):
                    vec[v][j * K + k] = sum(Fm[r * Wd + s, j] for r in rr for s in cc) / (len(rr) * len(cc))
        c, z = vec["a"], vec["b"]
        score = sum(z[i] * P[i, j] * c[j] for i in range(C * K) for j in range(C * K))
        g = q[n] * float(q[n] * score < 1)
        Pc = np.array([sum(P[i, j] * c[j] for j in range(C * K)) for i in range(C * K)])
        Rz = np.array([sum(P[i, j] * z[i] for i in range(C * K)) for j in range(C * K)])   # R = P^T
        for j in range(C):
            for p in range(Pn):
                r, s = divmod(p, Wd)
                ub = sum(Pc[j * K + k] / (len(rr) * len(cc)) for k, (rr, cc) in enumerate(bins) if r in rr and s in cc)
                ua = sum(Rz[j * K + k] / (len(rr) * len(cc)) for k, (rr, cc) in enumerate(bins) if r in rr and s in cc)
                mb, ma = ub * (a["b"][p, j] > 0), ua * (a["a"][p, j] > 0)
                dW[j] += lr * g * (mb * Xb[n, p] + ma * Xa[n, p]) / B
                db[j] += lr * g * (mb + ma) / B
        dP += lr_p * g * np.outer(z, c) / B
    return W + dW, b + db, P + dP


for grid, out in [((5, 5), (1, 1)), ((5, 5), (2, 2)), ((3, 4), (2, 2))]:
    Xa, Xb, W0, b0 = toy(B=3, grid=grid, C=3, d=4, seed=1)
    q = torch.tensor([1.0, -1.0, 1.0], dtype=D)
    U, Ut = ops(grid, out)
    P = 0.3 * torch.randn(3 * U.shape[0], 3 * U.shape[0], generator=torch.Generator().manual_seed(2), dtype=D)
    dW, db, dP, *_ = clapp_step(Xa, Xb, q, W0, b0, P, U, Ut)
    Wn, bn, Pn = numpy_step(Xa, Xb, q, W0, b0, P, grid, out, 0.5, 0.2)
    assert np.allclose((W0 + 0.5 * dW).numpy(), Wn, atol=1e-12) and np.allclose((b0 + 0.5 * db).numpy(), bn, atol=1e-12)
    assert np.allclose((P + 0.2 * dP).numpy(), Pn, atol=1e-12) and dW.abs().sum() > 0
print("3 um passo == numpy com lacos (1x1, 2x2 em 5x5 e 3x4): OK")

# 4. Ut = U^T (produto interno) e U == adaptive_avg_pool2d, inclusive 11x11 com bins desiguais
gen = torch.Generator().manual_seed(3)
for grid, out in [((11, 11), (1, 1)), ((11, 11), (2, 2)), ((24, 24), (2, 2)), ((5, 7), (2, 2))]:
    U, Ut = pool_op(grid, out)
    x, y = torch.randn(grid[0] * grid[1], generator=gen, dtype=D), torch.randn(U.shape[0], generator=gen, dtype=D)
    assert abs((U @ x) @ y - x @ (Ut @ y)) < 1e-12 and torch.equal(Ut, U.t())
    m = torch.randn(2, grid[0], grid[1], generator=gen, dtype=D)
    ref = F.adaptive_avg_pool2d(m, out).flatten(1)
    assert torch.allclose(m.flatten(1) @ U.t(), ref, atol=1e-12)
print("4 Ut = U^T e U == AdaptiveAvgPool (11x11 1x1/2x2, 24x24, 5x7): OK")

# 5. sem backward e sem .grad
torch.Tensor.backward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backward proibido"))
torch.autograd.backward = torch.Tensor.backward
Xa, Xb, W0, b0 = toy()
W0.requires_grad_(True)
W, b, _ = learn_layer_clapp([(Xa, Xb, 1.0), (Xa, Xb[[1, 2, 0]], -1.0)], W0, b0, (5, 5), {"lr": 0.1, "lr_p": 0.1, "saida": (2, 2)})
assert W0.grad is None and not W.requires_grad and not b.requires_grad
W0 = W0.detach()
print("5 sem backward/.grad: OK")

# 6. determinismo + mapas com dilatacao: unfold == conv2d
spec = {"lr": 0.05, "lr_p": 0.05, "saida": (2, 2), "p_init": 0.01, "kernel": 3, "dilation": 2}
gen = torch.Generator().manual_seed(4)
maps = [torch.randn(2, 2, 7, 7, generator=gen, dtype=D) for _ in range(3)]
W0m, b0m = torch.randn(4, 18, generator=gen, dtype=D), torch.zeros(4, dtype=D)
batches = [(maps[0], maps[1], 1.0), (maps[1], maps[2], torch.tensor([1.0, -1.0], dtype=D))]
r1 = learn_layer_clapp(batches, W0m, b0m, (7, 7), spec)
r2 = learn_layer_clapp(batches, W0m, b0m, (7, 7), spec)
assert torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1]) and r1[2] == r2[2]
a_rows = rows(maps[0], spec) @ W0m.t()
a_conv = F.conv2d(maps[0], W0m.view(4, 2, 3, 3), dilation=2, padding=2).flatten(2).transpose(1, 2)
assert torch.allclose(a_rows, a_conv, atol=1e-12)
print("6 determinismo + unfold dilatado == conv2d: OK", r1[2])
