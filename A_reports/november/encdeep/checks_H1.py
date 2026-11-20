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
"""checks_H1.py — §14 para flim/local_plasticity.py, em CPU com tensores minusculos."""
import numpy as np
import torch
import torch.nn.functional as F

from flim.local_plasticity import learn_layer_hebb

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
SPEC = {"steps": 5, "lr": 0.05, "inv_temp": 0.65, "delta_norm": "tensor", "init_radius": 2 ** 0.5}


def batches(seed, n=5, N=32, d=12):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, d, generator=g) for _ in range(n)]


g = torch.Generator().manual_seed(0)
Q0, b = F.normalize(torch.randn(6, 12, generator=g)), 0.1 * torch.randn(6, generator=g)

# 1. steps=0 devolve Q0 bit a bit
Q, _ = learn_layer_hebb(batches(1), Q0, b, {**SPEC, "steps": 0})
assert torch.equal(Q, Q0) and Q is not Q0
print("1 ok: steps=0 bit-identico")

# 2. uma atualizacao contra referencia numpy a mao
Q, info = learn_layer_hebb(batches(1, n=1), Q0, b, {**SPEC, "steps": 1})
z, w, bb = batches(1, n=1)[0].numpy(), Q0.numpy() * np.sqrt(2), b.numpy()
u = z @ w.T
race = 0.65 * (u + bb)
y = np.exp(race - race.max(1, keepdims=True))
y = -y / y.sum(1, keepdims=True)
win = (u + bb).argmax(1)
y[np.arange(len(y)), win] *= -1
delta = y.T @ z - (y * u).sum(0)[:, None] * w
delta /= np.abs(delta).max()
ref = w + (0.05 * np.sqrt(np.abs(np.linalg.norm(w, axis=1) - 1) + 1e-10))[:, None] * delta
assert np.allclose(Q.numpy(), ref, atol=1e-12), np.abs(Q.numpy() - ref).max()
assert info["steps"] == 1 and info["dead"] == int((np.bincount(win, minlength=6) == 0).sum())
print("2 ok: 1 passo = numpy, max|erro| =", np.abs(Q.numpy() - ref).max(), "| info:",
      {k: v for k, v in info.items() if k != "win_freq"})

# 3. sem autograd: backward proibido, nada de .grad, entrada com requires_grad recusada
orig = torch.Tensor.backward
torch.Tensor.backward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backward chamado"))
try:
    Qg = Q0.clone()
    Q, _ = learn_layer_hebb(batches(2), Qg, b, SPEC)
    assert Qg.grad is None and Q.grad is None and not Q.requires_grad
    try:
        learn_layer_hebb(batches(2), Q0.clone().requires_grad_(), b, SPEC)
        raise SystemExit("requires_grad aceito")
    except AssertionError:
        pass
finally:
    torch.Tensor.backward = orig
print("3 ok: sem backward, sem .grad, requires_grad recusado")

# 4. determinismo com a mesma semente
Qa, ia = learn_layer_hebb(batches(3), Q0, b, SPEC)
Qb, ib = learn_layer_hebb(batches(3), Q0, b, SPEC)
assert torch.equal(Qa, Qb) and ia == ib
print("4 ok: deterministico")

# 5. dobra §6.1: conv(K=QA, b-QAm) == Q(A(p-m))+b, com borda de zeros, dilatacao 1 e 2
Cin, k, C = 3, 3, 4
D = Cin * k * k
A, m = torch.randn(D, D, generator=g), torch.randn(D, generator=g)
Qr, br = torch.randn(C, D, generator=g), torch.randn(C, generator=g)
x = torch.randn(2, Cin, 7, 9, generator=g)
for dil in (1, 2):
    pad = dil * (k // 2)
    K = (Qr @ A).view(C, Cin, k, k)
    folded = F.conv2d(x, K, br - Qr @ A @ m, padding=pad, dilation=dil)
    p = F.unfold(x, k, dilation=dil, padding=pad).transpose(1, 2)          # (B, L, D), canal primeiro
    explicit = ((p - m) @ A.T @ Qr.T + br).transpose(1, 2).reshape(folded.shape)
    err = (folded - explicit).abs().max().item()
    assert err < 1e-10, err
    print(f"5 ok: dobra dil={dil}, max|erro| = {err:.1e} (bordas incluidas)")
