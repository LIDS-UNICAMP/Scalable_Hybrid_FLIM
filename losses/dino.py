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
"""DINOv2 sem iBOT para encoder CNN: cabeca, professor EMA, perda de autodestilacao e KoLeo.

Espelha github.com/facebookresearch/dinov2:
- dinov2/layers/dino_head.py            -> DINOHead
- dinov2/loss/dino_clstoken_loss.py     -> DINOLoss (softmax_center_teacher,
                                           sinkhorn_knopp_teacher, forward, update_center)
- dinov2/loss/koleo_loss.py             -> koleo_loss (pairwise_NNs_inner + PairwiseDistance)
- dinov2/configs/ssl_default_config.yaml -> student_temp 0.1, warmup_teacher_temp 0.04,
  teacher_temp 0.07, final_momentum_teacher 1.0, koleo_loss_weight 0.1 (peso do chamador).
Tamanhos menores que o artigo (2048/128/2048 em vez de 2048/256/65536): modelo de 1-5 M.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOHead(nn.Module):
    """dino_head.py: MLP de 3 camadas -> gargalo L2-normalizado -> prototipos com weight norm."""

    def __init__(self, in_dim: int, hidden: int = 512, bottleneck: int = 128,
                 n_prototypes: int = 2048) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        # parametrizations (e nao nn.utils.weight_norm) para o deepcopy do EMATeacher funcionar.
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, n_prototypes, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)  # weight_g = 1

    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(self.mlp(x), dim=-1, p=2, eps=1e-6)
        return self.last_layer(x)


def ema_momentum(step: int, total: int, base: float = 0.996, final: float = 1.0) -> float:
    """CosineScheduler do DINOv2 (momentum_teacher -> final_momentum_teacher)."""
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))


class EMATeacher(nn.Module):
    """Copia do aluno, sem gradiente; parametros por EMA, buffers (BN) copiados do aluno."""

    def __init__(self, student_encoder: nn.Module, student_head: nn.Module) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(student_encoder)
        self.head = copy.deepcopy(student_head)
        self.requires_grad_(False)

    @torch.no_grad()
    def update(self, student, momentum: float) -> None:
        """student = (encoder, head) do aluno. t <- m*t + (1-m)*s."""
        for t_mod, s_mod in zip((self.encoder, self.head), student):
            t_p, s_p = list(t_mod.parameters()), list(s_mod.parameters())
            torch._foreach_mul_(t_p, momentum)
            torch._foreach_add_(t_p, [p.detach() for p in s_p], alpha=1 - momentum)
            for t_b, s_b in zip(t_mod.buffers(), s_mod.buffers()):
                t_b.copy_(s_b)


class DINOLoss(nn.Module):
    """dino_clstoken_loss.py: CE entre professor (centralizado ou SK, afiado) e aluno."""

    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp_start: float = 0.04,
                 teacher_temp_end: float = 0.07, warmup_frac: float = 0.1,
                 center_momentum: float = 0.9, use_sinkhorn: bool = True, sk_iters: int = 3) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.t0, self.t1, self.warmup_frac = teacher_temp_start, teacher_temp_end, warmup_frac
        self.center_momentum = center_momentum
        self.use_sinkhorn, self.sk_iters = use_sinkhorn, sk_iters
        self.register_buffer("center", torch.zeros(1, out_dim))

    def teacher_temp(self, step: int, total: int) -> float:
        warm = self.warmup_frac * total
        return self.t1 if step >= warm else self.t0 + (self.t1 - self.t0) * step / warm

    @torch.no_grad()
    def sinkhorn_knopp(self, t: Tensor, temp: float) -> Tensor:
        """sinkhorn_knopp_teacher (sem all_reduce: uma GPU). t [N,K] -> linhas somando 1."""
        q = torch.exp((t.float() - t.float().max()) / temp).t()  # [K,N]; o max so estabiliza
        k, n = q.shape
        q /= q.sum()
        for _ in range(self.sk_iters):
            q /= q.sum(dim=1, keepdim=True)
            q /= k
            q /= q.sum(dim=0, keepdim=True)
            q /= n
        q *= n
        return q.t()

    @torch.no_grad()
    def softmax_center(self, t: Tensor, temp: float) -> Tensor:
        """softmax_center_teacher + update_center (EMA da media do lote)."""
        p = F.softmax((t - self.center) / temp, dim=-1)
        self.center.mul_(self.center_momentum).add_(t.mean(0, keepdim=True), alpha=1 - self.center_momentum)
        return p

    def forward(self, student_out: Tensor, teacher_out: Tensor, step: int, total: int) -> Tensor:
        """student_out [V,B,K] (as G globais primeiro), teacher_out [G,B,K] -> escalar.

        Media sobre pares (v,g) com v != g, como os V*G - G termos do ssl_meta_arch.
        """
        g, b, k = teacher_out.shape
        temp = self.teacher_temp(step, total)
        flat = teacher_out.detach().reshape(g * b, k)
        q = self.sinkhorn_knopp(flat, temp) if self.use_sinkhorn else self.softmax_center(flat, temp)
        q = q.reshape(g, b, k)
        lsm = F.log_softmax(student_out.float() / self.student_temp, dim=-1)
        total_loss, n_terms = 0.0, 0
        for v in range(student_out.shape[0]):
            for t in range(g):
                if v == t:
                    continue
                total_loss = total_loss - (q[t] * lsm[v]).sum(-1).mean()
                n_terms += 1
        return total_loss / n_terms


def koleo_loss(z: Tensor, eps: float = 1e-8) -> Tensor:
    """koleo_loss.py: -media log da distancia ao vizinho mais proximo, z [B,D] L2-normalizado."""
    z = F.normalize(z.float(), dim=-1, p=2, eps=eps)
    with torch.no_grad():
        dots = z @ z.t()
        dots.fill_diagonal_(-1)
        nn_idx = dots.argmax(dim=1)
    dist = F.pairwise_distance(z, z[nn_idx], p=2, eps=eps)
    return -torch.log(dist + eps).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    V, G, B, C = 8, 2, 16, 32
    enc = nn.Sequential(nn.Conv2d(3, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    head = DINOHead(C, hidden=64, bottleneck=16, n_prototypes=128)
    teacher = EMATeacher(enc, head)
    views = [torch.randn(B, 3, 32 if i < G else 16, 32 if i < G else 16) for i in range(V)]

    for sk in (True, False):
        crit = DINOLoss(128, use_sinkhorn=sk)
        s_out = torch.stack([head(enc(x)) for x in views])
        with torch.no_grad():
            t_out = torch.stack([teacher.head(teacher.encoder(x)) for x in views[:G]])
        loss = crit(s_out, t_out, step=5, total=100)
        loss.backward()
        print(f"sinkhorn={sk} loss={loss.item():.4f}")
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in head.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        enc.zero_grad(set_to_none=True), head.zero_grad(set_to_none=True)

    q = DINOLoss(128).sinkhorn_knopp(torch.randn(B * G, 128), 0.04)
    print(f"sinkhorn somas das linhas: min={q.sum(1).min():.4f} max={q.sum(1).max():.4f}")
    assert torch.allclose(q.sum(1), torch.ones(B * G), atol=1e-3)

    with torch.no_grad():
        for p in list(enc.parameters()) + list(head.parameters()):
            p.add_(torch.randn_like(p))
    s_vec = lambda m: torch.cat([p.flatten() for p in m.parameters()])
    student_vec = torch.cat([s_vec(enc), s_vec(head)])
    teacher_vec = lambda: torch.cat([s_vec(teacher.encoder), s_vec(teacher.head)])
    before = (teacher_vec() - student_vec).norm()
    teacher.update((enc, head), ema_momentum(0, 100))
    after = (teacher_vec() - student_vec).norm()
    print(f"EMA m={ema_momentum(0, 100):.3f} dist {before:.4f} -> {after:.4f}; m(fim)={ema_momentum(100, 100)}")
    assert after < before

    kl = koleo_loss(torch.randn(B, 64, requires_grad=True))
    print(f"koleo={kl.item():.4f}")
    assert torch.isfinite(kl)
    print("OK")
