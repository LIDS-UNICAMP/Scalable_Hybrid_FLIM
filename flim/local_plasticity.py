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
"""local_plasticity.py — H: SoftHebb (Journe et al., ICLR 2023) numa camada L4/L5, sem rotulo nem backprop.

Q e aprendido no espaco branqueado v = A(p - m) (A, m fixos; quem chama desdobra os patches com a dilatacao
real e aplica ZCA+N0). A regra e `hebb_step` de `eval/alt_encoders.py`, tal como esta, num bloco 1x1:
cada linha de v vira um "pixel" de d canais. `b_v` so entra na competicao. Numero de filtros constante.
`steps: 0` devolve Q0 exato (controle de zero atualizacao).
"""
from __future__ import annotations

from itertools import islice
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn



@torch.no_grad()
def learn_layer_hebb(v_batches: Iterable[torch.Tensor], Q0: torch.Tensor, b_v: torch.Tensor,
                     spec: dict) -> tuple[torch.Tensor, dict]:
    """spec: steps, lr, inv_temp, delta_norm ("tensor" por padrao), init_radius opcional (norma inicial das
    linhas, como `learn_softhebb`; sem ele, linhas de norma 1 dao lr*|‖w‖-1|^½ ~ 0)."""
    from eval.alt_encoders import hebb_step  # tardio: alt_encoders importa spf_grade, que importa este modulo
    assert not torch.is_grad_enabled() and not Q0.requires_grad and not b_v.requires_grad
    if not spec["steps"]:
        return Q0.clone(), {"steps": 0, "drift_cos_mean": 1.0, "drift_cos_min": 1.0, "rel_update": 0.0,
                            "win_freq": None, "entropy": None, "eff_filters": None, "dead": None}
    Q = Q0.clone()
    if spec.get("init_radius"):
        Q *= spec["init_radius"] / Q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    start = Q.clone()
    block = SimpleNamespace(bn=nn.Identity(), conv=SimpleNamespace(weight=Q.view(*Q.shape, 1, 1)))
    wins, steps = torch.zeros(len(Q), dtype=torch.long, device=Q.device), 0
    for v in islice(v_batches, spec["steps"]):
        assert not v.requires_grad
        wins += hebb_step(block, v.to(Q)[:, :, None, None], spec["lr"], spec["inv_temp"], b_v.to(Q),
                          spec.get("delta_norm", "tensor"))
        steps += 1
    cos = F.cosine_similarity(Q, Q0, dim=1)
    share = wins.double() / wins.sum().clamp_min(1)
    entropy = float(-(share * share.clamp_min(1e-300).log()).sum())
    return Q, {"steps": steps, "drift_cos_mean": float(cos.mean()), "drift_cos_min": float(cos.min()),
               "rel_update": float((Q - start).norm() / start.norm()), "win_freq": share.tolist(),
               "entropy": entropy, "eff_filters": float(torch.tensor(entropy).exp()),
               "dead": int((wins == 0).sum())}
