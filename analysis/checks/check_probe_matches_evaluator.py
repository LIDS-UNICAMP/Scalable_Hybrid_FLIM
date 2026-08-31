# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""check_probe_matches_evaluator.py — o probe de treino e o avaliador oficial sao o mesmo objeto.

Auto-teste com `assert`, sem pytest. Verifica duas coisas:

1. `_encode_pooled(model.encoder, x)` == `model.embed(x)` elemento a elemento, e um mesmo
   `SVC` alimentado pelos dois devolve kappa identico. Depois da troca em
   `validation_step`/`_embeddings`, o probe *chama* `_encode_pooled`, entao este assert e o
   que impede a equivalencia de se perder num refactor futuro.
2. Os dois modos de embedding devolvem a forma esperada: avgpool2d -> [B, 48],
   flatten -> [B, 27648].

Uso:  python -m analysis.checks.check_probe_matches_evaluator
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# analysis/checks/ esta a 2 niveis da raiz do repo (o arquivo veio de tools/, que era 1).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.svm import SVC                                             # noqa: E402

import eval.svm as _ev                                                  # noqa: E402
from core.metrics import compute_metrics                                # noqa: E402
from core.constants import NUM_CLASSES                                  # noqa: E402
from methods.autoencoder import AutoEncoderFlimModule                   # noqa: E402
from autoencoder_flim_ray import _arch_json, _flim_weights_path         # noqa: E402

# ponytail: um dataset/split fixo, sem CLI. O que se testa e a igualdade de dois caminhos de
# codigo, nao um numero do experimento — varrer a grade nao acrescentaria nada.
DATASET, SPLIT, B = "protozoan", 1, 8


def _kappa(X, y):
    clf = SVC(max_iter=-1, C=1e2, degree=3, gamma="auto", coef0=0,
              decision_function_shape="ovo", kernel="linear").fit(X, y)
    return compute_metrics(y, clf.predict(X), num_classes=NUM_CLASSES[DATASET])["kappa"]


def main() -> int:
    torch.manual_seed(0)
    model = AutoEncoderFlimModule(
        arch_json=_arch_json(DATASET, SPLIT),
        dataset=DATASET,
        flim_weights_path=_flim_weights_path(DATASET, SPLIT),
        num_classes=NUM_CLASSES[DATASET],
        image_size=200,
        freeze_encoder_flag=True,
    ).model.eval()

    # load_FLIM_encoder já deixa os kernels no device dele; o tensor segue o modelo.
    x = torch.rand(B, 3, 200, 200, device=next(model.parameters()).device)
    y = np.arange(B) % NUM_CLASSES[DATASET]

    with torch.no_grad():
        pooled = _ev._encode_pooled(model.encoder, x, mode="avgpool2d")
        embed = model.embed(x).detach().cpu()
    assert torch.allclose(pooled, embed, atol=1e-6), (pooled - embed).abs().max()
    assert _kappa(pooled.numpy(), y) == _kappa(embed.numpy(), y)
    assert pooled.shape == (B, 48), pooled.shape

    with torch.no_grad():
        flat = _ev._encode_pooled(model.encoder, x, mode="flatten")
    assert flat.shape == (B, 27648), flat.shape

    # Modo invalido nao pode mais cair silenciosamente no pooling.
    try:
        _ev._encode_pooled(model.encoder, x, mode="gap")
    except ValueError:
        pass
    else:
        raise AssertionError("_encode_pooled aceitou um modo invalido")

    print(f"OK  avgpool2d {tuple(pooled.shape)} == embed{tuple(embed.shape)} "
          f"(max |diff| {(pooled - embed).abs().max():.3e}, kappa identico) | "
          f"flatten {tuple(flat.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
