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
"""SPiFiL montado pelo A3: o `grow` de `flim/spifil.py` com o z-score dos filtros
tirado de TODOS os patches da camada anterior, e nao so dos ~24 escolhidos.

`spifil.nn.builder.build_filters` diz "z-score the dataset per feature", mas tira
media e desvio de `selected` — com 24 patches para 75+ features o desvio e ruido.
Aqui a dobra (z-score -> norma unitaria -> /desvio, bias = -k.media) e a mesma; so
a populacao das estatisticas muda. O pacote `spifil` nao e editado: o `grow`
roda com `build_filters` e `GrowOneLayer` trocados so dentro de `grow`.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import spifil.learner as _learner
from spifil.nn.builder import _UNIT_NORM_THRESHOLD, FilterBank

import flim.spifil as _base

_pool: list = []  # patches da camada anterior do grow em curso


class _StashPatches(_base.GrowOneLayer):
    def run(self, learn) -> None:
        _pool[:] = [learn.state[learn.n_layers - 1].patches.feats]
        super().run(learn)


def _build_filters_pop(selected, spec, *, stdev_factor: float = 0.01) -> FilterBank:
    span = 2 * (spec.kernel_size // 2) + 1
    feats, pop = selected.feats, _pool[0].to(selected.feats)
    mean = pop.mean(dim=0)
    stdev = torch.sqrt(((pop - mean) ** 2).mean(dim=0)) + stdev_factor
    zscored = (feats - mean) / stdev
    norms = torch.linalg.vector_norm(zscored, dim=1, keepdim=True)
    zscored = torch.where(norms > _UNIT_NORM_THRESHOLD, zscored / norms, zscored)
    kernels = zscored / stdev
    return FilterBank(
        weight=kernels.reshape(len(selected), -1, span, span).contiguous(),
        bias=(-(kernels * mean).sum(dim=1)).contiguous(),
        labels=selected.labels.clone(), mean=mean, stdev=stdev,
    )


@contextmanager
def _swapped():
    old = _learner.build_filters, _base.GrowOneLayer
    _learner.build_filters, _base.GrowOneLayer = _build_filters_pop, _StashPatches
    try:
        yield
    finally:
        _learner.build_filters, _base.GrowOneLayer = old
        _pool.clear()


def grow(**kwargs) -> int:
    """Mesma assinatura e retorno de `flim.spifil.grow`."""
    with _swapped():
        print("[grow a3] z-score dos filtros sobre todos os patches candidatos")
        return _base.grow(**kwargs)
