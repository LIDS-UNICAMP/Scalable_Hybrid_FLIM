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

"""Ponto representativo de cada superpixel: um dos 3 do SPiFiL ou uma combinacao deles.

`SeedPoints("medoid")` e o `Medoids()` de hoje. Nas combinacoes os 3 pontos saem do mesmo
superpixel, na mesma ordem de regiao:
- `union`: os 3 pontos, sem repetir pixel (ate 3 candidatos por superpixel).
- `mean`: os 3 pontos, repetidos ou nao, em linhas seguidas; `mean_rows` faz a media dos patches.
- `random`: um dos 3 por superpixel, sorteado por gerador proprio semeado.
"""

import numpy as np
from spifil.superpixels.centers import Centroids, GeodesicCenters, Medoids

SINGLE = {"medoid": Medoids(), "centroid": Centroids(), "geodesic": GeodesicCenters()}
MODES = (*SINGLE, "union", "mean", "random")


class SeedPoints:
    """`SeedExtractor` do SPiFiL (`spifil/superpixels/base.py`)."""

    def __init__(self, mode: str = "medoid", seed: int = 42) -> None:
        if mode not in MODES:
            raise ValueError(f"seed_point {mode!r} nao existe; use um de {MODES}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __call__(self, labels, features=None):
        if self.mode in SINGLE:
            return SINGLE[self.mode](labels, features)
        # np.stack levanta se algum extrator pular uma regiao: a ordem tem que casar.
        points = np.stack([f(labels, features) for f in SINGLE.values()], axis=1)
        if self.mode == "union":
            return np.unique(points.reshape(-1, 2), axis=0)
        if self.mode == "mean":
            return points.reshape(-1, 2)
        return points[np.arange(len(points)), self.rng.integers(3, size=len(points))]


def mean_rows(feats):
    """Modo `mean`: junta as 3 linhas seguidas de cada superpixel, (3R, D) -> (R, D)."""
    return feats.reshape(-1, 3, feats.shape[1]).mean(1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import torch
    from spifil import LayerSpec
    from spifil.color import LabNorm
    from spifil.data import Sample
    from spifil.patches import extract_patches
    from spifil.preparation import prepare_sample
    from spifil.superpixels.disf import DISF

    from flim.spifil import Cropped, rescale

    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/dados/home/moliveira/datasets/organized/flowers/orig/000001_00000001.jpg")
    data = Cropped([Sample(path.stem, path, None, 1)], 200, use_mask=False)

    def seeds(extractor):
        return prepare_sample(data, 0, superpixels=DISF(), seed_extractor=extractor,
                              n_superpixels=50, color=LabNorm())

    today, off = seeds(Medoids()), seeds(SeedPoints())
    assert today.seeds.coords.equal(off.seeds.coords)
    n = len(today.seeds)
    print(f"{path.name}: desligado = Medoids, {n} marcadores iguais, mesma ordem")

    both = seeds(SeedPoints("mean"))
    triples = both.seeds.coords.numpy().reshape(-1, 3, 2)
    assert len(triples) == n and (triples[:, 0] == today.seeds.coords.numpy()).all()
    per = np.array([len(np.unique(t, axis=0)) for t in triples])
    union = seeds(SeedPoints("union"))
    assert per.max() <= 3 and len(union.seeds) <= per.sum()
    print(f"union: {len(union.seeds)} pontos de {n} superpixels; superpixels com 1/2/3 "
          f"candidatos: {[int((per == k).sum()) for k in (1, 2, 3)]}")
    print(f"mesmo pixel: {3 * n - per.sum()} de {3 * n} dentro do superpixel, "
          f"{per.sum() - len(union.seeds)} entre superpixels vizinhos")
    for side in (99, 49, 24):
        moved = rescale(union.seeds, (side, side)).coords
        print(f"  grade {side}: {len(moved)} pontos em {len(moved.unique(dim=0))} posicoes")

    a, b, c = (seeds(SeedPoints("random", s)).seeds.coords for s in (42, 42, 7))
    assert a.equal(b) and len(a) == n and not a.equal(c)
    assert all((t == p).all(1).any() for t, p in zip(triples, a.numpy()))
    print(f"random: {n} pontos, mesma semente repete, semente diferente muda")

    spec = LayerSpec(kernel_size=5, out_channels=8)
    image = torch.as_tensor(both.features)
    rows = extract_patches(image, both.seeds, spec).feats
    assert rows[::3].equal(extract_patches(image, today.seeds, spec).feats)
    assert mean_rows(rows).shape == (n, rows.shape[1])
    print(f"mean: {len(rows)} patches viram {n}, um por superpixel")
