"""Testes de eval_backbone_head.py (CPU, sem dados reais).

    CUDA_VISIBLE_DEVICES="" python -m pytest -q research/spifil_b0b4/readout_x/test_eval_backbone_head.py
    (ou: python research/spifil_b0b4/readout_x/test_eval_backbone_head.py)

Cobre: dimensoes exatas 1280/4096 (so2 e cross-block, inclusive com canais < k), PCA so com
treino (o ajuste nao muda quando val/teste mudam; o fit so recebe o loader de treino),
determinismo (bitwise entre execucoes e independente do batch size).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_backbone_head as E  # noqa: E402


def _toy_model(chans=(8, 16, 96), seed=0):
    g = torch.Generator().manual_seed(seed)
    mods, cin = [], 3
    for c in chans:
        conv = torch.nn.Conv2d(cin, c, 3, padding=1)
        with torch.no_grad():
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * 0.3)
            conv.bias.zero_()
        mods.append(torch.nn.Sequential(conv, torch.nn.ReLU(), torch.nn.MaxPool2d(2)))
        cin = c
    mods = [m.eval() for m in mods]
    return E.Wrapped("toy", E._seq_blocks(mods),
                     lambda bl: torch.nn.functional.adaptive_avg_pool2d(bl[-1], 1).flatten(1),
                     mods, {})


class _DS(torch.utils.data.Dataset):
    def __init__(self, n, seed, n_cls=3):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 32, 32, generator=g)
        self.y = [i % n_cls for i in range(n)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


def _args(bs=4):
    return SimpleNamespace(batch_size=bs, num_workers=0, n_px=16)


def _splits(val_seed=2, test_seed=3):
    return {"train": _DS(12, 1), "validation": _DS(9, val_seed), "test": _DS(9, test_seed)}


def test_so2_dims_formula():
    assert E.so2_dim(E.SO2_K[1280]) == 1272 and E.so2_dim(E.SO2_K[1280] + 1) > 1280
    assert E.so2_dim(E.SO2_K[4096]) == 4092 and E.so2_dim(E.SO2_K[4096] + 1) > 4096


def test_exact_dims():
    feats, _, chans = E.extract(_toy_model(), _splits(), "cpu", _args(), log=lambda *a: None)
    want = {"gap1280": chans[-1], "so2_1280_pca": 1280, "so2_4096_pca": 4096,
            "so2_1280_rnd": 1280, "so2_4096_rnd": 4096, "xb_1280_pca": 1280, "xb_4096_pca": 4096}
    for s in ("train", "validation", "test"):
        assert set(feats[s]) == set(want)
        for h, d in want.items():
            assert feats[s][h].shape[1] == d, (s, h, feats[s][h].shape)
            assert feats[s][h].dtype == np.float32
            assert np.isfinite(feats[s][h]).all()
    # L2 = 1 nas leituras de 2a ordem
    for h in ("so2_1280_pca", "so2_4096_pca", "xb_1280_pca", "xb_4096_pca"):
        assert np.allclose(np.linalg.norm(feats["test"][h], axis=1), 1, atol=1e-5)


def test_exact_dims_last_block_wider_than_k():
    # ultimo bloco com 96 canais: k=48 cheio e k=88 cheio; 2 blocos so -> 1 par no cross-block
    feats, _, _ = E.extract(_toy_model((16, 96)), _splits(), "cpu", _args(), log=lambda *a: None)
    assert feats["test"]["so2_4096_pca"].shape[1] == 4096
    assert E.xb_kp([16, 96], 4096) == 64 and feats["test"]["xb_4096_pca"].shape[1] == 4096


def test_pca_only_train(monkeypatch=None):
    m = _toy_model()
    seen = []
    orig = E.Readouts.add_train

    def spy(self, blocks, ids):
        seen.append(blocks[0].shape[0])
        return orig(self, blocks, ids)

    E.Readouts.add_train = spy
    try:
        fa, _, _ = E.extract(m, _splits(2, 3), "cpu", _args(), log=lambda *a: None)
        fb, _, _ = E.extract(m, _splits(20, 30), "cpu", _args(), log=lambda *a: None)
    finally:
        E.Readouts.add_train = orig
    # o ajuste so viu as 12 imagens de treino (2 execucoes x 12)
    assert sum(seen) == 2 * 12
    # val/teste diferentes -> features de TREINO identicas (PCA nao depende de val/teste)
    for h in fa["train"]:
        assert np.array_equal(fa["train"][h], fb["train"][h]), h
    assert not np.array_equal(fa["test"]["so2_1280_pca"], fb["test"]["so2_1280_pca"])


def test_determinism_and_batch_independence():
    m = _toy_model()
    a, la, _ = E.extract(m, _splits(), "cpu", _args(4), log=lambda *a: None)
    b, lb, _ = E.extract(m, _splits(), "cpu", _args(4), log=lambda *a: None)
    c, _, _ = E.extract(m, _splits(), "cpu", _args(5), log=lambda *a: None)
    for s in a:
        assert np.array_equal(la[s], lb[s])
        for h in a[s]:
            assert np.array_equal(a[s][h], b[s][h]), (s, h)              # bitwise
            # outro batch: a soma fp64 dos momentos muda de ordem (~1e-16), o que pode girar
            # autovetores QUASE degenerados da cauda da PCA; leituras sem PCA saem iguais
            if not h.endswith("_pca"):
                assert np.allclose(a[s][h], c[s][h], atol=1e-5), (s, h)


def test_pixel_sampling_seeded_per_image():
    pm1, pm2 = E.PixelMoments(8, 0), E.PixelMoments(8, 0)
    f = torch.rand(6, 5, 4, 4)
    pm1.add(f, list(range(6)))
    pm2.add(f[:3], [0, 1, 2]); pm2.add(f[3:], [3, 4, 5])
    assert torch.equal(pm1.s1, pm2.s1) and torch.allclose(pm1.s2, pm2.s2)


def test_score_uses_repo_svm():
    feats, labels, _ = E.extract(_toy_model(), _splits(), "cpu", _args(), log=lambda *a: None)
    rows = E.score(feats, labels)
    for r in rows.values():
        assert r["svm_protocol"] == "scaler=none;kernel=linear;C=100;ovo;max_iter=-1"
        assert -1 <= r["kappa_test"] <= 1 and -1 <= r["kappa_val"] <= 1


if __name__ == "__main__":
    E._setup_path(Path(E.GROWTH_CODE if E.GROWTH_CODE.is_dir() else E.REPO))
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
