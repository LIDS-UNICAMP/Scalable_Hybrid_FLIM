import numpy as np
import pytest
import torch

from ga_search.modehb.supervised_fitness import stratified_subsample, svm_fitness

KEYS = {"kappa", "macro_f1", "balanced_acc", "quality", "fit_s", "n_train_used"}


def _blobs(n_per, k=3, d=8, sep=20.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(size=(n_per, d)) + sep * c for c in range(k)])
    y = np.repeat(np.arange(k), n_per)
    return X.astype(np.float32), y


def test_separable_kappa_one():
    Xtr, ytr = _blobs(40)
    Xva, yva = _blobs(20, seed=1)
    r = svm_fitness(Xtr, ytr + 1, Xva, yva + 1)  # 1-indexed labels are fine
    assert KEYS <= r.keys()
    assert r["kappa"] > 0.99 and r["macro_f1"] > 0.99 and r["quality"] > 0.99


def test_torch_input_and_standard_scaler():
    Xtr, ytr = _blobs(30)
    Xva, yva = _blobs(10, seed=1)
    r = svm_fitness(torch.from_numpy(Xtr), torch.from_numpy(ytr),
                    torch.from_numpy(Xva), torch.from_numpy(yva), scaler="standard")
    assert r["kappa"] > 0.99


def test_random_labels_kappa_near_zero():
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(size=(300, 16)), rng.normal(size=(600, 16))
    ytr, yva = rng.integers(0, 3, 300), rng.integers(0, 3, 600)
    r = svm_fitness(Xtr, ytr, Xva, yva, C=1.0)
    assert abs(r["kappa"]) < 0.12
    assert r["quality"] < 0.3


def test_stratified_subsample():
    y = np.array([0] * 800 + [1] * 150 + [2] * 50)
    idx = stratified_subsample(y, 100, seed=3)
    assert len(idx) == 100 and len(np.unique(idx)) == 100
    counts = np.bincount(y[idx])
    assert tuple(counts) == (80, 15, 5)
    assert np.array_equal(idx, stratified_subsample(y, 100, seed=3))
    assert not np.array_equal(idx, stratified_subsample(y, 100, seed=4))


def test_subsample_applied_validation_untouched():
    Xtr, ytr = _blobs(100)
    Xva, yva = _blobs(50, seed=1)
    r = svm_fitness(Xtr, ytr, Xva, yva, max_train=60)
    assert r["n_train_used"] == 60
    r = svm_fitness(Xtr, ytr, Xva, yva, max_train=None)
    assert r["n_train_used"] == 300


def test_quality_formula(monkeypatch):
    import core.metrics as cm
    Xtr, ytr = _blobs(10)
    for kappa, f1, w, expect in [(0.6, 0.8, 0.75, 0.75 * 0.6 + 0.25 * 0.8),
                                 (-0.2, 0.4, 0.75, 0.25 * 0.4),
                                 (0.5, 0.9, 0.5, 0.7)]:
        monkeypatch.setattr(cm, "compute_metrics",
                            lambda *a, _k=kappa, _f=f1, **kw: {"kappa": _k, "f1": _f, "acc": 0})
        r = svm_fitness(Xtr, ytr, Xtr, ytr, kappa_weight=w)
        assert r["quality"] == pytest.approx(expect)


def test_nonfinite_gives_zero():
    Xtr, ytr = _blobs(10)
    Xbad = Xtr.copy()
    Xbad[0, 0] = np.nan
    r = svm_fitness(Xbad, ytr, Xtr, ytr)
    assert r["quality"] == 0.0 and r["nonfinite"] is True
    Xbad[0, 0] = np.inf
    r = svm_fitness(Xtr, ytr, Xbad, ytr)
    assert r["quality"] == 0.0 and r["nonfinite"] is True
