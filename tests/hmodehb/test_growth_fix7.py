"""growth_fix7: SVMs do WiSE em paralelo (score_many) == sequencial; cache do re-score do pai; SIGUSR1."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.modehb.supervised_fitness import svm_fitness  # noqa: E402


class _FakeEv:
    """Só o que score/score_many/score_parent usam do HEvaluator; a SVM é a do repo (libsvm, C=100)."""

    def __init__(self, ytr, yva, workers=5):
        self.svm_train_y, self.val_y = ytr, yva
        self._readouts, self._svm_stamps = [], []
        self.workers = workers
        self.n_features = 0
        self.cfg = {"data": {}}

    def _fitness_speed(self, n_alphas=1):
        return {"svm_workers": max(1, min(n_alphas, self.workers)), "svm_blas_threads": 1}

    def _svm_workers_for(self, workers, tr):
        return workers

    def _svm_features(self, enc, head):
        self.n_features += 1
        self._readouts.append("head")
        return self._feats

    def _svm_score(self, tr, va):
        self._svm_stamps.append({"C": 100.0, "scaler": "none", "whiten": "none", "kappa_weight": 1.0,
                                 "max_train": 4000})
        return svm_fitness(tr, self.svm_train_y, va, self.val_y, C=100.0, scaler="none", whiten="none",
                           kappa_weight=1.0, max_train=4000, seed=42)


def _gev(ev):
    g = G.GrowthEvaluator.__new__(G.GrowthEvaluator)
    g.ev, g.embed_precision, g.fitness_sig = ev, "fp32_noautocast_notf32", "sig"
    import torch
    g.device = torch.device("cpu")
    return g


def _data(seed, n=600, nv=200, d=256, k=8):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(k, d)) * 0.6
    ytr, yva = rng.integers(0, k, n), rng.integers(0, k, nv)
    tr = (mu[ytr] + rng.normal(size=(n, d))).astype(np.float32)
    va = (mu[yva] + rng.normal(size=(nv, d))).astype(np.float32)
    return tr, va, ytr, yva


def _strip(sv):
    return {k: v for k, v in sv.items() if k != "fit_s"}


def test_score_many_equals_sequential():
    tr, va, ytr, yva = _data(0)
    ev = _FakeEv(ytr, yva)
    g = _gev(ev)
    feats = [(tr * (1 + 0.05 * i) + 0.01 * i, va * (1 + 0.05 * i) + 0.01 * i) for i in range(5)]
    seq = [g.svm(f) for f in feats]
    par = g.score_many(feats)
    assert [_strip(a) for a in seq] == [_strip(b) for b in par]       # kappa/f1/quality bit a bit
    assert len(ev._svm_stamps) == 10


def test_score_many_runs_concurrently():
    """libsvm solta o GIL: 5 fits em threads levam bem menos que 5x um fit."""
    tr, va, ytr, yva = _data(1, n=1500, nv=300, d=512, k=20)
    ev = _FakeEv(ytr, yva)
    g = _gev(ev)
    feats = [(tr + 0.001 * i, va + 0.001 * i) for i in range(5)]
    t0 = time.perf_counter()
    [g.svm(f) for f in feats]
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    g.score_many(feats)
    t_par = time.perf_counter() - t0
    assert t_par < 0.75 * t_seq, (t_seq, t_par)


def test_workers_one_is_sequential():
    tr, va, ytr, yva = _data(2)
    ev = _FakeEv(ytr, yva, workers=1)
    g = _gev(ev)
    feats = [(tr, va), (tr * 1.1, va * 1.1)]
    assert [_strip(a) for a in g.score_many(feats)] == [_strip(g.svm(f)) for f in feats]


def test_wise_select_parallel_matches_sequential(monkeypatch):
    """_wise_select (fix7) dá o mesmo alpha/kappas que o laço sequencial antigo (score por alpha)."""
    import torch
    tr, va, ytr, yva = _data(3)
    ev = _FakeEv(ytr, yva)
    g = _gev(ev)
    blend = {"a": 0.0}

    def fake_blend(model, anchor, trained, a):
        blend["a"] = a

    def fake_features(model, embed_batch_size=None):
        a = blend["a"]
        return (tr * (1 + a) + a * np.sin(np.arange(tr.shape[1], dtype=np.float32)), va * (1 + a)
                + a * np.sin(np.arange(va.shape[1], dtype=np.float32)))

    monkeypatch.setattr(G, "_blend_into", fake_blend)
    monkeypatch.setattr(g, "features", fake_features)
    model = torch.nn.Linear(1, 1)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best, kappas = G._wise_select(g, model, {}, {}, True, {"wise_alphas": alphas},
                                  {"embed_batch_size": None, "empty_cache": False})
    ref_best, ref = None, {}
    for a in alphas:                                   # laço sequencial pré-fix7
        blend["a"] = a
        sv = g.svm(fake_features(model))
        ref[str(a)] = float(sv["kappa"])
        if ref_best is None or sv["quality"] > ref_best[1]["quality"]:
            ref_best = (a, sv)
    assert kappas == ref and best[0] == ref_best[0] and _strip(best[1]) == _strip(ref_best[1])


def test_parent_rescore_cached():
    import torch
    tr, va, ytr, yva = _data(4)
    ev = _FakeEv(ytr, yva)
    ev._feats = (tr, va)
    g = _gev(ev)
    model = type("M", (), {})()
    model.blocks = [torch.nn.Identity()]
    model.head = torch.nn.Identity()
    model.to = lambda dev: model
    sv1, r1 = g.score_parent(model, "sha-A")
    sv2, r2 = g.score_parent(model, "sha-A")
    assert ev.n_features == 1                          # 2ª chamada não re-extrai nem re-ajusta
    assert _strip({k: v for k, v in sv1.items() if k != "parent_rescore_cached"}) == \
        _strip({k: v for k, v in sv2.items() if k != "parent_rescore_cached"})
    assert r1 == r2 and sv2["parent_rescore_cached"] and not sv1["parent_rescore_cached"]
    g.score_parent(model, "sha-B")                     # outro pai -> re-pontua
    assert ev.n_features == 2


def test_sigusr1_dumps_stacks():
    import faulthandler
    import signal
    src = Path(G.__file__).read_text()
    assert "faulthandler.register(signal.SIGUSR1, all_threads=True" in src
