"""growth v3.1 bloco 6: bootstrap pareado da val na aceitação (opção), época do AE = passada completa,
SPiFiL no pointwise novo do dwsep, svm_fitness com predições opt-in (SVM inalterada)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402


def test_kappa_np_equals_core_metrics():
    from core.metrics import compute_metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    p = np.where(rng.random(200) < 0.6, y, rng.integers(0, 5, 200))
    assert G._kappa_np(y, p, 5) == pytest.approx(compute_metrics(y, p, num_classes=5)["kappa"], abs=1e-6)


def test_bootstrap_delta_kappa():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 300)
    par = np.where(rng.random(300) < 0.5, y, rng.integers(0, 4, 300))
    good = np.where(rng.random(300) < 0.9, y, rng.integers(0, 4, 300))
    b = G.bootstrap_delta_kappa(par, good, y, n=300, seed=0)
    assert b["accept_ok"] and b["delta_kappa_ci"][0] > 0 and b["split"] == "validation"
    same = G.bootstrap_delta_kappa(par, par, y, n=100, seed=0)
    assert same["delta_kappa_point"] == 0.0 and same["accept_ok"]          # empate exato: Δ = 0 sempre
    noisy = np.where(rng.random(300) < 0.52, y, rng.integers(0, 4, 300))
    assert G.bootstrap_delta_kappa(par, noisy, y, n=300, seed=0)["accept_ok"] is False
    assert G.bootstrap_delta_kappa(par, good, y, n=300, seed=0) == b              # determinístico


def test_decide_with_bootstrap():
    r = {"status": "ok", "wise_alpha": 0.5, "kappa_val_orig": 0.61, "kappa_parent_rescored": 0.60,
         "preservation_verdict": "exact", "bootstrap": {"accept_ok": False}}
    assert G.decide(r, 0.0, False) == ("rejected_not_significant", False)
    r["bootstrap"]["accept_ok"] = True
    assert G.decide(r, 0.0, False) == ("accepted", True)
    r.pop("bootstrap")
    assert G.decide(r, 0.0, False) == ("accepted", True)                     # sem a opção: regra da v3


def test_svm_fitness_return_pred_same_kappa():
    from ga_search.modehb.supervised_fitness import svm_fitness
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(60, 8)), rng.integers(0, 3, 60)
    a = svm_fitness(X[:40], y[:40], X[40:], y[40:], C=100.0, scaler="none", kappa_weight=1.0, max_train=None)
    b = svm_fitness(X[:40], y[:40], X[40:], y[40:], C=100.0, scaler="none", kappa_weight=1.0, max_train=None,
                    return_pred=True)
    assert a["kappa"] == b["kappa"] and "pred" not in a and len(b["pred"]) == 20


def test_ssl_loader_full_epoch():
    from torch.utils.data import DataLoader, TensorDataset
    from types import SimpleNamespace
    ds = TensorDataset(torch.rand(50, 3, 8, 8))
    ev = SimpleNamespace(train_loader=DataLoader(ds, batch_size=16))
    ld = G._ssl_loader(ev, 16, 0)
    assert len(ld) == 3 and ld.batch_size == 16 and ld.drop_last
