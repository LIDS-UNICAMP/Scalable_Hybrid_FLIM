"""Tests for ga_search/modehb/ssl_trainers.py (CPU, tiny)."""
import math
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ga_search.modehb.ssl_trainers import (
    make_views, method_name, nt_xent, split_param_groups, train_global_ssl,
)


# --------------------------------------------------------------------- fixtures
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
            nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class TinyHead(nn.Module):
    def __init__(self, in_ch=6, d=8):
        super().__init__()
        self.fc = nn.Linear(in_ch, d)
        self.out_dim = d

    def forward(self, f):
        return self.fc(f.mean(dim=(2, 3))) * 5.0   # deliberately not unit-norm


def _head():
    try:
        from ga_search.modehb.pooling_head import EmbeddingHead
        return EmbeddingHead(6, 1280, "gap")
    except Exception:
        return TinyHead()


def _loader(n=16, bs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 16, 16, generator=g)
    y = torch.zeros(n, dtype=torch.long)
    return [(x[i:i + bs], y[i:i + bs]) for i in range(0, n, bs)]


class RecordingReg(nn.Module):
    calls = []

    def forward(self, z):
        RecordingReg.calls.append(z)
        return (z.mean(0).square().mean() + (z.std(0) - 1).square().mean())


@pytest.fixture
def fake_regularizers(monkeypatch):
    mod = types.ModuleType("ga_search.modehb.regularizers")
    RecordingReg.calls = []
    mod.build_regularizer = lambda name, **kw: None if name == "none" else RecordingReg()
    monkeypatch.setitem(sys.modules, "ga_search.modehb.regularizers", mod)
    return mod


class ListLogger:
    def __init__(self):
        self.records, self.alerts = [], []

    def log(self, d, step=None):
        self.records.append((step, dict(d)))

    def alert(self, title, text, level=None):
        self.alerts.append((title, text, level))


# --------------------------------------------------------------------- nt_xent
def _naive_nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n2 = z.shape[0]
    n = n2 // 2
    total = 0.0
    for i in range(n2):
        j = (i + n) % n2
        num = math.exp(float(z[i] @ z[j]) / tau)
        den = sum(math.exp(float(z[i] @ z[k]) / tau) for k in range(n2) if k != i)
        total += -math.log(num / den)
    return total / n2


@pytest.mark.parametrize("tau", [0.1, 0.2, 0.5])
def test_nt_xent_matches_naive(tau):
    torch.manual_seed(0)
    z1, z2 = torch.randn(5, 7), torch.randn(5, 7)
    assert float(nt_xent(z1, z2, tau)) == pytest.approx(_naive_nt_xent(z1, z2, tau), rel=1e-5)


def test_nt_xent_scale_invariant_and_low_for_identical():
    torch.manual_seed(1)
    z = torch.randn(6, 4)
    assert float(nt_xent(z, z * 3.0)) == pytest.approx(float(nt_xent(z, z)), rel=1e-5)
    assert float(nt_xent(z, z)) < float(nt_xent(z, torch.randn(6, 4)))


# --------------------------------------------------------------------- views
def test_make_views_deterministic_and_range():
    x = torch.rand(4, 3, 16, 16)
    a = make_views(x, 3, seed=7)
    b = make_views(x, 3, seed=7)
    c = make_views(x, 3, seed=8)
    assert len(a) == 3 and a[0].shape == x.shape
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not torch.equal(a[0], c[0])
    assert not torch.equal(a[0], a[1])
    assert all(v.min() >= 0 and v.max() <= 1 for v in a)


# --------------------------------------------------------------------- naming
def test_method_names():
    assert method_name("lejepa", "sigreg") == "lejepa"
    assert method_name("lejepa", "weak_sigreg") == "lejepa_weak"
    assert method_name("simclr", "none") == "simclr"
    assert method_name("simclr", "weak_sigreg") == "simclr+weak_sigreg"
    with pytest.raises(ValueError):
        method_name("lejepa", "none")
    with pytest.raises(ValueError):
        method_name("swav", "none")          # growth v3.1: byol passou a existir (bloco 7)


# --------------------------------------------------------------------- param groups
def test_param_groups_lrs():
    enc, head, proj = TinyEncoder(), TinyHead(), nn.Linear(8, 4)
    groups = split_param_groups(enc, head, [proj], ["layers.1"], lr_old=1e-4, lr_new=1e-3)
    by = {g["name"]: g for g in groups}
    assert by["old"]["lr"] == 1e-4 and by["new"]["lr"] == 1e-3
    old_ids = {id(p) for p in by["old"]["params"]}
    new_ids = {id(p) for p in by["new"]["params"]}
    assert old_ids == {id(p) for p in enc.layers[0].parameters()}
    expected_new = (list(enc.layers[1].parameters()) + list(head.parameters())
                    + list(proj.parameters()))
    assert new_ids == {id(p) for p in expected_new}
    assert not old_ids & new_ids


def test_train_uses_two_fixed_lrs(fake_regularizers):
    log = ListLogger()
    train_global_ssl(TinyEncoder(), TinyHead(), _loader(), "simclr", "none", epochs=2,
                     lr_old=1e-4, lr_new=2e-3, new_param_names=["layers.1"], device="cpu",
                     logger=log, log_every=1)
    lrs = [(r["lr/old"], r["lr/new"]) for _, r in log.records if "lr/old" in r]
    assert lrs and all(l == (1e-4, 2e-3) for l in lrs)


def test_frozen_params_get_unfrozen(fake_regularizers):
    enc = TinyEncoder()
    for p in enc.parameters():
        p.requires_grad_(False)
    w0 = enc.layers[0][0].weight.clone()
    train_global_ssl(enc, TinyHead(), _loader(), "simclr", "none", epochs=1, lr_old=1e-2,
                     lr_new=1e-2, new_param_names=["layers.1"], device="cpu")
    assert all(p.requires_grad for p in enc.parameters())
    assert not torch.equal(w0, enc.layers[0][0].weight)


# --------------------------------------------------------------------- reg pre-L2
def test_simclr_regularizer_on_head_embedding_pre_l2(fake_regularizers):
    head = TinyHead(d=8)
    train_global_ssl(TinyEncoder(), head, _loader(n=8), "simclr", "weak_sigreg", epochs=1,
                     lr_old=1e-3, lr_new=1e-3, new_param_names=["layers.1"], device="cpu")
    assert len(RecordingReg.calls) == 2           # one per view
    for z in RecordingReg.calls:
        assert z.shape == (8, head.out_dim)       # head dim, not projector dim (64)
        norms = z.detach().norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
        assert z.requires_grad                    # part of the graph


# --------------------------------------------------------------------- training
@pytest.mark.parametrize("ssl,reg", [("simclr", "none"), ("simclr", "sigreg"),
                                     ("lejepa", "sigreg"), ("lejepa", "weak_sigreg"),
                                     ("ae", "none")])
def test_tiny_training_finite_and_logs(fake_regularizers, ssl, reg):
    log = ListLogger()
    res = train_global_ssl(TinyEncoder(), _head(), _loader(), ssl, reg, epochs=2,
                           lr_old=1e-4, lr_new=1e-3, new_param_names=["layers.1"],
                           device="cpu", logger=log, log_every=1, seed=0)
    assert res["method"] == method_name(ssl, reg)
    assert math.isfinite(res["final_loss_total"])
    assert len(res["history"]) == 2
    rec = log.records[0][1]
    assert "ssl/loss_total" in rec and "perf/throughput_img_s" in rec
    assert "perf/dataloader_wait_s" in rec
    assert any(k.startswith("grad_norm/encoder/layers.0") for k in rec)
    assert any(k.startswith("weight_norm/") for k in rec)
    if ssl == "simclr":
        assert "ssl/loss_simclr" in rec and "ssl/pos_sim" in rec and "ssl/neg_sim" in rec
    if ssl == "lejepa":
        assert "ssl/loss_inv" in rec
        assert ("ssl/loss_sigreg" if reg == "sigreg" else "ssl/loss_weak_sigreg") in rec
    assert any("perf/epoch_time_s" in r for _, r in log.records)
    assert not log.alerts


def test_simclr_loss_decreases(fake_regularizers):
    torch.manual_seed(0)
    res = train_global_ssl(TinyEncoder(), TinyHead(), _loader(n=32, bs=16), "simclr", "none",
                           epochs=15, lr_old=3e-3, lr_new=3e-3, new_param_names=["layers.1"],
                           device="cpu", seed=0)
    h = [e["ssl/loss_total"] for e in res["history"]]
    assert all(math.isfinite(v) for v in h)
    assert min(h[-3:]) < h[0]


def test_deterministic_under_seed(fake_regularizers):
    def run():
        torch.manual_seed(123)
        enc, head = TinyEncoder(), TinyHead()
        return train_global_ssl(enc, head, _loader(), "lejepa", "weak_sigreg", epochs=1,
                                lr_old=1e-3, lr_new=1e-3, new_param_names=["layers.1"],
                                device="cpu", seed=3)["final_loss_total"]
    # RealSIGReg not involved here; weak reg mock is deterministic.
    assert run() == pytest.approx(run(), rel=1e-6)


def test_nan_raises_and_alerts(fake_regularizers):
    class NanHead(TinyHead):
        def forward(self, f):
            return super().forward(f) * float("nan")
    log = ListLogger()   # default: non-finite steps skipped + alerted, no crash
    res = train_global_ssl(TinyEncoder(), NanHead(), _loader(), "simclr", "none", epochs=1,
                           lr_old=1e-3, lr_new=1e-3, new_param_names=["layers.1"], device="cpu", logger=log)
    assert res["skipped_steps"] > 0 and log.alerts
    log = ListLogger()
    with pytest.raises(FloatingPointError):
        train_global_ssl(TinyEncoder(), NanHead(), _loader(), "simclr", "none", epochs=1,
                         lr_old=1e-3, lr_new=1e-3, new_param_names=["layers.1"],
                         device="cpu", logger=log, max_skip_frac=0.5)
    assert log.alerts and log.alerts[-1][2] == "ERROR"


def test_logger_none_ok(fake_regularizers):
    res = train_global_ssl(TinyEncoder(), TinyHead(), _loader(n=8), "ae", "none", epochs=1,
                           lr_old=1e-3, lr_new=1e-3, new_param_names=[], device="cpu")
    assert res["method"] == "ae" and math.isfinite(res["final_loss_total"])
