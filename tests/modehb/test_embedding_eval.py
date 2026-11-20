import math

import torch

from ga_search.modehb.embedding_eval import EmbeddingEvaluator, linear_cka


class _Log:
    def __init__(self):
        self.alerts = []

    def log(self, d, step=None):
        pass

    def alert(self, title, text, level):
        self.alerts.append((title, level))


def _ev(**kw):
    cfg = dict(cluster_k=4, cluster_n_init=2)
    cfg.update(kw)
    return EmbeddingEvaluator(cfg)


def test_isotropic_gaussian_high_rank_no_collapse():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 32, generator=g)
    views = torch.stack([x, x + 0.05 * torch.randn(512, 32, generator=g)])
    m = _ev().evaluate(views)
    assert m["rank_limit"] == 32
    assert m["effective_rank_ratio"] > 0.8
    assert m["rankme"] > 25
    assert not m["collapse_total"] and not m["collapse_dimensional"]
    assert m["frac_near_constant_dims"] == 0.0
    assert len(m["spectrum_topk"]) == 16


def test_rank1_dimensional_collapse():
    g = torch.Generator().manual_seed(1)
    u = torch.randn(512, 1, generator=g)
    w = torch.randn(1, 64, generator=g)
    x = u @ w
    m = _ev().evaluate(torch.stack([x, x]))
    assert m["effective_rank_ratio"] < 0.05
    assert m["rankme"] < 2.0
    assert m["collapse_dimensional"] and not m["collapse_total"]
    log = _Log()
    titles = [a["title"] for a in _ev().alerts(m, log)]
    assert "dimensional_collapse" in titles and log.alerts


def test_constant_total_collapse():
    x = torch.ones(256, 16) * 3.0
    m = _ev().evaluate(torch.stack([x, x]))
    assert m["collapse_total"] and m["collapse_dimensional"]
    assert m["quality"] == 0.0
    assert any(a["title"] == "total_collapse" for a in _ev().alerts(m, None))


def test_identical_views_consistency_one():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(300, 24, generator=g)
    m = _ev().evaluate(torch.stack([x, x.clone()]))
    assert math.isclose(m["knn_view_consistency"], 1.0)
    assert math.isclose(m["clustering_stability"], 1.0)
    assert math.isclose(m["quality"], 1.0)
    assert m["alignment"] < 1e-12


def test_cka_self_is_one_and_prev_other_init():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(200, 20, generator=g)
    assert math.isclose(linear_cka(x, x), 1.0, rel_tol=1e-9)
    assert math.isclose(linear_cka(x, 2 * x + 1), 1.0, rel_tol=1e-9)
    y = torch.randn(200, 20, generator=g)
    m = _ev().evaluate(torch.stack([x, x]), prev=x, other_init=y)
    assert math.isclose(m["cka_prev"], 1.0, rel_tol=1e-9)
    assert m["cka_other_init"] < 0.3
    assert abs(m["cos_other_init"]) < 0.2


def test_quality_invariant_to_loss():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(200, 16, generator=g)
    m = _ev().evaluate(torch.stack([x, x + 0.1 * torch.randn(200, 16, generator=g)]))
    q0 = EmbeddingEvaluator.quality(m)
    for loss in (0.0, 1e9, float("nan"), -5.0):
        assert EmbeddingEvaluator.quality({**m, "loss": loss, "val/loss": loss}) == q0
    assert 0.0 <= q0 <= 1.0


def test_spectral_every_and_alerts_history():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(128, 8, generator=g)
    ev = _ev(spectral_every=3)
    flags = [ev.evaluate(torch.stack([x, x]))["spectral_fresh"] for _ in range(4)]
    assert flags == [True, False, False, True]
    log = _Log()
    titles = [a["title"] for a in ev.alerts(
        {"var_mean": 1.0, "effective_rank_ratio": 0.9, "loss": 100.0, "quality": 0.5,
         "nonfinite_grads": True, "mean_abs_offdiag_corr": 0.95},
        log, history={"loss": [1.0, 1.0, 1.0, 1.0, 1.0], "quality": [0.5] * 5})]
    for t in ("loss_divergence", "non_finite", "excessive_redundancy", "quality_saturation"):
        assert t in titles
    assert len(log.alerts) == len(titles)


def test_lowrank_path_large_dim():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(256, 300, generator=g)
    m = _ev(max_full_dim=100, lowrank_q=64).evaluate(torch.stack([x, x]))
    assert math.isfinite(m["rankme"]) and m["rank_limit"] == 256
