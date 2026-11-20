"""SPiFiL Fisher-scorer OOM fallback (spifil_oom) + fit_dense encode skip: bit-identical results."""
import numpy as np
import pytest
import torch
from torch import nn

from ga_search.modehb import spifil_oom as O
from ga_search.modehb import spifil_ops as so

SIZE = 32
CFG = {"device": "cpu", "image_size": SIZE, "n_superpixels": 30, "impurities": True}


class _P:
    def __init__(self, feats, labels):
        self.feats, self.labels = feats, labels


def _patches(n=700, d=20, ncls=4, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, ncls, (n,), generator=g)
    feats = torch.randn(n, d, generator=g) + 0.4 * labels[:, None].float()
    labels[:3] = ncls + 1                       # a singleton class -> cnt_intra == 0 branch
    labels[3] = ncls + 1
    return _P(feats.to(dev), labels.to(dev))


def _metrics():
    from spifil.metrics import Euclidean, Mahalanobis
    return {"maha": lambda: Mahalanobis(chunk_size=128), "eucl": lambda: Euclidean(chunk_size=128),
            "maha_empirical_singular": lambda: Mahalanobis("empirical", chunk_size=128)}


@pytest.mark.parametrize("name", list(_metrics()))
@pytest.mark.parametrize("sub_rows", [1024, 50])
def test_blocked_is_bitwise_identical(monkeypatch, name, sub_rows):
    from spifil.scoring import FisherScorer
    monkeypatch.setattr(O, "SUB_ROWS", sub_rows)
    p = _patches(d=900 if "singular" in name else 20)   # N <= D -> empirical falls back to Euclidean
    make = _metrics()[name]
    with pytest.warns(UserWarning) if "singular" in name else _nullctx():
        ref = (O._ORIG or FisherScorer.__call__)(FisherScorer(), p, make())
    with pytest.warns(UserWarning) if "singular" in name else _nullctx():
        got = O.fisher_blocked(p, make())
    assert torch.equal(ref, got)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_install_idempotent_and_no_oom_is_original():
    from spifil.metrics import Mahalanobis
    from spifil.scoring import FisherScorer
    O.install()
    first = FisherScorer.__call__
    O.install()
    assert FisherScorer.__call__ is first and getattr(first, "_modehb_oom_fallback", False)
    p, before = _patches(), dict(O.STATS)
    assert torch.equal(FisherScorer()(p, Mahalanobis()), O._ORIG(FisherScorer(), p, Mahalanobis()))
    assert O.STATS == before


def test_oom_falls_back_identically_and_other_errors_propagate(monkeypatch):
    from spifil.metrics import Mahalanobis
    from spifil.scoring import FisherScorer
    O.install()
    p = _patches()
    ref = O._ORIG(FisherScorer(), p, Mahalanobis())
    orig = O._ORIG

    def oom(self, patches, metric):
        metric.fit(patches.feats)
        raise RuntimeError("CUDA out of memory. Tried to allocate 12.96 GiB")
    monkeypatch.setattr(O, "_ORIG", oom)
    n0 = O.STATS["fallbacks"]
    assert torch.equal(FisherScorer()(p, Mahalanobis()), ref)
    assert O.STATS["fallbacks"] == n0 + 1

    def other(self, patches, metric):
        raise RuntimeError("shape mismatch")
    monkeypatch.setattr(O, "_ORIG", other)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        FisherScorer()(p, Mahalanobis())
    monkeypatch.setattr(O, "_ORIG", orig)


def test_is_oom_wrapped():
    try:
        try:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        except Exception as e:
            raise ValueError("SPiFiL fit failed") from e
    except ValueError as w:
        assert O.is_oom(w)
    assert not O.is_oom(RuntimeError("index out of range"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_blocked_and_cpu_second_fallback(monkeypatch):
    from spifil.metrics import Mahalanobis
    from spifil.scoring import FisherScorer
    O.install()
    p = _patches(n=3000, d=27, dev="cuda")
    ref = O._ORIG(FisherScorer(), p, Mahalanobis())
    assert torch.equal(O.fisher_blocked(p, Mahalanobis()), ref)       # GPU blocked: bitwise

    def oom(self, patches, metric):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    real = O.fisher_blocked

    def blocked(patches, metric, device=None):
        if device is None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(patches, metric, device)
    monkeypatch.setattr(O, "_ORIG", oom)
    monkeypatch.setattr(O, "fisher_blocked", blocked)
    m = Mahalanobis()
    got = FisherScorer()(p, m)
    assert got.device.type == "cuda" and m._whitener.device.type == "cuda"
    rel = ((got - ref).abs() / ref.abs()).max().item()
    assert rel < 1e-4                                                  # CPU: fp rounding only


# ─── fit_dense skips the (unused) library encode, weights unchanged ───────────

@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    from skimage.io import imsave
    from spifil.data import Sample
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    out = []
    for i in range(6):
        lab = i % 2 + 1
        img = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
        if lab == 2:
            img[:, :, 0] = np.linspace(0, 255, 40)[None].astype(np.uint8)
        p = d / f"{lab:06d}_{i:08d}.png"
        imsave(p, img, check_contrast=False)
        out.append(Sample(p.stem, p, None, lab))
    return out


def _layer(**kw):
    base = dict(active=True, channels=8, kernel=3, conv="conv", residual=False,
                activation="relu", norm="none", pool="none", softhebb=False)
    base.update(kw)
    return base


@pytest.mark.parametrize("L", [_layer(kernel=7, pool="avg"), _layer(conv="inception", kernel=5, channels=12),
                               _layer(conv="dwsep", channels=6), _layer(channels=1),
                               _layer(residual=True, channels=5)])
def test_skip_encode_is_bitwise_identical(monkeypatch, samples, L):
    spec = so.to_ops_spec(L, 3)
    block, prov = so.fit_ops_block(None, samples, spec, {**CFG, "seed": 3})
    monkeypatch.setattr(so, "_SkipEncode", lambda: type("NoSkip", (), {})())   # library encode back on
    ref, prov_ref = so.fit_ops_block(None, samples, spec, {**CFG, "seed": 3})
    sa, sb = block.state_dict(), ref.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert prov["effective_spec"] == prov_ref["effective_spec"] and prov["stages"] == prov_ref["stages"]
