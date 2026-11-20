"""R-SPF readouts (gem_tta_pw / cat_tta_pw) of ga_search/modehb/svm_on_demand.py."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ga_search.modehb import svm_on_demand as S


def _encoder():
    torch.manual_seed(0)
    return nn.Sequential(nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()),
                         nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))).eval()


def _ref_gem_tta(enc, x):
    """discovery eval/svm.py l.332-334, written against nn.Sequential."""
    gem = lambda z: F.adaptive_avg_pool2d(z.clamp(min=1e-6).pow(3), 1).pow(1 / 3).flatten(1)
    return torch.stack([gem(enc(torch.rot90(v, k, (2, 3)))) for v in (x, x.flip(3)) for k in range(4)]).mean(0)


def test_gem_tta_matches_reference_and_is_rotation_invariant():
    enc, x = _encoder(), torch.rand(3, 3, 12, 12)
    with torch.no_grad():
        last = S._gem_tta_blocks(enc, x)[-1]
        torch.testing.assert_close(last, _ref_gem_tta(enc, x))
        torch.testing.assert_close(S._gem_tta_blocks(enc, torch.rot90(x, 1, (2, 3)))[-1], last, rtol=1e-4, atol=1e-5)


def test_pw_whiten_matches_formula():
    rng = np.random.default_rng(0)
    tr, te = rng.normal(size=(40, 6)), rng.normal(size=(10, 6))
    a, b = S.pw_whiten(tr, te)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-6)
    from sklearn.decomposition import PCA
    p = PCA().fit(tr)
    w = (p.explained_variance_ + 0.1 * p.explained_variance_.mean()) ** -0.5
    ref = p.transform(te) * w
    np.testing.assert_allclose(b, ref / np.linalg.norm(ref, axis=1, keepdims=True))


def test_cat_select():
    assert S.cat_select([64, 128, 256]) == [0, 1, 2]
    assert S.cat_select([500, 1200, 600]) == [1, 2]
    assert S.cat_select([64, 1900, 600]) == []
    assert S.cat_select([2500]) == []


def test_tta_features_shapes():
    enc = _encoder()
    loader = [(torch.rand(5, 3, 12, 12), torch.arange(5))]
    X, y = S._tta_features(enc, loader, "cpu", "gem_tta_pw", chunk=2)
    assert X.shape == (5, 16) and y.tolist() == list(range(5))
    X, _ = S._tta_features(enc, loader, "cpu", "cat_tta_pw")
    assert X.shape == (5, 24)


def test_cov_head_tta_variants():
    from ga_search.modehb.pooling_head import EmbeddingHead
    enc, head = _encoder(), EmbeddingHead(16, 1280, "gap").eval()
    data = {s: [(torch.rand(6, 3, 12, 12), torch.arange(6))] for s in ("train", "validation")}
    base = S._cov_features(enc, head, data, "cpu", k=4)
    tta = S._cov_features(enc, head, data, "cpu", k=4, tta=8, chunk=4)
    raw = S._cov_features(enc, head, data, "cpu", k=4, tta=8, pn=False)
    assert base[0].shape == tta[0].shape == raw[0].shape
    assert not np.allclose(base[0], tta[0]) and not np.allclose(tta[0], raw[0])
    one = S._cov_features(enc, head, data, "cpu", k=4, tta=1, chunk=2)
    np.testing.assert_allclose(one[0], base[0], rtol=1e-5, atol=1e-6)


def test_cov_fit_micro_same_pixels():
    from ga_search.modehb.pooling_head import CovReadout
    enc, x = _encoder(), torch.rand(10, 3, 12, 12)
    a = CovReadout(k=4, n_img=10, n_px=20).fit(enc, x, "cpu", bs=4)
    b = CovReadout(k=4, n_img=10, n_px=20).fit(enc, x, "cpu", bs=4, micro=1)
    for (ma, va), (mb, vb) in zip(a.proj, b.proj):
        torch.testing.assert_close(ma, mb)
        torch.testing.assert_close(va.abs(), vb.abs(), rtol=1e-4, atol=1e-5)
