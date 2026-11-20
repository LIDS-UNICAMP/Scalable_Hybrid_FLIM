import torch

from research.lejepa_diag import readouts as R


def test_cov_shape_pca_train_only(model, data):
    enc, head = model
    (X, y), (Xv, yv) = data
    cr = R.fit_cov(enc, X, k=64, micro=4)
    proj = [(m.clone(), V.clone()) for m, V in cr.proj]
    assert R.layer_k_eff(cr) == {"L1": 8, "L2": 12, "L3": 16, "L4": 20}   # k_eff = min(k, C_l)
    Fv = R.cov_features(cr, enc, head, Xv, batch_size=4)
    d = sum(k * (k + 1) // 2 for k in (8, 12, 16, 20)) + 1280
    assert Fv.shape == (10, d)
    # PCA so no treino: avaliar validacao (qualquer) nao altera a projecao nem as features do treino
    Ftr1 = R.cov_features(cr, enc, head, X, batch_size=4)
    R.cov_features(cr, enc, head, Xv * 100 + 3, batch_size=4)
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[0], b[0]) for a, b in zip(proj, cr.proj))
    assert torch.equal(Ftr1, R.cov_features(cr, enc, head, X, batch_size=4))
    # refit em outro treino muda a projecao (ela depende so do treino)
    cr2 = R.fit_cov(enc, X.flip(0) * 2, k=64)
    assert not torch.allclose(cr2.proj[0][1].abs(), cr.proj[0][1].abs())


def test_cov_readout_shortcut_equals_parts(model, data):
    enc, head = model
    (X, y), (Xv, yv) = data
    tr, va, cr = R.cov_readout(enc, head, X, Xv, batch_size=5)
    assert torch.allclose(va, R.cov_features(cr, enc, head, Xv, batch_size=5), atol=1e-6)


def test_layer_cov_equals_official_groups(model, data):
    enc, head = model
    (X, y), (Xv, yv) = data
    cr = R.fit_cov(enc, X, k=6)
    lf = R.layer_features(cr, enc, Xv, batch_size=4)
    off = R.cov_features(cr, enc, head, Xv, batch_size=4)
    n = 6 * 7 // 2
    for i, L in enumerate(["L1", "L2", "L3", "L4"]):
        assert lf[L]["cov"].shape == (10, n)
        assert torch.allclose(lf[L]["cov"], off[:, i * n:(i + 1) * n], atol=1e-6)
    assert lf["L2"]["gap"].shape == (10, 12)


def test_no_head_uses_gap(model, data):
    enc, _ = model
    (X, y), (Xv, yv) = data
    tr, va, _ = R.cov_readout(enc, None, X, Xv, k=4)
    assert va.shape == (10, 4 * 10 + 20)
