import numpy as np
import pytest

from research.lejepa_diag import analysis as A


def test_effective_rank_known():
    assert A.effective_rank(eig=np.ones(4)) == pytest.approx(4.0)
    assert A.effective_rank(eig=np.array([1.0, 0, 0])) == pytest.approx(1.0)
    assert A.effective_rank(eig=np.array([1.0, 1.0, 0, 0])) == pytest.approx(2.0)
    X = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)       # covariancia isotropica 2-d
    assert A.effective_rank(X) == pytest.approx(2.0)
    X2 = np.c_[X, np.zeros(4)]
    assert A.effective_rank(X2) == pytest.approx(2.0)


def test_spectrum_and_cumvar():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)) * np.array([5, 3, 1, 0.5, 0.1])
    p = A.normalized_spectrum(X)
    assert p.sum() == pytest.approx(1.0) and np.all(np.diff(p) <= 1e-12)
    c = A.cumulative_explained_variance(X)
    assert c[-1] == pytest.approx(1.0) and np.all(np.diff(c) >= -1e-12)
    assert A.n_components_for(X, 0.999) <= 5


@pytest.mark.parametrize("debiased", [False, True])
def test_cka_identity_and_rotation(debiased):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 10))
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    assert A.linear_cka(X, X, debiased) == pytest.approx(1.0, abs=1e-10)
    assert A.linear_cka(X, X @ Q, debiased) == pytest.approx(1.0, abs=1e-10)
    assert A.linear_cka(X, 3.0 * X + 2.0, debiased) == pytest.approx(1.0, abs=1e-10)
    Y = rng.standard_normal((60, 7))
    assert A.linear_cka(X, Y, debiased) < 0.6
    with pytest.raises(ValueError):
        A.linear_cka(X, X[:10])


def test_controls_same_subsample():
    i0, i1 = A.paired_sample_idx(100, 30, 3), A.paired_sample_idx(100, 30, 3)
    assert np.array_equal(i0, i1) and len(set(i0)) == 30
    assert np.array_equal(A.dim_idx(50, 10, 2), A.dim_idx(50, 10, 2))
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 20))
    g = A.controlled_geometry(X, X.copy(), n=40, d=10, reps=3)
    assert g["cka"] == pytest.approx(1.0) and g["erank_p0"] == pytest.approx(g["erank_p1"])
    assert g["n"] == 40 and g["d"] == 10
