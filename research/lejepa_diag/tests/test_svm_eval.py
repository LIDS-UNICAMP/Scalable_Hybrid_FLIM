import numpy as np

from research.lejepa_diag.svm_eval import evaluate_svm, public


def test_svm_separable_and_protocol():
    rng = np.random.default_rng(0)
    y = np.repeat([3, 5, 7], 20)                                  # rotulos nao 0-indexados
    X = rng.standard_normal((60, 4)) * 0.1 + np.eye(4)[np.repeat([0, 1, 2], 20)] * 5
    res = evaluate_svm(X[::2], y[::2], X[1::2], y[1::2])
    assert res["kappa"] == 1.0 and res["macro_f1"] == 1.0 and res["balanced_acc"] == 1.0
    p = public(res)
    assert "_clf" not in p and "scaler=none" in p["svm_protocol"] and "C=100" in p["svm_protocol"]
    assert "test_kappa" not in p                                   # teste desligado por padrao
    sc = public(evaluate_svm(X[::2], y[::2], X[1::2], y[1::2], scaler=True))
    assert "scaler=standard" in sc["svm_protocol"]
