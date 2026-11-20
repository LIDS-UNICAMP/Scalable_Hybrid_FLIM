"""SVM oficial do repo (core.svm.fit_svm: linear, C=100, ovo, max_iter=-1) reajustada por
representacao; metricas na VALIDACAO via core.metrics.compute_metrics (kappa, macro-F1) e
balanced accuracy. O teste NAO participa de nada; ``report_test=True`` (desligado por padrao)
so acrescenta colunas test_* para relatorio.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

C_DEFAULT = 100.0


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def evaluate_svm(X_train, y_train, X_val, y_val, *, scaler: bool = False, C: float = C_DEFAULT,
                 X_test=None, y_test=None, report_test: bool = False, tag: str = "LEJEPA_DIAG") -> dict:
    """Fit em treino, metricas em validacao. Retorna kappa, macro_f1, balanced_acc, fit_s, ..."""
    from core.metrics import compute_metrics
    from core.svm import fit_svm, svm_protocol
    Xtr, Xva = _np(X_train).astype(np.float64), _np(X_val).astype(np.float64)
    ytr, yva = _np(y_train).astype(int), _np(y_val).astype(int)
    labels = np.unique(np.concatenate([ytr, yva]))
    remap = {int(c): i for i, c in enumerate(labels)}               # 0-indexado (core.metrics)
    ytr0 = np.array([remap[int(v)] for v in ytr])
    yva0 = np.array([remap[int(v)] for v in yva])
    if not (np.isfinite(Xtr).all() and np.isfinite(Xva).all()):
        return {"kappa": float("nan"), "macro_f1": float("nan"), "balanced_acc": float("nan"),
                "nonfinite": True, "dim": int(Xtr.shape[1]), "scaler": bool(scaler), "C": float(C)}
    clf = fit_svm(Xtr, ytr0, C=float(C), scaler=bool(scaler), tag=tag)
    pred = clf.predict(Xva)
    m = compute_metrics(yva0, pred, num_classes=len(labels))
    res = {"kappa": m["kappa"], "macro_f1": m["f1"],
           "balanced_acc": float(balanced_accuracy_score(yva0, pred)),
           "fit_s": float(clf.fit_diagnostics_.get("svm_fit_seconds", float("nan"))),
           "svm_fit_status": clf.fit_diagnostics_.get("svm_fit_status"),
           "svm_protocol": svm_protocol(clf), "dim": int(Xtr.shape[1]),
           "n_train": int(len(ytr)), "n_val": int(len(yva)), "scaler": bool(scaler), "C": float(C),
           "nonfinite": False}
    if report_test and X_test is not None:
        yte = np.array([remap.get(int(v), -1) for v in _np(y_test).astype(int)])
        pt = clf.predict(_np(X_test).astype(np.float64))
        mt = compute_metrics(yte, pt, num_classes=len(labels))
        res.update(test_kappa=mt["kappa"], test_macro_f1=mt["f1"],
                   test_balanced_acc=float(balanced_accuracy_score(yte, pt)))
    res["_clf"] = clf
    return res


def public(res: dict) -> dict:
    """Remove chaves privadas (o estimador) para gravar em CSV."""
    return {k: v for k, v in res.items() if not k.startswith("_")}
