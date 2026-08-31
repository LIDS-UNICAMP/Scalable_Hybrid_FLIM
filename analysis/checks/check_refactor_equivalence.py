# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""check_refactor_equivalence.py — Self-check for the SVM/metrics/constants deduplication.

Run it after touching ``src/utils/evaluate.py:fit_svm``,
``src/metrics/classification.py:compute_metrics`` or the SVM probe in
``src/modules/autoencoder_flim_module.py``:

    python -m analysis.checks.check_refactor_equivalence

Every check is a bare ``assert`` against a hardcoded expectation or against the
*old* implementation spelled out inline, so a silent behaviour change fails
loudly instead of being absorbed by the shared helper it is supposed to test.
No pytest, no fixtures — this is one script, and it either prints OK or dies.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchmetrics.functional.classification import multiclass_accuracy

# analysis/checks/ esta a 2 niveis da raiz do repo (o arquivo veio de tools/, que era 1).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from core.metrics import compute_metrics  # noqa: E402
from eval.svm import fit_svm, svm_protocol  # noqa: E402

# The one and only SVM config, retyped here on purpose: if it is imported the
# comparison is circular and proves nothing.
_OLD_SVC_KWARGS = dict(
    max_iter=-1,
    C=1e2,
    degree=3,
    gamma="auto",
    coef0=0,
    decision_function_shape="ovo",
    kernel="linear",
)


def _fixture():
    """Deterministic 3-class, 8-d problem. RandomState(0), never bare np.random."""
    rs = np.random.RandomState(0)
    # Offset kept small on purpose: a perfectly separable fixture would make the
    # label-convention check below vacuous (kappa == 1.0 either way).
    X = np.vstack([rs.randn(20, 8) + c * 0.6 for c in range(3)])
    y = np.repeat(np.arange(3), 20)
    return X, y


# ── (a) metrics ───────────────────────────────────────────────────────────────

def check_metrics() -> None:
    # Deliberately class-imbalanced: 10 / 3 / 2 / 1.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 2, 3, 3])

    m = compute_metrics(y_true, y_pred, num_classes=4)
    for key, expected in (("kappa", 0.5761589407920837),
                          ("acc",   0.7416666746139526),
                          ("f1",    0.6688596606254578)):
        assert abs(m[key] - expected) < 1e-9, f"compute_metrics['{key}'] moved: {m[key]!r} != {expected!r}"

    # KNOWN QUIRK, pinned as current behaviour and NOT fixed here: compute_metrics
    # never passes average= to multiclass_accuracy, so torchmetrics defaults to
    # "macro".  Every `acc` in this repo is therefore BALANCED accuracy.
    macro = float(multiclass_accuracy(torch.from_numpy(y_pred), torch.from_numpy(y_true),
                                      num_classes=4, average="macro"))
    micro = float(accuracy_score(y_true, y_pred))
    assert abs(m["acc"] - macro) < 1e-9, f"acc stopped being macro/balanced: {m['acc']!r} vs {macro!r}"
    assert abs(macro - micro) > 1e-6, "fixture is no longer imbalanced — the macro/micro check is vacuous"
    assert abs(m["acc"] - micro) > 1e-6, f"acc became sklearn accuracy_score ({micro!r}) — convention changed"
    print(f"  (a) metrics      OK  kappa={m['kappa']:.6f} acc={m['acc']:.6f} (macro) f1={m['f1']:.6f}"
          f"  [sklearn micro acc would be {micro:.6f}]")


# ── (b) the default SVM arm ───────────────────────────────────────────────────

def check_svm_default() -> None:
    X, y = _fixture()

    new = fit_svm(X, y)
    old = SVC(**_OLD_SVC_KWARGS)
    old.fit(X, y)

    assert np.array_equal(new.predict(X), old.predict(X)), "fit_svm predictions differ from the old inline SVC"
    assert len(new.support_) == len(old.support_), \
        f"support vector count differs: {len(new.support_)} != {len(old.support_)}"
    assert svm_protocol(new) == svm_protocol(old), \
        f"protocol differs: {svm_protocol(new)!r} != {svm_protocol(old)!r}"
    assert "scaler=none" in svm_protocol(new), f"default arm must be unscaled: {svm_protocol(new)!r}"
    print(f"  (b) svm default  OK  {svm_protocol(new)}  n_sv={len(new.support_)}")


# ── (c) the scaler arm — the asymmetry that must survive ──────────────────────

def check_svm_scaler() -> None:
    X, y = _fixture()

    new = fit_svm(X, y, scaler=True)
    old = Pipeline([("scaler", StandardScaler()), ("svm", SVC(**_OLD_SVC_KWARGS))])
    old.fit(X, y)

    assert np.array_equal(new.predict(X), old.predict(X)), "scaled fit_svm predictions differ from the old inline Pipeline"
    assert len(new.named_steps["svm"].support_) == len(old.named_steps["svm"].support_), \
        "scaled support vector count differs"
    assert svm_protocol(new) == svm_protocol(old), \
        f"scaled protocol differs: {svm_protocol(new)!r} != {svm_protocol(old)!r}"
    assert "scaler=standard" in svm_protocol(new), f"scaler arm must report standard: {svm_protocol(new)!r}"
    assert "scaler=none" in svm_protocol(fit_svm(X, y)), "the two arms collapsed into one — asymmetry lost"
    print(f"  (c) svm scaler   OK  {svm_protocol(new)}  n_sv={len(new.named_steps['svm'].support_)}")


# ── (d) label-convention invariance ───────────────────────────────────────────

def check_label_convention() -> None:
    """+1-indexed labels must give the same metrics once shifted back.

    This is why the refactor could leave the mixed +1 / 0-indexed call sites alone.
    """
    X, y = _fixture()

    pred0 = fit_svm(X, y, tag="SVM_0IDX").predict(X)
    pred1 = fit_svm(X, y + 1, tag="SVM_1IDX").predict(X) - 1

    assert np.array_equal(pred0, pred1), "shifting labels by +1 changed the predictions"
    m0 = compute_metrics(y, pred0, num_classes=3)
    m1 = compute_metrics(y, pred1, num_classes=3)
    assert m0 == m1, f"metrics differ across label conventions: {m0} != {m1}"
    assert 0.0 < m0["kappa"] < 1.0, f"fixture became trivially separable — check is vacuous ({m0['kappa']})"
    print(f"  (d) label conv   OK  kappa={m0['kappa']:.6f} identical under +1/-1")


# ── (e) the known exception must not drift ────────────────────────────────────

_FIT_SVM_FILE = os.path.join(_ROOT, "src", "utils", "evaluate.py")
_PROBE_FILE = os.path.join(_ROOT, "src", "modules", "autoencoder_flim_module.py")


def _svc_kwargs(path: str, defaults_from: str | None = None) -> dict[str, str]:
    """Source-level kwargs of the single ``SVC(...)`` call in *path*.

    Bare names are resolved against the defaults of the function *defaults_from*,
    so ``max_iter=max_iter`` in ``fit_svm`` compares against the literal ``-1``.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())

    defaults: dict[str, str] = {}
    if defaults_from:
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == defaults_from)
        args = fn.args.args + fn.args.kwonlyargs
        vals = fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]
        for a, d in zip(args[len(args) - len(vals):], vals):
            defaults[a.arg] = ast.unparse(d)

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "SVC" or getattr(n.func, "id", None) == "SVC")]
    assert len(calls) == 1, f"{path}: expected exactly one SVC(...) call, found {len(calls)}"

    out = {}
    for kw in calls[0].keywords:
        src = ast.unparse(kw.value)
        out[kw.arg] = defaults.get(src, src)
    return out


def check_probe_config_matches() -> None:
    canonical = _svc_kwargs(_FIT_SVM_FILE, defaults_from="fit_svm")
    probe = _svc_kwargs(_PROBE_FILE)
    assert canonical == probe, (
        "src/modules/autoencoder_flim_module.py:_svm_probe drifted from fit_svm:\n"
        f"  fit_svm : {canonical}\n  _svm_probe: {probe}"
    )
    assert canonical["kernel"] == "'linear'" and canonical["C"] == "100.0", \
        f"canonical config itself changed: {canonical}"
    print(f"  (e) probe config OK  identical to fit_svm: {canonical}")


if __name__ == "__main__":
    print("check_refactor_equivalence")
    check_metrics()
    check_svm_default()
    check_svm_scaler()
    check_label_convention()
    check_probe_config_matches()
    print("ALL CHECKS PASSED")
