# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""svm.py — o ajuste de SVM do projeto, em UM lugar so.

Aqui moram `fit_svm`, `fit_svm_with_diagnostics` e `svm_protocol`, movidos
verbatim de `eval/svm.py`. Todo mundo (`eval/`, `eval/svm_variants/`,
`analysis/`) importa daqui: existe UMA definicao de `fit_svm` no repositorio.

POR QUE O PARAMETRO `scaler` EXISTE E O QUE ELE SEPARA
-------------------------------------------------------
Uma implementacao unica NAO quer dizer parametros unicos. O `scaler` e o
unico ponto em que os chamadores divergem de proposito, e ele marca as DUAS
REGUAS do projeto:

* `scaler=False` (default) — a regua do encoder, 48-d. As features saem do
  `AdaptiveAvgPool2d(1)` sobre o conv3, ja em escala comparavel entre canais.
  Padronizar aqui so adicionaria ruido. Usada por todos os braços menos dois.

* `scaler=True` — a regua da projecao, 1280-d. As duas arms de projecao
  (`eval/svm_variants/svm_real_flim.py` e
  `eval/svm_variants/svm_distill_with_projection.py`) precisam de
  `StandardScaler` para o libsvm convergir; sem ele o solver bate no teto.

As duas reguas NAO sao comparaveis entre si — e disso que trata
`analysis/distill/ruler_mismatch.py`. Por isso a assimetria nao e unificada,
so REGISTRADA: `svm_protocol` deriva o `scaler=none` / `scaler=standard` do
proprio estimador e grava em cada linha de CSV.

Quem for "limpar" isso colapsando as duas arms em uma quebra
`analysis/checks/check_refactor_equivalence.py`, que assegura em asserts
separados que o braco default reporta `scaler=none` e o braco da projecao
reporta `scaler=standard`. Os defaults (`max_iter=-1, C=1e2, scaler=False,
tag="SVM"`) sao contrato historico: os CSVs antigos foram gerados com eles.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _svc_step(estimator):
    """Return the ``SVC`` inside *estimator* — itself, or the step of a Pipeline.

    ``fit_status_`` / ``n_iter_`` / ``support_`` live on the ``SVC``, never on
    the wrapping ``Pipeline``.
    """
    steps = getattr(estimator, "named_steps", None)
    if steps is None:
        return estimator
    for step in steps.values():
        if isinstance(step, svm.SVC):
            return step
    return estimator


def svm_protocol(estimator) -> str:
    """One-line, auditable description of the SVM protocol of *estimator*.

    Derived from the estimator itself (never hardcoded), so the string cannot
    drift from the code.  Recorded in every result CSV so two experiment arms
    can be checked for pairing without reading the source — the scaler is
    deliberately NOT unified across arms, only reported.
    """
    svc   = _svc_step(estimator)
    steps = getattr(estimator, "named_steps", None) or {}
    scaler = "standard" if any(isinstance(s, StandardScaler) for s in steps.values()) else "none"
    return (
        f"scaler={scaler};kernel={svc.kernel};C={svc.C:g};"
        f"{svc.decision_function_shape};max_iter={svc.max_iter}"
    )


def fit_svm_with_diagnostics(estimator, X, y, tag: str = "SVM") -> dict:
    """Fit *estimator* on ``(X, y)`` and record how the libsvm solver terminated.

    Persisting this is the whole point: ``fit_status_=1`` (solver hit the
    iteration cap) went undetected for months because nothing in the repo ever
    saved it.  A non-zero status is logged as a WARNING.

    The returned dict is also stashed on the estimator as ``fit_diagnostics_``
    so callers that only receive the fitted classifier back can still write the
    columns without a signature change.

    Returns:
        dict with the ``svm_*`` columns (see ``SVM_DIAG_MISSING``).
    """
    _t0 = time.perf_counter()
    estimator.fit(X, y)
    elapsed = time.perf_counter() - _t0

    svc = _svc_step(estimator)
    n_iter = np.asarray(getattr(svc, "n_iter_", []))
    diag = {
        "svm_protocol":    svm_protocol(estimator),
        "svm_fit_status":  int(getattr(svc, "fit_status_", -1)),
        "svm_n_iter_max":  int(n_iter.max()) if n_iter.size else -1,
        "svm_n_iter_sum":  int(n_iter.sum()) if n_iter.size else -1,
        "svm_n_sv":        int(len(getattr(svc, "support_", ()))),
        "svm_fit_seconds": round(elapsed, 3),
    }

    msg = (
        f"[{tag}] {diag['svm_protocol']}  fit_status={diag['svm_fit_status']}  "
        f"n_iter(max/sum)={diag['svm_n_iter_max']}/{diag['svm_n_iter_sum']}  "
        f"n_sv={diag['svm_n_sv']}  fit={diag['svm_fit_seconds']:.1f}s"
    )
    if diag["svm_fit_status"] != 0:
        logging.warning("%s  <- SOLVER DID NOT CONVERGE (fit_status != 0)", msg)
        print(f"  [WARN] {msg}  <- SOLVER DID NOT CONVERGE")
    else:
        print(f"  {msg}")

    estimator.fit_diagnostics_ = diag
    return diag


# ─── SVM training ─────────────────────────────────────────────────────────────


def fit_svm(X, y, *, max_iter: int = -1, C: float = 1e2, scaler: bool = False,
            tag: str = "SVM"):
    """Build the repo's canonical linear SVM, fit it on ``(X, y)`` and return it.

    Every SVM in this repository uses exactly these hyperparameters, so they
    live here once instead of being retyped per call site.  ``degree`` and
    ``coef0`` are inert under ``kernel="linear"`` but are pinned anyway so the
    estimator repr matches the historical CSVs.

    *scaler* is the ONLY knob beyond the solver cap and ``C``: two arms (the
    1280-d projection ones) genuinely need ``StandardScaler`` to converge.  That
    asymmetry is deliberate — it is recorded per row by ``svm_protocol`` and
    must NOT be unified across arms.

    This function takes an already-built feature matrix; it does not extract
    features and it does not touch label indexing.  The ``+1`` / 0-indexed
    convention stays the caller's business, on purpose.

    Args:
        X:        Feature matrix ``(N, D)``.
        y:        Label vector ``(N,)`` — whatever indexing the caller chose.
        max_iter: Solver cap; ``-1`` = unbounded (default).
        C:        Regularisation parameter.
        scaler:   Wrap the SVC in ``Pipeline([StandardScaler, SVC])``.
        tag:      Prefix for the diagnostics line printed by
                  ``fit_svm_with_diagnostics``.

    Returns:
        The fitted estimator (``SVC`` or ``Pipeline``); solver diagnostics are
        attached as ``fit_diagnostics_``.
    """
    clf = svm.SVC(
        # Unbounded solver: results are deliberately NOT comparable with the
        # CSVs produced under the old max_iter cap.
        max_iter=max_iter,
        C=C,
        degree=3,
        gamma="auto",
        coef0=0,
        decision_function_shape="ovo",
        kernel="linear",
    )
    estimator = Pipeline([("scaler", StandardScaler()), ("svm", clf)]) if scaler else clf

    # Barra de progresso viva durante o fit: libsvm não reporta nada e um fit
    # longo parece travado.  Fica aqui para não ser reescrita em cada arm.
    _stop = threading.Event()

    def _progress():
        with tqdm(desc="SVM fit", unit="s", bar_format="{desc}: {elapsed} [{postfix}]") as pbar:
            while not _stop.wait(1.0):
                pbar.update(1)
            pbar.set_postfix_str("done")

    _thread = threading.Thread(target=_progress, daemon=True)
    _thread.start()
    try:
        fit_svm_with_diagnostics(estimator, X, y, tag=tag)
    finally:
        _stop.set()
        _thread.join()
    return estimator
