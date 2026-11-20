"""supervised_fitness.py — SVM-on-validation fitness for the MODEHB search.

Same protocol as ``svm_on_demand.py`` / the large-model baselines: the repo's
single ``core.svm.fit_svm`` (linear SVC, C=100, ovo, unbounded solver) and
``core.metrics.compute_metrics``. Train may be stratified-subsampled to
``max_train``; validation is never subsampled; test is never touched.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def stratified_subsample(y: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    """Indices of a stratified subsample of size ``max_n`` (sorted).

    Per-class quota proportional to frequency (largest-remainder rounding);
    every class present keeps at least one sample.
    """
    y = np.asarray(y)
    n = len(y)
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    raw = counts * (max_n / n)
    quota = np.maximum(np.floor(raw).astype(int), 1)
    quota = np.minimum(quota, counts)
    rem = max_n - quota.sum()
    if rem > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for i in order:
            if rem == 0:
                break
            if quota[i] < counts[i]:
                quota[i] += 1
                rem -= 1
    idx = []
    for c, q in zip(classes, quota):
        cidx = np.flatnonzero(y == c)
        idx.append(rng.choice(cidx, size=int(q), replace=False))
    return np.sort(np.concatenate(idx))


def whiten_features(Xtr, Xva, mode: str = "none"):
    """Readout whitening fitted on TRAIN only (port of eval/svm.py ``_pw`` / ``_pwlw``, the
    readout of the hybrid SPiFiL recipe that beat I-JEPA on the parasite sets):
    ``pw``  : PCA, scale by (var + 0.1*mean var)^-1/2 (shrinkage), then L2-normalise;
    ``pwlw``: Ledoit-Wolf covariance whitening, then L2-normalise; ``none``: unchanged."""
    if mode in (None, "none"):
        return Xtr, Xva
    from sklearn.preprocessing import normalize
    if mode == "pw":
        from sklearn.decomposition import PCA
        pca = PCA(svd_solver="full").fit(Xtr)
        s = (pca.explained_variance_ + 0.1 * pca.explained_variance_.mean()) ** -0.5
        return normalize(pca.transform(Xtr) * s), normalize(pca.transform(Xva) * s)
    if mode == "pwlw":
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(Xtr)
        ev, U = np.linalg.eigh(cov.covariance_)
        ev = np.clip(ev, 1e-12, None)
        return (normalize((Xtr - cov.location_) @ U / np.sqrt(ev)),
                normalize((Xva - cov.location_) @ U / np.sqrt(ev)))
    raise ValueError(f"whiten must be none|pw|pwlw, got {mode!r}")


def span_coordinates(Xtr, Xva, standardize: bool = True, device=None):
    """Exact isometry for a LINEAR kernel (opt-in ``svm_fitness(gram_reduce=True)``).

    With D > n the train rows span an n-dim subspace; Xs^T = Q R (thin QR) gives coordinates
    Xs Q = R^T (n x n) with the SAME Gram matrix (Xs Q)(Xs Q)^T = Xs Xs^T, and val rows map to Xv Q,
    whose products with the train rows are unchanged (Xv Q Q^T Xs^T = Xv Xs^T). The linear-SVM dual
    (libsvm) therefore sees the same kernel up to ~1e-15 rounding, while every kernel product costs n
    instead of D. ``standardize`` applies the same StandardScaler as ``fit_svm(scaler=True)``'s Pipeline
    (fit on train, fit_transform == fit + transform) BEFORE the rotation. QR in fp64 on ``device``
    (default: CUDA when available, else CPU)."""
    import torch
    if standardize:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    A = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float64).T.copy()).to(dev)
    Q, R = torch.linalg.qr(A, mode="reduced")
    Xr = np.ascontiguousarray(R.T.cpu().numpy())
    Xvr = np.ascontiguousarray((torch.from_numpy(np.ascontiguousarray(Xva, dtype=np.float64)).to(dev) @ Q).cpu().numpy())
    return Xr, Xvr


def svm_fitness(train_emb, train_y, val_emb, val_y, *, C: float = 100.0,
                scaler: str = "none", max_train: int | None = 10000, solver: str = "libsvm",
                seed: int = 0, kappa_weight: float = 0.75, whiten: str = "none",
                gram_reduce: bool = False) -> dict:
    """Fit the repo's linear SVM on train embeddings, score on validation.

    Returns {"kappa", "macro_f1", "balanced_acc", "quality", "fit_s",
    "n_train_used"} (+ "nonfinite": True when embeddings had NaN/inf).
    quality = clip(kappa_weight*max(kappa,0) + (1-kappa_weight)*macro_f1, 0, 1).

    ``gram_reduce`` (opt-in, default off): when D > n_train, fit the same ``core.svm.fit_svm``
    (linear, C, libsvm) on ``span_coordinates`` (scaler applied first, then an exact rotation onto the
    train span): same dual problem, same support vectors/iterations, predictions equal up to ~1e-14
    in the decision values; the libsvm fit costs n instead of D per kernel product.
    """
    from core.metrics import compute_metrics
    from core.svm import fit_svm
    from sklearn.metrics import balanced_accuracy_score

    if scaler not in ("none", "standard"):
        raise ValueError(f"scaler must be 'none' or 'standard', got {scaler!r}")
    Xtr, ytr = _to_np(train_emb).astype(np.float64), _to_np(train_y).reshape(-1)
    Xva, yva = _to_np(val_emb).astype(np.float64), _to_np(val_y).reshape(-1)
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xva = Xva.reshape(len(Xva), -1)

    if not (np.isfinite(Xtr).all() and np.isfinite(Xva).all()):
        log.warning("svm_fitness: non-finite embeddings -> quality=0")
        return {"kappa": 0.0, "macro_f1": 0.0, "balanced_acc": 0.0, "quality": 0.0,
                "fit_s": 0.0, "n_train_used": 0, "nonfinite": True}

    idx = stratified_subsample(ytr, max_train, seed)
    Xtr, ytr = Xtr[idx], ytr[idx]
    if len(idx) < len(train_y):
        log.info("svm_fitness: stratified subsample %d -> %d (seed=%d)",
                 len(train_y), len(idx), seed)

    Xtr, Xva = whiten_features(Xtr, Xva, whiten)

    labels = np.unique(np.concatenate([ytr, yva]))
    remap = {v: i for i, v in enumerate(labels.tolist())}
    ytr0 = np.array([remap[v] for v in ytr.tolist()], dtype=np.int64)
    yva0 = np.array([remap[v] for v in yva.tolist()], dtype=np.int64)

    t0 = time.perf_counter()
    if len(np.unique(ytr0)) < 2:  # degenerate: SVC can't fit one class
        pred = np.full_like(yva0, ytr0[0] if len(ytr0) else 0)
    else:
        if solver == "liblinear":
            # fast linear SVM for the search fitness (primal liblinear, one-vs-rest, squared hinge,
            # standardized features). Not the official libsvm/ovo protocol: official numbers come
            # from svm_on_demand on the final front.
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC
            clf = make_pipeline(StandardScaler(), LinearSVC(C=C, dual="auto", max_iter=1000, tol=1e-3))
            clf.fit(Xtr, ytr0)
        elif solver == "libsvm" and gram_reduce and Xtr.shape[1] > Xtr.shape[0]:
            Xtr, Xva = span_coordinates(Xtr, Xva, standardize=(scaler == "standard"))
            clf = fit_svm(Xtr, ytr0, C=C, scaler=False, tag="SVM_FITNESS")
        elif solver == "libsvm":
            clf = fit_svm(Xtr, ytr0, C=C, scaler=(scaler == "standard"), tag="SVM_FITNESS")
        else:
            raise ValueError(f"solver must be 'libsvm' or 'liblinear', got {solver!r}")
        pred = clf.predict(Xva)
    fit_s = time.perf_counter() - t0

    m = compute_metrics(yva0, np.asarray(pred, dtype=np.int64), num_classes=len(labels))
    kappa = float(m["kappa"])
    f1 = float(m["f1"])
    if not np.isfinite(kappa):
        kappa = 0.0
    if not np.isfinite(f1):
        f1 = 0.0
    q = kappa_weight * max(kappa, 0.0) + (1.0 - kappa_weight) * f1
    return {"kappa": kappa, "macro_f1": f1,
            "balanced_acc": float(balanced_accuracy_score(yva0, pred)),
            "quality": float(np.clip(q, 0.0, 1.0)),
            "fit_s": float(fit_s), "n_train_used": int(len(idx))}
