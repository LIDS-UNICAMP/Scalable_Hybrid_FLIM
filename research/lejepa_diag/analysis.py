"""Geometria das representacoes: espectro, rank efetivo, CKA linear e controles de n / dim.

Todas as funcoes recebem matrizes [N, D] (numpy ou torch) e trabalham em float64.
"""
from __future__ import annotations

import numpy as np


def _f64(X) -> np.ndarray:
    X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    return np.asarray(X, dtype=np.float64)


def eigenvalues(X, center: bool = True) -> np.ndarray:
    """Autovalores (decrescentes, >= 0) da covariancia amostral de X [N, D]."""
    X = _f64(X)
    if center:
        X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    return np.sort(s ** 2 / max(1, X.shape[0] - 1))[::-1]


def normalized_spectrum(X=None, *, eig=None) -> np.ndarray:
    """Autovalores normalizados para somar 1 (p_i = lambda_i / sum lambda)."""
    lam = np.clip(_f64(eig) if eig is not None else eigenvalues(X), 0, None)
    tot = lam.sum()
    return lam / tot if tot > 0 else np.zeros_like(lam)


def effective_rank(X=None, *, eig=None) -> float:
    """exp(H(p)), p = autovalores normalizados (Roy & Vetterli 2007). 0 log 0 = 0."""
    p = normalized_spectrum(X, eig=eig)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0


def cumulative_explained_variance(X=None, *, eig=None) -> np.ndarray:
    """Variancia explicada acumulada (curva de 1..D componentes)."""
    return np.cumsum(normalized_spectrum(X, eig=eig))


def n_components_for(X=None, frac: float = 0.9, *, eig=None) -> int:
    """Menor numero de componentes cuja variancia acumulada >= frac."""
    c = cumulative_explained_variance(X, eig=eig)
    return int(np.searchsorted(c, frac - 1e-12) + 1) if c.size else 0


def _hsic(K: np.ndarray, L: np.ndarray, debiased: bool) -> float:
    n = K.shape[0]
    if not debiased:
        H = np.eye(n) - 1.0 / n
        return float(np.sum((H @ K @ H) * L))  # tr(KHLH), escala cancela no CKA
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0), np.fill_diagonal(L, 0)
    t1 = np.sum(K * L)
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 * (K.sum(0) @ L.sum(0)) / (n - 2)
    return float((t1 + t2 - t3) / (n * (n - 3)))


def linear_cka(X, Y, debiased: bool = False) -> float:
    """CKA linear entre X [N, D1] e Y [N, D2] com amostras PAREADAS (mesmas linhas).

    debiased=True usa o estimador nao-viesado do HSIC (Song et al. 2012; Nguyen et al. 2021)."""
    X, Y = _f64(X), _f64(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA exige amostras pareadas: {X.shape[0]} != {Y.shape[0]}")
    if debiased and X.shape[0] < 4:
        raise ValueError("CKA debiased exige N >= 4")
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    if not debiased:   # forma em features: ||Y^T X||_F^2 / (||X^T X|| ||Y^T Y||)
        num = np.linalg.norm(Y.T @ X) ** 2
        den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
        return float(num / den) if den > 0 else float("nan")
    K, L = X @ X.T, Y @ Y.T
    hxy, hxx, hyy = _hsic(K, L, True), _hsic(K, K, True), _hsic(L, L, True)
    return float(hxy / np.sqrt(hxx * hyy)) if hxx > 0 and hyy > 0 else float("nan")


# ─── controles de numero de amostras e de dimensao ──────────────────────────────

def paired_sample_idx(n_total: int, n: int, seed: int = 0) -> np.ndarray:
    """Indices de amostras (ordenados) — os MESMOS aplicados a P0 e P1."""
    n = min(int(n), int(n_total))
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def dim_idx(d_total: int, d: int, seed: int = 0) -> np.ndarray:
    """Subconjunto aleatorio de coordenadas — o MESMO aplicado a P0 e P1 (dims iguais)."""
    d = min(int(d), int(d_total))
    return np.sort(np.random.default_rng(seed + 10_007).choice(d_total, d, replace=False))


def controlled_geometry(X0, X1, *, n: int | None = None, d: int | None = None, reps: int = 5,
                        seed: int = 0, debiased: bool = False) -> dict:
    """Rank efetivo de P0/P1 e CKA(P0, P1) com n e d iguais entre P0 e P1, media +- dp em ``reps``.

    Requer X0, X1 pareados (mesmas imagens nas mesmas linhas). ``d`` subamostra coordenadas
    apenas se X0 e X1 tem a mesma dimensao (senao cada um usa min(d, D_i) proprio)."""
    X0, X1 = _f64(X0), _f64(X1)
    if X0.shape[0] != X1.shape[0]:
        raise ValueError("P0 e P1 devem estar pareados")
    N = X0.shape[0]
    n = N if n is None else min(n, N)
    d_common = min(X0.shape[1], X1.shape[1]) if d is None else min(d, X0.shape[1], X1.shape[1])
    r0, r1, ck = [], [], []
    for r in range(reps):
        si = paired_sample_idx(N, n, seed + r)
        a, b = X0[si], X1[si]
        if d is not None:
            a = a[:, dim_idx(X0.shape[1], d_common, seed + r)]
            b = b[:, dim_idx(X1.shape[1], d_common, seed + r)]
        r0.append(effective_rank(a)), r1.append(effective_rank(b)), ck.append(linear_cka(a, b, debiased))
    return {"n": int(n), "d": int(d_common) if d is not None else None, "reps": int(reps),
            "erank_p0": float(np.mean(r0)), "erank_p0_sd": float(np.std(r0)),
            "erank_p1": float(np.mean(r1)), "erank_p1_sd": float(np.std(r1)),
            "cka": float(np.mean(ck)), "cka_sd": float(np.std(ck))}
