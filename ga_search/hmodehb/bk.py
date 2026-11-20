"""Subpopulacoes B0-B4: faixas FIXAS de parametros (encoder + cabeca >= 1280) e teto duro de 5,5M.

Faixas (lo, hi] como em ``strata.stratum_of``; 300000 -> B0, 300001 -> B1; > 5.500.000 -> None
(infeasivel, nunca gerado e nunca aceito). Sem fator 1,05 (o PNAS-CS usa 5,775M: viola a missao).
Decoders do AE local e projetores SSL ficam fora da contagem (registrados a parte pelo avaliador).
"""
from __future__ import annotations

from typing import Optional

BK_BANDS = (("B0", 0, 300_000), ("B1", 300_000, 800_000), ("B2", 800_000, 1_500_000),
            ("B3", 1_500_000, 3_000_000), ("B4", 3_000_000, 5_500_000))
HARD_CAP = 5_500_000
BK_NAMES = tuple(n for n, _, _ in BK_BANDS)
MIN_EMBED_DIM = 1280     # decisao do usuario 17/11: cabeca de alta dimensao obrigatoria


class BandError(ValueError):
    """Config de estratos diferente das faixas fixas B0-B4."""


def bands_dict() -> dict:
    return {n: [lo, hi] for n, lo, hi in BK_BANDS}


def bk_of(params) -> Optional[str]:
    """Faixa de uma contagem de parametros; None acima do teto duro."""
    if params is None:
        return None
    p = int(params)
    if p > HARD_CAP or p < 0:
        return None
    for n, lo, hi in BK_BANDS:
        if lo < p <= hi or (lo == 0 and p <= hi):
            return n
    return None


def _norm_strata(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = [(k, v[0], v[1]) for k, v in raw.items()]
    return tuple((str(n), int(lo), int(hi)) for n, lo, hi in raw)


def stratified_block(user: Optional[dict] = None) -> dict:
    """Bloco ``search.stratified`` com as faixas B0-B4 impostas.

    O usuario pode ajustar os knobs de promocao (min_share, explore_*, calibration, ...), mas NAO as
    faixas nem o teto: se vierem diferentes, ``BandError`` (nunca uma troca silenciosa)."""
    u = dict(user or {})
    got = _norm_strata(u.get("strata"))
    if got is not None and got != BK_BANDS:
        raise BandError(f"stratified.strata deve ser exatamente {bands_dict()}, veio {u.get('strata')}")
    cap = u.get("param_cap")
    if cap is not None and int(cap) != HARD_CAP:
        raise BandError(f"stratified.param_cap deve ser {HARD_CAP} (teto duro), veio {cap}")
    u.update(enabled=True, strata=bands_dict(), param_cap=HARD_CAP)
    ex = [s for s in (u.get("explore_strata") or ["B2", "B3", "B4"]) if s in BK_NAMES]
    u["explore_strata"] = ex
    return u


def check_strat(sc: dict) -> None:
    """Confere um ``strat_config`` ja resolvido (MODEHB.strat)."""
    if sc is None:
        raise BandError("H-MODEHB exige estratificacao B0-B4 (stratified.enabled)")
    if tuple(tuple(x) for x in sc["strata"]) != BK_BANDS or int(sc["param_cap"]) != HARD_CAP:
        raise BandError(f"faixas/teto resolvidos diferem de B0-B4/{HARD_CAP}: {sc['strata']} cap={sc['param_cap']}")
