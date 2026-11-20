"""Fitness v2 do H-MODEHB (desenho secao 5).

Objetivos (minimizados, em [0,1]):
  f0 = 1 - clip(kappa_val_ORIG, 0, 1)
  f1 = clip(log10(MACs/1e6) / log10(1e11/1e6), 0, 1)       (limites fixos: HV comparavel entre runs)

kappa_val_ORIG = ``core.svm.fit_svm`` linear, C=100, SEM scaler, SEM whitening, sobre o embedding
``head(encoder(x))`` (dimensao >= 1280), tta=1, kappa_weight=1.0, treino estratificado e semeado
(``max_train``), validacao INTEIRA. Teste nunca e carregado.

Auxiliares (COV/cov_head+scaler, GTP, macro-F1, label-free, kappa_val_alpha0, ...) vao SO para
``result["aux"]`` e nunca entram em F. A assinatura ``fitness_sig`` resume a regua; o estado,
cada job e o id do controller W&B a carregam, e o resume recusa misturar assinaturas.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional, Sequence

from ga_search.hmodehb.bk import MIN_EMBED_DIM

VERSION = "fitness_v2"
FLOPS_BOUNDS = (1e6, 1e11)
# valores obrigatorios do bloco eval.svm_fitness (a regua ORIG)
REQUIRED = {"C": 100.0, "scaler": "none", "readout": "head", "kappa_weight": 1.0, "tta": 1,
            "whiten": "none", "solver": "libsvm"}
DEFAULTS = {"max_train": 4000, "wise_alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "gram_reduce": False}
# chaves de eval.svm_fitness que nao mudam a regua (velocidade / memoria)
SPEED_KEYS = ("cov_eigh", "cov_eigh_workers", "svm_workers", "svm_ram_gb", "svm_blas_threads", "gram_reduce")
AUX_ONLY_KEYS = ("cov_k", "aux_cov")


class FitnessConfigError(ValueError):
    """Config de fitness que nao e a regua v2 (ex.: C=0.1, scaler=standard, readout=cov_head)."""


class FitnessSigMismatch(RuntimeError):
    """Resume/replay com uma regua de fitness diferente da gravada no estado."""


class NonFiniteFitness(RuntimeError):
    """Embedding ou loss nao finitos: o candidato falha com reason=nan."""


def _same(a, b) -> bool:
    if isinstance(b, float):
        try:
            return math.isclose(float(a), b)
        except (TypeError, ValueError):
            return False
    if isinstance(b, int) and not isinstance(b, bool):
        try:
            return int(a) == b
        except (TypeError, ValueError):
            return False
    return str(a).lower() == str(b).lower()


def validate_eval(ev: dict) -> dict:
    """Confere ``cfg["eval"]`` e devolve o bloco svm_fitness efetivo (v2). Erro em qualquer desvio."""
    if (ev or {}).get("fitness_mode") != "svm_kappa":
        raise FitnessConfigError("fitness v2 exige eval.fitness_mode: svm_kappa")
    sv = dict((ev or {}).get("svm_fitness") or {})
    bad = {k: sv[k] for k, v in REQUIRED.items() if k in sv and not _same(sv[k], v)}
    if bad:
        raise FitnessConfigError(f"eval.svm_fitness fora da regua v2 (ORIG): {bad}; exigido {REQUIRED}")
    if bool(sv.get("dimensional_collapse_penalty", False)):
        raise FitnessConfigError("fitness v2: penalidade de colapso dimensional proibida (so flag)")
    out = {**REQUIRED, **DEFAULTS, **sv}
    out["C"], out["kappa_weight"], out["tta"] = float(out["C"]), float(out["kappa_weight"]), int(out["tta"])
    out["max_train"] = None if out["max_train"] is None else int(out["max_train"])
    out["wise_alphas"] = [float(a) for a in (out.get("wise_alphas") or [])]
    return out


def spec(sv: dict, *, val_sha256: str, n_val: Optional[int], dataset: str, split=None) -> dict:
    """Tudo o que define a regua (o que muda o numero que vira f0/f1)."""
    return {"v": VERSION, "readout": "head", "min_embed_dim": MIN_EMBED_DIM, "svm": "core.svm.fit_svm",
            "C": float(sv["C"]), "scaler": "none", "kappa_weight": 1.0, "tta": 1, "whiten": "none",
            "solver": "libsvm", "max_train": sv.get("max_train"),
            "wise_alphas": [float(a) for a in sv.get("wise_alphas") or []],
            "val_sha256": str(val_sha256), "n_val": None if n_val is None else int(n_val),
            "dataset": str(dataset), "split": split, "flops_bounds": list(FLOPS_BOUNDS)}


def signature(sp: dict) -> str:
    blob = json.dumps(sp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def objectives(result: dict):
    """[f0, f1] com os limites fixos (falha -> [1, 1])."""
    from ga_search.modehb import modehb as M
    return M.objectives(result, False, {"flops_lo": FLOPS_BOUNDS[0], "flops_hi": FLOPS_BOUNDS[1]})


def quality_of(kappa: float) -> float:
    if kappa is None or not math.isfinite(float(kappa)):
        raise NonFiniteFitness(f"kappa nao finito: {kappa}")
    return float(min(max(float(kappa), 0.0), 1.0))


# chaves do resultado do Evaluator historico que sao so diagnostico -> aux
_AUX_FROM_BASE = ("quality_label_free", "fitness_macro_f1", "fitness_balanced_acc", "wise_kappas", "metrics",
                  "collapse_multiplier", "ssl_final_loss", "quality_raw")


def finalize(base: dict, *, kappa: float, nonfinite: bool = False, aux_extra: Optional[dict] = None,
             alphas: Sequence[float] = ()) -> dict:
    """Resultado v2 a partir do resultado do ``run_search.Evaluator``.

    ``quality`` = clip(kappa ORIG) SEM multiplicador de colapso; colapso total so vira flag.
    As chaves que o ``Evaluator.__call__`` historico le continuam presentes (quality_raw = quality,
    collapse_multiplier = 1.0); os valores antigos vao para ``aux``."""
    if nonfinite:
        raise NonFiniteFitness("embedding nao finito na fitness (svm_fitness.nonfinite)")
    loss = base.get("ssl_final_loss")
    if loss is not None and not math.isfinite(float(loss)):
        raise NonFiniteFitness(f"loss SSL final nao finita: {loss}")
    q = quality_of(kappa)
    aux = {k: base.get(k) for k in _AUX_FROM_BASE if k in base}
    aux["quality_legacy"] = base.get("quality")
    wk = base.get("wise_kappas") or {}
    if wk:
        a0 = wk.get("0.0", wk.get("0"))
        aux["kappa_val_alpha0"] = None if a0 is None else float(a0)
    aux.update(aux_extra or {})
    col = base.get("collapse") or {}
    out = {k: v for k, v in base.items() if k not in ("quality_label_free", "fitness_macro_f1",
                                                       "fitness_balanced_acc", "wise_kappas", "metrics")}
    out.update(quality=q, quality_raw=q, collapse_multiplier=1.0, fitness_kappa=float(kappa),
               kappa_val_orig=float(kappa), fitness_version=VERSION,
               collapse_total=bool(col.get("total", False)), collapse_dimensional=bool(col.get("dimensional", False)),
               aux=aux)
    return out


# ────────────────────────────────────────────────────────────────── regua em TODA decisao
# A1b (17/11): o fitness, o WiSE e a calibracao historicos decidiam por cov_head+StandardScaler enquanto o
# reportado e ORIG (rho(val COV, teste ORIG) = 0,09; o alpha do WiSE escolhido pelo COV piora o ORIG em 5/11).
# Aqui cada leitura SVM que DECIDE algo (alpha do WiSE e kappa final; selecao, promocao, calibracao e
# extensao leem ``quality``/F, derivados dela) carrega um carimbo ``ruler_sig``; o controller recusa
# (RulerViolation, fatal, nada gravado) qualquer resultado com carimbo diferente da regua da assinatura.
RULER_KEYS = ("readout", "C", "scaler", "whiten", "tta", "kappa_weight", "solver", "max_train")


class RulerViolation(FitnessSigMismatch):
    """Uma decisao da busca usou uma regua diferente da gravada (ex.: COV no WiSE)."""


def ruler(d: dict) -> dict:
    """Componentes da regua de um spec v2 OU de uma leitura efetiva (readout + kwargs do svm_fitness)."""
    mt = d.get("max_train")
    return {"readout": str(d.get("readout", "head")), "C": float(d.get("C", 100.0)),
            "scaler": str(d.get("scaler", "none")), "whiten": str(d.get("whiten", "none")),
            "tta": int(d.get("tta", 1)), "kappa_weight": float(d.get("kappa_weight", 1.0)),
            "solver": str(d.get("solver", "libsvm")), "max_train": None if mt is None else int(mt)}


def ruler_sig(d: dict) -> str:
    return signature(ruler(d))


def check_decisions(result: dict, sp: dict) -> None:
    """Falha se alguma decisao deste resultado nao foi tomada pela regua de ``sp`` (ORIG-val)."""
    if result.get("status", "ok") != "ok":
        return
    want = ruler_sig(sp)
    got = result.get("decision_rulers")
    if not got:
        raise RulerViolation("resultado ok sem carimbo de regua (decision_rulers): decisao nao auditavel")
    bad = sorted(set(got) - {want})
    if bad:
        raise RulerViolation(f"decisao com regua {bad} != {want} (ORIG-val da assinatura)")
    k = result.get("kappa_val_orig")
    if k is None or not math.isclose(float(result.get("quality", -1.0)), quality_of(k), abs_tol=1e-12):
        raise RulerViolation(f"quality {result.get('quality')} != clip(kappa_val_orig={k}): F nao vem da regua ORIG")
    wk = (result.get("aux") or {}).get("wise_kappas") or {}
    a = result.get("wise_alpha")
    if wk and a is not None:            # o WiSE escolhe pelo quality = clip(kappa ORIG), 1o maximo
        best = max(quality_of(v) for v in wk.values())
        ka = wk.get(str(a), wk.get(str(float(a))))
        if ka is None or not math.isclose(quality_of(ka), best, abs_tol=1e-12):
            raise RulerViolation(f"alpha do WiSE {a} nao e o argmax do kappa ORIG-val {wk}")
        if not math.isclose(float(ka), float(k), abs_tol=1e-12):
            raise RulerViolation(f"kappa final {k} != kappa ORIG-val do alpha escolhido {ka}")
