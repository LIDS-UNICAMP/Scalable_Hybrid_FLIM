#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""Testes pareados par-a-par + equivalencia (TOST) entre os 8 modelos oficiais.

Motivacao
---------
Os scripts `wilcoxon_{f1,acc,kappa}.py` sao um-contra-todos com o FLIM como baseline:
cobrem 7 pares e deixam 21 dos 28 pares possiveis sem teste. Varias afirmacoes de
"empate" ou "ganho desprezivel" do texto envolvem justamente pares nao cobertos —
notadamente **Distill 3 vs Distill 4** (o "+0,007 de F1 por +45% de params").

Alem disso, e o ponto central deste script: **ausencia de significancia nao e empate**.
Um Wilcoxon que nao rejeita diz "nao detectei diferenca", o que e compativel tanto com
"os modelos sao iguais" quanto com "o teste nao tem poder". Para o texto do artigo poder
escrever *empate*, e preciso um **teste de equivalencia**, que inverte o onus da prova:
a hipotese nula passa a ser "a diferenca e grande" e so se rejeita essa nula quando ha
evidencia positiva de que a diferenca e pequena.

Desenho
-------
* Unidade pareada: celula ``(dataset_short, pretrained_pct)``, 3 datasets x 6 fracoes de
  rotulos = **n = 18 pares**. Cada celula ja e a media sobre 3 splits. Pareamento pela
  chave, com aborto se as grades nao coincidirem.
* Convencao de sinal: ``diff = metrica(modelo_b) - metrica(modelo_a)``. Positivo favorece
  o **modelo_b**. Os dois nomes aparecem em toda linha, entao nao ha ambiguidade.
* **Superioridade**: ``scipy.stats.wilcoxon(a, b, alternative="two-sided",
  zero_method="wilcox", method="exact")``. Responde "existe diferenca?".
* **Equivalencia (TOST)**: dois testes unilaterais de Wilcoxon contra nulas deslocadas,
      H0_1: mediana(diff) >= +delta   testada por wilcoxon(diff - delta, alternative="less")
      H0_2: mediana(diff) <= -delta   testada por wilcoxon(diff + delta, alternative="greater")
  e ``p_TOST = max(p1, p2)``. Rejeitar ambas significa ``-delta < mediana(diff) < +delta``,
  isto e, **equivalencia demonstrada dentro da margem delta**. Responde "a diferenca e
  pequena o bastante para ser irrelevante?".
* **delta_min**: por busca binaria, a menor margem para a qual o TOST rejeita a alfa. E o
  numero mais informativo do relatorio — "estes dois modelos sao equivalentes dentro de
  +/- delta_min" — e dispensa o leitor de aceitar a nossa escolha de delta.

Margem de equivalencia
----------------------
``DELTA = 0.05`` (5 pontos da metrica), **pre-especificada**, nao escolhida depois de ver
os resultados. Justificativa dupla: (a) e o arredondamento convencional para "diferenca
praticamente irrelevante" em metricas de classificacao nesta escala; (b) fica em torno de
2x o desvio-padrao tipico entre os 3 splits da mesma celula (mediana ~0,026 em F1), ou
seja, uma diferenca menor que delta esta na ordem de grandeza do ruido de medicao do
proprio experimento. O piso de ruido e recalculado e impresso no relatorio.

Familias de correcao
--------------------
1. **Familia focal (primaria)**: os pares que sustentam afirmacoes explicitas de empate ou
   de ganho desprezivel no texto — ver ``FOCUS_PAIRS``, com a afirmacao anotada em cada um.
   Pre-especificada. Holm sobre esses pares, por metrica. E a tabela que vai para o artigo.
2. **Matriz completa (secundaria)**: os 28 pares, Holm sobre 28, por metrica. Existe para
   que nenhum par fique sem teste, respondendo a critica de cobertura. O preco e poder
   menor; a familia focal e que sustenta as conclusoes.

Correcao de multiplicidade sobre p de TOST e **conservadora na direcao segura**: dificulta
declarar equivalencia, nunca a facilita.

Uso
---
    cd /dados/home/moliveira/Scalable_Hybrid_FLIM
    LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
    /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
        -m analysis.stats.wilcoxon_equivalence

Saidas: statistics/tools/wilcoxon_equivalence.csv e statistics/tools/wilcoxon_equivalence.md
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# method -> (rotulo do artigo, filtro de init ou None, params)
MODELS: dict[str, tuple[str, str | None, int]] = {
    "SVM_FLIM": ("FLIM (59.504)", None, 59_504),
    "SVM_LeJEPA": ("LeJEPA (59.504)", "trunc_normal", 59_504),
    "SVM_IJEPA": ("I-JEPA (632M)", None, 632_000_000),
    "SVM_Distill_1x1BN": ("Distill 1 (123K)", "trunc_normal", 123_504),
    "SVM_Distill_1x1BN_flim_frozen_eval_loss": ("Distill 1 - FLIM init (123K)", None, 123_504),
    "SVM_Distill_2l400K": ("Distill 2 (402K)", None, 402_608),
    "SVM_Distill_3x3BN": ("Distill 3 (615K)", None, 615_024),
    "SVM_Distill_Proj1280": ("Distill 4 (889K)", None, 889_200),
}

# Familia focal: pares que sustentam afirmacoes de empate / ganho desprezivel no texto.
# (modelo_a, modelo_b, afirmacao que o par sustenta)
FOCUS_PAIRS: list[tuple[str, str, str]] = [
    ("SVM_FLIM", "SVM_IJEPA",
     "\"FLIM praticamente empatado com o teacher\" (F1 medio 0,938 vs 0,940 @100%)"),
    ("SVM_Distill_3x3BN", "SVM_Distill_Proj1280",
     "\"+45% de params rendem +0,007 de F1 e +0,002 de kappa\" (retornos decrescentes)"),
    ("SVM_Distill_1x1BN_flim_frozen_eval_loss", "SVM_Distill_2l400K",
     "\"Distill 1 FLIM init tem kappa 0,82 contra 0,83 do Distill 2, com 1/3 dos params\""),
    ("SVM_Distill_2l400K", "SVM_Distill_3x3BN",
     "curva de retornos decrescentes das proj heads (Distill 2 -> 3)"),
    ("SVM_FLIM", "SVM_Distill_Proj1280",
     "fronteira de Pareto: FLIM puro (60K) contra o melhor destilado (889K)"),
]

METRICS: list[tuple[str, str]] = [("f1", "F1"), ("kappa", "kappa"), ("acc", "Acc")]

DELTA = 0.05
DATASETS = ["eggs", "larvae", "protozoan"]
PCTS = [1, 5, 25, 50, 75, 100]
N_BOOT = 10_000
SEED = 42

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "artifacts" / "normalized" / "unified_svm_comparison.csv"
# Os .csv/.md gerados continuam caindo em statistics/tools/, ao lado dos que ja estao
# versionados la: mover o .py nao pode, sozinho, mudar onde o relatorio nasce. Mesmo
# OUT_DIR de analysis/stats/compute_cost.py:62 — um lugar so para redirecionar depois.
OUT_DIR = REPO / "statistics" / "tools"
OUT_CSV = OUT_DIR / "wilcoxon_equivalence.csv"
OUT_MD = OUT_DIR / "wilcoxon_equivalence.md"


def load_series(df: pd.DataFrame, method: str, metric: str,
                pcts: list[int] | None = None) -> pd.Series:
    """Serie de `metric` indexada por (dataset_short, pretrained_pct) para um metodo."""
    pcts = PCTS if pcts is None else pcts
    label, init_filter, _ = MODELS[method]
    sub = df[df["method"] == method]
    if init_filter is not None:
        sub = sub[sub["init"] == init_filter]
    sub = sub.copy()
    sub["pretrained_pct"] = sub["pretrained_pct"].astype(int)
    sub["dataset_short"] = sub["dataset_short"].astype(str)

    dup = sub.duplicated(subset=["dataset_short", "pretrained_pct"]).sum()
    if dup:
        raise SystemExit(
            f"[FALHA] {method} ({label}) tem {dup} celula(s) duplicada(s). "
            "Filtro de init insuficiente."
        )
    if sub[metric].isna().any():
        raise SystemExit(f"[FALHA] {method} ({label}) tem NaN na coluna {metric}.")

    sub = sub[sub["pretrained_pct"].isin(pcts)]
    ser = sub.set_index(["dataset_short", "pretrained_pct"])[metric].astype(float).sort_index()
    expected = sorted((d, p) for d in DATASETS for p in pcts)
    if sorted(ser.index.tolist()) != expected:
        raise SystemExit(
            f"[FALHA] grade incompleta para {method} ({label}): "
            f"{len(ser)} celulas em vez de {len(expected)}."
        )
    return ser


def _wilcoxon(x, y=None, alternative="two-sided"):
    """Wilcoxon exato com fallback para 'auto' se o scipy recusar (empates)."""
    try:
        return stats.wilcoxon(x, y, alternative=alternative,
                              zero_method="wilcox", method="exact"), "exact"
    except Exception:
        return stats.wilcoxon(x, y, alternative=alternative,
                              zero_method="wilcox", method="auto"), "auto"


def superiority(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon bilateral + tamanho de efeito. Convencao: diff = b - a."""
    diff = b - a
    nz = diff[diff != 0.0]
    ranks = stats.rankdata(np.abs(nz)) if nz.size else np.array([])
    w_pos = float(ranks[nz > 0].sum()) if nz.size else 0.0
    w_neg = float(ranks[nz < 0].sum()) if nz.size else 0.0
    res, how = _wilcoxon(a, b, "two-sided")
    denom = w_pos + w_neg
    return {
        "n_pairs": int(diff.size), "n_eff": int(nz.size),
        "W": float(res.statistic), "W_pos": w_pos, "W_neg": w_neg,
        "p_sup_raw": float(res.pvalue), "method_wilcoxon": how,
        "median_diff": float(np.median(diff)), "mean_diff": float(np.mean(diff)),
        "rank_biserial_r": (w_pos - w_neg) / denom if denom > 0 else float("nan"),
        "n_b_wins": int((diff > 0).sum()), "n_a_wins": int((diff < 0).sum()),
        "n_ties": int((diff == 0).sum()),
    }


def tost_p(diff: np.ndarray, delta: float) -> float:
    """p do TOST: max dos dois testes unilaterais contra as nulas deslocadas +/- delta."""
    if delta <= 0:
        return 1.0
    p_lo, _ = _wilcoxon(diff - delta, None, "less")      # H1: mediana < +delta
    p_hi, _ = _wilcoxon(diff + delta, None, "greater")   # H1: mediana > -delta
    return float(max(p_lo.pvalue, p_hi.pvalue))


def delta_min(diff: np.ndarray, alpha: float, hi: float = 1.0) -> float:
    """Menor margem para a qual o TOST rejeita a alfa (busca binaria; NaN se nem hi serve)."""
    if tost_p(diff, hi) > alpha:
        return float("nan")
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if tost_p(diff, mid) <= alpha:
            hi = mid
        else:
            lo = mid
    return float(hi)


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> dict:
    """IC95% percentil por bootstrap das celulas pareadas (mediana e media)."""
    idx = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
    s = diff[idx]
    lo_m, hi_m = np.percentile(np.median(s, axis=1), [2.5, 97.5])
    lo_a, hi_a = np.percentile(s.mean(axis=1), [2.5, 97.5])
    return {"median_ci_lo": float(lo_m), "median_ci_hi": float(hi_m),
            "mean_ci_lo": float(lo_a), "mean_ci_hi": float(hi_a)}


def verdict(sig_sup: bool, sig_tost: bool, median_diff: float,
            label_a: str, label_b: str, delta: float) -> tuple[str, str]:
    """Veredito 2x2 (superioridade x equivalencia) + leitura em uma linha."""
    vencedor = label_b if median_diff > 0 else (label_a if median_diff < 0 else "nenhum")
    if sig_sup and sig_tost:
        return ("diferenca real porem menor que a margem",
                f"{vencedor} vence de forma consistente, mas a diferenca e menor que {delta:g} "
                f"— real e desprezivel ao mesmo tempo")
    if sig_sup and not sig_tost:
        return ("diferenca real e nao-desprezivel",
                f"{vencedor} e melhor, e a diferenca NAO cabe na margem de {delta:g}")
    if not sig_sup and sig_tost:
        return ("EQUIVALENTES",
                f"empate demonstrado: a diferenca esta comprovadamente dentro de +/-{delta:g}")
    return ("inconclusivo",
            f"sem poder para decidir: nao se detectou diferenca NEM se demonstrou empate "
            f"(tendencia a favor de {vencedor})")


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def fmt_d(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.4f}"


def analyse(df: pd.DataFrame, pairs: list[tuple[str, str, str]], familia: str,
            alpha: float, rng: np.random.Generator,
            pcts: list[int] | None = None) -> pd.DataFrame:
    """Roda superioridade + TOST para uma lista de pares, em todas as metricas."""
    pcts = PCTS if pcts is None else pcts
    rows: list[dict] = []
    for metric, mlabel in METRICS:
        for method_a, method_b, claim in pairs:
            sa = load_series(df, method_a, metric, pcts)
            sb = load_series(df, method_b, metric, pcts)
            if list(sa.index) != list(sb.index):
                raise SystemExit(f"[FALHA] grades divergem: {method_a} vs {method_b}")
            a, b = sa.to_numpy(), sb.to_numpy()
            diff = b - a

            row = {
                "familia": familia, "metric": metric, "metric_label": mlabel,
                "model_a": method_a, "label_a": MODELS[method_a][0],
                "params_a": MODELS[method_a][2],
                "model_b": method_b, "label_b": MODELS[method_b][0],
                "params_b": MODELS[method_b][2],
                "afirmacao_sustentada": claim,
                "delta": DELTA, "pcts": "+".join(str(p) for p in pcts),
                "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            }
            row.update(superiority(a, b))
            row.update(bootstrap_ci(diff, rng))
            row["p_tost_raw"] = tost_p(diff, DELTA)
            row["delta_min"] = delta_min(diff, alpha)
            rows.append(row)

    res = pd.DataFrame(rows)

    # Holm/Bonferroni dentro de cada metrica, separadamente para superioridade e TOST.
    for col, tag in (("p_sup_raw", "sup"), ("p_tost_raw", "tost")):
        for meth in ("holm", "bonferroni"):
            res[f"p_{tag}_{meth}"] = np.nan
            res[f"sig_{tag}_{meth}"] = False
        for metric, _ in METRICS:
            mask = res["metric"] == metric
            for meth in ("holm", "bonferroni"):
                rej, padj, _, _ = multipletests(res.loc[mask, col].to_numpy(),
                                                alpha=alpha, method=meth)
                res.loc[mask, f"p_{tag}_{meth}"] = padj
                res.loc[mask, f"sig_{tag}_{meth}"] = rej

    v = [verdict(s, t, m, la, lb, DELTA) for s, t, m, la, lb in
         zip(res["sig_sup_holm"], res["sig_tost_holm"], res["median_diff"],
             res["label_a"], res["label_b"])]
    res["veredito"] = [x[0] for x in v]
    res["leitura"] = [x[1] for x in v]
    return res


def wilcoxon_equivalence(csv: Path = DEFAULT_CSV, alpha: float = 0.05) -> int:
    # O corpo abaixo continua lendo `args.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    df = pd.read_csv(args.csv)
    needed = {"method", "init", "dataset_short", "pretrained_pct"} | {m for m, _ in METRICS}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"[FALHA] colunas ausentes em {args.csv}: {missing}")

    rng = np.random.default_rng(SEED)

    focal = analyse(df, FOCUS_PAIRS, "focal", args.alpha, rng)

    # Mesma familia focal descartando a fracao de 1%. Motivo: em varios pares quase toda a
    # diferenca agregada vem do regime de rotulo escasso, onde alguns modelos colapsam. Sem o
    # 1% (n = 15) ve-se o que sobra no regime de operacao realista.
    pcts_sem1 = [p for p in PCTS if p != 1]
    focal_sem1 = analyse(df, FOCUS_PAIRS, "focal_sem_1pct", args.alpha, rng, pcts_sem1)

    all_pairs = [(a, b, "") for a, b in itertools.combinations(MODELS, 2)]
    completa = analyse(df, all_pairs, "completa", args.alpha, rng)

    # Decomposicao descritiva por fracao de rotulos, para os pares focais.
    dec_rows: list[dict] = []
    for metric, mlabel in METRICS:
        for method_a, method_b, _ in FOCUS_PAIRS:
            sa = load_series(df, method_a, metric)
            sb = load_series(df, method_b, metric)
            for pct in PCTS:
                a_p = sa.xs(pct, level="pretrained_pct")
                b_p = sb.xs(pct, level="pretrained_pct")
                dec_rows.append({
                    "metric": metric, "metric_label": mlabel,
                    "label_a": MODELS[method_a][0], "label_b": MODELS[method_b][0],
                    "model_a": method_a, "model_b": method_b, "pct": pct,
                    "mean_a": float(a_p.mean()), "mean_b": float(b_p.mean()),
                    "delta": float(b_p.mean() - a_p.mean()),
                    "n_b_wins": int((b_p.to_numpy() > a_p.to_numpy()).sum()),
                })
    dec = pd.DataFrame(dec_rows)

    # Consistencia por dataset (mediana das 6 fracoes), pares focais.
    ds_rows: list[dict] = []
    for metric, mlabel in METRICS:
        for method_a, method_b, _ in FOCUS_PAIRS:
            d_ser = (load_series(df, method_b, metric) - load_series(df, method_a, metric))
            for ds in DATASETS:
                sub_d = d_ser.loc[ds]
                ds_rows.append({
                    "metric": metric, "metric_label": mlabel,
                    "label_a": MODELS[method_a][0], "label_b": MODELS[method_b][0],
                    "model_a": method_a, "model_b": method_b, "dataset": ds,
                    "median_diff": float(np.median(sub_d)),
                    "n_b_wins": int((sub_d > 0).sum()), "n_cells": int(len(sub_d)),
                })
    per_ds = pd.DataFrame(ds_rows)

    out = pd.concat([focal, focal_sem1, completa], ignore_index=True)
    out["sign_convention"] = "diff = metrica(model_b) - metrica(model_a); >0 favorece model_b"
    out.to_csv(OUT_CSV, index=False)

    # piso de ruido: std entre os 3 splits, nas celulas dos 8 modelos oficiais
    noise = {}
    for metric, _ in METRICS:
        col = f"{metric}_std"
        if col in df.columns:
            sub = df[df["method"].isin(MODELS)]
            sub = pd.concat([
                sub[(sub["method"] == m) & ((sub["init"] == MODELS[m][1]) if MODELS[m][1] else True)]
                for m in MODELS
            ])
            noise[metric] = (float(sub[col].median()), float(sub[col].quantile(0.9)))

    # ---------------- Markdown ----------------
    L: list[str] = []
    L.append("# Testes par-a-par e de equivalencia (TOST) entre os 8 modelos oficiais")
    L.append("")
    L.append(
        "Este relatorio existe por duas razoes. Primeira: os testes de `wilcoxon_f1.md`, "
        "`wilcoxon_acc.md` e `wilcoxon_kappa.md` sao um-contra-todos com o FLIM como baseline, "
        "logo cobrem 7 dos 28 pares possiveis — **Distill 3 vs Distill 4**, entre outros, ficava "
        "sem teste. Segunda, e mais importante: **ausencia de significancia nao e empate**."
    )
    L.append("")
    L.append(
        "Um Wilcoxon que nao rejeita diz apenas \"nao detectei diferenca\", o que e compativel "
        "tanto com \"os modelos sao iguais\" quanto com \"o teste nao tem poder\". Para escrever "
        "*empate* no artigo e preciso um **teste de equivalencia**, que inverte o onus da prova: "
        "a hipotese nula passa a ser \"a diferenca e grande\", e so se rejeita essa nula diante "
        "de evidencia positiva de que a diferenca e pequena. E o que o TOST faz aqui."
    )
    L.append("")

    L.append("## Desenho")
    L.append("")
    L.append(
        "Unidade pareada: celula `(dataset, % de rotulos)`, 3 datasets x 6 fracoes = **n = 18 "
        "pares**. Cada celula ja e a media sobre 3 splits. Pareamento pela chave, verificado "
        "antes de cada teste."
    )
    L.append("")
    L.append("**Convencao de sinal:** `diff = metrica(modelo B) - metrica(modelo A)`. Positivo favorece o **modelo B**.")
    L.append("")
    L.append("Cada par recebe **dois** testes, que respondem a perguntas diferentes:")
    L.append("")
    L.append("| Teste | Pergunta | H0 | Rejeitar significa |")
    L.append("|---|---|---|---|")
    L.append("| **Superioridade** (Wilcoxon bilateral) | Existe diferenca? | mediana = 0 | ha diferenca |")
    L.append(f"| **Equivalencia** (TOST) | A diferenca e pequena? | \\|mediana\\| >= {DELTA:g} | a diferenca cabe em +/-{DELTA:g} |")
    L.append("")
    L.append("Combinando os dois, cada par cai em uma de quatro caixas:")
    L.append("")
    L.append("| | TOST rejeita (dif. pequena) | TOST nao rejeita |")
    L.append("|---|---|---|")
    L.append("| **Superioridade rejeita** | diferenca real porem menor que a margem | diferenca real e nao-desprezivel |")
    L.append("| **Superioridade nao rejeita** | **EQUIVALENTES** (empate demonstrado) | inconclusivo (sem poder) |")
    L.append("")
    L.append(
        "Apenas a caixa **EQUIVALENTES** autoriza a palavra \"empate\" no texto. A caixa "
        "\"inconclusivo\" e onde caem hoje varias afirmacoes do artigo."
    )
    L.append("")

    L.append("## Margem de equivalencia")
    L.append("")
    L.append(
        f"`delta = {DELTA:g}` ({DELTA * 100:g} pontos da metrica), **pre-especificada** — escolhida "
        "antes de ver os resultados, nao ajustada depois. Justificativa dupla:"
    )
    L.append("")
    L.append(f"1. E o arredondamento convencional para \"diferenca praticamente irrelevante\" nesta escala de metrica.")
    if noise:
        parts = ", ".join(f"{m}: {v[0]:.4f}" for m, v in noise.items())
        L.append(
            f"2. Fica em torno de **2x o ruido de medicao do proprio experimento**: o "
            f"desvio-padrao tipico entre os 3 splits da mesma celula e ({parts}); no percentil 90 "
            f"chega a ({', '.join(f'{m}: {v[1]:.4f}' for m, v in noise.items())}). Uma diferenca "
            f"menor que {DELTA:g} esta na ordem de grandeza do ruido entre splits."
        )
    L.append("")
    L.append(
        "Para nao obrigar o leitor a aceitar essa escolha, toda linha traz tambem **`delta_min`**: "
        "a menor margem para a qual a equivalencia se sustenta a alfa = "
        f"{args.alpha}. Leitura direta: *\"estes dois modelos sao equivalentes dentro de "
        "+/- delta_min\"*. Quanto menor, mais forte o empate. `n/a` significa que a equivalencia "
        "nao se sustenta em margem nenhuma."
    )
    L.append("")

    L.append("## 1. Familia focal (tabela do artigo)")
    L.append("")
    L.append(
        "Os pares que sustentam afirmacoes explicitas de empate ou de ganho desprezivel no texto. "
        f"Familia pre-especificada de {len(FOCUS_PAIRS)} pares, correcao de Holm por metrica, "
        f"alfa = {args.alpha}."
    )
    L.append("")
    for metric, mlabel in METRICS:
        sub = focal[focal["metric"] == metric]
        L.append(f"### {mlabel}")
        L.append("")
        L.append("| A | B | Mediana B-A | IC95% | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |")
        L.append("|---|---|---:|---|---:|---:|---:|---:|---|")
        for _, r in sub.iterrows():
            L.append(
                f"| {r['label_a']} | {r['label_b']} | {r['median_diff']:+.4f} "
                f"| [{r['median_ci_lo']:+.3f}, {r['median_ci_hi']:+.3f}] "
                f"| {r['n_b_wins']}/{r['n_pairs']} "
                f"| {fmt_p(r['p_sup_holm'])} | {fmt_p(r['p_tost_holm'])} "
                f"| {fmt_d(r['delta_min'])} | **{r['veredito']}** |"
            )
        L.append("")

    L.append("### Afirmacao que cada par sustenta")
    L.append("")
    L.append("| A | B | Afirmacao no texto | F1 | kappa | Acc |")
    L.append("|---|---|---|---|---|---|")
    for method_a, method_b, claim in FOCUS_PAIRS:
        vs = []
        for metric, _ in METRICS:
            r = focal[(focal["metric"] == metric) & (focal["model_a"] == method_a)
                      & (focal["model_b"] == method_b)].iloc[0]
            vs.append(r["veredito"])
        L.append(f"| {MODELS[method_a][0]} | {MODELS[method_b][0]} | {claim} | {vs[0]} | {vs[1]} | {vs[2]} |")
    L.append("")

    L.append("## 1b. Decomposicao por fracao de rotulos (descritiva)")
    L.append("")
    L.append(
        "A mediana agregada sobre as 18 celulas esconde a estrutura mais importante do "
        "experimento: em varios pares, quase toda a diferenca vem do regime de rotulo escasso. "
        "As tabelas abaixo mostram a diferenca media (B - A) em cada fracao, e `B vence` conta "
        "em quantos dos 3 datasets o modelo B ficou a frente naquela fracao."
    )
    L.append("")
    for metric, mlabel in METRICS:
        L.append(f"### {mlabel}")
        L.append("")
        header = "| Par (A vs B) | " + " | ".join(f"{p}%" for p in PCTS) + " |"
        L.append(header)
        L.append("|---|" + "---:|" * len(PCTS))
        for method_a, method_b, _ in FOCUS_PAIRS:
            sub = dec[(dec["metric"] == metric) & (dec["model_a"] == method_a)
                      & (dec["model_b"] == method_b)].set_index("pct")
            cells = " | ".join(
                f"{sub.loc[p, 'delta']:+.3f} ({int(sub.loc[p, 'n_b_wins'])}/3)" for p in PCTS
            )
            L.append(f"| {MODELS[method_a][0]} vs {MODELS[method_b][0]} | {cells} |")
        L.append("")

    L.append("## 1c. Familia focal descartando a fracao de 1% (n = 15)")
    L.append("")
    L.append(
        "Repeticao integral da familia focal sobre as 5 fracoes de 5% a 100%. Serve para "
        "responder a pergunta \"a diferenca sobrevive fora do regime de colapso?\". Onde um "
        "veredito muda entre esta tabela e a de §1, a diferenca agregada era sustentada "
        "principalmente pela fracao de 1%."
    )
    L.append("")
    for metric, mlabel in METRICS:
        sub = focal_sem1[focal_sem1["metric"] == metric]
        sub18 = focal[focal["metric"] == metric]
        L.append(f"### {mlabel}")
        L.append("")
        L.append("| A | B | Mediana B-A | B vence | p superior. (Holm) | delta_min | Veredito (n=15) | Veredito (n=18) |")
        L.append("|---|---|---:|---:|---:|---:|---|---|")
        for _, r in sub.iterrows():
            r18 = sub18[(sub18["model_a"] == r["model_a"])
                        & (sub18["model_b"] == r["model_b"])].iloc[0]
            mudou = "" if r["veredito"] == r18["veredito"] else " ⚠"
            L.append(
                f"| {r['label_a']} | {r['label_b']} | {r['median_diff']:+.4f} "
                f"| {r['n_b_wins']}/{r['n_pairs']} | {fmt_p(r['p_sup_holm'])} "
                f"| {fmt_d(r['delta_min'])} | **{r['veredito']}**{mudou} | {r18['veredito']} |"
            )
        L.append("")

    L.append("## 1d. Consistencia por dataset (pares focais)")
    L.append("")
    L.append(
        "Mediana da diferenca dentro de cada dataset (6 fracoes cada). Um par cujo **sinal muda** "
        "entre datasets tem vantagem dependente de dominio, e a mediana agregada esconde isso."
    )
    L.append("")
    for metric, mlabel in METRICS:
        L.append(f"### {mlabel}")
        L.append("")
        L.append("| Par (A vs B) | " + " | ".join(DATASETS) + " | sinal consistente? |")
        L.append("|---|" + "---:|" * len(DATASETS) + "---|")
        for method_a, method_b, _ in FOCUS_PAIRS:
            sub = per_ds[(per_ds["metric"] == metric) & (per_ds["model_a"] == method_a)
                         & (per_ds["model_b"] == method_b)].set_index("dataset")
            vals = [sub.loc[ds, "median_diff"] for ds in DATASETS]
            cells = " | ".join(
                f"{v:+.3f} ({int(sub.loc[ds, 'n_b_wins'])}/6)" for ds, v in zip(DATASETS, vals)
            )
            consistente = "sim" if (all(v > 0 for v in vals) or all(v < 0 for v in vals)) \
                else "**NAO — troca de sinal**"
            L.append(f"| {MODELS[method_a][0]} vs {MODELS[method_b][0]} | {cells} | {consistente} |")
        L.append("")

    L.append("## 2. Matriz completa dos 28 pares")
    L.append("")
    L.append(
        "Nenhum par fica sem teste. Holm sobre os 28 pares, por metrica — correcao bem mais "
        "severa que a da familia focal, logo os vereditos aqui sao mais conservadores. Ordenado "
        "por magnitude da mediana da diferenca."
    )
    L.append("")
    for metric, mlabel in METRICS:
        sub = completa[completa["metric"] == metric].copy()
        sub["abs_med"] = sub["median_diff"].abs()
        sub = sub.sort_values("abs_med")
        L.append(f"### {mlabel} (28 pares)")
        L.append("")
        L.append("| A | B | Mediana B-A | B vence | p superior. (Holm) | p TOST (Holm) | delta_min | Veredito |")
        L.append("|---|---|---:|---:|---:|---:|---:|---|")
        for _, r in sub.iterrows():
            L.append(
                f"| {r['label_a']} | {r['label_b']} | {r['median_diff']:+.4f} "
                f"| {r['n_b_wins']}/{r['n_pairs']} | {fmt_p(r['p_sup_holm'])} "
                f"| {fmt_p(r['p_tost_holm'])} | {fmt_d(r['delta_min'])} | {r['veredito']} |"
            )
        L.append("")

    # ---- leitura ----
    def get(fam, metric, a, b):
        d = fam[(fam["metric"] == metric) & (fam["model_a"] == a) & (fam["model_b"] == b)]
        return d.iloc[0] if len(d) else None

    L.append("## 3. O que isso muda no texto do artigo")
    L.append("")

    f1i = get(focal, "f1", "SVM_FLIM", "SVM_IJEPA")
    kai = get(focal, "kappa", "SVM_FLIM", "SVM_IJEPA")
    aci = get(focal, "acc", "SVM_FLIM", "SVM_IJEPA")
    _f15 = focal_sem1[(focal_sem1["metric"] == "f1") & (focal_sem1["model_a"] == "SVM_FLIM")
                      & (focal_sem1["model_b"] == "SVM_IJEPA")].iloc[0]
    if _f15["veredito"] == "EQUIVALENTES":
        L.append("### 3.1 \"FLIM praticamente empatado com o I-JEPA\" — verdadeiro, mas precisa de escopo")
    else:
        L.append("### 3.1 \"FLIM praticamente empatado com o I-JEPA\" — nao se sustenta")
    L.append("")
    L.append(
        f"Em **F1** o par e **{f1i['veredito']}**: nao se detecta diferenca "
        f"(p Holm = {fmt_p(f1i['p_sup_holm'])}) *e* nao se demonstra equivalencia "
        f"(p TOST Holm = {fmt_p(f1i['p_tost_holm'])}). O `delta_min` = "
        f"**{fmt_d(f1i['delta_min'])}** diz o tamanho do problema: so seria possivel declarar "
        f"empate admitindo uma margem de +/-{f1i['delta_min']:.2f} de F1, o que e largo demais "
        "para ter sentido pratico. Com n = 18 o experimento simplesmente nao distingue os dois."
    )
    L.append("")
    L.append(
        f"Em **kappa** a situacao e pior para a afirmacao: o I-JEPA vence em "
        f"{kai['n_b_wins']}/{kai['n_pairs']} celulas com mediana {kai['median_diff']:+.4f} e "
        f"p Holm = {fmt_p(kai['p_sup_holm'])} — **diferenca real e nao-desprezivel**. Em "
        f"**Acc**, {aci['veredito']} ({fmt_p(aci['p_sup_holm'])})."
    )
    L.append("")
    # decomposicao do gap de kappa por fracao, gerada dos dados
    kd = dec[(dec["metric"] == "kappa") & (dec["model_a"] == "SVM_FLIM")
             & (dec["model_b"] == "SVM_IJEPA")].set_index("pct")
    k15 = focal_sem1[(focal_sem1["metric"] == "kappa") & (focal_sem1["model_a"] == "SVM_FLIM")
                     & (focal_sem1["model_b"] == "SVM_IJEPA")].iloc[0]
    kds = per_ds[(per_ds["metric"] == "kappa") & (per_ds["model_a"] == "SVM_FLIM")
                 & (per_ds["model_b"] == "SVM_IJEPA")].set_index("dataset")
    L.append(
        "O tamanho dessa vantagem em kappa depende quase inteiramente da quantidade de rotulos "
        f"(§1b): {kd.loc[1, 'delta']:+.3f} a 1%, {kd.loc[5, 'delta']:+.3f} a 5%, "
        f"{kd.loc[25, 'delta']:+.3f} a 25% e {kd.loc[100, 'delta']:+.3f} a 100% "
        f"(kappa {kd.loc[100, 'mean_a']:.3f} do FLIM contra {kd.loc[100, 'mean_b']:.3f} do "
        "teacher). Quase toda a superioridade se concentra no regime de rotulo escasso, onde o "
        "FLIM colapsa, e estabiliza em torno de 0,07 a partir de 25%."
    )
    L.append("")
    L.append(
        f"Mas ela **nao desaparece** fora desse regime: descartando a fracao de 1% (§1c), o "
        f"I-JEPA ainda vence em {k15['n_b_wins']}/{k15['n_pairs']} celulas, com mediana "
        f"{k15['median_diff']:+.4f} e p Holm = {fmt_p(k15['p_sup_holm'])}. E e consistente nos "
        "tres dominios (§1d): "
        + ", ".join(f"{ds} {kds.loc[ds, 'median_diff']:+.3f}" for ds in DATASETS)
        + " — sem a troca de sinal entre datasets que o F1 apresenta."
    )
    L.append("")
    f1i15 = focal_sem1[(focal_sem1["metric"] == "f1") & (focal_sem1["model_a"] == "SVM_FLIM")
                       & (focal_sem1["model_b"] == "SVM_IJEPA")].iloc[0]
    aci15 = focal_sem1[(focal_sem1["metric"] == "acc") & (focal_sem1["model_a"] == "SVM_FLIM")
                       & (focal_sem1["model_b"] == "SVM_IJEPA")].iloc[0]
    if f1i15["veredito"] == "EQUIVALENTES":
        L.append(
            "**O empate existe, mas so fora do regime de colapso e so em F1/Acc.** Esta e a "
            "descoberta mais util deste relatorio para a redacao do artigo. Descartando a fracao "
            f"de 1%, o par FLIM vs I-JEPA passa a **{f1i15['veredito']}** em F1 "
            f"(mediana {f1i15['median_diff']:+.4f}, superioridade p = {fmt_p(f1i15['p_sup_holm'])}, "
            f"TOST p = {fmt_p(f1i15['p_tost_holm'])}, `delta_min` = **{fmt_d(f1i15['delta_min'])}**) "
            f"e tambem em Acc (`delta_min` = {fmt_d(aci15['delta_min'])}). O `delta_min` de "
            f"{f1i15['delta_min']:.3f} e uma margem **apertada** — cerca de um terco dos "
            f"{DELTA:g} pre-especificados e da ordem do ruido entre splits. Ou seja: a partir de "
            "5% dos rotulos, FLIM e I-JEPA sao equivalentes em F1 e acuracia dentro de uma margem "
            "muito estreita, e isso agora esta **demonstrado**, nao apenas nao-refutado."
        )
        L.append("")
        L.append(
            "O contraste com o agregado de 18 celulas nao e contradicao: as 3 celulas de 1% "
            f"sozinhas ({kd.loc[1, 'delta']:+.3f} de kappa, e +0,516 de F1) carregavam toda a "
            "diferenca e ao mesmo tempo inflavam a dispersao, o que impedia o TOST de concluir "
            "qualquer coisa. Separar os dois regimes resolve as duas pontas."
        )
        L.append("")
        L.append(
            "> **Redacao sugerida.** Trocar \"estatisticamente empatados\" por: *\"a partir de 5% "
            "dos rotulos, o FLIM e estatisticamente equivalente ao teacher I-JEPA em F1 e "
            f"acuracia, dentro de uma margem de +/-{f1i15['delta_min']:.3f} "
            f"(TOST pareado, n = 15, p = {fmt_p(f1i15['p_tost_holm'])}), com ~10.600x menos "
            f"parametros. O teacher mantem vantagem em kappa ({k15['median_diff']:+.3f}, "
            f"p = {fmt_p(k15['p_sup_holm'])}), e domina no regime de 1% de rotulos, onde o FLIM "
            "colapsa.\"* Essa formulacao e mais forte que a original, porque troca uma afirmacao "
            "de empate sem teste por uma equivalencia demonstrada com escopo declarado."
        )
    else:
        L.append(
            "> **Redacao sugerida.** Trocar \"estatisticamente empatados\" por: *\"o FLIM alcanca "
            "F1 medio equiparavel ao do teacher (0,938 vs 0,940 a 100% dos rotulos) com ~10.600x "
            "menos parametros; o teste pareado nao detecta diferenca em F1 "
            f"(p = {fmt_p(f1i['p_sup_holm'])}), mas tambem nao estabelece equivalencia "
            f"estatistica, e o I-JEPA mantem vantagem consistente em kappa "
            f"({kai['median_diff']:+.3f}, p = {fmt_p(kai['p_sup_holm'])}). A comparacao deve ser "
            "lida como custo-beneficio, nao como empate.\"*"
        )
    L.append("")

    L.append("### 3.2 \"Distill 3 vs Distill 4: ganho praticamente zero\" — quase certo, mas o numero esta errado")
    L.append("")
    f1d = get(focal, "f1", "SVM_Distill_3x3BN", "SVM_Distill_Proj1280")
    kad = get(focal, "kappa", "SVM_Distill_3x3BN", "SVM_Distill_Proj1280")
    acd = get(focal, "acc", "SVM_Distill_3x3BN", "SVM_Distill_Proj1280")
    L.append(
        f"Este e o par mais interessante do relatorio, e cai na caixa **{f1d['veredito']}** nas "
        "tres metricas. Em F1: o Distill 4 vence em "
        f"{f1d['n_b_wins']}/{f1d['n_pairs']} celulas, com p Holm = {fmt_p(f1d['p_sup_holm'])} — "
        "a diferenca **existe e e consistente**. Mas o TOST tambem rejeita "
        f"(p Holm = {fmt_p(f1d['p_tost_holm'])}, `delta_min` = {fmt_d(f1d['delta_min'])}): ela "
        f"esta **comprovadamente abaixo de {f1d['delta_min']:.3f} de F1**. Mesmo padrao em kappa "
        f"({fmt_d(kad['delta_min'])}) e Acc ({fmt_d(acd['delta_min'])})."
    )
    L.append("")
    cd3 = completa[(completa["model_a"] == "SVM_Distill_3x3BN")
                   & (completa["model_b"] == "SVM_Distill_Proj1280")]
    n_small_c = int((cd3["veredito"] == "diferenca real porem menor que a margem").sum())
    if n_small_c < 3:
        perdidas = ", ".join(
            cd3[cd3["veredito"] != "diferenca real porem menor que a margem"]["metric_label"]
        )
        L.append(
            f"Sob a correcao mais severa da matriz completa (Holm sobre 28 pares), a metade de "
            f"equivalencia sobrevive em {n_small_c} das 3 metricas — em {perdidas} o TOST deixa de "
            "rejeitar. A conclusao a citar e a da familia focal, que e pre-especificada; a queda "
            "na matriz completa e o custo de testar tudo, nao um resultado contraditorio."
        )
        L.append("")
    L.append(
        f"O numero \"+0,007 de F1\" do texto vem so do recorte a **100% dos rotulos**, com n = 3 "
        "datasets — sem poder para teste algum. Sobre as 18 celulas a mediana e "
        f"{f1d['median_diff']:+.4f}, cerca de 4x maior. A conclusao qualitativa (\"nao vale a pena "
        "pagar +45% de params\") se mantem, e agora com respaldo estatistico bem mais forte do "
        "que a versao original — mas o numero citado precisa mudar."
    )
    L.append("")
    L.append(
        "> **Redacao sugerida.** *\"Subir de Distill 3 (615K) para Distill 4 (889K) — +45% de "
        f"parametros — produz um ganho consistente porem minusculo: mediana de {f1d['median_diff']:+.3f} "
        f"de F1 sobre as 18 celulas (p = {fmt_p(f1d['p_sup_holm'])}), com equivalencia demonstrada "
        f"dentro de +/-{f1d['delta_min']:.3f} (TOST, p = {fmt_p(f1d['p_tost_holm'])}). Ou seja: a "
        "diferenca e real, e e comprovadamente desprezivel.\"* Essa formulacao e mais forte que "
        "\"praticamente zero\", porque poe um limite superior demonstrado no tamanho do ganho."
    )
    L.append("")

    L.append("### 3.3 Distill 1 FLIM init (123K) vs Distill 2 (402K)")
    L.append("")
    f1e = get(focal, "f1", "SVM_Distill_1x1BN_flim_frozen_eval_loss", "SVM_Distill_2l400K")
    kae = get(focal, "kappa", "SVM_Distill_1x1BN_flim_frozen_eval_loss", "SVM_Distill_2l400K")
    ace = get(focal, "acc", "SVM_Distill_1x1BN_flim_frozen_eval_loss", "SVM_Distill_2l400K")
    L.append(
        f"O resultado e **dependente da metrica**, e o texto precisa dizer isso. Em **kappa** o "
        f"veredito e **{kae['veredito']}**: mediana {kae['median_diff']:+.4f} (a favor de "
        f"{kae['label_a'] if kae['median_diff'] < 0 else kae['label_b']}), superioridade "
        f"p = {fmt_p(kae['p_sup_holm'])}, TOST p = {fmt_p(kae['p_tost_holm'])}, "
        f"delta_min = {fmt_d(kae['delta_min'])}."
    )
    L.append("")
    if kae["veredito"] == "EQUIVALENTES":
        L.append(
            "Esse e o **unico empate genuinamente demonstrado** da familia focal, e favorece a "
            "tese do artigo: em kappa, 123K params com init FLIM congelada equivalem a 402K "
            "params com init aleatoria — 3,3x menos parametros para o mesmo kappa."
        )
    else:
        L.append(
            f"E o par que **mais perto chega** de um empate demonstrado em todo o estudo, e ainda "
            f"assim nao chega: o TOST bruto da p = {fmt_p(kae['p_tost_raw'])} (passaria sozinho), "
            f"mas nao sobrevive a correcao de Holm sobre os {len(FOCUS_PAIRS)} pares da familia "
            f"({fmt_p(kae['p_tost_holm'])}). O `delta_min` = {fmt_d(kae['delta_min'])} explica "
            f"por que e tao apertado: a equivalencia se sustentaria com uma margem de "
            f"+/-{kae['delta_min']:.3f}, logo abaixo dos {DELTA:g} pre-especificados. "
            "**Nao escreva \"mesmo kappa\" no artigo** — escreva que a diferenca em kappa e "
            f"indistinguivel de zero nesta bateria (mediana {kae['median_diff']:+.3f}, "
            f"p = {fmt_p(kae['p_sup_holm'])}) e que o experimento nao tem poder para converter "
            "isso em equivalencia formal."
        )
    L.append("")
    L.append(
        f"Em **F1** o par e {f1e['veredito']} (p = {fmt_p(f1e['p_sup_holm'])}) e em **Acc** o "
        f"Distill 2 leva vantagem real ({ace['median_diff']:+.4f}, p = {fmt_p(ace['p_sup_holm'])}, "
        f"delta_min = {fmt_d(ace['delta_min'])}). A afirmacao \"1/3 dos parametros pelo mesmo "
        "desempenho\" so e defensavel em kappa, e mesmo la apenas como ausencia de diferenca "
        "detectavel — nao como equivalencia. Declarar essa dependencia de metrica evita a critica "
        "de cherry-picking."
    )
    L.append("")

    n_equiv = int((completa["veredito"] == "EQUIVALENTES").sum())
    n_incon = int((completa["veredito"] == "inconclusivo").sum())
    n_small = int((completa["veredito"] == "diferenca real porem menor que a margem").sum())
    n_real = int((completa["veredito"] == "diferenca real e nao-desprezivel").sum())
    L.append("### 3.4 Panorama dos 28 pares")
    L.append("")
    L.append(f"Sobre as {len(completa)} linhas da matriz completa (28 pares x 3 metricas):")
    L.append("")
    L.append(f"- **{n_real}** sao diferencas reais e nao-despreziveis;")
    L.append(f"- **{n_small}** sao diferencas reais porem menores que a margem de {DELTA:g};")
    L.append(f"- **{n_equiv}** sao equivalencias demonstradas;")
    L.append(f"- **{n_incon}** ficam inconclusivas.")
    L.append("")
    n_equiv15 = int((focal_sem1["veredito"] == "EQUIVALENTES").sum())
    if n_equiv == 0:
        L.append(
            f"**Sobre as 18 celulas completas, em nenhum dos 28 pares e em nenhuma das 3 metricas "
            f"ha empate estatisticamente demonstrado.** Nesse recorte a palavra \"empate\" nao "
            "deveria aparecer no texto. Restam duas construcoes defensaveis:"
        )
        L.append("")
        small = completa[completa["veredito"] == "diferenca real porem menor que a margem"]
        quais = sorted({f"{r['label_a']} vs {r['label_b']}" for _, r in small.iterrows()})
        onde = "; ".join(
            f"{q} ({', '.join(small[(small['label_a'] + ' vs ' + small['label_b']) == q]['metric_label'])})"
            for q in quais
        ) if len(small) else "nenhuma"
        L.append(
            f"1. **\"diferenca real porem limitada a X\"** — quando o TOST rejeita junto com a "
            f"superioridade. Ocorre em {n_small} celula(s): {onde}. E a formulacao mais forte "
            "disponivel, porque poe um teto demonstrado no tamanho do ganho."
        )
        L.append(
            f"2. **\"nao se detectou diferenca, e o experimento nao tem poder para estabelecer "
            f"equivalencia\"** — nas {n_incon} celulas inconclusivas, entre elas FLIM vs I-JEPA "
            "em F1 e Acc."
        )
    else:
        L.append(
            f"As {n_equiv} equivalencias demonstradas sao as unicas ocorrencias em que o texto "
            "pode usar a palavra \"empate\"."
        )
    L.append("")
    if n_equiv15:
        eq15 = focal_sem1[focal_sem1["veredito"] == "EQUIVALENTES"]
        quais15 = "; ".join(
            f"{r['label_a']} vs {r['label_b']} em {r['metric_label']} "
            f"(delta_min {fmt_d(r['delta_min'])})" for _, r in eq15.iterrows()
        )
        L.append(
            f"**Mas o quadro muda ao separar o regime de rotulo escasso.** Descartando a fracao "
            f"de 1% (§1c), {n_equiv15} celula(s) da familia focal atingem equivalencia "
            f"demonstrada: {quais15}. A licao metodologica e que a fracao de 1% estava "
            "envenenando o agregado nas duas direcoes — inflava as diferencas medias e, por "
            "aumentar a dispersao, impedia o TOST de concluir. **Qualquer afirmacao de "
            "equivalencia no artigo deve declarar o regime de supervisao a que se aplica.**"
        )
        L.append("")
    L.append(
        f"As {n_real} celulas de diferenca real e nao-desprezivel mostram que o problema nao e "
        "falta de poder generalizada — a bateria separa bem os modelos distantes entre si. O que "
        "ela dificilmente consegue, no agregado de 18 celulas, e **provar igualdade** entre os "
        "modelos proximos, que e exatamente o que as afirmacoes de empate do texto exigem."
    )
    L.append("")

    L.append("## 4. Ressalvas")
    L.append("")
    L.append(
        "- **A coluna `f1` e F1 ponderado, nao macro.** `scripts/normalize_reports.py:114` mapeia "
        "`test_f1_weighted -> f1`; `src/evaluate/svm_classification_flim.py:144` confirma "
        "`average=\"weighted\"`. O cabecalho de `metrics_distillation/params_vs_quality_flim.md` "
        "diz \"F1 (macro)\" e **esta errado**. Isso importa para a leitura deste relatorio: F1 "
        "ponderado e inflado pela classe majoritaria em dados desbalanceados, enquanto o kappa "
        "desconta concordancia por acaso. Quando as duas metricas divergem — como em FLIM vs "
        "I-JEPA, inconclusivo em F1 e diferenca real em kappa — o kappa e o numero mais honesto, "
        "e a divergencia provavelmente reflete desempenho pior nas classes minoritarias."
    )
    L.append(
        "- **As 18 celulas nao sao independentes.** Dentro de um dataset, os subconjuntos de "
        "treino sao estritamente encaixados (1% ⊂ 5% ⊂ ... ⊂ 100%, contencao verificada nos JSONs "
        "de `data/to_mateus/splits_incremental/`) e o conjunto de teste e identico nas 6 fracoes. "
        "O Wilcoxon supoe pares independentes, logo n = 18 e otimista para **ambos** os testes. "
        "Para a superioridade isso e anticonservador (p otimista); para o TOST tambem, o que "
        "significa que as equivalencias declaradas aqui devem ser lidas como o limite superior "
        "do que os dados sustentam."
    )
    L.append(
        "- **A margem delta e uma escolha, nao um fato.** Por isso `delta_min` esta em toda "
        "linha: quem discordar de "
        f"{DELTA:g} le a coluna e aplica o proprio criterio."
    )
    L.append(
        "- **Equivalencia nao e identidade.** Declarar que dois modelos sao equivalentes dentro "
        f"de +/-{DELTA:g} nao os torna intercambiaveis para todo fim — apenas afirma que a "
        "diferenca media de qualidade nesta bateria esta abaixo desse limiar."
    )
    L.append(
        "- Os `.ckpt` das runs Distill 1/2/3/4 nao estao mais no disco; contagens de parametros "
        "vem da topologia reinstanciada (secao 5 do `README.md` deste diretorio)."
    )
    L.append(
        "- A matriz completa usa Holm sobre 28 pares. Um par pode aparecer como significativo na "
        "familia focal e nao-significativo na matriz completa: nao e contradicao, e o preco da "
        "cobertura exaustiva. A familia focal e pre-especificada e e a que sustenta as conclusoes."
    )
    L.append("")

    L.append("## 5. Reproducao")
    L.append("")
    L.append("```bash")
    L.append("cd /dados/home/moliveira/Scalable_Hybrid_FLIM")
    L.append("LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \\")
    L.append("/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \\")
    L.append("    -m analysis.stats.wilcoxon_equivalence")
    L.append("```")
    L.append("")
    L.append(
        f"Versoes: scipy {__import__('scipy').__version__}, "
        f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}, "
        f"pandas {pd.__version__}."
    )
    L.append("")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT_CSV.relative_to(REPO)}")
    print(f"[ok] {OUT_MD.relative_to(REPO)}")
    print()
    print("Familia focal:")
    print(focal[["metric", "label_a", "label_b", "median_diff", "p_sup_holm",
                 "p_tost_holm", "delta_min", "veredito"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    wilcoxon_equivalence()
