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
"""Wilcoxon pareado de sinais (signed-rank) na metrica F1: baseline vs cada modelo.

Desenho experimental
--------------------
Pedido do Revisor 3 (SIBGRAPI camera-ready): teste estatistico pareado comparando
cada modelo contra um baseline, com correcao para multiplas comparacoes.

* Unidade pareada: a celula ``(dataset_short, pretrained_pct)``.
  3 datasets (eggs, larvae, protozoan) x 6 fracoes de pre-treino (1, 5, 25, 50, 75, 100)
  => n = 18 pares por comparacao. Cada celula do CSV de entrada ja e a media sobre
  3 splits, logo nao ha re-agregacao aqui.
  O pareamento e feito pela chave (dataset, fracao), nunca por posicao de linha;
  o script falha alto se as chaves de baseline e modelo nao coincidirem exatamente.
* Baseline: ``SVM_FLIM`` (rotulo do artigo "FLIM (59.504)"). O artigo e uma comparacao
  *contra o FLIM*, portanto o desenho e um-contra-todos: 7 comparacoes
  (FLIM vs cada um dos outros 7 modelos oficiais).
* ``SVM_lejepa_view`` aparece no CSV com 5 inicializacoes diferentes (90 linhas). Somente
  ``init == "trunc_normal"`` entra no artigo; o filtro esta codificado em MODELS.
* Teste: ``scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided",
  zero_method="wilcox")``. Tenta ``method="exact"`` primeiro (n <= 18 permite) e cai
  para ``method="auto"`` se o scipy recusar; o metodo efetivamente usado e reportado.
* Correcao de multiplicidade: ``statsmodels.stats.multitest.multipletests`` com
  ``holm`` (principal) e ``bonferroni`` (referencia conservadora), sobre a familia das
  7 comparacoes. alpha = 0.05.
* Tamanho de efeito: correlacao rank-biserial pareada r = (W+ - W-) / (W+ + W-),
  calculada sobre as diferencas ``modelo - baseline``, mais a mediana e a media das
  diferencas com IC95% por bootstrap percentil (10.000 reamostragens das 18 celulas
  pareadas, ``numpy.random.default_rng(42)``).

Convencao de sinal (usada em todo o script e nos relatorios)
------------------------------------------------------------
    diferenca = F1(modelo) - F1(baseline FLIM)
    diferenca > 0  =>  o modelo e melhor que o FLIM
    diferenca < 0  =>  o FLIM e melhor que o modelo
O mesmo vale para r rank-biserial: r > 0 favorece o modelo, r < 0 favorece o FLIM.
A coluna ``melhor`` do CSV/MD nomeia explicitamente o vencedor.

Analise secundaria (exploratoria)
---------------------------------
O mesmo Wilcoxon repetido dentro de cada dataset (n = 6 fracoes). Serve apenas para
inspecionar consistencia entre datasets; com n = 6 o poder e minimo e o p minimo
alcancavel pelo teste exato bilateral e 0.03125, logo nenhuma correcao de
multiplicidade e aplicada nessa tabela e ela nao sustenta conclusao.

Por que nao ``scikit-posthocs``
-------------------------------
O ``scikit_posthocs.posthoc_wilcoxon`` roda todos-contra-todos (all-vs-all): com 8
modelos seriam 28 comparacoes. O desenho aqui e um-contra-todos (baseline vs cada
modelo), com familia de 7 hipoteses. Usar a rotina all-vs-all mudaria a familia de
correcao e inflaria desnecessariamente o ajuste de p. Por isso o teste e a correcao
sao montados diretamente com scipy + statsmodels.

Uso
---
    cd /dados/home/moliveira/Scalable_Hybrid_FLIM
    LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
    /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
        statistics/tools/wilcoxon_f1.py

Saidas: statistics/tools/wilcoxon_f1.csv e statistics/tools/wilcoxon_f1.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRIC = "f1"
METRIC_LABEL = "F1"

# method -> (rotulo do artigo, filtro de init ou None)
MODELS: dict[str, tuple[str, str | None]] = {
    "SVM_FLIM": ("FLIM (59.504)", None),
    "SVM_lejepa_view": ("lejepa_view", "trunc_normal"),
    "SVM_IJEPA": ("I-JEPA (632M)", None),
    "SVM_Distill_Proj1280": ("Distill 4 (889K)", None),
    "SVM_Distill_3x3BN": ("Distill 3 (615K)", None),
    "SVM_Distill_1x1BN": ("Distill 1 (123K)", None),
    "SVM_Distill_2l400K": ("Distill 2 (402K)", None),
    "SVM_Distill_1x1BN_flim_frozen_eval_loss": ("Distill 1 - FLIM init (123K)", None),
}

DATASETS = ["eggs", "larvae", "protozoan"]
PCTS = [1, 5, 25, 50, 75, 100]
N_BOOT = 10_000
SEED = 42

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "artifacts" / "normalized" / "unified_svm_comparison.csv"
OUT_CSV = Path(__file__).resolve().parent / "wilcoxon_f1.csv"
OUT_MD = Path(__file__).resolve().parent / "wilcoxon_f1.md"


def load_series(df: pd.DataFrame, method: str) -> pd.Series:
    """Serie de F1 indexada por (dataset_short, pretrained_pct) para um metodo."""
    label, init_filter = MODELS[method]
    sub = df[df["method"] == method]
    if init_filter is not None:
        sub = sub[sub["init"] == init_filter]
    sub = sub.copy()
    sub["pretrained_pct"] = sub["pretrained_pct"].astype(int)
    sub["dataset_short"] = sub["dataset_short"].astype(str)

    dup = sub.duplicated(subset=["dataset_short", "pretrained_pct"]).sum()
    if dup:
        raise SystemExit(
            f"[FALHA] {method} ({label}) tem {dup} celula(s) (dataset, pct) duplicada(s). "
            "Filtro de init insuficiente."
        )
    if sub[METRIC].isna().any():
        raise SystemExit(f"[FALHA] {method} ({label}) tem NaN na coluna {METRIC}.")

    ser = sub.set_index(["dataset_short", "pretrained_pct"])[METRIC].astype(float)
    ser = ser.sort_index()

    expected = sorted((d, p) for d in DATASETS for p in PCTS)
    got = sorted(ser.index.tolist())
    if got != expected:
        faltando = sorted(set(expected) - set(got))
        sobrando = sorted(set(got) - set(expected))
        raise SystemExit(
            f"[FALHA] grade incompleta para {method} ({label}): "
            f"{len(got)} celulas em vez de 18. Faltando={faltando} Sobrando={sobrando}"
        )
    return ser


def signed_rank_parts(diff: np.ndarray) -> tuple[float, float, int]:
    """W+, W- e n efetivo sobre `diff` (zeros descartados, zero_method='wilcox')."""
    nz = diff[diff != 0.0]
    if nz.size == 0:
        return 0.0, 0.0, 0
    ranks = stats.rankdata(np.abs(nz))
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    return w_pos, w_neg, int(nz.size)


def wilcoxon_pair(base: np.ndarray, model: np.ndarray) -> dict:
    """Wilcoxon bilateral + efeito. Convencao: diff = model - base."""
    diff = model - base
    w_pos, w_neg, n_eff = signed_rank_parts(diff)

    method_used = "exact"
    try:
        res = stats.wilcoxon(
            base, model, alternative="two-sided", zero_method="wilcox", method="exact"
        )
    except Exception:
        method_used = "auto"
        res = stats.wilcoxon(
            base, model, alternative="two-sided", zero_method="wilcox", method="auto"
        )

    denom = w_pos + w_neg
    rrb = (w_pos - w_neg) / denom if denom > 0 else float("nan")
    return {
        "n_pairs": int(diff.size),
        "n_eff": n_eff,
        "W": float(res.statistic),
        "W_pos": w_pos,
        "W_neg": w_neg,
        "p_raw": float(res.pvalue),
        "method_wilcoxon": method_used,
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "rank_biserial_r": rrb,
        "n_model_wins": int((diff > 0).sum()),
        "n_base_wins": int((diff < 0).sum()),
        "n_ties": int((diff == 0).sum()),
    }


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> dict:
    """IC95% percentil por bootstrap das celulas pareadas (mediana e media da diferenca)."""
    n = diff.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    samples = diff[idx]
    med = np.median(samples, axis=1)
    mea = samples.mean(axis=1)
    lo_m, hi_m = np.percentile(med, [2.5, 97.5])
    lo_a, hi_a = np.percentile(mea, [2.5, 97.5])
    return {
        "median_ci_lo": float(lo_m),
        "median_ci_hi": float(hi_m),
        "mean_ci_lo": float(lo_a),
        "mean_ci_hi": float(hi_a),
    }


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV unificado de entrada")
    ap.add_argument("--baseline", default="SVM_FLIM", help="method usado como baseline")
    ap.add_argument("--alpha", type=float, default=0.05, help="nivel de significancia")
    args = ap.parse_args()

    if args.baseline not in MODELS:
        raise SystemExit(f"[FALHA] baseline '{args.baseline}' nao esta em MODELS.")

    df = pd.read_csv(args.csv)
    for col in ("method", "init", "dataset_short", "pretrained_pct", METRIC):
        if col not in df.columns:
            raise SystemExit(f"[FALHA] coluna '{col}' ausente em {args.csv}")

    base_label = MODELS[args.baseline][0]
    base_ser = load_series(df, args.baseline)
    others = [m for m in MODELS if m != args.baseline]

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    diffs: dict[str, pd.Series] = {}

    for method in others:
        ser = load_series(df, method)
        if list(ser.index) != list(base_ser.index):
            raise SystemExit(
                f"[FALHA] chaves de pareamento divergem entre {args.baseline} e {method}."
            )
        base_v = base_ser.to_numpy()
        mod_v = ser.to_numpy()
        d = mod_v - base_v
        diffs[method] = pd.Series(d, index=base_ser.index)

        row = {"method": method, "label": MODELS[method][0], "baseline": args.baseline,
               "baseline_label": base_label, "metric": METRIC}
        row.update(wilcoxon_pair(base_v, mod_v))
        row.update(bootstrap_ci(d, rng))
        row["baseline_median"] = float(np.median(base_v))
        row["model_median"] = float(np.median(mod_v))
        rows.append(row)

    res = pd.DataFrame(rows)

    for meth in ("holm", "bonferroni"):
        rej, padj, _, _ = multipletests(res["p_raw"].to_numpy(), alpha=args.alpha, method=meth)
        res[f"p_{meth}"] = padj
        res[f"sig_{meth}"] = rej

    def winner(r: pd.Series) -> str:
        if r["median_diff"] > 0:
            return r["label"]
        if r["median_diff"] < 0:
            return base_label
        return "empate"

    res["melhor"] = res.apply(winner, axis=1)
    res["conclusao_holm"] = [
        (f"{w} melhor" if s else "sem diferenca detectada")
        for w, s in zip(res["melhor"], res["sig_holm"])
    ]
    res = res.sort_values("p_raw").reset_index(drop=True)

    # ---- analise secundaria: por dataset (exploratoria, n = 6) ----
    sec_rows: list[dict] = []
    for method in others:
        for ds in DATASETS:
            sub = diffs[method].loc[ds]
            base_v = base_ser.loc[ds].to_numpy()
            mod_v = base_v + sub.to_numpy()
            r = wilcoxon_pair(base_v, mod_v)
            sec_rows.append({
                "method": method, "label": MODELS[method][0], "dataset": ds,
                "n_pairs": r["n_pairs"], "n_eff": r["n_eff"], "W": r["W"],
                "p_raw": r["p_raw"], "median_diff": r["median_diff"],
                "rank_biserial_r": r["rank_biserial_r"],
                "n_model_wins": r["n_model_wins"], "n_base_wins": r["n_base_wins"],
                "method_wilcoxon": r["method_wilcoxon"],
            })
    sec = pd.DataFrame(sec_rows)

    # ---------------- CSV ----------------
    cols = ["metric", "baseline", "baseline_label", "method", "label", "melhor",
            "n_pairs", "n_eff", "n_model_wins", "n_base_wins", "n_ties",
            "baseline_median", "model_median", "median_diff",
            "median_ci_lo", "median_ci_hi", "mean_diff", "mean_ci_lo", "mean_ci_hi",
            "W", "W_pos", "W_neg", "rank_biserial_r", "method_wilcoxon",
            "p_raw", "p_holm", "p_bonferroni", "sig_holm", "sig_bonferroni",
            "conclusao_holm"]
    res[cols].to_csv(OUT_CSV, index=False)

    # ---------------- Markdown ----------------
    wm = res["method_wilcoxon"].unique().tolist()
    n_sig_holm = int(res["sig_holm"].sum())
    n_sig_bonf = int(res["sig_bonferroni"].sum())
    vencedores_holm = res.loc[res["sig_holm"], ["label", "melhor"]]

    L: list[str] = []
    L.append(f"# Wilcoxon pareado de sinais em {METRIC_LABEL}: {base_label} como baseline")
    L.append("")
    L.append(f"Métrica: coluna `{METRIC}` de `artifacts/normalized/unified_svm_comparison.csv`. "
             f"Baseline: `{args.baseline}` ({base_label}). "
             f"Família de {len(res)} comparações um-contra-todos, alfa = {args.alpha}.")
    L.append("")
    L.append("Unidade pareada: célula `(dataset, fração de pré-treino)`, "
             "3 datasets x 6 frações = **n = 18 pares** por comparação. "
             "Cada célula já é a média sobre 3 splits. "
             "O pareamento usa a chave `(dataset, fração)`, verificada antes de cada teste.")
    L.append("")
    L.append(f"Teste: `scipy.stats.wilcoxon(baseline, modelo, alternative=\"two-sided\", "
             f"zero_method=\"wilcox\")`, método `{'`, `'.join(wm)}`. "
             "Correção de multiplicidade por Holm (principal) e Bonferroni (referência), "
             "via `statsmodels.stats.multitest.multipletests`. "
             f"IC95% por bootstrap percentil das 18 células pareadas "
             f"({N_BOOT} reamostragens, seed {SEED}).")
    L.append("")
    L.append(f"**Convenção de sinal**: diferença = {METRIC_LABEL}(modelo) - {METRIC_LABEL}(FLIM). "
             "Valor positivo significa que o modelo supera o FLIM; negativo, que o FLIM supera o "
             "modelo. O mesmo vale para `r` rank-biserial. A coluna *Melhor* nomeia o vencedor "
             "pelo sinal da mediana, independentemente de haver significância.")
    L.append("")
    L.append("## Tabela principal (n = 18 pares por linha)")
    L.append("")
    L.append("| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap (mediana) | Melhor | W | p bruto | p Holm | p Bonferroni | r rank-biserial | Sig. Holm? |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for _, r in res.iterrows():
        L.append(
            f"| {r['label']} | {r['median_diff']:+.4f} | "
            f"[{r['median_ci_lo']:+.4f}, {r['median_ci_hi']:+.4f}] | {r['melhor']} | "
            f"{r['W']:.1f} | {fmt_p(r['p_raw'])} | {fmt_p(r['p_holm'])} | "
            f"{fmt_p(r['p_bonferroni'])} | {r['rank_biserial_r']:+.3f} | "
            f"{'sim' if r['sig_holm'] else 'nao'} |"
        )
    L.append("")
    L.append(f"Mediana de {METRIC_LABEL} do baseline {base_label} nas 18 células: "
             f"{res['baseline_median'].iloc[0]:.4f}.")
    L.append("")
    L.append("Detalhe dos sinais e da média das diferenças:")
    L.append("")
    L.append("| Modelo | Vitórias do modelo | Vitórias do FLIM | Empates | W+ | W- | Média da dif. | IC95% bootstrap (média) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in res.iterrows():
        L.append(
            f"| {r['label']} | {r['n_model_wins']} | {r['n_base_wins']} | {r['n_ties']} | "
            f"{r['W_pos']:.1f} | {r['W_neg']:.1f} | {r['mean_diff']:+.4f} | "
            f"[{r['mean_ci_lo']:+.4f}, {r['mean_ci_hi']:+.4f}] |"
        )
    L.append("")
    L.append("O teste exato depende apenas de W e de n, logo comparações com o mesmo W recebem "
             "o mesmo p bruto. As quatro linhas com W = 51 não são erro de cálculo: I-JEPA tem "
             "W- = 51 (13 vitórias contra 5) e Distill 2, 3 e 4 têm W+ = 51 (3 vitórias contra "
             "15, justamente as três de maior magnitude). A magnitude do efeito difere entre "
             "elas, o p não.")
    L.append("")
    L.append("## Tabela secundária por dataset (exploratória)")
    L.append("")
    L.append("Wilcoxon dentro de cada dataset, n = 6 frações. Serve para checar consistência do "
             "efeito entre datasets. Com n = 6 o menor p bilateral alcançável pelo teste exato é "
             "0.03125, o poder é mínimo e nenhuma correção de multiplicidade foi aplicada aqui: "
             "estes p não sustentam conclusão isolada.")
    L.append("")
    L.append("| Modelo | Dataset | Mediana da dif. | Vitórias mod./FLIM | W | p bruto (não corrigido) | r rank-biserial |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    order = {m: i for i, m in enumerate(res["method"])}
    sec_sorted = sec.assign(_o=sec["method"].map(order)).sort_values(["_o", "dataset"])
    for _, r in sec_sorted.iterrows():
        rr = "n/a" if not np.isfinite(r["rank_biserial_r"]) else f"{r['rank_biserial_r']:+.3f}"
        L.append(
            f"| {r['label']} | {r['dataset']} | {r['median_diff']:+.4f} | "
            f"{r['n_model_wins']}/{r['n_base_wins']} | {r['W']:.1f} | "
            f"{fmt_p(r['p_raw'])} | {rr} |"
        )
    L.append("")
    L.append("## Leitura dos resultados")
    L.append("")
    n_pairs = int(res["n_pairs"].iloc[0])
    low = min(PCTS)
    pcts_txt = ", ".join(f"{p}%" for p in PCTS)

    # ---- 1. o que o teste faz, em linguagem simples ----
    L.append("### O que este teste faz, em linguagem simples")
    L.append("")
    L.append(
        f"Cada modelo enfrentou o {base_label} em **{n_pairs} situações idênticas**: "
        f"{len(DATASETS)} conjuntos de imagens ({', '.join(DATASETS)}) x {len(PCTS)} "
        f"quantidades de pré-treino ({pcts_txt}). Como os dois rodam exatamente na mesma "
        "condição, dá para comparar um a um quem obteve o "
        f"{METRIC_LABEL} maior — é o mesmo raciocínio de testar dois remédios no *mesmo* "
        "paciente em vez de comparar dois grupos de pacientes diferentes. Chamamos cada uma "
        "dessas situações de *célula*."
    )
    L.append("")
    L.append(
        "O teste de Wilcoxon olha o placar dessas 18 comparações e responde a uma única "
        "pergunta: **um resultado tão desequilibrado assim apareceria por puro acaso?** O valor "
        "`p` é essa resposta em número — quanto menor, mais difícil explicar o resultado por "
        "sorte. Ele não conta apenas quantas células cada lado venceu: ordena as 18 diferenças "
        "da menor para a maior e dá mais peso às vitórias mais folgadas. É isso que a palavra "
        "*posto* (rank) significa. O que ele **não** faz é somar as diferenças — uma vitória "
        "gigantesca não vale por dez, vale por ficar em primeiro lugar na fila."
    )
    L.append("")
    L.append(
        f"A mesma pergunta foi feita {len(res)} vezes (o {base_label} contra cada um dos "
        f"{len(res)} modelos). Fazer muitas perguntas aumenta a chance de uma delas dar "
        "\"positivo\" por acaso, como comprar sete bilhetes de loteria em vez de um. As "
        "correções de **Holm** e **Bonferroni** compensam isso exigindo evidência mais forte de "
        "cada comparação; Bonferroni é a mais rígida e entra só como checagem. Por isso a coluna "
        "que decide é `p Holm`, não `p bruto`."
    )
    L.append("")
    L.append(
        f"**Regra do sinal**: número negativo = {base_label} melhor; positivo = o outro modelo "
        f"melhor. A diferença está na escala do {METRIC_LABEL}, que vai de 0 a 1, então "
        f"-0.10 quer dizer 10 pontos de {METRIC_LABEL} abaixo do {base_label}."
    )
    L.append("")

    # ---- 2. o que ficou provado ----
    L.append("### O que o teste conseguiu demonstrar")
    L.append("")
    if n_sig_holm == 0:
        L.append(
            f"Nenhuma das {len(res)} comparações sobrevive à correção de Holm em "
            f"{METRIC_LABEL} ao nível alfa = {args.alpha} "
            f"(menor p ajustado = {fmt_p(res['p_holm'].min())}). Em linguagem direta: com este "
            "conjunto de experimentos não é possível afirmar que qualquer um dos modelos seja "
            f"diferente do {base_label}. Isso não é o mesmo que dizer que são iguais — veja a "
            "seção seguinte."
        )
        L.append("")
    else:
        nomes = ", ".join(f"{lab} (a favor de {w})" for lab, w in
                          vencedores_holm.itertuples(index=False))
        L.append(
            f"Após a correção de Holm, {n_sig_holm} das {len(res)} comparações diferem do "
            f"{base_label} em {METRIC_LABEL} ao nível alfa = {args.alpha}: {nomes}. Sob "
            f"Bonferroni, a régua mais rígida, sobrevivem {n_sig_bonf}. Em detalhe:"
        )
        L.append("")
        for _, r in res[res["sig_holm"]].iterrows():
            venc = r["melhor"]
            n_venc = int(r["n_base_wins"] if venc == base_label else r["n_model_wins"])
            L.append(
                f"- **{r['label']}** — o {venc} vence em {n_venc} das {int(r['n_pairs'])} "
                f"células, com diferença típica de {r['median_diff']:+.3f} de {METRIC_LABEL} "
                f"(mediana). {METRIC_LABEL} mediano: {r['baseline_median']:.3f} do "
                f"{base_label} contra {r['model_median']:.3f}. p Holm = {fmt_p(r['p_holm'])}."
            )
        L.append("")
        L.append(
            "Nesses dois casos a diferença não é marginal nem depende de detalhe estatístico: "
            "ela é grande, aparece na maioria das células e sobrevive à régua mais conservadora. "
            f"Nenhum modelo supera o {base_label} com significância."
            if all(v == base_label for v in vencedores_holm["melhor"])
            else "Essas diferenças sobrevivem à correção de multiplicidade."
        )
        L.append("")

    # ---- 3. o que ficou inconclusivo ----
    nao_sig = res[~res["sig_holm"]]
    if len(nao_sig):
        quase = nao_sig[nao_sig["p_raw"] < args.alpha]
        L.append("### O que ficou sem veredito (e o que isso *não* quer dizer)")
        L.append("")
        L.append(
            f"As outras {len(nao_sig)} comparações ({', '.join(nao_sig['label'])}) ficam "
            "**inconclusivas**. Inconclusivo não é empate. É como pesar duas malas numa balança "
            "de banheiro: se as duas marcam 20 kg, isso não prova que têm o mesmo peso, prova "
            f"apenas que a balança não enxerga a diferença. Com {n_pairs} células e uma família "
            f"de {len(res)} perguntas, o teste só detecta diferenças grandes ou muito "
            "consistentes; as pequenas passam despercebidas."
        )
        L.append("")
        if len(quase):
            q = quase.iloc[0]
            L.append(
                f"O caso mais ilustrativo é **{q['label']}**: p bruto = {fmt_p(q['p_raw'])}, "
                f"abaixo de {args.alpha}. Sozinha, essa comparação passaria. Dentro da família "
                f"de {len(res)} perguntas feitas ao mesmo tempo, não passa "
                f"(p Holm = {fmt_p(q['p_holm'])}). É exatamente o preço de comprar sete bilhetes."
            )
            L.append("")
        L.append(
            "Nessas linhas, a leitura útil é a mediana da diferença e o IC95%, não o `p`. Uma "
            "ressalva honesta: esse IC vem de reamostrar as 18 células (*bootstrap*) e não é o "
            "mesmo procedimento do teste. Ele pode excluir o zero enquanto o Wilcoxon não "
            "rejeita, e nesse caso a conclusão conservadora — a que vai para o artigo — é a do "
            "Wilcoxon corrigido."
        )
        L.append("")

    # ---- 4. onde as vitorias se concentram ----
    conc: list[str] = []
    for _, r in res.iterrows():
        d = diffs[r["method"]]
        wins = d[d > 0]
        if 0 < len(wins) <= 4 and all(k[1] == low for k in wins.index):
            conc.append(r["label"])
    if conc:
        segunda = sorted(PCTS)[1]
        L.append(f"### Onde o {base_label} perde: só quando quase não há pré-treino")
        L.append("")
        L.append(
            f"Para {len(conc)} dos {len(res)} modelos ({', '.join(conc)}), *todas* as células em "
            f"que o modelo bate o {base_label} estão na fração de {low}% de pré-treino. De "
            f"{segunda}% em diante o {base_label} vence em todas as células, sem exceção. Ou "
            "seja: a vantagem dos concorrentes existe apenas no regime de dado escasso e "
            f"desaparece assim que há pré-treino disponível. Essa é a leitura prática do "
            "experimento, e ela não depende de nenhum valor de `p`."
        )
        L.append("")
        # modelos em que o IC da media cruza o zero mas o da mediana nao:
        # sintoma de poucas vitorias grandes contra muitas derrotas pequenas
        dv = res[(res["mean_ci_lo"] < 0) & (res["mean_ci_hi"] > 0)
                 & ((res["median_ci_lo"] > 0) | (res["median_ci_hi"] < 0))]
        if len(dv):
            r0 = dv.iloc[0]
            n_win = int(r0["n_model_wins"])
            n_loss = int(r0["n_base_wins"])
            L.append("### Por que a mediana e a média discordam em alguns modelos")
            L.append("")
            L.append(
                "A **mediana** é o caso típico: enfileire as 18 diferenças e pegue a do meio. A "
                "**média** soma tudo e divide por 18, então poucos valores extremos a puxam — é "
                "o efeito de \"a renda média do bar sobe quando entra um bilionário\". Em "
                f"{', '.join(dv['label'])} acontece exatamente isso: {n_win} vitórias grandes "
                f"(todas em {low}%) contra {n_loss} derrotas pequenas. A mediana é negativa (no "
                f"caso típico o {base_label} é melhor), enquanto a média fica perto de zero ou "
                "positiva."
            )
            L.append("")
            L.append(
                "O Wilcoxon se comporta como a mediana, não como a média: ele vê "
                f"{n_win} vitórias contra {n_loss} derrotas e a folga dessas {n_win} compra, no "
                "máximo, os primeiros lugares da fila. Daí o veredito \"inconclusivo\" para "
                f"{'esses modelos' if len(dv) > 1 else 'esse modelo'}, apesar da média favorável."
            )
            L.append("")

    # ---- 5. consistencia entre datasets ----
    flip: list[str] = []
    for _, r in res.iterrows():
        sgn = {np.sign(sec[(sec["method"] == r["method"]) & (sec["dataset"] == ds)]
                       ["median_diff"].iloc[0]) for ds in DATASETS}
        if len({s for s in sgn if s != 0}) > 1:
            flip.append(r["label"])
    L.append("### O resultado se repete nos três conjuntos de imagens?")
    L.append("")
    L.append(
        "A tabela secundária repete a mesma disputa dentro de cada dataset (6 células cada) para "
        "responder a uma pergunta simples: o efeito vale nos três domínios ou é peculiaridade de "
        "um só? Se um modelo ganha num dataset e perde noutro, a comparação agregada sobre as 18 "
        "células mistura as duas coisas e esconde essa dependência."
    )
    L.append("")
    if flip:
        for lab in flip:
            met = res.loc[res["label"] == lab, "method"].iloc[0]
            det = ", ".join(
                f"{ds} {sec[(sec['method'] == met) & (sec['dataset'] == ds)]['median_diff'].iloc[0]:+.3f}"
                for ds in DATASETS
            )
            agg = res.loc[res["label"] == lab, "median_diff"].iloc[0]
            L.append(
                f"- **{lab}** troca de sinal entre datasets ({det}). A mediana agregada "
                f"({agg:+.3f}) esconde esse comportamento: a vantagem depende do domínio, não é "
                "geral."
            )
        L.append("")
        L.append(
            f"Nos outros {len(res) - len(flip)} modelos o sinal é o mesmo nos três conjuntos, o "
            "que indica efeito consistente e não artefato de um dataset isolado."
        )
    else:
        L.append(
            "Nenhum modelo troca de sinal entre datasets: o sentido do efeito é o mesmo nos três "
            "domínios, o que sustenta ler a comparação agregada como consistente."
        )
    L.append("")

    # ---- 6. resumo ----
    pos = res[res["median_diff"] > 0]["label"].tolist()
    neg = res[res["median_diff"] < 0]["label"].tolist()
    def plural(n: int) -> str:
        return "modelo fica" if n == 1 else "modelos ficam"

    L.append("### Resumo")
    L.append("")
    L.append(
        f"Pelo sinal da mediana, {len(pos)} {plural(len(pos))} acima do {base_label} em "
        f"{METRIC_LABEL} ({', '.join(pos) if pos else 'nenhum'}) e {len(neg)} "
        f"{plural(len(neg))} abaixo ({', '.join(neg) if neg else 'nenhum'}). "
        f"Com correção de multiplicidade, {n_sig_holm} dessas diferenças se sustentam "
        f"estatisticamente"
        + (f" (todas a favor do {base_label})"
           if n_sig_holm and all(v == base_label for v in vencedores_holm["melhor"]) else "")
        + f"; as demais {len(res) - n_sig_holm} ficam sem veredito. Onde os concorrentes ganham, "
        f"ganham apenas na fração de {low}% de pré-treino."
    )
    L.append("")
    L.append("## Reprodução")
    L.append("")
    L.append("```bash")
    L.append("cd /dados/home/moliveira/Scalable_Hybrid_FLIM")
    L.append("LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \\")
    L.append("/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \\")
    L.append("    statistics/tools/wilcoxon_f1.py")
    L.append("```")
    L.append("")
    L.append(f"Versões: scipy {__import__('scipy').__version__}, "
             f"statsmodels {__import__('statsmodels').__version__}, "
             f"numpy {np.__version__}, pandas {pd.__version__}.")
    L.append("")
    L.append("`scikit-posthocs` não foi usado: seu `posthoc_wilcoxon` é all-vs-all "
             f"(28 comparações com 8 modelos), enquanto o desenho pedido é um-contra-todos "
             f"({len(res)} comparações contra o FLIM). A família de correção seria diferente e o "
             "ajuste de p ficaria mais conservador sem necessidade.")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(res[["label", "median_diff", "W", "p_raw", "p_holm", "p_bonferroni",
               "rank_biserial_r", "sig_holm", "melhor"]].to_string(index=False))
    print(f"\n[ok] {OUT_CSV}")
    print(f"[ok] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
