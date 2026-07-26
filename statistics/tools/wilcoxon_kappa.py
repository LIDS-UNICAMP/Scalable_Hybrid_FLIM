#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Teste de Wilcoxon pareado de sinais (signed-rank) na metrica kappa de Cohen.

Desenho experimental
--------------------
Pedido do Revisor 3 (SIBGRAPI camera-ready): comparacao estatistica pareada de
cada modelo contra um baseline, com correcao para multiplas comparacoes.

* Fonte: ``artifacts/normalized/unified_svm_comparison.csv``. Cada linha e a
  media sobre 3 splits para um (metodo, dataset, fracao de pre-treino).
* Unidades pareadas: as 18 celulas ``(dataset_short, pretrained_pct)`` =
  3 datasets (eggs, larvae, protozoan) x 6 fracoes (1, 5, 25, 50, 75, 100).
  n = 18 pares por comparacao. O pareamento e feito pela chave, nunca por
  indice de linha; o script aborta se as chaves de baseline e modelo nao
  coincidirem exatamente.
* Baseline: ``SVM_FLIM`` (rotulo do artigo "FLIM (59.504)"). O artigo e uma
  comparacao contra o FLIM, logo o desenho e um-contra-todos: 7 comparacoes
  (FLIM vs cada um dos outros 7 modelos oficiais).
* ``SVM_LeJEPA`` aparece no CSV com 5 inicializacoes (90 linhas). Somente
  ``init == "trunc_normal"`` entra no artigo; o filtro e obrigatorio.
* Teste: ``scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided",
  zero_method="wilcox")``. Zeros exatos sao descartados (por isso reportamos
  o n efetivo). Usa ``method="exact"`` quando n efetivo <= 25, senao "approx".
* Correcao de multiplas comparacoes: ``statsmodels.stats.multitest.multipletests``
  com ``holm`` (principal) e ``bonferroni`` (referencia conservadora), sobre a
  familia das 7 comparacoes. alpha = 0.05.
* Tamanho de efeito: correlacao rank-biserial pareada
  ``r = (W+ - W-) / (W+ + W-)`` e mediana das diferencas, com IC 95% por
  bootstrap percentil (10.000 reamostragens das 18 celulas pareadas,
  ``numpy.random.default_rng(42)``).
* Convencao de sinal: ``diferenca = modelo - baseline``. Diferenca positiva
  significa que o **modelo** e melhor que o FLIM; negativa, que o FLIM e
  melhor. O r rank-biserial segue a mesma convencao (r > 0 => modelo melhor).
* Analise secundaria (exploratoria): Wilcoxon por dataset, n = 6 fracoes cada.
  Poder estatistico muito baixo, sem correcao adicional entre datasets; serve
  apenas para checar consistencia do sinal, nao para inferencia confirmatoria.

Por que NAO ``scikit-posthocs``
-------------------------------
``scikit_posthocs.posthoc_wilcoxon`` executa todos-contra-todos (all-vs-all),
produzindo 28 comparacoes para 8 modelos. Aqui o desenho e um-contra-todos
(baseline vs cada modelo), com familia de 7 hipoteses. Usar a versao all-vs-all
mudaria a familia de correcao (e portanto os p ajustados) e responderia uma
pergunta diferente da que o revisor fez. O teste em si e o mesmo
``scipy.stats.wilcoxon``, entao a dependencia extra nao traz beneficio.

Execucao
--------
    cd /dados/home/moliveira/Scalable_Hybrid_FLIM
    LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
    /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
        statistics/tools/wilcoxon_kappa.py

Saidas: ``statistics/tools/wilcoxon_kappa.csv`` e
``statistics/tools/wilcoxon_kappa.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

METRIC = "kappa"

# method -> (rotulo do artigo, filtro extra de init ou None)
MODELS: "dict[str, tuple[str, str | None]]" = {
    "SVM_FLIM": ("FLIM (59.504)", None),
    "SVM_LeJEPA": ("LeJEPA (59.504)", "trunc_normal"),
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
OUT_CSV = Path(__file__).resolve().parent / f"wilcoxon_{METRIC}.csv"
OUT_MD = Path(__file__).resolve().parent / f"wilcoxon_{METRIC}.md"


def die(msg: str) -> None:
    print(f"ERRO: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_series(df: pd.DataFrame, method: str, init: "str | None") -> pd.Series:
    """Serie da metrica indexada por (dataset_short, pretrained_pct), ordenada."""
    sub = df[df["method"] == method]
    if init is not None:
        sub = sub[sub["init"] == init]
    sub = sub.copy()
    sub["dataset_short"] = sub["dataset_short"].astype(str)
    sub["pretrained_pct"] = sub["pretrained_pct"].astype(int)

    keys = list(zip(sub["dataset_short"], sub["pretrained_pct"]))
    if len(set(keys)) != len(keys):
        die(f"{method}: chaves (dataset, pct) duplicadas apos filtro init={init}")

    expected = [(d, p) for d in DATASETS for p in PCTS]
    s = pd.Series(
        sub[METRIC].to_numpy(dtype=float),
        index=pd.MultiIndex.from_tuples(keys, names=["dataset_short", "pretrained_pct"]),
    )
    missing = [k for k in expected if k not in s.index]
    if missing:
        die(f"{method}: celulas ausentes {missing}")
    extra = [k for k in s.index if k not in expected]
    if extra:
        die(f"{method}: celulas inesperadas {extra}")
    s = s.reindex(expected)
    if s.isna().any():
        die(f"{method}: NaN em {METRIC} nas celulas {list(s.index[s.isna()])}")
    return s


def rank_biserial(diff: np.ndarray) -> "tuple[float, float, float, int]":
    """r = (W+ - W-)/(W+ + W-) sobre diferencas nao nulas. diff = modelo - baseline."""
    d = diff[diff != 0]
    n_eff = int(d.size)
    if n_eff == 0:
        return float("nan"), 0.0, 0.0, 0
    ranks = rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    total = w_pos + w_neg
    r = (w_pos - w_neg) / total if total > 0 else float("nan")
    return float(r), w_pos, w_neg, n_eff


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, stat=np.median):
    n = diff.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = stat(diff[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def run_wilcoxon(base: np.ndarray, model: np.ndarray) -> "tuple[float, float, str, int]":
    diff = model - base
    n_eff = int(np.count_nonzero(diff))
    if n_eff == 0:
        return float("nan"), 1.0, "n/a (todas as diferencas nulas)", 0
    method = "exact" if n_eff <= 25 else "approx"
    res = stats.wilcoxon(
        base,
        model,
        alternative="two-sided",
        zero_method="wilcox",
        method=method,
    )
    return float(res.statistic), float(res.pvalue), method, n_eff


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Wilcoxon pareado um-contra-todos em {METRIC}")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV unificado de entrada")
    ap.add_argument("--baseline", default="SVM_FLIM", help="method usado como baseline")
    ap.add_argument("--alpha", type=float, default=0.05, help="nivel de significancia")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        die(f"CSV nao encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ("method", "init", "dataset_short", "pretrained_pct", METRIC):
        if col not in df.columns:
            die(f"coluna ausente no CSV: {col}")

    if args.baseline not in MODELS:
        die(f"baseline {args.baseline} nao esta na lista dos 8 modelos oficiais")

    series = {m: load_series(df, m, init) for m, (_, init) in MODELS.items()}
    base_s = series[args.baseline]
    base_label = MODELS[args.baseline][0]
    rng = np.random.default_rng(SEED)

    others = [m for m in MODELS if m != args.baseline]

    rows = []
    for m in others:
        s = series[m]
        if list(s.index) != list(base_s.index):
            die(f"chaves pareadas nao batem entre {args.baseline} e {m}")
        base = base_s.to_numpy()
        mod = s.to_numpy()
        diff = mod - base

        W, p_raw, method_used, n_eff = run_wilcoxon(base, mod)
        r, w_pos, w_neg, _ = rank_biserial(diff)
        med_lo, med_hi = bootstrap_ci(diff, rng, np.median)
        mean_lo, mean_hi = bootstrap_ci(diff, rng, np.mean)

        rows.append(
            {
                "metric": METRIC,
                "baseline_method": args.baseline,
                "baseline_label": base_label,
                "model_method": m,
                "model_label": MODELS[m][0],
                "n_pairs": int(diff.size),
                "n_effective": n_eff,
                "baseline_mean": float(base.mean()),
                "model_mean": float(mod.mean()),
                "baseline_median": float(np.median(base)),
                "model_median": float(np.median(mod)),
                "median_diff": float(np.median(diff)),
                "median_diff_ci95_lo": med_lo,
                "median_diff_ci95_hi": med_hi,
                "mean_diff": float(diff.mean()),
                "mean_diff_ci95_lo": mean_lo,
                "mean_diff_ci95_hi": mean_hi,
                "n_model_better": int((diff > 0).sum()),
                "n_baseline_better": int((diff < 0).sum()),
                "n_ties": int((diff == 0).sum()),
                "W": W,
                "W_pos": w_pos,
                "W_neg": w_neg,
                "wilcoxon_method": method_used,
                "p_raw": p_raw,
                "rank_biserial_r": r,
            }
        )

    res = pd.DataFrame(rows)
    pvals = res["p_raw"].to_numpy()
    for meth, col in (("holm", "p_holm"), ("bonferroni", "p_bonferroni")):
        rej, padj, _, _ = multipletests(pvals, alpha=args.alpha, method=meth)
        res[col] = padj
        res[f"reject_{meth}"] = rej

    res["better_model"] = np.where(
        res["median_diff"] > 0, res["model_label"], base_label
    )
    res["better_model"] = np.where(
        res["median_diff"] == 0, "empate", res["better_model"]
    )
    res["alpha"] = args.alpha
    res["sign_convention"] = "diff = modelo - baseline (positivo => modelo melhor)"

    # ordena pela mediana da diferenca (do mais favoravel ao modelo ao menos)
    res = res.sort_values("median_diff", ascending=False).reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)

    # ---------------- analise secundaria por dataset ----------------
    sec_rows = []
    rng2 = np.random.default_rng(SEED)
    for ds in DATASETS:
        for m in others:
            base = base_s.loc[ds].reindex(PCTS).to_numpy()
            mod = series[m].loc[ds].reindex(PCTS).to_numpy()
            diff = mod - base
            W, p_raw, method_used, n_eff = run_wilcoxon(base, mod)
            r, _, _, _ = rank_biserial(diff)
            sec_rows.append(
                {
                    "metric": METRIC,
                    "dataset_short": ds,
                    "model_method": m,
                    "model_label": MODELS[m][0],
                    "n_pairs": int(diff.size),
                    "n_effective": n_eff,
                    "median_diff": float(np.median(diff)),
                    "W": W,
                    "wilcoxon_method": method_used,
                    "p_raw": p_raw,
                    "rank_biserial_r": r,
                    "n_model_better": int((diff > 0).sum()),
                }
            )
    sec = pd.DataFrame(sec_rows)

    # ---------------- relatorio markdown ----------------
    a = args.alpha
    n_boot_txt = f"{N_BOOT:,}".replace(",", ".")
    lines = []
    lines.append("# Wilcoxon pareado (signed-rank) na métrica kappa de Cohen")
    lines.append("")
    lines.append(
        f"Baseline: **{base_label}** (`{args.baseline}`). Desenho um-contra-todos, "
        f"{len(others)} comparações. Unidades pareadas: 18 células "
        f"`(dataset, fração de pré-treino)` = 3 datasets x 6 frações "
        f"(1, 5, 25, 50, 75, 100 %). alfa = {a}."
    )
    lines.append("")
    lines.append(
        "Convenção de sinal: `diferença = modelo - baseline`. Diferença positiva e "
        "`r` positivo indicam que o modelo supera o FLIM; valores negativos indicam "
        "que o FLIM supera o modelo. `LeJEPA` restrito a `init == trunc_normal`. "
        'Teste bilateral, `zero_method="wilcox"`, p exato quando n efetivo <= 25. '
        "Correção de múltiplas comparações por Holm (principal) e Bonferroni "
        f"(referência conservadora) sobre a família das {len(others)} comparações. "
        "IC 95% da mediana da diferença por bootstrap percentil "
        f"({n_boot_txt} reamostragens das 18 células, semente {SEED})."
    )
    lines.append("")
    lines.append("## Tabela principal (n = 18 pares)")
    lines.append("")
    hdr = (
        "| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap | W | p bruto | "
        "p Holm | p Bonferroni | r rank-biserial | Melhor | Signif. (Holm) |"
    )
    lines.append(hdr)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r_ in res.iterrows():
        sig = "sim" if r_["reject_holm"] else "nao"
        lines.append(
            f"| {r_['model_label']} | {r_['median_diff']:+.4f} | "
            f"[{r_['median_diff_ci95_lo']:+.4f}, {r_['median_diff_ci95_hi']:+.4f}] | "
            f"{r_['W']:.1f} | {fmt_p(r_['p_raw'])} | {fmt_p(r_['p_holm'])} | "
            f"{fmt_p(r_['p_bonferroni'])} | {r_['rank_biserial_r']:+.3f} | "
            f"{r_['better_model']} | {sig} |"
        )
    lines.append("")
    lines.append(
        "Colunas auxiliares (n efetivo após descarte de zeros, W+/W-, médias, "
        "contagem de células em que cada lado vence, IC da diferença média) estão em "
        f"`{OUT_CSV.name}`."
    )
    lines.append("")
    lines.append("### Suporte por contagem de células")
    lines.append("")
    lines.append(
        "| Modelo | Células modelo > FLIM | Células FLIM > modelo | Empates | "
        "n efetivo | Método p | Média kappa modelo | Média kappa FLIM |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r_ in res.iterrows():
        lines.append(
            f"| {r_['model_label']} | {r_['n_model_better']}/18 | "
            f"{r_['n_baseline_better']}/18 | {r_['n_ties']} | {r_['n_effective']} | "
            f"{r_['wilcoxon_method']} | {r_['model_mean']:.4f} | {r_['baseline_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Análise secundária por dataset (exploratória, n = 6)")
    lines.append("")
    lines.append(
        "Wilcoxon dentro de cada dataset, pareando pelas 6 frações de pré-treino. "
        "Com n = 6 o menor p bilateral atingível é 0,0312, e nenhum resultado "
        f"sobreviveria à correção para {len(others) * len(DATASETS)} testes. Os p "
        "abaixo são **brutos** e servem só para inspecionar se o sinal do efeito é "
        "consistente entre datasets ou vem de um deles. Não use esta tabela para "
        "inferência confirmatória."
    )
    lines.append("")
    lines.append(
        "| Modelo | Dataset | Mediana da dif. | Células modelo > FLIM | W | "
        "p bruto | r rank-biserial |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    order = list(res["model_method"])
    for m in order:
        for ds in DATASETS:
            r_ = sec[(sec["model_method"] == m) & (sec["dataset_short"] == ds)].iloc[0]
            lines.append(
                f"| {r_['model_label']} | {ds} | {r_['median_diff']:+.4f} | "
                f"{r_['n_model_better']}/6 | {r_['W']:.1f} | {fmt_p(r_['p_raw'])} | "
                f"{r_['rank_biserial_r']:+.3f} |"
            )
    lines.append("")

    # ---- leitura dos resultados (gerada a partir dos numeros) ----
    n_sig = int(res["reject_holm"].sum())
    sig_names = list(res.loc[res["reject_holm"], "model_label"])
    raw_sig = list(res.loc[res["p_raw"] < a, "model_label"])
    pos = res[res["median_diff"] > 0]
    neg = res[res["median_diff"] < 0]

    sig_above = list(
        res.loc[res["reject_holm"] & (res["median_diff"] > 0), "model_label"]
    )
    sig_below = list(
        res.loc[res["reject_holm"] & (res["median_diff"] < 0), "model_label"]
    )
    ns_names = list(res.loc[~res["reject_holm"], "model_label"])
    min_p_exact = 2.0 / (2 ** 18)

    lines.append("## Leitura dos resultados")
    lines.append("")
    para = []
    para.append(
        f"Na métrica kappa, {len(pos)} dos {len(others)} modelos apresentam mediana "
        f"de diferença positiva contra o FLIM e {len(neg)} apresentam mediana negativa."
    )
    if raw_sig:
        para.append(
            "Antes da correção, " + ", ".join(raw_sig) + f" ficam abaixo de {a}."
        )
    else:
        para.append(f"Nenhuma comparação fica abaixo de {a} nem antes da correção.")
    if n_sig == 0:
        para.append(
            f"Depois de Holm sobre a família de {len(others)} comparações, nenhuma "
            "diferença permanece significativa. Com 18 células pareadas os dados não "
            "sustentam afirmação de superioridade estatística de nenhum modelo sobre "
            "o FLIM em kappa."
        )
    else:
        plural = "comparações permanecem" if n_sig > 1 else "comparação permanece"
        frase = f"Depois de Holm, {n_sig} {plural} significativa"
        frase += "s" if n_sig > 1 else ""
        detalhe = []
        if sig_above:
            detalhe.append("acima do FLIM: " + ", ".join(sig_above))
        if sig_below:
            detalhe.append("abaixo do FLIM: " + ", ".join(sig_below))
        para.append(frase + " (" + "; ".join(detalhe) + ").")
    if ns_names:
        para.append("As demais (" + ", ".join(ns_names) + ") não se separam do FLIM.")
    n_ci_excl = int((res["median_diff_ci95_lo"] * res["median_diff_ci95_hi"] > 0).sum())
    ci_excl_names = list(
        res.loc[
            res["median_diff_ci95_lo"] * res["median_diff_ci95_hi"] > 0, "model_label"
        ]
    )
    if n_ci_excl:
        para.append(
            f"O IC95% bootstrap da mediana da diferença exclui zero em {n_ci_excl} "
            f"de {len(others)} comparações ("
            + ", ".join(ci_excl_names)
            + f"); nas {len(others) - n_ci_excl} restantes o intervalo contém zero, "
            "de acordo com os p ajustados."
        )
    else:
        para.append(
            "O IC95% bootstrap da mediana da diferença contém zero em todas as "
            "comparações."
        )
    lines.append(" ".join(para))
    lines.append("")

    # consistencia entre datasets, calculada da tabela secundaria
    inconsist = []
    for m in order:
        signs = sec.loc[sec["model_method"] == m, "median_diff"].to_numpy()
        if (signs > 0).any() and (signs < 0).any():
            inconsist.append(MODELS[m][0])
    consist = [MODELS[m][0] for m in order if MODELS[m][0] not in inconsist]
    sec_txt = []
    if inconsist:
        sec_txt.append(
            "Na análise por dataset o sinal da mediana da diferença troca entre "
            "datasets para " + ", ".join(inconsist) + "."
        )
    if consist:
        sec_txt.append(
            "Mantém o mesmo sinal nos três datasets: " + ", ".join(consist) + ", "
            "ou seja, nesses casos o resultado agregado não vem de um único dataset."
        )
    sig_inconsist = [n for n in sig_names if n in inconsist]
    if sig_inconsist:
        sec_txt.append(
            "Ressalva: " + ", ".join(sig_inconsist) + " aparece como significativo no "
            "teste agregado apesar de inverter o sinal em pelo menos um dataset, "
            "logo o efeito não é uniforme entre os três problemas."
        )
    lines.append(" ".join(sec_txt))
    lines.append("")
    lines.append(
        f"Limitação de poder: n = 18 pares, teste bilateral e família de {len(others)} "
        "hipóteses corrigida por Holm. O menor p bilateral exato alcançável com "
        f"n = 18 é {min_p_exact:.2e}, portanto o teste só detecta efeito quando o "
        "sinal da diferença é quase uniforme entre as 18 células. Diferenças "
        "moderadas mas inconsistentes ao longo das frações de pré-treino ficam "
        "indetectáveis, e ausência de significância não é evidência de equivalência. "
        "As células pareadas agregam 3 splits cada (média já calculada no CSV de "
        "entrada), logo a variabilidade entre splits não entra no teste: o Wilcoxon "
        "trata a heterogeneidade de dataset e de fração de pré-treino como a única "
        "fonte de variação pareada."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # ---- resumo no stdout ----
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(
            res[
                [
                    "model_label",
                    "n_effective",
                    "median_diff",
                    "median_diff_ci95_lo",
                    "median_diff_ci95_hi",
                    "W",
                    "p_raw",
                    "p_holm",
                    "p_bonferroni",
                    "rank_biserial_r",
                    "reject_holm",
                    "better_model",
                ]
            ].to_string(index=False)
        )
    print()
    print(f"escrito: {OUT_CSV}")
    print(f"escrito: {OUT_MD}")


if __name__ == "__main__":
    main()
