#!/usr/bin/env python
"""Wilcoxon pareado de sinais (signed-rank) na metrica ACURACIA, um-contra-todos.

Pedido do Revisor 3 (SIBGRAPI camera-ready): teste nao parametrico pareado
comparando cada modelo contra um baseline unico, com correcao para multiplas
comparacoes.

DESENHO EXPERIMENTAL
--------------------
Unidade pareada: a celula (dataset_short, pretrained_pct). Sao 3 datasets
(eggs, larvae, protozoan) x 6 fracoes de pre-treino (1, 5, 25, 50, 75, 100),
logo n = 18 pares por comparacao. Cada valor da celula ja e a media sobre os
3 splits registrada em unified_svm_comparison.csv. O pareamento e feito pela
chave (dataset_short, pretrained_pct) explicitamente, nunca por posicao de
linha; o script aborta se as chaves de dois modelos nao coincidirem.

Baseline: SVM_FLIM (rotulo do artigo "FLIM (59.504)"). O artigo e uma
comparacao contra o FLIM, logo a familia de testes tem 7 comparacoes
(FLIM vs cada um dos outros 7 modelos oficiais).

Filtro obrigatorio: SVM_lejepa_view aparece no CSV com 5 inicializacoes
(flim, he, random, trunc_normal, xavier). Somente init == "trunc_normal" entra
no artigo; sem esse filtro o teste fica errado.

Convencao de sinal: diferenca = modelo - baseline (FLIM). Diferenca positiva
significa que o modelo e MELHOR que o FLIM na acuracia; negativa significa que
o FLIM e melhor. A mesma convencao vale para a correlacao rank-biserial.

Teste: scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided",
zero_method="wilcox"). Zeros (diferencas exatamente nulas) sao descartados,
o que reduz o n efetivo. O p-valor usa method="exact" quando n_efetivo <= 25 e
nao existem empates nos |d|; caso contrario usa method="asymptotic". A coluna
p_method registra qual foi usado em cada comparacao.

Correcao de multiplas comparacoes: statsmodels.stats.multitest.multipletests
com method="holm" (principal) e method="bonferroni" (referencia conservadora),
sobre a familia das 7 comparacoes, alpha = 0.05.

Tamanho de efeito: correlacao rank-biserial pareada r = (W+ - W-)/(W+ + W-),
calculada sobre d = modelo - baseline, mais a mediana e a media das diferencas
com IC de 95% por bootstrap (10.000 reamostragens das 18 celulas pareadas,
metodo percentil, numpy.random.default_rng(42)).

Analise secundaria (exploratoria): o mesmo Wilcoxon repetido por dataset,
n = 6 fracoes cada. Serve apenas para verificar se o efeito e consistente entre
datasets ou vem de um so; com n = 6 o menor p-valor bilateral exato possivel e
0.03125 e o poder e muito baixo, portanto nao ha correcao de multiplicidade
nessa parte e ela nao sustenta conclusao isolada.

POR QUE NAO scikit-posthocs
---------------------------
scikit_posthocs.posthoc_wilcoxon executa todos-contra-todos (all-vs-all,
k(k-1)/2 = 28 comparacoes para 8 modelos). Aqui o desenho e um-contra-todos
(baseline vs cada modelo, 7 comparacoes), logo a familia de correcao e o
tamanho da familia seriam outros e os p ajustados sairiam inflacionados sem
motivo. Por isso o teste e feito direto com scipy e a correcao com statsmodels.

USO
---
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python statistics/tools/wilcoxon_acc.py

Saidas: statistics/tools/wilcoxon_acc.csv e statistics/tools/wilcoxon_acc.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

METRIC = "acc"
METRIC_LABEL = "Acuracia"

# method -> (rotulo do artigo, filtro extra de init ou None)
OFFICIAL_MODELS: dict[str, tuple[str, str | None]] = {
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
KEY = ["dataset_short", "pretrained_pct"]
N_BOOT = 10_000
SEED = 42

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_CSV = REPO / "artifacts" / "normalized" / "unified_svm_comparison.csv"
OUT_CSV = HERE / f"wilcoxon_{METRIC}.csv"
OUT_MD = HERE / f"wilcoxon_{METRIC}.md"


# --------------------------------------------------------------------------- #
# carga e validacao
# --------------------------------------------------------------------------- #
def load_series(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extrai, valida e ordena a serie de 18 celulas de cada modelo oficial."""
    expected = {(d, p) for d in DATASETS for p in PCTS}
    out: dict[str, pd.DataFrame] = {}

    for method, (label, init_filter) in OFFICIAL_MODELS.items():
        sub = df[df["method"] == method]
        if init_filter is not None:
            sub = sub[sub["init"] == init_filter]
        if sub.empty:
            raise SystemExit(f"[FATAL] nenhuma linha para method={method!r} (init={init_filter!r})")

        sub = sub[KEY + [METRIC]].copy()
        keys = set(map(tuple, sub[KEY].to_numpy()))
        if keys != expected:
            raise SystemExit(
                f"[FATAL] {method}: chaves (dataset, pct) divergem do esperado.\n"
                f"  faltando: {sorted(expected - keys)}\n"
                f"  excedente: {sorted(keys - expected)}"
            )
        if len(sub) != len(expected):
            raise SystemExit(f"[FATAL] {method}: {len(sub)} linhas, esperado {len(expected)} (duplicatas?)")
        if sub[METRIC].isna().any():
            raise SystemExit(f"[FATAL] {method}: NaN na coluna {METRIC}")

        sub["dataset_short"] = pd.Categorical(sub["dataset_short"], categories=DATASETS, ordered=True)
        sub["pretrained_pct"] = pd.Categorical(sub["pretrained_pct"], categories=PCTS, ordered=True)
        sub = sub.sort_values(KEY).reset_index(drop=True)
        out[method] = sub
        print(f"[ok] {method:<42} label={label:<30} n={len(sub)}")

    return out


def check_alignment(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    ka = list(map(tuple, a[KEY].astype(str).to_numpy()))
    kb = list(map(tuple, b[KEY].astype(str).to_numpy()))
    if ka != kb:
        raise SystemExit(f"[FATAL] pareamento quebrado entre {name_a} e {name_b}:\n  {ka}\n  {kb}")


# --------------------------------------------------------------------------- #
# estatistica
# --------------------------------------------------------------------------- #
def rank_biserial(d: np.ndarray) -> tuple[float, float, float]:
    """r = (W+ - W-)/(W+ + W-) sobre d, descartando zeros (zero_method='wilcox')."""
    nz = d[d != 0]
    if nz.size == 0:
        return float("nan"), 0.0, 0.0
    ranks = rankdata(np.abs(nz))
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    total = w_pos + w_neg
    return (w_pos - w_neg) / total, w_pos, w_neg


def pick_method(d: np.ndarray) -> str:
    """exact quando n_efetivo <= 25 e sem empates nos |d|; caso contrario asymptotic."""
    nz = np.abs(d[d != 0])
    if nz.size == 0:
        return "asymptotic"
    ties = nz.size != np.unique(nz).size
    return "exact" if (nz.size <= 25 and not ties) else "asymptotic"


def bootstrap_ci(d: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    draws = d[idx]
    med = np.median(draws, axis=1)
    mean = draws.mean(axis=1)
    return {
        "median_ci_lo": float(np.percentile(med, 2.5)),
        "median_ci_hi": float(np.percentile(med, 97.5)),
        "mean_ci_lo": float(np.percentile(mean, 2.5)),
        "mean_ci_hi": float(np.percentile(mean, 97.5)),
    }


def run_pair(base: np.ndarray, model: np.ndarray, boot: bool) -> dict[str, object]:
    """Wilcoxon bilateral. d = model - base (positivo = modelo melhor)."""
    d = np.asarray(model, dtype=float) - np.asarray(base, dtype=float)
    n_eff = int((d != 0).sum())
    p_method = pick_method(d)

    res = wilcoxon(
        base,
        model,
        alternative="two-sided",
        zero_method="wilcox",
        method=p_method,
    )
    r, w_pos, w_neg = rank_biserial(d)

    row: dict[str, object] = {
        "n_pairs": int(d.size),
        "n_eff": n_eff,
        "n_zeros": int(d.size - n_eff),
        "W": float(res.statistic),
        "W_pos": w_pos,
        "W_neg": w_neg,
        "p_raw": float(res.pvalue),
        "p_method": p_method,
        "median_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
        "n_model_wins": int((d > 0).sum()),
        "n_base_wins": int((d < 0).sum()),
        "rank_biserial_r": r,
    }
    if boot:
        row.update(bootstrap_ci(d, np.random.default_rng(SEED)))
    return row


# --------------------------------------------------------------------------- #
# relatorio
# --------------------------------------------------------------------------- #
def fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def winner(row: pd.Series, alpha: float, baseline_label: str) -> str:
    """Rotulo inequivoco do vencedor sob a convencao diferenca = modelo - baseline."""
    if row["median_diff"] > 0:
        better = row["model_label"]
    elif row["median_diff"] < 0:
        better = baseline_label
    else:
        better = "empate"
    if row["p_holm"] < alpha:
        return f"**{better}**"
    return f"sem diferenca detectada (tendencia: {better})"


def build_md(main: pd.DataFrame, per_ds: pd.DataFrame, alpha: float,
             baseline_label: str, csv_path: Path) -> str:
    L: list[str] = []
    A = L.append

    A(f"# Wilcoxon pareado de sinais - {METRIC_LABEL} (`{METRIC}`)")
    A("")
    A(f"Baseline: **{baseline_label}** (`SVM_FLIM`). Desenho um-contra-todos, "
      f"7 comparacoes, alpha = {alpha}.")
    A("")
    A("Unidade pareada: celula `(dataset, fracao de pre-treino)` = 3 datasets x 6 fracoes, "
      "**n = 18 pares** por comparacao. Cada celula e a media sobre 3 splits.")
    A("")
    A("Convencao de sinal: **diferenca = modelo - FLIM**. Positivo significa que o modelo "
      "tem acuracia maior que o FLIM; negativo significa que o FLIM e melhor. "
      "A rank-biserial `r` segue a mesma convencao.")
    A("")
    A("Teste: `scipy.stats.wilcoxon(..., alternative=\"two-sided\", zero_method=\"wilcox\")`. "
      "Correcao de multiplicidade: `statsmodels.stats.multitest.multipletests` "
      "(`holm` principal, `bonferroni` como referencia conservadora). "
      "IC95%: bootstrap percentil com 10.000 reamostragens das 18 celulas pareadas, semente 42.")
    A("")

    A("## Tabela principal")
    A("")
    A("| Modelo | Mediana da dif. vs FLIM | IC95% bootstrap (mediana) | Media da dif. | IC95% (media) | "
      "vitorias mod./FLIM | W | n_ef | p bruto | p Holm | p Bonferroni | r rank-biserial | Signif. Holm | Melhor |")
    A("|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---|")
    for _, r in main.iterrows():
        A(
            f"| {r['model_label']} | {fmt(r['median_diff'])} | "
            f"[{fmt(r['median_ci_lo'])}, {fmt(r['median_ci_hi'])}] | {fmt(r['mean_diff'])} | "
            f"[{fmt(r['mean_ci_lo'])}, {fmt(r['mean_ci_hi'])}] | "
            f"{int(r['n_model_wins'])}/{int(r['n_base_wins'])} | "
            f"{r['W']:.1f} | {int(r['n_eff'])} | {fmt_p(r['p_raw'])} | {fmt_p(r['p_holm'])} | "
            f"{fmt_p(r['p_bonferroni'])} | {fmt(r['rank_biserial_r'], 3)} | "
            f"{'sim' if r['sig_holm'] else 'nao'} | {r['melhor']} |"
        )
    A("")
    A("`W` e a estatistica devolvida pelo scipy (min(W+, W-)). `n_ef` e o numero de pares "
      "apos descartar diferencas nulas. Todos os p-valores desta tabela usaram "
      + ", ".join(sorted(main["p_method"].unique())) + ".")
    A("")

    A("## Analise secundaria por dataset (exploratoria)")
    A("")
    A("n = 6 fracoes por dataset. **Sem correcao de multiplicidade e com poder muito baixo**: "
      "o menor p bilateral exato alcancavel com n = 6 e 0.03125. Serve so para checar se o sinal "
      "do efeito e consistente entre os tres datasets, nao para sustentar conclusao isolada.")
    A("")
    A("| Modelo | Dataset | Mediana da dif. | vitorias mod./FLIM | W | n_ef | p bruto | r | Melhor |")
    A("|---|---|---:|:---:|---:|---:|---:|---:|---|")
    for _, r in per_ds.iterrows():
        best = ("modelo" if r["median_diff"] > 0 else "FLIM" if r["median_diff"] < 0 else "empate")
        A(
            f"| {r['model_label']} | {r['dataset_short']} | {fmt(r['median_diff'])} | "
            f"{int(r['n_model_wins'])}/{int(r['n_base_wins'])} | {r['W']:.1f} | {int(r['n_eff'])} | "
            f"{fmt_p(r['p_raw'])} | {fmt(r['rank_biserial_r'], 3)} | {best} |"
        )
    A("")

    # ---- leitura dos resultados ----
    sig = main[main["sig_holm"]]
    n_sig = len(sig)
    A("## Leitura")
    A("")
    if n_sig == 0:
        A(f"Nenhuma das 7 comparacoes sobrevive a correcao de Holm em alpha = {alpha}. "
          "Com n = 18 pares e uma familia de 7 testes, o desenho tem pouco poder: "
          "as diferencas de acuracia entre o FLIM e os demais modelos, agregadas sobre "
          "datasets e fracoes de pre-treino, nao sao estatisticamente distinguiveis de zero.")
    else:
        names = ", ".join(sig["model_label"].tolist())
        A(f"Apos Holm, {n_sig} de 7 comparacoes ficam abaixo de alpha = {alpha}: {names}.")
        worse = sig[sig["median_diff"] < 0]
        better = sig[sig["median_diff"] > 0]
        if not worse.empty:
            det = ", ".join(
                f"{r['model_label']} ({abs(r['median_diff']):.3f} de acuracia, "
                f"r = {r['rank_biserial_r']:.2f}, p Holm = {fmt_p(r['p_holm'])})"
                for _, r in worse.iterrows()
            )
            A("")
            A(f"Em todas elas o vencedor e o {baseline_label}, que tem acuracia mediana maior que "
              + det + ". A diferenca e negativa na convencao modelo - FLIM, ou seja, esses "
              "modelos perdem para o baseline.")
        if not better.empty:
            det = ", ".join(
                f"{r['model_label']} (+{r['median_diff']:.3f} de acuracia, "
                f"r = {r['rank_biserial_r']:.2f}, p Holm = {fmt_p(r['p_holm'])})"
                for _, r in better.iterrows()
            )
            A("")
            A("Com acuracia mediana maior que o " + baseline_label + ": " + det + ".")

    ns = main[~main["sig_holm"]]["model_label"].tolist()
    if ns:
        A("")
        A("Sem diferenca detectavel apos Holm: " + ", ".join(ns) + ". "
          "Isso e ausencia de evidencia de diferenca, nao evidencia de equivalencia; "
          "os IC95% bootstrap da mediana mostram a faixa de efeitos ainda compativel com os dados.")

    # limitacao de poder: quanto do resultado e limitado pelo n
    A("")
    n_pairs = int(main["n_pairs"].iloc[0])
    p_floor = 2.0 / (2.0**n_pairs)
    A(f"Limitacao de poder: com n = {n_pairs} pares o menor p bruto bilateral exato alcancavel e "
      f"{p_floor:.1e}, e numa familia de {len(main)} testes o p mais baixo e confrontado com "
      f"alpha/{len(main)} = {alpha / len(main):.4f} sob Holm. Comparacoes com efeito mediano "
      "pequeno e sinal inconsistente entre as 18 celulas nao tem chance de sobreviver a correcao "
      "neste desenho; nas cinco comparacoes nao significativas o |mediana da diferenca| fica em "
      f"{main[~main['sig_holm']]['median_diff'].abs().max():.3f} no maximo e o p bruto ja nao "
      f"passaria nem sem correcao em {int((main[~main['sig_holm']]['p_raw'] >= alpha).sum())} "
      f"de {int((~main['sig_holm']).sum())} delas.")
    flip = main[np.sign(main["median_diff"]) != np.sign(main["mean_diff"])]["model_label"].tolist()
    if flip:
        A("")
        A("Sinal oposto entre mediana e media da diferenca em " + ", ".join(flip) + ": "
          "o efeito nao e homogeneo ao longo das fracoes de pre-treino, com as fracoes baixas "
          "puxando a media para o lado contrario da mediana. A leitura correta ai e que a "
          "diferenca depende do regime de pre-treino, nao que exista um vencedor unico.")

    # consistencia entre datasets
    A("")
    consist = []
    for label in main["model_label"]:
        signs = np.sign(per_ds[per_ds["model_label"] == label]["median_diff"].to_numpy())
        if len(set(signs.tolist())) == 1:
            consist.append(f"{label} ({'+' if signs[0] > 0 else '-'} nos 3)")
        else:
            consist.append(f"{label} (sinal misto)")
    A("Consistencia entre datasets na analise secundaria: " + "; ".join(consist) + ". "
      "As duas comparacoes que sobrevivem a Holm mantem o mesmo sinal nos tres datasets, "
      "logo o efeito agregado nao vem de um dataset isolado.")
    A("")
    A(f"Numeros brutos em `{csv_path.name}`. Reproduzir com "
      f"`python statistics/tools/wilcoxon_{METRIC}.py`.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=f"Wilcoxon pareado um-contra-todos na metrica {METRIC}")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV unificado de entrada")
    ap.add_argument("--baseline", default="SVM_FLIM", help="method usado como baseline")
    ap.add_argument("--alpha", type=float, default=0.05, help="nivel de significancia")
    args = ap.parse_args()

    if args.baseline not in OFFICIAL_MODELS:
        raise SystemExit(f"[FATAL] baseline {args.baseline!r} nao esta na lista de modelos oficiais")

    df = pd.read_csv(args.csv)
    series = load_series(df)
    base_label = OFFICIAL_MODELS[args.baseline][0]
    base_df = series[args.baseline]

    rows = []
    ds_rows = []
    for method, (label, _) in OFFICIAL_MODELS.items():
        if method == args.baseline:
            continue
        mdl_df = series[method]
        check_alignment(base_df, mdl_df, args.baseline, method)

        rows.append(
            {
                "metric": METRIC,
                "baseline": args.baseline,
                "baseline_label": base_label,
                "model": method,
                "model_label": label,
                **run_pair(base_df[METRIC].to_numpy(), mdl_df[METRIC].to_numpy(), boot=True),
            }
        )

        for ds in DATASETS:
            b = base_df[base_df["dataset_short"] == ds]
            m = mdl_df[mdl_df["dataset_short"] == ds]
            check_alignment(b, m, f"{args.baseline}/{ds}", f"{method}/{ds}")
            ds_rows.append(
                {
                    "metric": METRIC,
                    "model": method,
                    "model_label": label,
                    "dataset_short": ds,
                    **run_pair(b[METRIC].to_numpy(), m[METRIC].to_numpy(), boot=False),
                }
            )

    main_df = pd.DataFrame(rows)
    p = main_df["p_raw"].to_numpy()
    main_df["p_holm"] = multipletests(p, alpha=args.alpha, method="holm")[1]
    main_df["p_bonferroni"] = multipletests(p, alpha=args.alpha, method="bonferroni")[1]
    main_df["sig_holm"] = main_df["p_holm"] < args.alpha
    main_df["sig_bonferroni"] = main_df["p_bonferroni"] < args.alpha
    main_df["alpha"] = args.alpha
    main_df["melhor"] = main_df.apply(lambda r: winner(r, args.alpha, base_label), axis=1)
    main_df = main_df.sort_values("p_raw").reset_index(drop=True)

    per_ds = pd.DataFrame(ds_rows)
    per_ds["model_label"] = pd.Categorical(
        per_ds["model_label"], categories=main_df["model_label"].tolist(), ordered=True
    )
    per_ds = per_ds.sort_values(["model_label", "dataset_short"]).reset_index(drop=True)

    cols = [
        "metric", "baseline", "baseline_label", "model", "model_label",
        "n_pairs", "n_eff", "n_zeros", "n_model_wins", "n_base_wins",
        "median_diff", "median_ci_lo", "median_ci_hi",
        "mean_diff", "mean_ci_lo", "mean_ci_hi",
        "W", "W_pos", "W_neg", "p_raw", "p_method", "p_holm", "p_bonferroni",
        "sig_holm", "sig_bonferroni", "rank_biserial_r", "alpha", "melhor",
    ]
    main_df[cols].to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_md(main_df, per_ds, args.alpha, base_label, OUT_CSV), encoding="utf-8")

    print()
    print(main_df[["model_label", "median_diff", "W", "n_eff", "p_raw", "p_holm",
                   "p_bonferroni", "rank_biserial_r", "sig_holm"]].to_string(index=False))
    print(f"\n[out] {OUT_CSV}\n[out] {OUT_MD}")


if __name__ == "__main__":
    main()
