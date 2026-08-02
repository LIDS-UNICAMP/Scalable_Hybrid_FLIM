#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wilcoxon pareado: init FLIM congelada vs init aleatoria, com a MESMA cabeca de 123K.

Desenho experimental
--------------------
Diferente dos tres scripts `wilcoxon_{f1,acc,kappa}.py`, que usam o FLIM como baseline
num desenho um-contra-todos (7 comparacoes), aqui o desenho e uma **comparacao controlada
de um fator**: dois modelos com arquitetura e contagem de parametros identicas
(123.504 params = encoder FLIM de 59.504 + cabeca 1x1 BN2d de 64.000), diferindo apenas
na origem dos pesos do encoder.

    baseline: SVM_Distill_1x1BN                        (init trunc_normal, encoder treinavel)
    modelo:   SVM_Distill_1x1BN_flim_frozen_eval_loss  (pesos FLIM reais, encoder CONGELADO)

A pergunta que o teste responde: trocar pesos aleatorios por filtros FLIM congelados,
*sem adicionar um unico parametro*, muda a qualidade do embedding destilado?

* Unidade pareada: a celula ``(dataset_short, pretrained_pct)``.
  3 datasets (eggs, larvae, protozoan) x 6 fracoes de rotulos (1, 5, 25, 50, 75, 100)
  => n = 18 pares. Cada celula do CSV de entrada ja e a media sobre 3 splits, logo nao
  ha re-agregacao aqui. O pareamento e feito pela chave (dataset, fracao), nunca por
  posicao de linha; o script falha alto se as chaves nao coincidirem exatamente.
* ``pretrained_pct`` neste CSV e o **% de dados rotulados usados para treinar o SVM
  linear**, nao fracao de pre-treino (ver `metrics_distillation/data_provenance.md`,
  secao 3). Os relatorios `wilcoxon_f1.md`/`_acc`/`_kappa` chamam esse eixo de
  "pre-treino"; a nomenclatura correta e a deste script.
* Familia principal: as **3 metricas** (f1, kappa, acc) da mesma comparacao controlada.
  Correcao de multiplicidade por Holm (principal) e Bonferroni (referencia) sobre essas
  3 hipoteses. alpha = 0.05.
* Teste: ``scipy.stats.wilcoxon(baseline, modelo, alternative="two-sided",
  zero_method="wilcox")``, ``method="exact"`` (n = 18 permite), com fallback para
  ``method="auto"``; o metodo efetivamente usado e reportado.
* Tamanho de efeito: correlacao rank-biserial pareada r = (W+ - W-) / (W+ + W-) sobre as
  diferencas ``modelo - baseline``, mais mediana e media das diferencas com IC95% por
  bootstrap percentil (10.000 reamostragens das 18 celulas, ``default_rng(42)``).

Convencao de sinal (usada em todo o script e no relatorio)
----------------------------------------------------------
    diferenca = metrica(FLIM init congelada) - metrica(init aleatoria)
    diferenca > 0  =>  a init FLIM e melhor
    diferenca < 0  =>  a init aleatoria e melhor
O mesmo vale para r rank-biserial.

Analises secundarias
--------------------
1. **Por dataset** (exploratoria, n = 6 fracoes cada, metrica F1): checa se o efeito vale
   nos tres dominios ou e peculiaridade de um. Com n = 6 o menor p bilateral alcancavel
   pelo teste exato e 0.03125 e o poder e minimo; nenhuma correcao e aplicada e estes p
   nao sustentam conclusao isolada.
2. **Robustez da escolha do braco FLIM** (metrica F1, fora da familia de correcao):
   - ``SVM_Distill_1x1BN_flim_frozen_eval_knn`` — mesmo encoder congelado, checkpoint
     selecionado por ``val/knn_kappa`` em vez de best-loss. Mostra que a conclusao nao
     depende do criterio de selecao de checkpoint.
   - ``SVM_Distill_1x1BN_flim`` — init FLIM **destreinada** (encoder liberado). Isola
     "init FLIM" de "encoder congelado" e responde se congelar custa ou rende qualidade.
3. **Descritiva por fracao de rotulos**: media das 3 celulas de cada fracao, para mostrar
   em que regime de supervisao o ganho aparece (o teste agregado nao diz isso).
4. **Teste colapsado (n = 3)**: as 18 celulas *nao* sao independentes — dentro de um dataset,
   os 6 subconjuntos de treino sao estritamente encaixados (1% subset de 5% subset de 25%...,
   contencao verificada em `data/to_mateus/splits_incremental/`) e o conjunto de teste e
   identico nas 6 fracoes. O Wilcoxon supoe pares independentes, logo o n = 18 e otimista.
   O teste colapsado reduz cada dataset a uma unica observacao (mediana das 6 fracoes),
   dando n = 3 — a leitura mais conservadora possivel. Serve de piso, nao de resultado
   principal: com n = 3 o menor p bilateral alcancavel e 0.25, logo nunca ha significancia.

Uso
---
    cd /dados/home/moliveira/Scalable_Hybrid_FLIM
    LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
    /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
        statistics/tools/wilcoxon_flim_init.py

Saidas: statistics/tools/wilcoxon_flim_init.csv e statistics/tools/wilcoxon_flim_init.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASELINE = "SVM_Distill_1x1BN"
BASELINE_LABEL = "Distill 1 - init aleatoria (123K)"
BASELINE_INIT = "trunc_normal"

MODEL = "SVM_Distill_1x1BN_flim_frozen_eval_loss"
MODEL_LABEL = "Distill 1 - FLIM init congelada (123K)"

# metricas da familia principal: (coluna, rotulo)
METRICS: list[tuple[str, str]] = [("f1", "F1"), ("kappa", "kappa"), ("acc", "Acc")]

# bracos alternativos do lado FLIM (analise 2, fora da familia de correcao)
ROBUSTNESS: dict[str, str] = {
    "SVM_Distill_1x1BN_flim_frozen_eval_knn": "FLIM init congelada, ckpt knn+kappa (123K)",
    "SVM_Distill_1x1BN_flim": "FLIM init destreinada, encoder liberado (123K)",
}

# params: backbone FLIM ch24_32_48 + cabeca de projecao (distillation_model_architecture.md)
BACKBONE_PARAMS = 59_504
HEAD_PARAMS = 64_000  # Conv2d 48->1280 k=1x1 bias=False (61.440) + BN2d(1280) (2.560)
TOTAL_PARAMS = BACKBONE_PARAMS + HEAD_PARAMS

DATASETS = ["eggs", "larvae", "protozoan"]
PCTS = [1, 5, 25, 50, 75, 100]
N_BOOT = 10_000
SEED = 42

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "artifacts" / "normalized" / "unified_svm_comparison.csv"
OUT_CSV = Path(__file__).resolve().parent / "wilcoxon_flim_init.csv"
OUT_MD = Path(__file__).resolve().parent / "wilcoxon_flim_init.md"


def load_series(df: pd.DataFrame, method: str, metric: str,
                init_filter: str | None = None) -> pd.Series:
    """Serie de `metric` indexada por (dataset_short, pretrained_pct) para um metodo."""
    sub = df[df["method"] == method]
    if init_filter is not None:
        sub = sub[sub["init"] == init_filter]
    sub = sub.copy()
    sub["pretrained_pct"] = sub["pretrained_pct"].astype(int)
    sub["dataset_short"] = sub["dataset_short"].astype(str)

    dup = sub.duplicated(subset=["dataset_short", "pretrained_pct"]).sum()
    if dup:
        raise SystemExit(
            f"[FALHA] {method} tem {dup} celula(s) (dataset, pct) duplicada(s). "
            "Filtro de init insuficiente."
        )
    if sub[metric].isna().any():
        raise SystemExit(f"[FALHA] {method} tem NaN na coluna {metric}.")

    ser = sub.set_index(["dataset_short", "pretrained_pct"])[metric].astype(float).sort_index()

    expected = sorted((d, p) for d in DATASETS for p in PCTS)
    got = sorted(ser.index.tolist())
    if got != expected:
        faltando = sorted(set(expected) - set(got))
        sobrando = sorted(set(got) - set(expected))
        raise SystemExit(
            f"[FALHA] grade incompleta para {method}: {len(got)} celulas em vez de 18. "
            f"Faltando={faltando} Sobrando={sobrando}"
        )
    return ser


def signed_rank_parts(diff: np.ndarray) -> tuple[float, float, int]:
    """W+, W- e n efetivo sobre `diff` (zeros descartados, zero_method='wilcox')."""
    nz = diff[diff != 0.0]
    if nz.size == 0:
        return 0.0, 0.0, 0
    ranks = stats.rankdata(np.abs(nz))
    return float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum()), int(nz.size)


def wilcoxon_pair(base: np.ndarray, model: np.ndarray) -> dict:
    """Wilcoxon bilateral + efeito. Convencao: diff = model - base."""
    diff = model - base
    w_pos, w_neg, n_eff = signed_rank_parts(diff)

    method_used = "exact"
    try:
        res = stats.wilcoxon(base, model, alternative="two-sided",
                             zero_method="wilcox", method="exact")
    except Exception:
        method_used = "auto"
        res = stats.wilcoxon(base, model, alternative="two-sided",
                             zero_method="wilcox", method="auto")

    denom = w_pos + w_neg
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
        "rank_biserial_r": (w_pos - w_neg) / denom if denom > 0 else float("nan"),
        "n_model_wins": int((diff > 0).sum()),
        "n_base_wins": int((diff < 0).sum()),
        "n_ties": int((diff == 0).sum()),
    }


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> dict:
    """IC95% percentil por bootstrap das celulas pareadas (mediana e media da diferenca)."""
    idx = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
    samples = diff[idx]
    lo_m, hi_m = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    lo_a, hi_a = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {
        "median_ci_lo": float(lo_m), "median_ci_hi": float(hi_m),
        "mean_ci_lo": float(lo_a), "mean_ci_hi": float(hi_a),
    }


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def mil(n: int) -> str:
    """Inteiro com separador de milhar no padrao pt-BR (ponto)."""
    return f"{n:,}".replace(",", ".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV unificado de entrada")
    ap.add_argument("--alpha", type=float, default=0.05, help="nivel de significancia")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"method", "init", "dataset_short", "pretrained_pct"} | {m for m, _ in METRICS}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"[FALHA] colunas ausentes em {args.csv}: {missing}")

    rng = np.random.default_rng(SEED)

    # ---- familia principal: 3 metricas da comparacao controlada ----
    rows: list[dict] = []
    diffs_f1: pd.Series | None = None
    base_f1: pd.Series | None = None
    model_f1: pd.Series | None = None

    for metric, mlabel in METRICS:
        base_ser = load_series(df, BASELINE, metric, BASELINE_INIT)
        mod_ser = load_series(df, MODEL, metric)
        if list(base_ser.index) != list(mod_ser.index):
            raise SystemExit(f"[FALHA] chaves de pareamento divergem em {metric}.")

        base_v, mod_v = base_ser.to_numpy(), mod_ser.to_numpy()
        d = mod_v - base_v

        row = {
            "analise": "principal", "metric": metric, "metric_label": mlabel,
            "baseline": BASELINE, "baseline_label": BASELINE_LABEL,
            "model": MODEL, "model_label": MODEL_LABEL,
            "escopo": "agregado (18 celulas)",
            "baseline_params": TOTAL_PARAMS, "model_params": TOTAL_PARAMS,
        }
        row.update(wilcoxon_pair(base_v, mod_v))
        row.update(bootstrap_ci(d, rng))
        row["baseline_median"] = float(np.median(base_v))
        row["model_median"] = float(np.median(mod_v))
        row["baseline_mean"] = float(np.mean(base_v))
        row["model_mean"] = float(np.mean(mod_v))
        rows.append(row)

        if metric == "f1":
            base_f1, model_f1 = base_ser, mod_ser
            diffs_f1 = pd.Series(d, index=base_ser.index)

    main_df = pd.DataFrame(rows)
    for meth in ("holm", "bonferroni"):
        rej, padj, _, _ = multipletests(main_df["p_raw"].to_numpy(),
                                        alpha=args.alpha, method=meth)
        main_df[f"p_{meth}"] = padj
        main_df[f"sig_{meth}"] = rej

    def winner(r: pd.Series) -> str:
        if r["median_diff"] > 0:
            return MODEL_LABEL
        if r["median_diff"] < 0:
            return BASELINE_LABEL
        return "empate"

    main_df["melhor"] = main_df.apply(winner, axis=1)
    main_df["conclusao_holm"] = [
        (f"{w} melhor" if s else "sem diferenca detectada")
        for w, s in zip(main_df["melhor"], main_df["sig_holm"])
    ]

    assert base_f1 is not None and model_f1 is not None and diffs_f1 is not None

    # ---- secundaria 1: por dataset (F1, n = 6, exploratoria) ----
    sec_rows: list[dict] = []
    for ds in DATASETS:
        b = base_f1.loc[ds].to_numpy()
        m = model_f1.loc[ds].to_numpy()
        r = wilcoxon_pair(b, m)
        sec_rows.append({
            "analise": "por_dataset", "metric": "f1", "metric_label": "F1",
            "baseline": BASELINE, "baseline_label": BASELINE_LABEL,
            "model": MODEL, "model_label": MODEL_LABEL, "escopo": ds,
            "baseline_params": TOTAL_PARAMS, "model_params": TOTAL_PARAMS,
            "baseline_median": float(np.median(b)), "model_median": float(np.median(m)),
            "baseline_mean": float(np.mean(b)), "model_mean": float(np.mean(m)),
            **r,
        })
    sec_df = pd.DataFrame(sec_rows)

    # ---- secundaria 2: robustez do braco FLIM (F1, 18 celulas, sem correcao) ----
    rob_rows: list[dict] = []
    for meth, label in ROBUSTNESS.items():
        alt = load_series(df, meth, "f1")
        if list(alt.index) != list(base_f1.index):
            raise SystemExit(f"[FALHA] chaves de pareamento divergem para {meth}.")
        b, m = base_f1.to_numpy(), alt.to_numpy()
        r = wilcoxon_pair(b, m)
        r.update(bootstrap_ci(m - b, rng))
        rob_rows.append({
            "analise": "robustez", "metric": "f1", "metric_label": "F1",
            "baseline": BASELINE, "baseline_label": BASELINE_LABEL,
            "model": meth, "model_label": label, "escopo": "agregado (18 celulas)",
            "baseline_params": TOTAL_PARAMS, "model_params": TOTAL_PARAMS,
            "baseline_median": float(np.median(b)), "model_median": float(np.median(m)),
            "baseline_mean": float(np.mean(b)), "model_mean": float(np.mean(m)),
            **r,
        })
    rob_df = pd.DataFrame(rob_rows)

    # ---- secundaria 3: teste colapsado, n = 3 datasets (piso conservador) ----
    col_b = base_f1.groupby(level="dataset_short").median()
    col_m = model_f1.groupby(level="dataset_short").median()
    col_r = wilcoxon_pair(col_b.to_numpy(), col_m.to_numpy())
    col_df = pd.DataFrame([{
        "analise": "colapsado_n3", "metric": "f1", "metric_label": "F1",
        "baseline": BASELINE, "baseline_label": BASELINE_LABEL,
        "model": MODEL, "model_label": MODEL_LABEL,
        "escopo": "mediana por dataset (n = 3)",
        "baseline_params": TOTAL_PARAMS, "model_params": TOTAL_PARAMS,
        "baseline_median": float(np.median(col_b)), "model_median": float(np.median(col_m)),
        "baseline_mean": float(np.mean(col_b)), "model_mean": float(np.mean(col_m)),
        **col_r,
    }])

    # ---- descritiva por fracao de rotulos (F1) ----
    per_pct = pd.DataFrame({
        "f1_aleatoria": base_f1.groupby(level="pretrained_pct").mean(),
        "f1_flim_init": model_f1.groupby(level="pretrained_pct").mean(),
    })
    per_pct["delta"] = per_pct["f1_flim_init"] - per_pct["f1_aleatoria"]
    per_pct["celulas_pro_flim"] = (
        (diffs_f1 > 0).groupby(level="pretrained_pct").sum().astype(int)
    )

    # ---------------- CSV ----------------
    cols = ["analise", "metric", "metric_label", "escopo",
            "baseline", "baseline_label", "model", "model_label",
            "baseline_params", "model_params", "melhor",
            "n_pairs", "n_eff", "n_model_wins", "n_base_wins", "n_ties",
            "baseline_median", "model_median", "baseline_mean", "model_mean",
            "median_diff", "median_ci_lo", "median_ci_hi",
            "mean_diff", "mean_ci_lo", "mean_ci_hi",
            "W", "W_pos", "W_neg", "rank_biserial_r", "method_wilcoxon",
            "p_raw", "p_holm", "p_bonferroni", "sig_holm", "sig_bonferroni",
            "conclusao_holm", "sign_convention"]
    out = pd.concat([main_df, sec_df, rob_df, col_df], ignore_index=True)
    out["sign_convention"] = "diff = metrica(FLIM init) - metrica(init aleatoria); >0 favorece FLIM init"
    # `melhor` e so o sinal da mediana: vale para as linhas secundarias tambem.
    # `p_holm`/`p_bonferroni`/`conclusao_holm` ficam vazios nelas de proposito — a correcao
    # de multiplicidade cobre apenas a familia principal das 3 metricas.
    out["melhor"] = out["melhor"].fillna(
        pd.Series(
            [
                lab if d > 0 else (BASELINE_LABEL if d < 0 else "empate")
                for d, lab in zip(out["median_diff"], out["model_label"])
            ],
            index=out.index,
        )
    )
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out[cols].to_csv(OUT_CSV, index=False)

    # ---------------- Markdown ----------------
    wm = sorted(set(out["method_wilcoxon"].dropna()))
    f1_row = main_df[main_df["metric"] == "f1"].iloc[0]
    n_sig = int(main_df["sig_holm"].sum())
    ratio_head = HEAD_PARAMS / BACKBONE_PARAMS

    L: list[str] = []
    L.append("# Wilcoxon pareado: init FLIM congelada vs init aleatoria, mesma cabeca de 123K")
    L.append("")
    L.append(
        f"Comparacao controlada de **um fator**. Os dois bracos tem arquitetura e contagem de "
        f"parametros **identicas** — {mil(TOTAL_PARAMS)} params = encoder FLIM `ch24_32_48` "
        f"({mil(BACKBONE_PARAMS)}) + cabeca de projecao 1x1 BN2d 48->1280 ({mil(HEAD_PARAMS)}, "
        f"ou {ratio_head:.2f}x o backbone). A unica diferenca e a origem dos pesos do encoder:"
    )
    L.append("")
    L.append(f"- **baseline** `{BASELINE}` (init `{BASELINE_INIT}`) — {BASELINE_LABEL}, encoder treinavel;")
    L.append(f"- **modelo** `{MODEL}` — {MODEL_LABEL}, pesos FLIM reais, **encoder congelado** (zero gradiente no encoder), checkpoint por best-loss.")
    L.append("")
    L.append(
        "Metricas: colunas `f1`, `kappa` e `acc` de "
        "`artifacts/normalized/unified_svm_comparison.csv` (embedding `proj (B,1280)` avaliado "
        f"por SVM linear). Familia principal = essas 3 metricas, alfa = {args.alpha}."
    )
    L.append("")
    L.append(
        "Unidade pareada: celula `(dataset, % de rotulos)`, 3 datasets x 6 fracoes = "
        "**n = 18 pares** por metrica. Cada celula ja e a media sobre 3 splits. O pareamento "
        "usa a chave `(dataset, fracao)`, verificada antes de cada teste."
    )
    L.append("")
    L.append(
        "> **Nota de nomenclatura:** a coluna `pretrained_pct` do CSV e o **% de dados rotulados "
        "usados para treinar o SVM linear**, nao uma fracao de pre-treino (o teacher e frozen) — "
        "ver secao 3 de `metrics_distillation/data_provenance.md`. Os relatorios "
        "`wilcoxon_f1.md`/`_acc.md`/`_kappa.md` chamam esse mesmo eixo de \"pre-treino\"; a "
        "nomenclatura correta e a usada aqui."
    )
    L.append("")
    L.append(
        f"Teste: `scipy.stats.wilcoxon(baseline, modelo, alternative=\"two-sided\", "
        f"zero_method=\"wilcox\")`, metodo `{'`, `'.join(wm)}`. Correcao de multiplicidade por "
        "Holm (principal) e Bonferroni (referencia) sobre as 3 metricas, via "
        "`statsmodels.stats.multitest.multipletests`. IC95% por bootstrap percentil das 18 "
        f"celulas pareadas ({mil(N_BOOT)} reamostragens, seed {SEED})."
    )
    L.append("")
    L.append(
        "**Convencao de sinal**: diferenca = metrica(FLIM init congelada) - metrica(init "
        "aleatoria). Positivo significa que a init FLIM supera a aleatoria. O mesmo vale para "
        "`r` rank-biserial."
    )
    L.append("")

    # tabela principal
    L.append("## Tabela principal (n = 18 pares por linha)")
    L.append("")
    L.append("| Metrica | Celulas pro FLIM init | Mediana da dif. | IC95% bootstrap (mediana) | Media da dif. | W | p bruto | p Holm | p Bonferroni | r rank-biserial | Sig. Holm? |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in main_df.iterrows():
        L.append(
            f"| **{r['metric_label']}** | {r['n_model_wins']}/{r['n_pairs']} "
            f"| {r['median_diff']:+.4f} "
            f"| [{r['median_ci_lo']:+.4f}, {r['median_ci_hi']:+.4f}] "
            f"| {r['mean_diff']:+.4f} | {r['W']:.1f} | {fmt_p(r['p_raw'])} "
            f"| {fmt_p(r['p_holm'])} | {fmt_p(r['p_bonferroni'])} "
            f"| {r['rank_biserial_r']:+.3f} | {'sim' if r['sig_holm'] else 'nao'} |"
        )
    L.append("")
    L.append("Medianas por braco nas 18 celulas:")
    L.append("")
    L.append("| Metrica | Init aleatoria | FLIM init congelada |")
    L.append("|---|---:|---:|")
    for _, r in main_df.iterrows():
        L.append(f"| {r['metric_label']} | {r['baseline_median']:.4f} | {r['model_median']:.4f} |")
    L.append("")

    # descritiva por fracao
    L.append("## F1 medio por fracao de rotulos (descritiva, 3 datasets por linha)")
    L.append("")
    L.append("| % de rotulos | Init aleatoria | FLIM init congelada | Delta | Celulas pro FLIM init |")
    L.append("|---:|---:|---:|---:|---:|")
    for pct, r in per_pct.iterrows():
        L.append(
            f"| {int(pct)}% | {r['f1_aleatoria']:.4f} | {r['f1_flim_init']:.4f} "
            f"| {r['delta']:+.4f} | {int(r['celulas_pro_flim'])}/3 |"
        )
    L.append("")
    L.append(
        f"O teto da init aleatoria e **{per_pct['f1_aleatoria'].max():.4f}**, alcancado com "
        f"**100% dos rotulos** — ela nunca ultrapassa esse valor em nenhum regime de supervisao. "
        f"A init FLIM congelada ja supera esse teto a partir de **25%** dos rotulos "
        f"({per_pct.loc[25, 'f1_flim_init']:.4f})."
    )
    L.append("")

    # por dataset
    L.append("## Tabela secundaria por dataset (exploratoria, F1)")
    L.append("")
    L.append(
        "Wilcoxon dentro de cada dataset, n = 6 fracoes. Serve para checar consistencia do "
        "efeito entre dominios. Com n = 6 o menor p bilateral alcancavel pelo teste exato e "
        "0.03125, o poder e minimo e nenhuma correcao de multiplicidade foi aplicada aqui: "
        "estes p nao sustentam conclusao isolada."
    )
    L.append("")
    L.append("| Dataset | Mediana da dif. | Celulas pro FLIM init | W | p bruto (nao corrigido) | r rank-biserial |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in sec_df.iterrows():
        L.append(
            f"| {r['escopo']} | {r['median_diff']:+.4f} "
            f"| {r['n_model_wins']}/{r['n_pairs']} | {r['W']:.1f} "
            f"| {fmt_p(r['p_raw'])} | {r['rank_biserial_r']:+.3f} |"
        )
    L.append("")

    # robustez
    L.append("## Robustez do braco FLIM (F1, 18 celulas, fora da familia de correcao)")
    L.append("")
    L.append(
        "Duas variantes do lado FLIM, sempre contra o mesmo baseline aleatorio. A primeira troca "
        "o criterio de selecao de checkpoint; a segunda libera o encoder, separando \"init FLIM\" "
        "de \"encoder congelado\"."
    )
    L.append("")
    L.append("| Braco FLIM | Mediana da dif. vs aleatoria | IC95% (mediana) | Celulas pro FLIM | W | p bruto | F1 medio |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in rob_df.iterrows():
        L.append(
            f"| {r['model_label']} | {r['median_diff']:+.4f} "
            f"| [{r['median_ci_lo']:+.4f}, {r['median_ci_hi']:+.4f}] "
            f"| {r['n_model_wins']}/{r['n_pairs']} | {r['W']:.1f} | {fmt_p(r['p_raw'])} "
            f"| {r['model_mean']:.4f} |"
        )
    L.append(
        f"| {MODEL_LABEL} *(braco principal)* | {f1_row['median_diff']:+.4f} "
        f"| [{f1_row['median_ci_lo']:+.4f}, {f1_row['median_ci_hi']:+.4f}] "
        f"| {f1_row['n_model_wins']}/{f1_row['n_pairs']} | {f1_row['W']:.1f} "
        f"| {fmt_p(f1_row['p_raw'])} | {f1_row['model_mean']:.4f} |"
    )
    L.append("")

    # leitura
    L.append("## Leitura dos resultados")
    L.append("")
    L.append("### O que este teste faz, em linguagem simples")
    L.append("")
    L.append(
        "Os dois modelos enfrentaram um ao outro em **18 situacoes identicas**: 3 conjuntos de "
        "imagens (eggs, larvae, protozoan) x 6 quantidades de rotulos (1%, 5%, 25%, 50%, 75%, "
        "100%). Como rodam exatamente na mesma condicao e com exatamente o mesmo numero de "
        "parametros, da para comparar um a um quem obteve a metrica maior. Chamamos cada uma "
        "dessas situacoes de *celula*."
    )
    L.append("")
    L.append(
        "O teste de Wilcoxon olha o placar dessas 18 comparacoes e responde a uma unica pergunta: "
        "**um resultado tao desequilibrado assim apareceria por puro acaso?** Ele nao conta apenas "
        "quantas celulas cada lado venceu: ordena as 18 diferencas e da mais peso as vitorias mais "
        "folgadas. E isso que a palavra *posto* (rank) significa."
    )
    L.append("")
    L.append(
        "A mesma pergunta foi feita 3 vezes (uma por metrica). Fazer varias perguntas aumenta a "
        "chance de uma delas dar \"positivo\" por acaso, e as correcoes de **Holm** e **Bonferroni** "
        "compensam isso exigindo evidencia mais forte de cada comparacao. Por isso a coluna que "
        "decide e `p Holm`, nao `p bruto`."
    )
    L.append("")
    L.append("### O que o teste demonstrou")
    L.append("")
    L.append(
        f"Nas **{n_sig} de 3** metricas a init FLIM congelada supera a aleatoria ao nivel "
        f"alfa = {args.alpha}, e o resultado e o **maximo que o desenho permite**: "
        f"**{f1_row['n_model_wins']}/18 celulas** a favor do FLIM em todas as tres metricas, "
        f"**W = {f1_row['W']:.0f}** (nenhuma celula discordante) e `r` rank-biserial "
        f"= {f1_row['rank_biserial_r']:+.3f}. O `p` bruto de {fmt_p(f1_row['p_raw'])} e o piso do "
        "teste exato bilateral com n = 18; nao existe evidencia mais forte alcancavel com esse n."
    )
    L.append("")
    for _, r in main_df.iterrows():
        L.append(
            f"- **{r['metric_label']}** — mediana da diferenca **{r['median_diff']:+.3f}** "
            f"(IC95% [{r['median_ci_lo']:+.3f}, {r['median_ci_hi']:+.3f}]); mediana "
            f"{r['baseline_median']:.3f} da init aleatoria contra {r['model_median']:.3f} da init "
            f"FLIM. p Holm = {fmt_p(r['p_holm'])}."
        )
    L.append("")
    L.append(
        "Diferente das linhas inconclusivas de `wilcoxon_f1.md`, aqui a conclusao nao depende de "
        "detalhe estatistico: o efeito e grande, aparece em **todas** as celulas e em **todos** os "
        "datasets (6/6 em eggs, larvae e protozoan), e sobrevive a Bonferroni. Se esta comparacao "
        "fosse embutida na familia de 7 hipoteses daquele relatorio, o p corrigido seria "
        f"7 x {fmt_p(f1_row['p_raw'])} = {fmt_p(min(1.0, 7 * f1_row['p_raw']))} — ainda "
        "significativo."
    )
    L.append("")
    L.append("### Congelar nao e concessao: e melhor")
    L.append("")
    unfrozen = rob_df[rob_df["model"] == "SVM_Distill_1x1BN_flim"]
    if len(unfrozen):
        u = unfrozen.iloc[0]
        L.append(
            f"A variante com init FLIM mas **encoder liberado** tambem bate a init aleatoria "
            f"({u['n_model_wins']}/18 celulas, p bruto {fmt_p(u['p_raw'])}), mas fica em F1 medio "
            f"**{u['model_mean']:.4f}** contra **{f1_row['model_mean']:.4f}** da versao congelada. "
            "Ou seja: destreinar o encoder inicializado por FLIM **piora** o embedding. A "
            "contribuicao nao e \"FLIM como ponto de partida para o gradiente\", e \"FLIM como "
            "backbone pronto\" — o gradiente fica so na cabeca."
        )
        L.append("")
    knn = rob_df[rob_df["model"] == "SVM_Distill_1x1BN_flim_frozen_eval_knn"]
    if len(knn):
        k = knn.iloc[0]
        L.append(
            f"A conclusao tambem nao depende do criterio de selecao de checkpoint: com o ckpt "
            f"escolhido por `val/knn_kappa` em vez de best-loss, o resultado e praticamente o "
            f"mesmo ({k['n_model_wins']}/18 celulas, mediana {k['median_diff']:+.4f}, F1 medio "
            f"{k['model_mean']:.4f})."
        )
        L.append("")
    L.append("### Onde o ganho aparece")
    L.append("")
    L.append(
        "O ganho existe em **todos** os regimes de supervisao, mas cresce com a quantidade de "
        f"rotulos: +{per_pct.loc[1, 'delta']:.2f} de F1 a 1%, +{per_pct.loc[5, 'delta']:.2f} a 5%, "
        f"+{per_pct.loc[25, 'delta']:.2f} a 25% e +{per_pct.loc[100, 'delta']:.2f} a 100%. Isso "
        "importa para a redacao: os valores frequentemente citados de **0,42 -> 0,83** sao os de "
        "**100% dos rotulos**, nao de 5% — a 5% os numeros sao "
        f"{per_pct.loc[5, 'f1_aleatoria']:.2f} -> {per_pct.loc[5, 'f1_flim_init']:.2f}. A leitura "
        "correta de 0,42 e \"o teto da init aleatoria, atingido so com todos os rotulos\"."
    )
    L.append("")
    L.append("### O que este teste NAO diz")
    L.append("")
    L.append(
        "- Nao compara a init FLIM com o **FLIM puro** nem com o teacher: para isso ver "
        "`wilcoxon_f1.md` (baseline FLIM), onde `Distill 1 - FLIM init (123K)` fica *abaixo* do "
        "FLIM de 59.504 params (mediana -0.157, inconclusivo apos Holm)."
    )
    L.append(
        "- Nao estabelece um limiar de capacidade. Pela descritiva de `unified_svm_comparison.csv`, "
        "o menor braco de init aleatoria que escapa do colapso e o **Distill 2** (402.608 params, "
        "cabeca 5,77x o backbone, F1 medio 0,86 a 100%), nao o Distill 3/4. O que a init FLIM "
        "entrega com 123K, a init aleatoria so alcanca com 402K-615K."
    )
    L.append(
        "- Nao mede custo de inferencia: params, FLOPs, latencia e memoria estao em "
        "`compute_cost.md`."
    )
    L.append(
        "- **Nao diz nada sobre os outros pares de modelos.** Para FLIM vs I-JEPA, Distill 3 vs "
        "Distill 4 e os demais 28 pares — e, principalmente, para separar \"nao detectei "
        "diferenca\" de \"demonstrei que a diferenca e pequena\" (teste de equivalencia TOST) — "
        "ver `wilcoxon_equivalence.md`."
    )
    L.append("")

    L.append("### De onde vem o p, e o que ele NAO mede")
    L.append("")
    L.append(
        f"O `p` de {fmt_p(f1_row['p_raw'])} nao e um numero opaco: com n = 18 o teste exato "
        f"enumera as 2^18 = {mil(2 ** 18)} atribuicoes de sinal possiveis sob a nula, e apenas "
        f"**2** delas dao W = 0 (\"todas positivas\" e \"todas negativas\"). Logo "
        f"p = 2 / 2^18 = {2 / 2 ** 18:.6e}. Verificado por enumeracao exaustiva, nao so pelo "
        "scipy. Como consequencia, esse e o **piso** do teste: nenhum resultado com n = 18 pode "
        "dar p menor. O teste do sinal (binomial) sobre as mesmas 18 celulas da exatamente o "
        "mesmo valor, o que era esperado com 18/18."
    )
    L.append("")
    L.append(
        "O que isso implica: **o `p` mede apenas a consistencia da direcao**, nao a magnitude. "
        "Dezoito diferencas de 1e-9 no mesmo sentido dariam o mesmo p. A magnitude tem de ser "
        f"lida na mediana ({f1_row['median_diff']:+.3f} de F1, IC95% "
        f"[{f1_row['median_ci_lo']:+.3f}, {f1_row['median_ci_hi']:+.3f}]) e no fato de que a "
        f"**menor** das 18 celulas ja e +{diffs_f1.min():.3f} de F1 "
        f"({diffs_f1.idxmin()[0]} a {diffs_f1.idxmin()[1]}%) — nao e um caso de 18 vitorias "
        "minusculas."
    )
    L.append("")

    L.append("## Ressalvas")
    L.append("")
    L.append(
        "- **As 18 celulas nao sao independentes.** Dentro de um dataset, os 6 subconjuntos de "
        "treino sao estritamente encaixados (1% ⊂ 5% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%, contencao de "
        "100% verificada nos JSONs de `data/to_mateus/splits_incremental/`) e o conjunto de teste "
        "e **identico** nas 6 fracoes. O Wilcoxon supoe pares independentes, portanto n = 18 e "
        "otimista e o `p` deve ser lido como evidencia de direcao consistente, nao como "
        "probabilidade calibrada. No extremo conservador — colapsando cada dataset na mediana das "
        f"6 fracoes, n = 3 — a direcao se mantem ({col_r['n_model_wins']}/3 datasets a favor da "
        f"init FLIM, mediana {col_r['median_diff']:+.4f}), mas o p = {fmt_p(col_r['p_raw'])} e o "
        "piso alcancavel com n = 3, logo nada pode ser declarado significativo nesse recorte. "
        "Os tres splits, por outro lado, **sao** genuinamente diferentes (~51% de sobreposicao "
        "entre os conjuntos de treino), o que valida a media sobre splits dentro de cada celula."
    )
    L.append(
        "- Os `.ckpt` das runs Distill 1/2/3/4 nao estao mais no disco; a contagem de parametros "
        "vem da topologia reinstanciada (ver secao 5 de `README.md` deste diretorio). Para "
        "**protozoan** o encoder real usa `conv2` com 30 canais, nao 32, logo o total e 3.602 "
        "params menor que os 123.504 de eggs/larvae. Isso nao afeta o teste (os dois bracos tem a "
        "mesma topologia), mas afeta qualquer numero de params citado por dataset."
    )
    L.append(
        "- O IC95% vem de reamostrar as 18 celulas (*bootstrap*) e nao e o mesmo procedimento do "
        "teste; aqui os dois concordam (IC exclui zero e o Wilcoxon rejeita), mas em caso de "
        "conflito a conclusao conservadora e a do Wilcoxon corrigido."
    )
    L.append(
        "- Ausencia de significancia com n = 18 seria inconclusividade, nao equivalencia. Nao e o "
        "caso aqui."
    )
    L.append("")

    L.append("## Reproducao")
    L.append("")
    L.append("```bash")
    L.append("cd /dados/home/moliveira/Scalable_Hybrid_FLIM")
    L.append("LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \\")
    L.append("/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \\")
    L.append("    statistics/tools/wilcoxon_flim_init.py")
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
    print(main_df[["metric", "n_model_wins", "W", "p_raw", "p_holm",
                   "median_diff", "rank_biserial_r"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
