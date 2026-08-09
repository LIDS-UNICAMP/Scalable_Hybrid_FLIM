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
"""plot_comparacao_flim_protocolo.py — curva de kappa: registrado x reproducoes.

SCRIPT VERSIONADO (nao e descartavel). Re-executavel.

Small multiples, um painel por dataset. Ate tres series por painel:

  1. FLIM registrado        CSV externo (data/reports_felipe/svm/), via
                            artifacts/normalized/unified_svm_comparison.csv
  2. conv3 achatado         27.648-d + LAB[0,1] + solver convergido
                            (o protocolo original reproduzido)
  3. GAP 48-d               48-d + LAB[0,1] + solver convergido
                            (so as correcoes de entrada e de solver)

A serie 3 e opcional: se `eval_48d_norm_off.csv` nao existir, ela some e o
grafico sai com as duas primeiras, identicas as de antes.

Le e escreve apenas dentro de artifacts/plots/comparacao_flim_protocolo_original/.
Nao toca em src/, scripts/, configs/ nem em CSV registrado.

Uso:
    conda run -n scalable_FLIM python tools/plot_comparacao_flim_protocolo.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import pandas as pd                                                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIR = os.path.join(_ROOT, "artifacts", "plots",
                    "comparacao_flim_protocolo_original")
_CSV_MAIN = os.path.join(_DIR, "comparacao_unified_svm_com_reproducao.csv")
_CSV_48D = os.path.join(_DIR, "eval_48d_norm_off.csv")
_OUT = os.path.join(_DIR, "comparacao_kappa_registrado_vs_reproducao.png")

# Paleta de referencia (skill dataviz), modo claro. Slots 1, 2 e 3.
SURFACE, GRID = "#fcfcfb", "#e4e3dd"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

PCTS = [1, 5, 25, 50, 75, 100]
DATASETS = [("protozoan", "protozoan-cysts (7 classes)"),
            ("eggs", "helminth-eggs (9 classes)"),
            ("larvae", "helminth-larvae (2 classes)")]


def _series48(path: str):
    """Agrega o CSV por celula em media/desvio por (dataset, percentage)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return (df.groupby(["dataset", "percentage"])["kappa"]
              .agg(kappa="mean", kappa_std=lambda s: s.std(ddof=1))
              .reset_index())


def main() -> None:
    if not os.path.exists(_CSV_MAIN):
        sys.exit(f"[ERRO] ausente: {_CSV_MAIN}\n"
                 f"       rode antes: build_comparacao_csv.py")
    df = pd.read_csv(_CSV_MAIN)
    ref = df[df["method"] == "SVM_FLIM"]
    rep = df[df["method"] == "SVM_FLIM_repro"]
    d48 = _series48(_CSV_48D)
    print(f"[INFO] serie 48-d: {'presente' if d48 is not None else 'ausente (grafico com 2 curvas)'}")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = list(range(len(PCTS)))

    for ax, (ds, title) in zip(axes, DATASETS):
        ax.set_facecolor(SURFACE)
        # pct1: celula degenerada da referencia, nao comparavel.
        ax.axvspan(-0.42, 0.42, color=INK_MUTED, alpha=0.09, zorder=0, linewidth=0)

        r = ref[ref["dataset_short"] == ds].set_index("pretrained_pct").reindex(PCTS)
        p = rep[rep["dataset_short"] == ds].set_index("pretrained_pct").reindex(PCTS)

        ax.errorbar(x, r["kappa"], yerr=r["kappa_std"], color=S1, linewidth=2,
                    marker="o", markersize=8, markerfacecolor=S1,
                    markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1.4,
                    capsize=3, ecolor=S1, alpha=0.95, zorder=3,
                    label="FLIM registrado (CSV externo)")

        ax.errorbar(x, p["kappa"], yerr=p["kappa_std"], color=S2, linewidth=2,
                    linestyle=(0, (5, 2.5)), marker="s", markersize=8,
                    markerfacecolor=S2, markeredgecolor=SURFACE, markeredgewidth=2,
                    elinewidth=1.4, capsize=3, ecolor=S2, alpha=0.95, zorder=4,
                    label="conv3 achatado 27.648-d, LAB cru, max_iter=-1")

        if d48 is not None:
            q = (d48[d48["dataset"] == ds].set_index("percentage").reindex(PCTS))
            ax.errorbar(x, q["kappa"], yerr=q["kappa_std"], color=S3, linewidth=2,
                        linestyle=(0, (1.6, 2.2)), marker="^", markersize=8,
                        markerfacecolor=S3, markeredgecolor=SURFACE,
                        markeredgewidth=2, elinewidth=1.4, capsize=3, ecolor=S3,
                        alpha=0.95, zorder=5,
                        label="GAP 48-d, LAB cru, max_iter=-1")

        ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}%" for v in PCTS], fontsize=11, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(PCTS) - 0.5)
        ax.set_ylim(-0.08, 1.0)
        ax.set_xlabel("rótulos usados no treino do SVM", fontsize=11,
                      color=INK_SECONDARY)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, length=0)

        if d48 is not None and not pd.isna(q.loc[75, "kappa"]):
            gap = p.loc[75, "kappa"] - q.loc[75, "kappa"]
            ax.annotate(f"@75%  achatado − 48-d = {gap:+.4f}", xy=(4, 0),
                        xytext=(0.30, 0.045), textcoords="axes fraction",
                        fontsize=10, color=INK_SECONDARY)

    axes[0].set_ylabel("Cohen's κ  (teste)", fontsize=12, color=INK_PRIMARY)
    axes[0].tick_params(axis="y", labelsize=11)
    axes[0].annotate("1%: célula\ndegenerada\nna referência", xy=(0, 0.02),
                     xytext=(0.62, 0.14), fontsize=9.5, color=INK_MUTED,
                     ha="left", va="bottom", linespacing=1.35)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0.008), ncol=3, frameon=False,
                     fontsize=11.5, handlelength=2.6, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.suptitle("Encoder FLIM congelado + SVM linear — o que cada correção recupera",
                 fontsize=15, color=INK_PRIMARY, y=0.985)
    fig.text(0.5, 0.925,
             "mesmos pesos · fit no train, score no test · 3 splits (barra = desvio) · "
             "48-d isola o efeito da dimensão",
             ha="center", fontsize=11, color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.115, 1, 0.895])
    fig.savefig(_OUT, dpi=170, facecolor=SURFACE)
    print(f"[OK] {_OUT}")


if __name__ == "__main__":
    main()
