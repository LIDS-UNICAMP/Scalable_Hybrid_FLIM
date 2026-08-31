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
"""Gera os 6 PNGs que ilustram o achatamento ReLU -> Sigmoid da cabeça sigmoid2l.

Três gráficos vêm do forward direto de checkpoints representativos (split1 pct75
unfrozen por dataset); três vêm dos CSVs agregados. Descrição de cada plot:

  A_hist_norms.png          Histograma das normas ||P|| vs ||S|| (renormalizadas
                            pela mediana): a largura da distribuição = quanto da
                            magnitude de P sobrevive na Sigmoid.
  C_scatter_normP_normS.png Scatter ||P|| vs ||S|| por amostra: nuvem plana /
                            horizontal (Spearman ~ 0) => o "quão longe" não sobrevive.
  E_hist_sigmoid_vals.png   Histograma dos valores das unidades S: acúmulo em 0 e 1
                            evidencia a saturação da sigmoide.
  B_saturation_bar.png      Fração saturada por dataset x modo (frozen/unfrozen),
                            média +- desvio sobre os splits.
  CV_dumbbell.png           CV das normas P->S por checkpoint (uma linha cada):
                            visualiza o colapso do spread de magnitude.
  D_trainability_gap.png    Acurácia real do modelo treinado vs linear-probe sobre S:
                            pontos abaixo da diagonal = a info existe em S mas a
                            layer2 não a lê (gap de treinabilidade).

Backend matplotlib Agg (sem display). Uso:
    python -m analysis.plots.plot_sigmoid_saturation

Sem CLI: os antigos --csv, --test-csv e --out-dir sao parametros nomeados de ``main()``,
com os mesmos defaults. Para outra entrada, chame a funcao:
    main(csv="results/relu_vs_sigmoid_flatten.csv",
         test_csv="results/sigmoid2l_test_results.csv",
         out_dir="results/plots_flatten")
"""
from __future__ import annotations
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
sys.path.insert(0, _ROOT)
from core.data import ParasiteDataModule
from methods.classification import ClassificationFlimModule
from core.constants import PARASITE_NAME
import torch

_ART = os.path.join(_ROOT, "artifacts", "classification_flim")
DATASETS = ["eggs", "larvae", "protozoan"]
COL_P, COL_S = "#1f77b4", "#d62728"   # azul = P (após ReLU), vermelho = S (após Sigmoid)


@torch.no_grad()
def extract(run_dir, device):
    """Forward do test set de um checkpoint -> (P [N,48], S [N,24]) em numpy."""
    meta = json.load(open(os.path.join(run_dir, "run_metadata.json")))
    ck = os.path.join(run_dir, "checkpoints", "best_kappa.ckpt")
    mod = ClassificationFlimModule.load_from_checkpoint(ck, map_location=device).eval().to(device)
    enc, head = mod.model.encoder, mod.model.head
    dm = ParasiteDataModule(
        parasite_name=PARASITE_NAME[meta["dataset"]], split=int(meta["split"]),
        percentage=int(meta["percentage"]), image_size=200, V_train=1, V_eval=1, batch_size=32,
        num_workers=4, pin_memory=True, persistent_workers=False, loader="ift_lab",
        imagenet_norm=not bool(meta.get("no_imagenet_norm", False)))
    dm.setup("test")
    Ps, Ss = [], []
    for views, y in dm.test_dataloader():
        # Trata views 5D [B,V,C,H,W] (pega view 0), lista/tupla, ou tensor 4D simples.
        x = (views[:, 0] if (not isinstance(views, (list, tuple)) and views.ndim == 5) else
             (views[0] if isinstance(views, (list, tuple)) else views)).to(device)
        feat = enc(x)
        P = head.pool(feat).flatten(1)
        S = head.sigmoid(head.layer1(P))
        Ps.append(P.cpu().numpy())
        Ss.append(S.cpu().numpy())
    return np.concatenate(Ps), np.concatenate(Ss)


def rep_run(ds):
    """Checkpoint representativo por dataset: split1 pct75 unfrozen (ou None)."""
    p = os.path.join(_ART, f"sigmoid2l_classhead_{ds}_split1_pct75_sigmoid2l")
    return p if os.path.isdir(p) else None


def main(csv: str = os.path.join(_ROOT, "results", "relu_vs_sigmoid_flatten.csv"),
         test_csv: str = os.path.join(_ROOT, "results", "sigmoid2l_test_results.csv"),
         out_dir: str = os.path.join(_ROOT, "results", "plots_flatten")):
    """Gera os 6 PNGs do achatamento ReLU->Sigmoid em results/plots_flatten/.

    Um parametro por flag do argparse antigo, com o mesmo default: `csv` e o CSV de
    metricas agregadas (saida do analyze_sigmoid_saturation.py), `test_csv` traz o
    test_accuracy do modelo treinado (plot D) e `out_dir` recebe os PNGs.
    """
    # O corpo ja le tudo por `args.x`: o shim nasce so dos parametros, entao esta e a
    # primeira linha viva e locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())
    OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_ROOT, args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {ds: extract(rep_run(ds), dev) for ds in DATASETS}

    # ---- A: histograma das normas (normalizadas pela mediana -> compara SPREAD) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, ds in zip(axes, DATASETS):
        P, S = data[ds]
        nP = np.linalg.norm(P, axis=1)
        nS = np.linalg.norm(S, axis=1)
        nP = nP / np.median(nP)
        nS = nS / np.median(nS)
        ax.hist(nP, bins=40, alpha=0.6, color=COL_P, density=True,
                label=f"||P|| apos ReLU (CV={nP.std()/nP.mean():.2f})")
        ax.hist(nS, bins=40, alpha=0.6, color=COL_S, density=True,
                label=f"||S|| apos Sigmoid (CV={nS.std()/nS.mean():.2f})")
        ax.axvline(1, color="k", ls=":", lw=1)
        ax.set_title(f"{ds} (split1 pct75 unfrozen)")
        ax.set_xlabel("norma / mediana")
        ax.legend(fontsize=8)
    fig.suptitle("A. Distribuicao das normas dos vetores (largura = magnitude preservada)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/A_hist_norms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- C: scatter ||P|| vs ||S|| ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, DATASETS):
        P, S = data[ds]
        nP = np.linalg.norm(P, axis=1)
        nS = np.linalg.norm(S, axis=1)
        rho = spearmanr(nP, nS).correlation
        print(f"[C] {ds}: Spearman(||P||,||S||) = {rho:.3f}  (n={len(nP)})")
        ax.scatter(nP, nS, s=10, alpha=0.5, color="#555")
        ax.set_title(f"{ds}: Spearman rho={rho:.2f}")
        ax.set_xlabel("||P|| (apos ReLU)")
        ax.set_ylabel("||S|| (apos Sigmoid)")
    fig.suptitle("C. ||P|| vs ||S|| por amostra — nuvem plana/horizontal = 'quao longe' nao sobrevive", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/C_scatter_normP_normS.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- E: histograma dos valores de S (saturacao) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, ds in zip(axes, DATASETS):
        _, S = data[ds]
        v = S.ravel()
        sat = ((v < 0.01) | (v > 0.99)).mean()
        print(f"[E] {ds}: fracao saturada (<0.01 ou >0.99) = {sat:.3f}")
        ax.hist(v, bins=50, range=(0, 1), color=COL_S, alpha=0.8)
        ax.set_title(f"{ds}: {sat*100:.0f}% saturado (<0.01 ou >0.99)")
        ax.set_xlabel("valor da unidade sigmoide")
    fig.suptitle("E. Valores das unidades sigmoides — acumulo em 0 e 1 = saturacao", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/E_hist_sigmoid_vals.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- agregados a partir dos CSVs ----
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_ROOT, args.csv)
    df = pd.read_csv(csv_path)

    # B: barra fracao saturada por dataset x modo
    fig, ax = plt.subplots(figsize=(7, 4.5))
    piv = df.groupby(["dataset", "mode"])["sigmoid_sat_frac"].agg(["mean", "std"]).unstack("mode")
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w/2, piv[("mean", "frozen")], w, yerr=piv[("std", "frozen")].fillna(0),
           label="frozen", color="#8888cc", capsize=3)
    ax.bar(x + w/2, piv[("mean", "unfrozen")], w, yerr=piv[("std", "unfrozen")].fillna(0),
           label="unfrozen", color="#cc8844", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("fracao de unidades saturadas")
    ax.axhspan(0.6, 0.7, color="red", alpha=0.08)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("B. Saturacao da sigmoide (media +- dp sobre 3 splits)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/B_saturation_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # CV dumbbell: P->S por checkpoint
    fig, ax = plt.subplots(figsize=(7, 8))
    d = df.sort_values(["dataset", "mode", "split", "pct"]).reset_index(drop=True)
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.cv_P, r.cv_S], [yi, yi], color="#bbb", lw=1, zorder=1)
    ax.scatter(d.cv_P, y, color=COL_P, s=22, label="CV apos ReLU (P)", zorder=2)
    ax.scatter(d.cv_S, y, color=COL_S, s=22, label="CV apos Sigmoid (S)", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.dataset[:4]} s{r.split} p{r.pct} {r['mode'][:2]}"
                        for _, r in d.iterrows()], fontsize=6)
    ax.set_xlabel("CV das normas (spread de magnitude)")
    ax.legend()
    ax.set_title("CV. Colapso da magnitude P->S (cada linha = 1 checkpoint)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/CV_dumbbell.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # D: trainability gap — acc real do modelo vs probe sobre S
    test_path = args.test_csv if os.path.isabs(args.test_csv) else os.path.join(_ROOT, args.test_csv)
    test = pd.read_csv(test_path)
    test = test.rename(columns={"percentage": "pct", "encoder_mode": "mode"})
    m = df.merge(test[["dataset", "split", "pct", "mode", "test_accuracy"]],
                 on=["dataset", "split", "pct", "mode"], how="inner")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for mode, mk, cc in [("frozen", "o", "#3355aa"), ("unfrozen", "s", "#cc7722")]:
        sub = m[m["mode"] == mode]
        ax.scatter(sub.probe_acc_S, sub.test_accuracy, marker=mk, s=55, color=cc, alpha=0.8, label=mode)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (modelo le tudo)")
    ax.set_xlabel("linear-probe sobre S  (informacao PRESENTE em S)")
    ax.set_ylabel("acc real do modelo treinado  (o que a layer2 LE)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("D. Gap treinabilidade vs informacao\n(pontos abaixo da diagonal: info existe mas o modelo nao le)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/D_trainability_gap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n[B] fracao saturada por dataset x modo (media sobre splits):")
    print(df.groupby(["dataset", "mode"])["sigmoid_sat_frac"].mean().to_string())
    print("\n[D] gap de treinabilidade (probe_acc_S vs test_accuracy):")
    print(m.sort_values(["mode", "dataset"])[
        ["dataset", "split", "pct", "mode", "probe_acc_S", "test_accuracy"]].to_string(index=False))

    print("\nplots salvos em", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
