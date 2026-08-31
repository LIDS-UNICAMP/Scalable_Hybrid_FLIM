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
"""Quem acende? Análise POR NEURÔNIO da camada oculta sigmoide (24 unidades).

O plot E mostrou que os valores de S se acumulam em 0 e 1 (saturação). Mas isso
não diz QUAIS unidades acendem nem SE o padrão liga/desliga codifica a classe.
Esta ferramenta responde isso, a partir dos pesos já congelados.

Para um checkpoint sigmoid2l representativo (split1, pct75) — frozen e unfrozen —
roda o test set e coleta S = Sigmoid(layer1(P))  ->  [N_amostras x 24].
Calcula, por UNIDADE j (j=0..23):
  - mean_act        : ativação média sobre todas as amostras
  - frac_on         : fração de amostras em que a unidade fica ~1 (>0.99)
  - frac_off        : fração ~0 (<0.01)
  - frac_mid        : fração responsiva (entre 0.01 e 0.99)
  - role            : STUCK_ON / STUCK_OFF / SWITCHING / RESPONSIVE
  - F_anova, p      : ANOVA da ativação da unidade entre as classes (discriminância)
E a matriz M[24 x C] = ativação média da unidade j na classe c (o "código" por classe).

Saídas:
  results/unit_activation_stats.csv                    (uma linha por dataset x modo x unidade)
  results/plots_units/U_heatmap_{dataset}.png          heatmap unidades x classes (frozen | unfrozen)
  results/plots_units/U_unit_roles.png                 quantas unidades presas-on / presas-off / úteis
  results/plots_units/U_selectivity_{dataset}.png      F-ANOVA por unidade (quem discrimina classe)
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

_ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
sys.path.insert(0, _ROOT)
from core.data import ParasiteDataModule
from methods.classification import ClassificationFlimModule
from core.constants import PARASITE_NAME
import torch

_ART = os.path.join(_ROOT, "artifacts", "classification_flim")
OUT = os.path.join(_ROOT, "results", "plots_units"); os.makedirs(OUT, exist_ok=True)
DATASETS = ["eggs", "larvae", "protozoan"]
ON, OFF = 0.99, 0.01


@torch.no_grad()
def extract_S(run_dir, device):
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
    Ss, ys = [], []
    for views, y in dm.test_dataloader():
        x = (views[:, 0] if (not isinstance(views, (list, tuple)) and views.ndim == 5) else
             (views[0] if isinstance(views, (list, tuple)) else views)).to(device)
        S = head.sigmoid(head.layer1(head.pool(enc(x)).flatten(1)))
        Ss.append(S.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(Ss), np.concatenate(ys).astype(int)


def role(frac_on, frac_off, frac_mid):
    if frac_on > 0.95:  return "STUCK_ON"
    if frac_off > 0.95: return "STUCK_OFF"
    if frac_mid < 0.20: return "SWITCHING"   # quase sempre saturada, mas alterna 0/1
    return "RESPONSIVE"


def rep_run(ds, mode):
    suf = "_sigmoid2l_frozen" if mode == "frozen" else "_sigmoid2l"
    p = os.path.join(_ART, f"sigmoid2l_classhead_{ds}_split1_pct75{suf}")
    return p if os.path.isdir(p) else None


def per_unit(S, y):
    U = S.shape[1]
    F, p = f_classif(S, y)                       # ANOVA por unidade (discriminância entre classes)
    rows = []
    for j in range(U):
        col = S[:, j]
        f_on = float((col > ON).mean()); f_off = float((col < OFF).mean())
        f_mid = 1.0 - f_on - f_off
        rows.append(dict(unit=j, mean_act=float(col.mean()), frac_on=f_on, frac_off=f_off,
                         frac_mid=float(f_mid), role=role(f_on, f_off, f_mid),
                         F_anova=float(F[j]) if np.isfinite(F[j]) else 0.0,
                         p_anova=float(p[j]) if np.isfinite(p[j]) else 1.0))
    classes = np.unique(y)
    M = np.stack([S[y == c].mean(0) for c in classes], axis=1)   # [U x C]
    return pd.DataFrame(rows), M, classes


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for ds in DATASETS:
        Ms = {}
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        for ax, mode in zip(axes, ["frozen", "unfrozen"]):
            rd = rep_run(ds, mode)
            if rd is None:
                ax.set_visible(False); continue
            S, y = extract_S(rd, dev)
            dfu, M, classes = per_unit(S, y)
            dfu.insert(0, "mode", mode); dfu.insert(0, "dataset", ds)
            all_rows.append(dfu); Ms[mode] = (M, classes)
            # ordena unidades pela classe de maior ativação (revela blocos)
            order = np.argsort(M.argmax(1) * 1000 + M.max(1))
            im = ax.imshow(M[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{ds} — {mode}\n(unidades ordenadas por classe preferida)")
            ax.set_xlabel("classe"); ax.set_ylabel("unidade oculta (0..23)")
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
            n_sw = int((dfu.role.isin(["STUCK_ON", "STUCK_OFF"])).sum())
            ax.text(0.5, -0.14, f"{n_sw}/24 presas (on/off); "
                    f"{int((dfu.p_anova<0.05).sum())}/24 discriminam classe (p<0.05)",
                    transform=ax.transAxes, ha="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, label="ativação média")
        fig.suptitle(f"U. Quem acende — ativação média por unidade x classe ({ds})", y=1.02)
        fig.tight_layout(); fig.savefig(f"{OUT}/U_heatmap_{ds}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(os.path.join(_ROOT, "results", "unit_activation_stats.csv"), index=False)

    # U_unit_roles: contagem de papéis por dataset x modo
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv = df.groupby(["dataset", "mode", "role"]).size().unstack("role").fillna(0)
    piv = piv.reindex(columns=["STUCK_OFF", "STUCK_ON", "SWITCHING", "RESPONSIVE"], fill_value=0)
    labels = [f"{d}\n{m}" for d, m in piv.index]
    bottom = np.zeros(len(piv)); colors = {"STUCK_OFF": "#334", "STUCK_ON": "#e8b", "SWITCHING": "#4a8", "RESPONSIVE": "#fc6"}
    for role_name in piv.columns:
        ax.bar(labels, piv[role_name], bottom=bottom, label=role_name, color=colors[role_name])
        bottom += piv[role_name].values
    ax.set_ylabel("nº de unidades (de 24)"); ax.legend(ncol=4, fontsize=8)
    ax.set_title("U. Papel das 24 unidades ocultas por dataset x modo")
    fig.tight_layout(); fig.savefig(f"{OUT}/U_unit_roles.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # U_selectivity: F-ANOVA por unidade (eggs/protozoan mais informativos)
    for ds in ["eggs", "protozoan"]:
        sub = df[(df.dataset == ds)]
        fig, ax = plt.subplots(figsize=(9, 4))
        for mode, cc in [("frozen", "#3355aa"), ("unfrozen", "#cc7722")]:
            s = sub[sub["mode"] == mode].sort_values("unit")
            if len(s): ax.plot(s.unit, np.log10(s.F_anova + 1e-6), "o-", color=cc, label=mode, ms=4)
        ax.set_xlabel("unidade oculta"); ax.set_ylabel("log10 F-ANOVA (discriminância)")
        ax.set_title(f"U. Discriminância por unidade — {ds}"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{OUT}/U_selectivity_{ds}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # resumo em texto
    print("\n=== RESUMO: papel das 24 unidades (rep: split1 pct75) ===")
    for (ds, mode), g in df.groupby(["dataset", "mode"]):
        vc = g.role.value_counts().to_dict()
        disc = int((g.p_anova < 0.05).sum())
        print(f"{ds:9s} {mode:8s} | presas_off={vc.get('STUCK_OFF',0):2d} presas_on={vc.get('STUCK_ON',0):2d} "
              f"switching={vc.get('SWITCHING',0):2d} responsive={vc.get('RESPONSIVE',0):2d} "
              f"| discriminam_classe(p<.05)={disc}/24  mean_act_medio={g.mean_act.mean():.2f}")
    print("\nCSV -> results/unit_activation_stats.csv ; plots -> results/plots_units/")


if __name__ == "__main__":
    main()
