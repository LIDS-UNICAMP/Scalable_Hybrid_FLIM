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
"""Sigmoid ativa MENOS ou ativa INCORRETAMENTE (vs ReLU)?

Duas hipóteses distintas para o mau desempenho da cabeça sigmoide:
  (1) "ativa menos"        -> dispara em menos unidades / com força menor.
  (2) "ativa incorretamente" -> dispara nas unidades ERRADAS para a classe
                                (o padrão de disparo não separa as classes).

Como os checkpoints da cabeça ReLU estão vazios, comparamos de forma controlada
usando o MESMO sinal de entrada real do modelo sigmoid treinado:
  P = avgpool(encoder(x))        (48-d, saída ReLU do encoder — real)
  z = layer1(P)                  (24-d pré-ativação — real, pesos treinados)
  S = Sigmoid(z)                 (o que o modelo realmente usa)
  R = ReLU(z)                    (contrafactual: a MESMA entrada, outra função)

Observação-chave: ReLU dispara quando z>0; Sigmoid passa de 0.5 quando z>0.
Logo o CONJUNTO de unidades que "acende" é o mesmo limiar (z>0). O que difere é
a FORÇA graduada (ReLU = z, ilimitado; Sigmoid = σ(z), esmagado em (0,1)).

Mede, por checkpoint:
  relu_active_frac   fração de ativações com z>0 (unidade acende)  [ReLU]
  relu_zero_frac     fração exatamente 0 (esparsidade da ReLU)
  sig_off/on/mid     fração de S <0.01 / >0.99 / no meio
  firing_agreement   fração em que (R>0) == (S>0.5)  -> ~1.0 se disparam nas MESMAS unidades
  mean_R, mean_S     ativação média
  probe_R, probe_S   (repres.) acurácia de linear-probe sobre R e sobre S no MESMO z
                     -> se probe_S << probe_R, a sigmoid "ativa incorretamente"

Saídas:
  results/firing_relu_vs_sigmoid.csv
  results/plots_units/F_firing_dist.png     histogramas R vs S (força de disparo)
  results/plots_units/F_rate_bars.png       taxa de disparo e esparsidade por dataset x modo
  results/plots_units/F_discrimination.png  probe R vs S no mesmo z (dispara certo?)
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

_ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
sys.path.insert(0, _ROOT)
from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited
from src.modules.classification_flim_module import ClassificationFlimModule, _dataset_short_to_parasite_name
import torch

_ART = os.path.join(_ROOT, "artifacts", "classification_flim")
OUT = os.path.join(_ROOT, "results", "plots_units"); os.makedirs(OUT, exist_ok=True)
DATASETS = ["eggs", "larvae", "protozoan"]
COL_R, COL_S = "#2ca02c", "#d62728"   # verde=ReLU, vermelho=Sigmoid


@torch.no_grad()
def extract(run_dir, device):
    meta = json.load(open(os.path.join(run_dir, "run_metadata.json")))
    ck = os.path.join(run_dir, "checkpoints", "best_kappa.ckpt")
    mod = ClassificationFlimModule.load_from_checkpoint(ck, map_location=device).eval().to(device)
    enc, head = mod.model.encoder, mod.model.head
    dm = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(meta["dataset"]), split=int(meta["split"]),
        percentage=int(meta["percentage"]), image_size=200, V_train=1, V_eval=1, batch_size=32,
        num_workers=4, pin_memory=True, persistent_workers=False, loader="ift_lab",
        imagenet_norm=not bool(meta.get("no_imagenet_norm", False)))
    dm.setup("test")
    Zs, ys = [], []
    for views, y in dm.test_dataloader():
        x = (views[:, 0] if (not isinstance(views, (list, tuple)) and views.ndim == 5) else
             (views[0] if isinstance(views, (list, tuple)) else views)).to(device)
        z = head.layer1(head.pool(enc(x)).flatten(1))      # pré-ativação (mesma p/ ambas)
        Zs.append(z.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(Zs), np.concatenate(ys).astype(int), meta


def probe(V, y):
    try:
        k = min(5, int(np.bincount(y).min()))
        if k < 2: return float("nan")
        Vs = StandardScaler().fit_transform(V)
        return float(cross_val_score(LogisticRegression(max_iter=2000), Vs, y, cv=k).mean())
    except Exception:
        return float("nan")


def stats(z, y, with_probe):
    R = np.maximum(0.0, z); S = 1.0 / (1.0 + np.exp(-z))
    d = dict(
        relu_active_frac=float((z > 0).mean()),
        relu_zero_frac=float((z <= 0).mean()),
        sig_off_frac=float((S < 0.01).mean()),
        sig_on_frac=float((S > 0.99).mean()),
        sig_mid_frac=float(((S >= 0.01) & (S <= 0.99)).mean()),
        firing_agreement=float(((R > 0) == (S > 0.5)).mean()),
        mean_R=float(R.mean()), mean_S=float(S.mean()),
    )
    if with_probe:
        d["probe_R"] = probe(R, y); d["probe_S"] = probe(S, y)
    return d, R, S


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted(d for d in glob.glob(os.path.join(_ART, "sigmoid2l_classhead_*")) if os.path.isdir(d))
    rep = {(ds, mode): os.path.join(_ART, f"sigmoid2l_classhead_{ds}_split1_pct75"
                                    f"{'_sigmoid2l_frozen' if mode=='frozen' else '_sigmoid2l'}")
           for ds in DATASETS for mode in ["frozen", "unfrozen"]}
    rep_paths = set(rep.values())

    rows, rep_RS = [], {}
    for rd in runs:
        try:
            z, y, meta = extract(rd, dev)
        except Exception as e:
            print(f"[ERR] {os.path.basename(rd)}: {e}"); continue
        is_rep = rd in rep_paths
        d, R, S = stats(z, y, with_probe=is_rep)
        d.update(dataset=meta["dataset"], split=int(meta["split"]), pct=int(meta["percentage"]),
                 mode="frozen" if bool(meta.get("freeze_encoder", False)) else "unfrozen",
                 run=os.path.basename(rd))
        rows.append(d)
        if is_rep:
            rep_RS[(meta["dataset"], d["mode"])] = (R, S)
        print(f"  {os.path.basename(rd):52s} active={d['relu_active_frac']:.2f} "
              f"sig_mid={d['sig_mid_frac']:.2f} agree={d['firing_agreement']:.3f}"
              + (f" probeR={d['probe_R']:.2f} probeS={d['probe_S']:.2f}" if is_rep else ""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_ROOT, "results", "firing_relu_vs_sigmoid.csv"), index=False)

    # --- F_firing_dist: histogramas R vs S (força de disparo), repres. unfrozen ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, ds in zip(axes, DATASETS):
        R, S = rep_RS[(ds, "unfrozen")]
        ax.hist(R.ravel(), bins=60, alpha=0.6, color=COL_R, density=True, label="ReLU(z) = max(0,z)")
        ax.hist(S.ravel(), bins=60, range=(0, S.max()), alpha=0.6, color=COL_S, density=True, label="Sigmoid(z)")
        ax.axvline(0, color="k", ls=":", lw=1); ax.set_yscale("log")
        ax.set_title(f"{ds} (unfrozen)"); ax.set_xlabel("valor da ativação"); ax.legend(fontsize=8)
    fig.suptitle("F. Força de disparo — ReLU (cauda longa) vs Sigmoid (teto em 1)  [mesmo z]", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/F_firing_dist.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- F_rate_bars: taxa de disparo (z>0) e "ligado forte" ---
    g = df.groupby(["dataset", "mode"]).agg(active=("relu_active_frac", "mean"),
                                            sig_on=("sig_on_frac", "mean"),
                                            sig_off=("sig_off_frac", "mean"),
                                            agree=("firing_agreement", "mean")).reset_index()
    labels = [f"{r.dataset}\n{r['mode']}" for _, r in g.iterrows()]
    x = np.arange(len(g)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w/2, g.active, w, color=COL_R, label="ReLU: fração que dispara (z>0)")
    ax.bar(x + w/2, g.sig_on, w, color=COL_S, label="Sigmoid: fração 'ligado forte' (>0.99)")
    ax.plot(x, g.agree, "ks--", label="concordância R>0 vs S>0.5 (mesmas unidades?)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fração das ativações"); ax.legend(fontsize=8)
    ax.set_title("F. Taxa de disparo: 'ativam menos'? (e disparam nas MESMAS unidades?)")
    fig.tight_layout(); fig.savefig(f"{OUT}/F_rate_bars.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- F_discrimination: probe R vs S no mesmo z (dispara CERTO?) ---
    pr = df.dropna(subset=["probe_R", "probe_S"]) if "probe_R" in df else pd.DataFrame()
    if len(pr):
        pr = pr.groupby(["dataset", "mode"]).agg(probe_R=("probe_R", "mean"), probe_S=("probe_S", "mean")).reset_index()
        labels = [f"{r.dataset}\n{r['mode']}" for _, r in pr.iterrows()]
        x = np.arange(len(pr)); w = 0.4
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - w/2, pr.probe_R, w, color=COL_R, label="probe sobre ReLU(z)")
        ax.bar(x + w/2, pr.probe_S, w, color=COL_S, label="probe sobre Sigmoid(z)")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_ylabel("acurácia linear-probe (mesmo z)"); ax.legend(fontsize=8)
        ax.set_title("F. Disparam nas unidades CERTAS? separabilidade de ReLU(z) vs Sigmoid(z)")
        fig.tight_layout(); fig.savefig(f"{OUT}/F_discrimination.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    print("\n=== RESUMO ===")
    print(g.to_string(index=False))
    if len(pr):
        print("\nProbe (mesmo z, repres.):")
        print(pr.to_string(index=False))
    print("\nCSV -> results/firing_relu_vs_sigmoid.csv ; plots -> results/plots_units/F_*.png")


if __name__ == "__main__":
    main()
