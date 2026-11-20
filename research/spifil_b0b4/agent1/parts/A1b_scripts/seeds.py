"""A1b — curvas das sementes SPiFiL-pequeno: L0 (só SPiFiL) × L6 (SPiFiL + LeJEPA/SIGReg). Só treino/val."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
G = "/dados/home/moliveira/exps/spifil_growth/ssl_pequeno_lejepa"
rows, curves = [], []
for d in sorted(glob.glob(f"{G}/*")):
    if not os.path.isdir(d) or d.endswith("__pycache__"):
        continue
    name = os.path.basename(d)
    l0 = f"{d}/L0/epoch_metrics.csv"
    k0 = pd.read_csv(l0).val_kappa.iloc[-1] if os.path.exists(l0) else np.nan
    for L in ("L6", "L6_kappa"):
        f = f"{d}/{L}/epoch_metrics.csv"
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f)
        if e.empty or "val_kappa" not in e:
            continue
        e = e.dropna(subset=["val_kappa"])
        rho = spearmanr(e.train_loss, e.val_kappa)[0] if len(e) > 3 else np.nan
        rows.append(dict(run=name, stage=L, source=f, L0_val_kappa=k0, n_epochs=len(e),
                         ep0_val_kappa=e.val_kappa.iloc[0], max_val_kappa=e.val_kappa.max(),
                         argmax_epoch=int(e.epoch.iloc[e.val_kappa.values.argmax()]),
                         last_val_kappa=e.val_kappa.iloc[-1], gain_max_vs_L0=e.val_kappa.max() - k0,
                         gain_last_vs_L0=e.val_kappa.iloc[-1] - k0,
                         loss_first=e.train_loss.iloc[0], loss_last=e.train_loss.iloc[-1],
                         rho_trainloss_valkappa=rho,
                         val_kappa_std_last10=e.val_kappa.iloc[-10:].std()))
        e = e.assign(run=name, stage=L, L0=k0)
        curves.append(e[["run", "stage", "epoch", "train_loss", "val_loss", "val_kappa", "train_sigreg", "train_invariance", "L0"]])
R = pd.DataFrame(rows)
R.to_csv("A1b_seed_curves_summary.csv", index=False)
pd.concat(curves).to_csv("A1b_seed_curves_long.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(R.drop(columns=["source"]).round(3).to_string())
print(R[["gain_max_vs_L0", "gain_last_vs_L0", "rho_trainloss_valkappa", "argmax_epoch"]].describe().round(3))
