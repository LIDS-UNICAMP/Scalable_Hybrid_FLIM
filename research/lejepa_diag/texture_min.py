"""Etapa 4 (textura) do diagnostico LeJEPA: driver sobre a biblioteca comum (textura, readouts,
svm_eval), estendendo ``run.stage_texture`` com o readout COV oficial e estatisticas de 2a ordem.

Por par (P0 = ck['anchor'], P1 = ck['ssl_state'] do MO-DEHB encoder.pt) e por readout
(GAP = embedding oficial head(encoder(x)); COV = CovReadout oficial k=64 ajustado so no treino):
  * SVM original (core.svm.fit_svm, C=100, SEM scaler) ajustada no treino LIMPO;
  * validacao perturbada (blur sigma, ruido de mesma energia, down-up) -> kappa / F1 / bacc;
  * cosseno linha a linha embedding(limpo) x embedding(perturbado);
  * ultima camada: distancia de Frobenius normalizada ||C_p - C_c||_F / ||C_c||_F entre as
    covariancias de canais (cheia, centrada nos pixels) limpa e perturbada, por imagem; idem para a
    covariancia projetada no PCA-k do CovReadout; e a mudanca relativa da media espacial (1a ordem).
As perturbacoes sao geradas UMA vez por (tipo, sigma) e reusadas por P0/P1 e GAP/COV.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 nice -n 19 python -m research.lejepa_diag.texture_min \
        --pair "remote_sensing:modehb_hybrid:c00004/f45"
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.lejepa_diag import readouts as R                     # noqa: E402
from research.lejepa_diag import textura as T                      # noqa: E402
from research.lejepa_diag.extract import fp32_strict               # noqa: E402
from research.lejepa_diag.run import load_split_data                # noqa: E402
from research.lejepa_diag.svm_eval import evaluate_svm             # noqa: E402

MANIFEST = REPO / "reports/lejepa_diagnostics/data/checkpoint_manifest.json"
OUT = REPO / "reports/lejepa_diagnostics/data/texture_metrics.csv"


def load_pair_models(pair):
    ck = torch.load(pair["P0"]["path"], map_location="cpu", weights_only=False)
    out = {}
    for name in ("P0", "P1"):
        key = pair[name]["key"]
        enc, head = copy.deepcopy(ck["encoder"]), copy.deepcopy(ck["head"])
        enc.load_state_dict(ck[key]["encoder"])
        head.load_state_dict(ck[key]["head"])
        out[name] = (enc.float().eval(), head.float().eval())
    return out


@torch.no_grad()
def features(cr, enc, head, X, bs=8, keep_last_cov=False):
    """Um forward por lote -> GAP oficial (head), COV oficial, media espacial e covariancias da
    ultima camada (cheia e projetada no PCA-k do CovReadout)."""
    gap, cov, mean_last, full_c, proj_c = [], [], [], [], []
    with fp32_strict():
        for b in X.split(bs):
            outs = cr._blocks(enc, b)
            covs = cr._covs(outs)
            hz = head(outs[-1]).float()
            cov.append(cr._finish(covs, cr._eig(covs), hz, cr._pn_l2).float())
            gap.append(hz)
            z = outs[-1]
            f = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1]).double()
            mean_last.append(f.mean(1))
            if keep_last_cov:
                fc = f - f.mean(1, keepdim=True)
                full_c.append(fc.transpose(1, 2) @ fc / fc.shape[1])
                proj_c.append(covs[-1])
    out = {"gap": torch.cat(gap), "cov": torch.cat(cov), "mean_last": torch.cat(mean_last)}
    if keep_last_cov:
        out["full_c"], out["proj_c"] = torch.cat(full_c), torch.cat(proj_c)
    return out


def rel_fro(A, B):
    """||A - B||_F / ||B||_F por imagem (B = limpo)."""
    num = (A - B).flatten(1).norm(dim=1)
    return (num / B.flatten(1).norm(dim=1).clamp_min(1e-30)).numpy()


def kappa_boot_std(y, pred, n_classes, seed, reps=300):
    """Desvio-padrao bootstrap (reamostragem das imagens de validacao) do kappa."""
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        ks.append(cohen_kappa_score(y[i], pred[i]) if len(np.unique(y[i])) > 1 else np.nan)
    return float(np.nanstd(ks))


def make_perturbed(Xva, sigmas, seed):
    sets = {("clean", 0.0): Xva}
    for kind in T.KINDS:
        for s in sigmas:
            if s <= 0:
                continue
            fn = T.perturb(kind, s, seed=seed)
            sets[(kind, s)] = torch.cat([fn(b) for b in Xva.split(16)])
    return sets


def run_pair(pair, args):
    from core.metrics import compute_metrics
    from sklearn.metrics import balanced_accuracy_score
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    lp = {"loader": "modehb", "dataset": pair["dataset"], "split": int(pair["split"]),
          "config": pair["config"], "percentage": int(pair.get("percentage", 100))}
    (Xtr, ytr), (Xva, yva) = load_split_data(lp)
    print(f"[data] {pair['pair_id']} train={tuple(Xtr.shape)} val={tuple(Xva.shape)} "
          f"range=[{Xva.min():.3f},{Xva.max():.3f}] {time.perf_counter() - t0:.1f}s", flush=True)
    sets = make_perturbed(Xva, args.sigmas, args.seed)
    energy = {k: (T.removed_energy_frac(Xva, k[1]) if k[0] != "clean" else 0.0) for k in sets}
    labels = np.unique(np.concatenate([ytr.numpy(), yva.numpy()]))
    remap = {int(c): i for i, c in enumerate(labels)}
    y0 = np.array([remap[int(v)] for v in yva])
    models = load_pair_models(pair)
    rows = []
    for name, (enc, head) in models.items():
        t1 = time.perf_counter()
        cr = R.fit_cov(enc, Xtr, "cpu", k=R.COV_K, seed=args.seed, micro=args.bs)
        ftr = features(cr, enc, head, Xtr, args.bs)
        clean = features(cr, enc, head, Xva, args.bs, keep_last_cov=True)
        clfs, clean_kappa = {}, {}
        for ro in ("gap", "cov"):
            res = evaluate_svm(ftr[ro], ytr, clean[ro], yva, scaler=False, C=args.C)
            clfs[ro], clean_kappa[ro] = res["_clf"], res["kappa"]
        print(f"[fit] {pair['pair_id']} {name} kappa_clean GAP={clean_kappa['gap']:.4f} "
              f"COV={clean_kappa['cov']:.4f} ({time.perf_counter() - t1:.1f}s)", flush=True)
        for (kind, s), Xp in sets.items():
            fp = clean if kind == "clean" else features(cr, enc, head, Xp, args.bs, keep_last_cov=True)
            fro_full = rel_fro(fp["full_c"], clean["full_c"])
            fro_proj = rel_fro(fp["proj_c"], clean["proj_c"])
            dmean = rel_fro(fp["mean_last"], clean["mean_last"])
            for ro in ("gap", "cov"):
                pred = clfs[ro].predict(fp[ro].double().numpy())
                m = compute_metrics(y0, pred, num_classes=len(labels))
                cos = T.cosine_rows(clean[ro], fp[ro]).numpy()
                mu = clean[ro].double().mean(0, keepdim=True)          # centro = media da val limpa
                cosc = T.cosine_rows(clean[ro].double() - mu, fp[ro].double() - mu).numpy()
                kstd = kappa_boot_std(y0, pred, len(labels), args.seed)
                kinds = T.KINDS if kind == "clean" else (kind,)   # sigma=0 replicado em cada curva
                for kd in kinds:
                    rows.append({
                        "pair_id": pair["pair_id"], "dataset": pair["dataset"], "split": pair["split"],
                        "search": pair["pair_id"].split(":")[1], "pooling": pair["genome_globals"]["pooling"],
                        "delta_kappa_fitness": pair["val_kappa_P1_alpha1"] - pair["val_kappa_P0_alpha0"],
                        "model": name, "ckpt_key": pair[name]["key"], "readout": ro.upper(),
                        "kind": kd, "sigma": s,
                        "downup_factor": T.downup_factor(s) if kd == "downup" else None,
                        "removed_energy_frac": energy[(kind, s)],
                        "kappa": m["kappa"], "macro_f1": m["f1"],
                        "balanced_acc": float(balanced_accuracy_score(y0, pred)),
                        "kappa_clean": clean_kappa[ro], "kappa_drop": clean_kappa[ro] - m["kappa"],
                        "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
                        "cos_p10": float(np.percentile(cos, 10)),
                        "cos_centered_mean": float(cosc.mean()),
                        "kappa_boot_std": kstd,
                        "lastcov_fro_rel_mean": float(fro_full.mean()),
                        "lastcov_fro_rel_median": float(np.median(fro_full)),
                        "lastcovk_fro_rel_mean": float(fro_proj.mean()),
                        "lastmean_rel_l2_mean": float(dmean.mean()),
                        "last_channels": int(clean["full_c"].shape[1]),
                        "cov_k_last": int(clean["proj_c"].shape[1]), "dim": int(fp[ro].shape[1]),
                        "n_train": len(ytr), "n_val": len(yva), "C": args.C, "scaler": False,
                        "seed": args.seed, "device": "cpu", "dtype": "fp32",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[ok] {pair['pair_id']} {name} ({time.perf_counter() - t1:.1f}s)", flush=True)
    return rows


def write_rows(rows, pair_id):
    """Grava em texture_metrics.csv (unico arquivo), substituindo as linhas do par; lock de arquivo
    para processos concorrentes."""
    import fcntl
    import os
    import pandas as pd
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        new = pd.DataFrame(rows)
        if OUT.exists():
            old = pd.read_csv(OUT)
            new = pd.concat([old[old["pair_id"] != pair_id], new], ignore_index=True)
        tmp = OUT.with_suffix(".csv.tmp")
        new.to_csv(tmp, index=False)
        os.replace(tmp, OUT)
    print(f"[write] {pair_id}: {len(rows)} linhas -> {OUT}", flush=True)

FIG = REPO / "reports/lejepa_diagnostics/figures"
COL = {"P0": "#2a78d6", "P1": "#eb6834"}                      # paleta de referencia (slots 1, 2)
LS = {"blur": "-", "noise": "--", "downup": ":"}
LBL = {"blur": "blur gaussiano", "noise": "ruído de mesma energia (controle)",
       "downup": "downsample-upsample (controle)"}


def plot(dataset):
    """06_texture_sensitivity: linhas = pares (maior dk+ / maior dk-), colunas = kappa GAP,
    kappa COV, cosseno GAP, cosseno COV, Frobenius rel. da cov. da ultima camada."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    df = pd.read_csv(OUT)
    df = df[df["dataset"] == dataset]
    pairs = (df.drop_duplicates("pair_id").sort_values("delta_kappa_fitness", ascending=False)
             ["pair_id"].tolist())
    cols = [("GAP", "kappa", "κ val (SVM limpa) · GAP"), ("COV", "kappa", "κ val (SVM limpa) · COV"),
            ("GAP", "cos_centered_mean", "cosseno centrado limpo×pert. · GAP"),
            ("COV", "cos_centered_mean", "cosseno centrado limpo×pert. · COV"),
            ("GAP", "lastcov_fro_rel_mean", "‖ΔΣ‖F/‖Σ‖F · última camada")]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(3.3 * len(cols), 3.0 * len(pairs) + 0.9),
                             squeeze=False)
    for i, pid in enumerate(pairs):
        d = df[df["pair_id"] == pid]
        dk = d["delta_kappa_fitness"].iloc[0]
        for j, (ro, met, title) in enumerate(cols):
            ax = axes[i, j]
            for m in ("P0", "P1"):
                for kind in ("blur", "noise", "downup"):
                    q = d[(d["readout"] == ro) & (d["model"] == m) & (d["kind"] == kind)].sort_values("sigma")
                    ax.plot(q["sigma"], q[met], LS[kind], color=COL[m], lw=2 if kind == "blur" else 1.5,
                            marker="o" if kind == "blur" else None, ms=4,
                            alpha=1.0 if kind == "blur" else 0.85)
            ax.set_xscale("symlog", linthresh=0.5)
            ax.set_xticks([0, 0.5, 1, 2, 4])
            ax.set_xticklabels(["0", "0,5", "1", "2", "4"])
            if met.startswith("lastcov"):
                ax.set_yscale("symlog", linthresh=0.01, linscale=0.3)
                ax.set_ylim(bottom=0)
            ax.grid(True, color="#e4e3df", lw=0.6)
            if i == 0:
                ax.set_title(title, fontsize=9.5, color="#0b0b0b")
            if i == len(pairs) - 1:
                ax.set_xlabel("σ (px)")
            if j == 0:
                ax.set_ylabel(f"{pid.split(':', 1)[1]}\nΔκ fitness {dk:+.3f}", fontsize=8.5)
        lo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        hi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for j in (0, 1):
            axes[i, j].set_ylim(lo, hi)
        lo = min(axes[i, 2].get_ylim()[0], axes[i, 3].get_ylim()[0])
        for j in (2, 3):
            axes[i, j].set_ylim(lo, 1.005)
    handles = [Line2D([], [], color=COL["P0"], lw=2, label="P0 (antes do LeJEPA)"),
               Line2D([], [], color=COL["P1"], lw=2, label="P1 (depois do LeJEPA)")]
    handles += [Line2D([], [], color="#52514e", ls=LS[k], lw=1.6, marker="o" if k == "blur" else None,
                       ms=4, label=LBL[k]) for k in ("blur", "noise", "downup")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{dataset}: sensibilidade à textura (perturbação só na validação; SVM C=100 sem scaler "
                 "ajustada no treino limpo)", y=1.045, fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / dataset
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"06_texture_sensitivity.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {out}/06_texture_sensitivity.(png|svg)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--plot", action="append", default=[], help="dataset a plotar")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(T.SIGMAS))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    pairs = {p["pair_id"]: p for p in json.loads(MANIFEST.read_text())["pairs"]}
    for pid in args.pair:
        write_rows(run_pair(pairs[pid], args), pid)
    for ds in args.plot:
        plot(ds)


if __name__ == "__main__":
    main()
