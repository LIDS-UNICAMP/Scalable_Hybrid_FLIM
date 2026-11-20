"""Etapa 3 do diagnostico LeJEPA: metricas POR CAMADA (L1..Ln) em P0 (anchor) e P1 (ssl_state).

Usa a biblioteca comum (readouts.fit_cov/layer_features, svm_eval.evaluate_svm, analysis.*) e
acrescenta: loader do formato MO-DEHB {anchor, ssl_state}, selecao automatica dos pares (maior
Delta-kappa positivo e maior negativo por base), matriz CKA camada x camada (P0 x P1) completa,
espectros normalizados e figuras.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 nice -n 10 python -m research.lejepa_diag.layer_diag \
        --datasets remote_sensing larvae --k 32 --geom-n 500

Saida unica: reports/lejepa_diagnostics/data/layer_metrics.csv (substitui por pair_id). A matriz CKA
completa vai em colunas largas (linha P0/Li: cka_P0Li_vs_P1Lj; linha P1/Lj: cka_P0Li_vs_P1Lj por i) e o
espectro normalizado em ``eig_norm_json``. Figuras em figures/<dataset>/.
Controles: SVM usa o MESMO treino/validacao completos em P0 e P1; geometria (rank, espectro, CKA)
usa o MESMO subconjunto de validacao (n = min(geom_n, N_val), indices fixos), a dimensao de cada
camada e identica em P0/P1 (mesma arquitetura; COV com k_eff = min(k, C_l) igual nos dois).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.lejepa_diag import analysis as A          # noqa: E402
from research.lejepa_diag import readouts as R          # noqa: E402
from research.lejepa_diag.run import _write, load_split_data  # noqa: E402
from research.lejepa_diag.svm_eval import evaluate_svm, public  # noqa: E402

MANIFEST = REPO / "reports/lejepa_diagnostics/data/checkpoint_manifest.json"
OUT = REPO / "reports/lejepa_diagnostics/data"
FIG = REPO / "reports/lejepa_diagnostics/figures"


def select_pairs(manifest, datasets):
    pairs = json.loads(Path(manifest).read_text())["pairs"]
    sel = []
    for ds in datasets:
        cand = [p for p in pairs if p["dataset"] == ds and p.get("val_kappa_P0_alpha0") is not None
                and p.get("val_kappa_P1_alpha1") is not None]
        for p in cand:
            p["delta_kappa_manifest"] = p["val_kappa_P1_alpha1"] - p["val_kappa_P0_alpha0"]
        pos = max(cand, key=lambda p: p["delta_kappa_manifest"])
        neg = min(cand, key=lambda p: p["delta_kappa_manifest"])
        sel += [dict(pos, role="delta_pos"), dict(neg, role="delta_neg")]
    return sel


def load_pair_models(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for name, key in (("P0", "anchor"), ("P1", "ssl_state")):
        enc = copy.deepcopy(ck["encoder"])
        enc.load_state_dict(ck[key]["encoder"])
        out[name] = enc.float().eval()
    return out


def run_pair(pair, args):
    t0 = time.perf_counter()
    data_pair = {"loader": "modehb", "dataset": pair["dataset"], "split": pair["split"],
                 "config": pair["config"], "percentage": pair["percentage"]}
    (Xtr, ytr), (Xva, yva) = load_split_data(data_pair)
    models = load_pair_models(pair["checkpoint_file"])
    n_geom = min(args.geom_n, len(Xva))
    gidx = A.paired_sample_idx(len(Xva), n_geom, args.seed)          # mesmas linhas P0/P1
    tidx = A.paired_sample_idx(len(Xtr), min(args.geom_n, len(Xtr)), args.seed)
    feats, kef, chans, rows = {}, {}, {}, []
    for m, enc in models.items():
        cr = R.fit_cov(enc, Xtr, "cpu", k=args.k, seed=args.seed, micro=args.batch_size)
        kef[m] = R.layer_k_eff(cr)
        ftr = R.layer_features(cr, enc, Xtr, "cpu", args.batch_size)
        fva = R.layer_features(cr, enc, Xva, "cpu", args.batch_size)
        feats[m] = (ftr, fva)
        chans[m] = {L: int(ftr[L]["gap"].shape[1]) for L in ftr}
        print(f"  [{m}] features ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    assert kef["P0"] == kef["P1"] and chans["P0"] == chans["P1"], "dimensao difere entre P0 e P1"
    layers = list(feats["P0"][0].keys())
    cka_rows = []
    for feat in ("gap", "cov"):
        for m in ("P0", "P1"):
            ftr, fva = feats[m]
            for li, L in enumerate(layers):
                res = public(evaluate_svm(ftr[L][feat], ytr, fva[L][feat], yva, scaler=False, C=args.C))
                Xg = fva[L][feat][gidx].double().numpy()
                eig = A.eigenvalues(Xg)
                p = A.normalized_spectrum(eig=eig)
                cum = np.cumsum(p)
                er = A.effective_rank(eig=eig)
                er_tr = A.effective_rank(ftr[L][feat][tidx])
                cap = min(n_geom - 1, Xg.shape[1])
                rows.append({
                    "pair_id": pair["pair_id"], "dataset": pair["dataset"], "split": pair["split"],
                    "role": pair["role"], "delta_kappa_manifest": pair["delta_kappa_manifest"],
                    "model": m, "ckpt_key": "anchor" if m == "P0" else "ssl_state",
                    "layer": L, "layer_idx": li + 1, "block": pair["architecture_blocks"][li],
                    "channels": chans[m][L], "feature": feat, "dim": int(Xg.shape[1]),
                    "k_eff": kef[m][L] if feat == "cov" else None, "pca_k_requested": args.k,
                    **{k: res[k] for k in ("kappa", "macro_f1", "balanced_acc", "n_train", "n_val",
                                           "svm_fit_status", "C", "scaler")},
                    "geom_split": "validation", "geom_n": n_geom, "erank": er,
                    "erank_norm": er / cap if cap > 0 else np.nan, "erank_cap": cap,
                    "ncomp90": A.n_components_for(eig=eig, frac=0.9),
                    "ncomp99": A.n_components_for(eig=eig, frac=0.99),
                    "top1_var_frac": float(p[0]), "top5_cumvar": float(cum[min(4, len(cum) - 1)]),
                    "erank_train_n": er_tr, "train_geom_n": len(tidx),
                    "ckpt_path": pair["checkpoint_file"], "ckpt_md5": pair["md5"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
                rows[-1]["eig_norm_json"] = json.dumps([float(f"{v:.6g}") for v in p])
        # CKA camada x camada P0 x P1 (pareado, mesmo subconjunto de validacao)
        for a in layers:
            for b in layers:
                X0 = feats["P0"][1][a][feat][gidx]
                X1 = feats["P1"][1][b][feat][gidx]
                lin = A.linear_cka(X0, X1)
                deb = A.linear_cka(X0, X1, debiased=True)
                cka_rows.append({"pair_id": pair["pair_id"], "dataset": pair["dataset"], "role": pair["role"],
                                 "feature": feat, "layer_P0": a, "layer_P1": b, "cka_linear": lin,
                                 "cka_debiased": deb, "n": n_geom, "split": "validation"})
        # CKA diagonal tambem no treino (controle de n)
        for L in layers:
            ctr = A.linear_cka(feats["P0"][0][L][feat][tidx], feats["P1"][0][L][feat][tidx])
            for r in rows:
                if r["layer"] == L and r["feature"] == feat:
                    r["cka_diag_train_n"] = ctr
    # CKA diagonal / matriz larga / delta-kappa nas linhas
    cmap = {(c["feature"], c["layer_P0"], c["layer_P1"]): c for c in cka_rows}
    kap = {(r["model"], r["feature"], r["layer"]): r for r in rows}
    for r in rows:
        c = cmap[(r["feature"], r["layer"], r["layer"])]
        r["cka_diag"], r["cka_diag_debiased"] = c["cka_linear"], c["cka_debiased"]
        for o in layers:   # linha da matriz (P0: Li fixo; P1: Lj fixo)
            a, b = (r["layer"], o) if r["model"] == "P0" else (o, r["layer"])
            cc = cmap[(r["feature"], a, b)]
            r[f"cka_P0{a}_vs_P1{b}"] = cc["cka_linear"]
            r[f"ckadeb_P0{a}_vs_P1{b}"] = cc["cka_debiased"]
        r["delta_kappa_layer"] = kap[("P1", r["feature"], r["layer"])]["kappa"] - kap[("P0", r["feature"], r["layer"])]["kappa"]
        r["delta_erank_norm"] = kap[("P1", r["feature"], r["layer"])]["erank_norm"] - kap[("P0", r["feature"], r["layer"])]["erank_norm"]
        r["stage_time_s"] = time.perf_counter() - t0
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["remote_sensing", "larvae"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--geom-n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    torch.manual_seed(args.seed)
    pairs = select_pairs(MANIFEST, args.datasets)
    if not args.plots_only:
        for pair in pairs:
            print(f"[par] {pair['pair_id']} ({pair['role']}, dk={pair['delta_kappa_manifest']:+.4f})", flush=True)
            rows = run_pair(pair, args)
            _write(OUT / "layer_metrics.csv", rows, pair["pair_id"])
            print(f"[ok] {pair['pair_id']}", flush=True)
    from research.lejepa_diag.layer_plots import plot_all
    for ds in args.datasets:
        plot_all(ds, OUT, FIG / ds)


if __name__ == "__main__":
    main()
