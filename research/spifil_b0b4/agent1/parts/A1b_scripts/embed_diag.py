"""A1b — diagnóstico do embedding (CPU, só treino→validação; NUNCA teste, NUNCA seleção).

Por checkpoint (encoder.pt de um candidato MO-DEHB) e variante de pesos (saved = α escolhido pelo WiSE;
a0 = âncora SPiFiL(+AE) com head inicial; a1 = estado SSL puro), mede:
  * geometria do embedding 1280 (ORIG = head(encoder(x))): norma, escala por dimensão, espectro,
    rank efetivo (participation ratio, mesma def. de embedding_eval), RankMe, condicionamento, posto,
    fração de dims mortas, razão ||média||/dispersão; posto/espectro do W do head 1×1;
  * a mesma geometria do GAP do último bloco;
  * sondas SVM (core.svm.fit_svm, linear, C=100, libsvm ovo) treino→val em várias leituras:
    ORIG (sem scaler), ORIG+scaler, ORIG+sqrt/L2, ORIG+pw_whiten, GAP/GeM por bloco (+scaler),
    concat GAP de todos os blocos (+scaler), COV completo (CovReadout k=64 + scaler = leitura do fitness),
    COV sem o grupo head, COV só do último bloco, head-group do COV (head com sqrt/L2 + scaler).
Uso: OMP_NUM_THREADS=4 nice -n 15 CUDA_VISIBLE_DEVICES="" python embed_diag.py --dataset remote_sensing \
        --ckpt <encoder.pt> [...] --variants saved,a0,a1 --out <jsonl>
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

ROOT = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM")
sys.path.insert(0, str(ROOT))
SCRATCH = Path("/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad")


def stratified(y, max_n, seed=42):
    from ga_search.modehb.run_search import _stratified_indices
    return _stratified_indices(np.asarray(y), max_n, seed)


def load_xy(dataset, max_train, max_val):
    from ga_search.modehb.run_search import load_config
    from ga_search.progressive_data import load_data
    cfg = load_config(str(ROOT / "ga_search/config/modehb/search_v3.yaml"), dataset)
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.update(num_workers=4, seed=cfg["seed"])
    tr, va, _, ident = load_data(d, SCRATCH / "a1b_data")
    out = {}
    for name, loader, mx in (("train", tr, max_train), ("val", va, max_val)):
        xs, ys = [], []
        for x, y, *_ in loader:
            xs.append(x), ys.append(torch.as_tensor(y))
        X, Y = torch.cat(xs), torch.cat(ys)
        idx = stratified(Y.numpy(), mx, 42)
        out[name] = (X[torch.as_tensor(idx)], Y[torch.as_tensor(idx)].numpy())
    return out, ident["sha256"], cfg


def blend(sd_anchor, sd_ssl, a):
    out = {}
    for n, t in sd_ssl.items():
        a0 = sd_anchor.get(n)
        out[n] = ((1 - a) * a0 + a * t) if (a0 is not None and t.is_floating_point() and a0.shape == t.shape) else t
    return out


def variants(ck, which):
    enc, head = ck["encoder"].float().eval(), ck["head"].float().eval()
    res = {}
    for v in which:
        if v == "saved":
            res[v] = (enc, head)
        elif v in ("a0", "a1") and ck.get("anchor") is not None and ck.get("ssl_state") is not None:
            a = 0.0 if v == "a0" else 1.0
            e2, h2 = copy.deepcopy(enc), copy.deepcopy(head)
            e2.load_state_dict(blend(ck["anchor"]["encoder"], ck["ssl_state"]["encoder"], a))
            h2.load_state_dict(blend(ck["anchor"]["head"], ck["ssl_state"]["head"], a))
            res[v] = (e2.eval(), h2.eval())
    return res


@torch.no_grad()
def features(enc, head, X, cr, bs=32):
    from ga_search.modehb.pooling_head import gem_pool
    gap, gem, H, COV = None, None, [], []
    for b in X.split(bs):
        outs, h = [], b
        for blk in enc:
            h = blk(h)
            outs.append(h.float())
        g = [o.mean((2, 3)) for o in outs]
        m = [gem_pool(o) for o in outs]
        gap = [[] for _ in outs] if gap is None else gap
        gem = [[] for _ in outs] if gem is None else gem
        for i in range(len(outs)):
            gap[i].append(g[i]), gem[i].append(m[i])
        H.append(head(outs[-1]).float())
        if cr is not None:
            COV.append(cr(enc, head, b))   # recomputes the blocks (simple; CPU time acceptable)
    cat = lambda L: torch.cat(L).numpy().astype(np.float64)
    return [cat(x) for x in gap], [cat(x) for x in gem], cat(H), (cat(COV) if COV else None)


def geom(Z):
    """Geometry of an embedding matrix [N, D] (float64)."""
    Z = np.asarray(Z, np.float64)
    mu = Z.mean(0)
    Xc = Z - mu
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / max(len(Z) - 1, 1)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 0.0
    p = s / s.sum() + 1e-7 if s.sum() > 0 else s
    rankme = float(np.exp(-(p * np.log(p)).sum())) if s.sum() > 0 else 0.0
    tol = s.max() * max(Z.shape) * np.finfo(np.float64).eps if s.size else 0
    rank = int((s > tol).sum())
    sd = Xc.std(0)
    nz = sd[sd > 1e-8]
    cum = np.cumsum(np.sort(lam)[::-1]) / lam.sum() if lam.sum() > 0 else np.array([1.0])
    norms = np.linalg.norm(Z, axis=1)
    return {"dim": int(Z.shape[1]), "n": int(Z.shape[0]), "norm_mean": float(norms.mean()),
            "norm_cv": float(norms.std() / max(norms.mean(), 1e-12)),
            "mean_norm_over_spread": float(np.linalg.norm(mu) / max(np.sqrt((Xc ** 2).sum(1).mean()), 1e-12)),
            "dim_std_median": float(np.median(sd)), "dim_std_max": float(sd.max()),
            "dim_std_max_over_median": float(sd.max() / max(np.median(sd), 1e-12)),
            "dim_std_p99_over_p50": float(np.percentile(sd, 99) / max(np.percentile(sd, 50), 1e-12)),
            "frac_dead_dims": float((sd <= 1e-8).mean()),
            "eff_rank_pr": pr, "rankme": rankme, "num_rank": rank,
            "top1_var_frac": float(cum[0]), "n_dims_90var": int(np.searchsorted(cum, 0.90) + 1),
            "n_dims_99var": int(np.searchsorted(cum, 0.99) + 1),
            "cond_nonzero": float(lam[0] / lam[rank - 1]) if rank > 0 and lam[rank - 1] > 0 else float("inf"),
            "spectrum_top8_frac": [float(x) for x in (lam[:8] / lam.sum())] if lam.sum() > 0 else []}


def probe(Xtr, ytr, Xva, yva, scaler, C=100.0):
    from core.svm import fit_svm
    from core.metrics import compute_metrics
    labels = np.unique(np.concatenate([ytr, yva]))
    remap = {int(c): i for i, c in enumerate(labels)}
    a, b = np.vectorize(remap.get)(ytr), np.vectorize(remap.get)(yva)
    t0 = time.time()
    clf = fit_svm(Xtr, a, C=C, scaler=scaler, tag="A1B_DIAG")
    pred = clf.predict(Xva)
    m = compute_metrics(b, np.asarray(pred, np.int64), num_classes=len(labels))
    dg = getattr(clf, "fit_diagnostics_", {}) or {}
    return {"kappa": float(m["kappa"]), "f1": float(m["f1"]), "fit_s": round(time.time() - t0, 2),
            "n_iter_max": dg.get("svm_n_iter_max"), "fit_status": dg.get("svm_fit_status"),
            "n_sv": dg.get("svm_n_sv")}


def pn_l2(Z):
    Z = np.sign(Z) * np.sqrt(np.abs(Z))
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--variants", default="saved")
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-val", type=int, default=100000)
    ap.add_argument("--no-cov", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    data, sha, cfg = load_xy(a.dataset, a.max_train, a.max_val)
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    print(f"[{a.dataset}] train {len(ytr)} val {len(yva)} sha {sha[:12]}", flush=True)
    from ga_search.modehb.pooling_head import CovReadout
    from ga_search.modehb.supervised_fitness import stratified_subsample  # noqa: F401 (protocol ref)
    from ga_search.modehb.svm_on_demand import pw_whiten
    for path in a.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for vname, (enc, head) in variants(ck, a.variants.split(",")).items():
            t0 = time.time()
            cr = None if a.no_cov else CovReadout(k=64, seed=42).fit(enc, Xtr, torch.device("cpu"))
            gtr, mtr, Htr, Ctr = features(enc, head, Xtr, cr)
            gva, mva, Hva, Cva = features(enc, head, Xva, cr)
            nb = len(gtr)
            W = head.proj[0].weight.detach().double().reshape(head.proj[0].weight.shape[0], -1).numpy()
            sW = np.linalg.svd(W, compute_uv=False)
            rec = {"dataset": a.dataset, "checkpoint": str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else path,
                   "cand_id": ck.get("cand_id"), "fidelity": ck.get("fidelity"), "variant": vname,
                   "wise_alpha_saved": ck.get("wise_alpha"), "pooling": ck["genome"].get("pooling"),
                   "embed_dim": int(Htr.shape[1]), "n_blocks": nb, "block_channels": [int(g.shape[1]) for g in gtr],
                   "params": int(sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())),
                   "data_sha256": sha, "n_train": int(len(ytr)), "n_val": int(len(yva)),
                   "head_W_shape": list(W.shape), "head_W_rank": int((sW > sW.max() * max(W.shape) * 1e-12).sum()),
                   "head_W_cond": float(sW[0] / sW[-1]) if sW[-1] > 0 else float("inf"),
                   "geom_head": geom(np.vstack([Htr, Hva])), "geom_lastgap": geom(np.vstack([gtr[-1], gva[-1]])),
                   "geom_gap_blocks": [geom(np.vstack([gtr[i], gva[i]])) for i in range(nb)], "probes": {}}
            P = rec["probes"]
            P["orig_noscaler"] = probe(Htr, ytr, Hva, yva, False)
            P["orig_scaler"] = probe(Htr, ytr, Hva, yva, True)
            P["orig_pnl2"] = probe(pn_l2(Htr), ytr, pn_l2(Hva), yva, False)
            P["orig_pnl2_scaler"] = probe(pn_l2(Htr), ytr, pn_l2(Hva), yva, True)
            wt, wv = pw_whiten(Htr, Hva)
            P["orig_pw"] = probe(wt, ytr, wv, yva, False)
            P["lastgap_noscaler"] = probe(gtr[-1], ytr, gva[-1], yva, False)
            P["lastgap_scaler"] = probe(gtr[-1], ytr, gva[-1], yva, True)
            for i in range(nb):
                P[f"gap_b{i}_scaler"] = probe(gtr[i], ytr, gva[i], yva, True)
                P[f"gem_b{i}_scaler"] = probe(mtr[i], ytr, mva[i], yva, True)
            if nb > 1:
                P["catgap_scaler"] = probe(np.hstack(gtr), ytr, np.hstack(gva), yva, True)
                P["catgem_scaler"] = probe(np.hstack(mtr), ytr, np.hstack(mva), yva, True)
            if Ctr is not None:
                # group layout: per block k_i(k_i+1)/2 triu dims, then head (embed_dim)
                ks = [min(64, c) for c in rec["block_channels"]]
                sizes = [k * (k + 1) // 2 for k in ks] + [Htr.shape[1]]
                assert sum(sizes) == Ctr.shape[1], (sizes, Ctr.shape)
                cuts = np.cumsum([0] + sizes)
                grp = lambda Z, i: Z[:, cuts[i]:cuts[i + 1]]
                P["cov_full_scaler"] = probe(Ctr, ytr, Cva, yva, True)
                P["cov_full_noscaler"] = probe(Ctr, ytr, Cva, yva, False)
                P["cov_nohead_scaler"] = probe(Ctr[:, :cuts[-2]], ytr, Cva[:, :cuts[-2]], yva, True)
                P["cov_last_scaler"] = probe(grp(Ctr, nb - 1), ytr, grp(Cva, nb - 1), yva, True)
                P["cov_headgroup_scaler"] = probe(grp(Ctr, nb), ytr, grp(Cva, nb), yva, True)
                for i in range(nb):
                    P[f"cov_b{i}_scaler"] = probe(grp(Ctr, i), ytr, grp(Cva, i), yva, True)
                rec["cov_dim"] = int(Ctr.shape[1])
                rec["geom_cov"] = geom(np.vstack([Ctr, Cva]))
            rec["seconds"] = round(time.time() - t0, 1)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {rec['cand_id']} f{rec['fidelity']} {vname}: ORIG {P['orig_noscaler']['kappa']:.3f} "
                  f"ORIG+sc {P['orig_scaler']['kappa']:.3f} lastGAP+sc {P['lastgap_scaler']['kappa']:.3f} "
                  f"COV {P.get('cov_full_scaler', {}).get('kappa', float('nan')):.3f} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
