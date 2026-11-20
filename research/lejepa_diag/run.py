"""CLI do diagnostico LeJEPA: le um manifest de pares (P0, P1) e grava CSVs idempotentes.

    CUDA_VISIBLE_DEVICES="" nice -n 19 python -m research.lejepa_diag.run \
        --manifest reports/lejepa_diagnostics/data/checkpoint_manifest.json \
        --stage matrix --stage layers --stage texture --device cpu --batch-size 8

Manifest: lista de pares ou {"pairs": [...]}. Campos por par:
    p0_path, p1_path   checkpoints (MO-DEHB encoder.pt: {"encoder", "head", ...}; ou nn.Module)
    loader             "modehb" (ga_search.modehb.run_search.load_config + progressive_data.load_data,
                       mesmo caminho de svm_on_demand) | "synthetic" (testes)
    dataset, split     ex.: "protozoan", 1
    opcionais: pair_id, ckpt_loader ("pkg.mod:fn", fn(path) -> (encoder, head|None)), config (yaml do MO-DEHB, padrao ga_search/config/modehb/search.yaml),
               percentage, image_size, n_train/n_val/n_classes (synthetic)
Saidas em --out-dir (padrao reports/lejepa_diagnostics/data/): metrics.csv, layer_metrics.csv,
texture_metrics.csv. Cada unidade (pair_id, stage) com a mesma impressao digital dos checkpoints
e pulada se ja existe (``--force`` refaz). Teste nunca e lido (``--report-test`` nao existe aqui
de proposito: a avaliacao e na validacao).
"""
from __future__ import annotations

import argparse
import hashlib
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
from research.lejepa_diag import textura as T           # noqa: E402
from research.lejepa_diag.extract import extract, load_checkpoint, load_images  # noqa: E402
from research.lejepa_diag.svm_eval import evaluate_svm, public                   # noqa: E402

OUT_DEFAULT = REPO / "reports/lejepa_diagnostics/data"
CSV = {"matrix": "metrics.csv", "layers": "layer_metrics.csv", "texture": "texture_metrics.csv"}
DEFAULT_CONFIG = "ga_search/config/modehb/search.yaml"


# ─── manifest / dados ─────────────────────────────────────────────────────────

def read_manifest(path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    pairs = raw.get("pairs", raw) if isinstance(raw, dict) else raw
    out = []
    for p in pairs:
        miss = [k for k in ("p0_path", "p1_path", "loader", "dataset", "split") if k not in p]
        if miss:
            raise ValueError(f"par sem campos {miss}: {p}")
        p = dict(p)
        p.setdefault("pair_id", f"{p['dataset']}_split{p['split']}_" +
                     hashlib.sha1(f"{p['p0_path']}|{p['p1_path']}".encode()).hexdigest()[:8])
        out.append(p)
    return out


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def ckpt_fingerprint(p0, p1) -> str:
    parts = []
    for p in (p0, p1):
        st = _abs(p).stat()
        parts.append(f"{_abs(p)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _synthetic(pair):
    g = torch.Generator().manual_seed(int(pair.get("seed", 0)))
    s, nc = int(pair.get("image_size", 32)), int(pair.get("n_classes", 3))
    mk = lambda n: (torch.randn(n, 3, s, s, generator=g), torch.arange(n) % nc)
    return mk(int(pair.get("n_train", 24))), mk(int(pair.get("n_val", 12)))


def load_split_data(pair, num_workers=0):
    """-> ((X_train, y_train), (X_val, y_val)) em RAM (imagens ja normalizadas do pipeline)."""
    if pair["loader"] == "synthetic":
        return _synthetic(pair)
    if pair["loader"] not in ("modehb", "progressive"):
        raise ValueError(f"loader desconhecido {pair['loader']!r} (use modehb|synthetic)")
    from ga_search.modehb.run_search import load_config
    from ga_search.progressive_data import load_data
    cfg = load_config(str(_abs(pair.get("config", DEFAULT_CONFIG))), pair["dataset"])
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.update(num_workers=num_workers, seed=cfg["seed"], split=int(pair["split"]))
    for k in ("percentage", "image_size"):
        if k in pair:
            d[k] = pair[k]
    tr, va, _, _ident = load_data(d, Path(cfg["output_dir"]) / "svm_data")
    Xtr, ytr, _ = load_images(tr)
    Xva, yva, _ = load_images(va)
    return (Xtr, ytr), (Xva, yva)


def _load_model(pair, path):
    """(encoder, head) de um checkpoint. ``pair["ckpt_loader"] = "pkg.mod:fn"`` (fn(path) ->
    (encoder nn.Sequential de blocos, head|None)) cobre formatos state_dict; senao extract.load_checkpoint."""
    spec = pair.get("ckpt_loader")
    if spec:
        import importlib
        mod, fn = spec.split(":")
        enc, head = getattr(importlib.import_module(mod), fn)(str(_abs(path)))[:2]
        return enc, head
    enc, head, _meta = load_checkpoint(_abs(path))
    return enc, head


def _sub(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y
    i = torch.as_tensor(A.paired_sample_idx(len(X), n, seed))
    return X[i], y[i]


# ─── CSV idempotente ──────────────────────────────────────────────────────────

def _done(csv: Path, pair_id: str, fp: str) -> bool:
    if not csv.exists():
        return False
    import pandas as pd
    df = pd.read_csv(csv, usecols=lambda c: c in ("pair_id", "ckpt_fp"))
    return bool(((df["pair_id"] == pair_id) & (df["ckpt_fp"] == fp)).any())


def _write(csv: Path, rows: list[dict], pair_id: str):
    import pandas as pd
    new = pd.DataFrame(rows)
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old["pair_id"] != pair_id]          # substitui a unidade (force / fp novo)
        new = pd.concat([old, new], ignore_index=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, csv)


# ─── etapas ───────────────────────────────────────────────────────────────────

def _base(pair, fp, model):
    return {"pair_id": pair["pair_id"], "dataset": pair["dataset"], "split": pair["split"],
            "model": model, "ckpt_path": pair["p0_path" if model == "P0" else "p1_path"],
            "ckpt_fp": fp, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def stage_matrix(pair, models, data, args, fp):
    (Xtr, ytr), (Xva, yva) = data
    rows, val_feats = [], {}
    for name, (enc, head) in models.items():
        etr = extract(enc, head, (Xtr, ytr), device=args.device, batch_size=args.batch_size, layers=False)
        eva = extract(enc, head, (Xva, yva), device=args.device, batch_size=args.batch_size, layers=False)
        t0 = time.perf_counter()
        ctr, cva, _cr = R.cov_readout(enc, head, Xtr, Xva, args.device, k=args.cov_k, seed=args.seed,
                                      batch_size=args.batch_size)
        cov_s = time.perf_counter() - t0
        feats = {"GAP": (etr.embedding, eva.embedding, False, etr.time_s + eva.time_s),
                 "COV": (ctr, cva, False, cov_s), "COV+scaler": (ctr, cva, True, cov_s)}
        for ro, (ftr, fva, sc, ts) in feats.items():
            res = public(evaluate_svm(ftr, ytr, fva, yva, scaler=sc, C=args.C))
            val_feats[(name, ro)] = fva
            rows.append({**_base(pair, fp, name), "readout": ro, **res, "extract_time_s": ts,
                         "peak_mem_bytes": max(etr.peak_mem_bytes, eva.peak_mem_bytes),
                         "mem_kind": etr.mem_kind, "erank_val": A.effective_rank(fva),
                         "n_comp90_val": A.n_components_for(fva, 0.9)})
    for r in rows:   # CKA(P0, P1) pareado na validacao, com controle de n/d
        if r["model"] == "P1" and ("P0", r["readout"]) in val_feats:
            g = A.controlled_geometry(val_feats[("P0", r["readout"])], val_feats[("P1", r["readout"])],
                                      n=args.ctrl_n, d=args.ctrl_d, reps=args.ctrl_reps, seed=args.seed)
            r.update({f"ctrl_{k}": v for k, v in g.items()})
    return rows


def stage_layers(pair, models, data, args, fp):
    (Xtr, ytr), (Xva, yva) = data
    rows, val = [], {}
    for name, (enc, _head) in models.items():
        t0 = time.perf_counter()
        cr = R.fit_cov(enc, Xtr, args.device, k=args.layer_k, seed=args.seed, micro=args.batch_size)
        kef = R.layer_k_eff(cr)
        ftr = R.layer_features(cr, enc, Xtr, args.device, args.batch_size)
        fva = R.layer_features(cr, enc, Xva, args.device, args.batch_size)
        ts = time.perf_counter() - t0
        for L in ftr:
            for feat in ("gap", "cov"):
                res = public(evaluate_svm(ftr[L][feat], ytr, fva[L][feat], yva, scaler=False, C=args.C))
                val[(name, L, feat)] = fva[L][feat]
                rows.append({**_base(pair, fp, name), "layer": L, "feature": feat,
                             "k_eff": kef[L] if feat == "cov" else None, **res,
                             "erank_val": A.effective_rank(fva[L][feat]), "stage_time_s": ts})
    for r in rows:
        key0 = ("P0", r["layer"], r["feature"])
        if r["model"] == "P1" and key0 in val:
            g = A.controlled_geometry(val[key0], val[("P1", r["layer"], r["feature"])],
                                      n=args.ctrl_n, d=args.ctrl_d, reps=args.ctrl_reps, seed=args.seed)
            r.update({f"ctrl_{k}": v for k, v in g.items()})
    return rows


def stage_texture(pair, models, data, args, fp):
    """SVM ajustada no treino LIMPO (GAP oficial); validacao perturbada -> cosseno + kappa."""
    from core.metrics import compute_metrics
    from sklearn.metrics import balanced_accuracy_score
    (Xtr, ytr), (Xva, yva) = data
    rows = []
    for name, (enc, head) in models.items():
        etr = extract(enc, head, (Xtr, ytr), device=args.device, batch_size=args.batch_size, layers=False)
        clean = extract(enc, head, (Xva, yva), device=args.device, batch_size=args.batch_size, layers=False)
        clf = evaluate_svm(etr.embedding, ytr, clean.embedding, yva, scaler=False, C=args.C)["_clf"]
        labels = np.unique(np.concatenate([ytr.numpy(), yva.numpy()]))
        remap = {int(c): i for i, c in enumerate(labels)}               # mesmo remap de svm_eval
        y0 = np.array([remap[int(v)] for v in yva])
        for kind in T.KINDS:
            for s in args.sigmas:
                pert = extract(enc, head, (Xva, yva), device=args.device, batch_size=args.batch_size,
                               layers=False, transform=T.perturb(kind, s, seed=args.seed))
                cos = T.cosine_rows(clean.embedding, pert.embedding).numpy()
                pred = clf.predict(pert.embedding.double().numpy())
                m = compute_metrics(y0, pred, num_classes=len(labels))
                rows.append({**_base(pair, fp, name), "kind": kind, "sigma": s,
                             "downup_factor": T.downup_factor(s) if kind == "downup" else None,
                             "removed_energy_frac": T.removed_energy_frac(Xva[:64], s),
                             "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
                             "cos_p10": float(np.percentile(cos, 10)),
                             "kappa_clean_svm": m["kappa"], "macro_f1_clean_svm": m["f1"],
                             "balanced_acc_clean_svm": float(balanced_accuracy_score(y0, pred)),
                             "n_val": int(len(yva)), "extract_time_s": pert.time_s})
    return rows


STAGES = {"matrix": stage_matrix, "layers": stage_layers, "texture": stage_texture}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostico LeJEPA (P0 vs P1)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", action="append", choices=list(STAGES), required=True)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pair-filter", default=None, help="substring de pair_id")
    ap.add_argument("--force", action="store_true", help="refaz unidades ja gravadas")
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--cov-k", type=int, default=R.COV_K)
    ap.add_argument("--layer-k", type=int, default=R.COV_K, help="k fixo (igual P0/P1) por camada")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="subamostra (smoke); None = tudo")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--ctrl-n", type=int, default=None, help="n de amostras igual P0/P1 (rank/CKA)")
    ap.add_argument("--ctrl-d", type=int, default=None, help="dimensao igual P0/P1 (rank/CKA)")
    ap.add_argument("--ctrl-reps", type=int, default=1)
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(T.SIGMAS))
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    pairs = [p for p in read_manifest(args.manifest) if not args.pair_filter or args.pair_filter in p["pair_id"]]
    cache: dict = {}
    for pair in pairs:
        fp = ckpt_fingerprint(pair["p0_path"], pair["p1_path"])
        todo = [s for s in args.stage if args.force or not _done(out / CSV[s], pair["pair_id"], fp)]
        for s in set(args.stage) - set(todo):
            print(f"[skip] {pair['pair_id']} {s} (ja em {CSV[s]})")
        if not todo:
            continue
        key = (pair["loader"], pair["dataset"], pair["split"], pair.get("config"), pair.get("percentage"))
        if key not in cache:
            cache = {key: load_split_data(pair, args.num_workers)}       # 1 dataset em RAM por vez
        (Xtr, ytr), (Xva, yva) = cache[key]
        data = (_sub(Xtr, ytr, args.max_train, args.seed), _sub(Xva, yva, args.max_val, args.seed))
        models = {m: _load_model(pair, pair[k]) for m, k in (("P0", "p0_path"), ("P1", "p1_path"))}
        for s in todo:
            t0 = time.perf_counter()
            rows = STAGES[s](pair, models, data, args, fp)
            _write(out / CSV[s], rows, pair["pair_id"])
            print(f"[ok] {pair['pair_id']} {s}: {len(rows)} linhas em {time.perf_counter() - t0:.1f}s "
                  f"-> {out / CSV[s]}")


if __name__ == "__main__":
    main()
