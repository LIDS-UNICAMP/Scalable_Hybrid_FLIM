"""On-demand SVM probe of saved MO-DEHB encoders (prompt §17.1). NEVER feeds the fitness.

    nice -n 19 python -m ga_search.modehb.svm_on_demand --dataset remote_sensing \
        --checkpoint artifacts/modehb/.../encoder.pt  [--artifact ophira-ai/progressive_mo_dehb/encoder-<id>:latest ...]

Adapter only: the official repo SVM (``eval/svm.py``) reads FLIM encoders through
``conv1..convN`` and cannot see a MO-DEHB ``encoder.pt`` ({"encoder": nn.Sequential,
"head": EmbeddingHead}). This script extracts ``head(encoder(x))`` (the searched >=1280-d
embedding) and then uses the repo's single canonical ``core.svm.fit_svm`` (with the
StandardScaler that ``core/svm.py`` documents for the 1280-d arms) and ``core.metrics``.
Fit on the training split, score on VALIDATION by default. ``--split test`` scores the
held-out test split instead (explicit user request; reporting only, never read by the search). Results go to a W&B
``external_eval`` run + the "SVM sob demanda" table + the local JSONL; the search never
reads them back.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml


HEAD_CHUNK = None     # images per forward in the head readout (None = loader batch)


def _features(encoder, head, loader, device):
    encoder.eval(), head.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            for b in x.split(HEAD_CHUNK or len(x)):
                xs.append(head(encoder(b.to(device))).float().cpu())
            ys.append(torch.as_tensor(y))
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def _resolve(args):
    """-> list of (local checkpoint path, parent run id, artifact metadata)."""
    out = [(p, None, {}) for p in args.checkpoint]
    if args.artifact:
        import wandb
        api = wandb.Api()
        for name in args.artifact:
            art = api.artifact(name, type="encoder")
            d = Path(art.download(root=str(Path(args.download_dir) / art.name.replace(":", "_"))))
            meta = dict(art.metadata or {})
            out.append((str(next(d.glob("*.pt"))), meta.get("run_id"), meta))
    return out


def _test_loader(d):
    """Test-split loader built from the same dataset YAML (reporting only)."""
    from torch.utils.data import DataLoader
    from ga_search import progressive_data as PD
    ds = dict(d["dataset_config"])
    for k in ("images_dir", "masks_dir", "split_json"):
        if ds.get(k):
            ds[k] = ds[k].replace("{root}", str(ds.get("root", "")))
    te = PD.existing_data._samples(ds, int(d.get("percentage", 100)), int(d.get("split", 1)), "test")
    dset = PD.ReconstructionDataset(te, int(d.get("image_size", 200)), bool(d.get("imagenet_norm", False)))
    return DataLoader(dset, batch_size=int(d.get("batch_size", 64)),
                      num_workers=int(d.get("num_workers", 0)), shuffle=False)


def _cov_features(enc, head, data, device, k=64, tta=1, pn=True, chunk=32, fit_micro=None):
    """pooling_head.CovReadout (label-free, fitted on train pixels) for train / validation.
    ``tta=8``: mean of the readout over pooling_head.tta_transforms(8) (rot90 x hflip; the
    CovReadout PCA is still fitted on the untransformed train images). ``pn=False``: no
    per-group signed-sqrt/L2 (the caller whitens instead)."""
    from ga_search.modehb.pooling_head import CovReadout, tta_transforms
    views = tta_transforms(tta)
    enc.eval(), head.eval()
    xs, ys = {}, {}
    for split in ("train", "validation"):
        xx, yy = [], []
        for x, y, *_ in data[split]:
            xx.append(x), yy.append(torch.as_tensor(y))
        xs[split], ys[split] = torch.cat(xx), torch.cat(yy).numpy()
    cr = CovReadout(k=k).fit(enc, xs["train"], device, micro=fit_micro)
    def emb(b):
        return torch.stack([cr(enc, head, t(b), pn=pn).float() for t in views]).mean(0).cpu()
    f = lambda X: torch.cat([emb(b.to(device)) for b in X.split(chunk)]).numpy()
    return f(xs["train"]), ys["train"], f(xs["validation"]), ys["validation"]


# --- R-SPF ruler of the discovery repo (/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery) ---
# gem_tta_pw = eval/svm.py `_encode_pooled(mode="gem_tta")` (l.332-334: GeM p=3, clamp 1e-6, of the
# LAST layer, mean over the 8 views rot90(k=0..3) x {id, hflip} of the IMAGE) + `pw_whiten` (l.381-389)
# + core.svm.fit_svm(C=100, scaler=False) (eval/l1_fast.py l.144-147, eval/spf_grade.py score_full
# l.1046-1049). cat_tta_pw = spf_grade.py l.1662-1672: per-layer gem_tta concatenated from the last
# layer backwards while the total stays < MAX_DIM=2000 (only if >= 2 layers fit), then pw_whiten.
PW_SHRINK = 0.1      # eval/svm.py l.388 / spf_grade.py SHRINK
CAT_MAX_DIM = 2000   # spf_grade.py MAX_DIM


def pw_whiten(X_train, *others):
    """Copy of discovery eval/svm.py l.381-389: PCA-whitening with shrinkage + L2, fitted on train."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    pca = PCA().fit(X_train)
    s = (pca.explained_variance_ + PW_SHRINK * pca.explained_variance_.mean()) ** -0.5
    return [normalize(pca.transform(z) * s) for z in (X_train, *others)]


def _gem_tta_blocks(encoder, x):
    """[B,C_i] per block: GeM p=3 of each block output, averaged over the 8 rot90 x flip views."""
    from ga_search.modehb.pooling_head import gem_pool
    views = []
    for v in (x, x.flip(3)):
        for k in range(4):
            h, feats = torch.rot90(v, k, (2, 3)), []
            for blk in encoder:
                h = blk(h)
                feats.append(gem_pool(h))
            views.append(feats)
    return [torch.stack([f[i] for f in views]).mean(0) for i in range(len(views[0]))]


def cat_select(dims):
    """Block indices that cat_tta concatenates (spf_grade.py l.1662-1668); [] = no cat."""
    fit, total = [], 0
    for i in reversed(range(len(dims))):
        total += dims[i]
        if total >= CAT_MAX_DIM:
            break
        fit.insert(0, i)
    return fit if len(fit) > 1 else []


def _tta_features(encoder, loader, device, readout, chunk=None):
    chunk = chunk or GEM_TTA_CHUNK
    encoder.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            for b in x.split(chunk):
                blocks = [f.float().cpu() for f in _gem_tta_blocks(encoder, b.to(device))]
                if readout == "gem_tta_pw":
                    xs.append(blocks[-1])
                else:
                    idx = cat_select([f.shape[1] for f in blocks])
                    if not idx:
                        raise ValueError(f"cat_tta_pw: no >=2-layer cat below {CAT_MAX_DIM}-d "
                                         f"(dims {[f.shape[1] for f in blocks]})")
                    xs.append(torch.cat([blocks[i] for i in idx], 1))
            ys.append(torch.as_tensor(y))
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


PW_READOUTS = ("gem_tta_pw", "cat_tta_pw")
COV_TTA_CHUNK = 8    # images per forward in the cov_head TTA readouts (8 views each)
COV_CHUNK = 32       # images per forward in cov_head (historical value)
COV_FIT_MICRO = None # CovReadout.fit micro-batch (None = 32); same sampled pixels either way
GEM_TTA_CHUNK = 16   # images per forward (x8 views) in gem_tta_pw / cat_tta_pw


def evaluate_checkpoint(ckpt_path, data, device, *, scaler=True, readout="head", C=100.0):
    from core.metrics import compute_metrics
    from core.svm import fit_svm, svm_protocol
    from sklearn.metrics import balanced_accuracy_score, f1_score
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc, head = ck["encoder"].to(device), ck["head"].to(device)
    if readout == "cov_head":
        Xtr, ytr, Xva, yva = _cov_features(enc, head, data, device, chunk=COV_CHUNK, fit_micro=COV_FIT_MICRO)
    elif readout == "cov_head_tta":            # same normalisation as cov_head (+ --scaler)
        Xtr, ytr, Xva, yva = _cov_features(enc, head, data, device, tta=8, chunk=COV_TTA_CHUNK, fit_micro=COV_FIT_MICRO)
    elif readout == "cov_head_tta_pw":         # pw_whiten + L2 replaces signed-sqrt/L2 + scaler
        Xtr, ytr, Xva, yva = _cov_features(enc, head, data, device, tta=8, pn=False, chunk=COV_TTA_CHUNK, fit_micro=COV_FIT_MICRO)
        Xtr, Xva = pw_whiten(Xtr, Xva)
        scaler = False
    elif readout in PW_READOUTS:                                     # R-SPF: no StandardScaler
        Xtr, ytr = _tta_features(enc, data["train"], device, readout)
        Xva, yva = _tta_features(enc, data["validation"], device, readout)
        Xtr, Xva = pw_whiten(Xtr, Xva)
        scaler = False
    else:
        Xtr, ytr = _features(enc, head, data["train"], device)
        Xva, yva = _features(enc, head, data["validation"], device)
    labels = np.unique(np.concatenate([ytr, yva]))
    remap = {int(c): i for i, c in enumerate(labels)}                # 0-indexed for torchmetrics
    ytr0, yva0 = np.vectorize(remap.get)(ytr), np.vectorize(remap.get)(yva)
    clf = fit_svm(Xtr, ytr0, C=float(C), scaler=scaler, tag="SVM_MODEHB")
    pred = clf.predict(Xva)
    m = compute_metrics(yva0, pred, num_classes=len(labels))
    per_class = f1_score(yva0, pred, labels=list(range(len(labels))), average=None, zero_division=0)
    return {"macro_f1": m["f1"], "kappa": m["kappa"],
            "balanced_acc": float(balanced_accuracy_score(yva0, pred)),
            "per_class": {f"f1_class{int(c)}": float(v) for c, v in zip(labels, per_class)},
            "svm_protocol": svm_protocol(clf), "embed_dim": int(Xtr.shape[1]),
            "n_train": int(len(ytr)), "n_validation": int(len(yva)), "readout": readout,
            "C": float(C), "scaler": bool(scaler), "fit_s": clf.fit_diagnostics_["svm_fit_seconds"],
            "genome": ck.get("genome"), "fidelity": ck.get("fidelity"), "cand_id": ck.get("cand_id")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="ga_search/config/modehb/search.yaml")
    ap.add_argument("--checkpoint", action="append", default=[], help="local encoder.pt (repeatable)")
    ap.add_argument("--artifact", action="append", default=[], help="W&B encoder artifact (repeatable)")
    ap.add_argument("--download-dir", default="artifacts/modehb/svm_downloads")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="split to score (test = reporting only)")
    ap.add_argument("--scaler", choices=["standard", "none"], default="standard")
    ap.add_argument("--readout", choices=["head", "cov_head", "cov_head_tta", "cov_head_tta_pw", *PW_READOUTS], default="head",
                    help="cov_head = second-order readout (use with --scaler standard --C 0.1); "
                         "gem_tta_pw / cat_tta_pw = discovery R-SPF ruler (use --C 100; scaler forced off)")
    ap.add_argument("--C", type=float, default=100.0)
    args = ap.parse_args(argv)
    try:
        os.nice(19)                                  # low priority: never compete with the search
    except OSError:
        pass
    if not (args.checkpoint or args.artifact):
        ap.error("give --checkpoint and/or --artifact")
    from ga_search.modehb.run_search import load_config
    from ga_search.progressive_data import load_data
    cfg = load_config(args.config, args.dataset)
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.update(num_workers=args.num_workers, seed=cfg["seed"])
    tr, va, _, ident = load_data(d, Path(cfg["output_dir"]) / "svm_data")
    data = {"train": tr, "validation": _test_loader(d) if args.split == "test" else va}
    device = torch.device(args.device)
    for path, parent, meta in _resolve(args):
        res = evaluate_checkpoint(path, data, device, scaler=args.scaler == "standard",
                                  readout=args.readout, C=args.C)
        res["split_eval"] = args.split
        parent = parent or res.get("cand_id") or Path(path).parent.parent.name
        print(json.dumps({"checkpoint": path, "parent_id": parent,
                          **{k: v for k, v in res.items() if k != "genome"}}, default=str))
        if args.no_wandb:
            continue
        from ga_search.modehb import wandb_logger as WL
        run = WL.ExternalEvalRun(parent, dataset=args.dataset, project=cfg["wandb"]["project"],
                                 entity=cfg["wandb"]["entity"],
                                 config={"checkpoint": path, "split_eval": args.split,
                                         "split_fit": "train", "data_sha256": ident["sha256"],
                                         "svm_protocol": res["svm_protocol"], "artifact_meta": meta,
                                         "readout": args.readout, "C": args.C,
                                         "genome": res["genome"], "never_feeds_fitness": True})
        run.add_result(macro_f1=res["macro_f1"], kappa=res["kappa"], balanced_acc=res["balanced_acc"],
                       per_class=res["per_class"], fidelity=meta.get("fidelity", res["fidelity"]),
                       generation=meta.get("generation"))
        print(f"[svm_on_demand] {run.run.url}")
        run.finish()


if __name__ == "__main__":
    main()
