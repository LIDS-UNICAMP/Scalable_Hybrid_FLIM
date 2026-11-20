"""E0–E8 da atencao fixa nos MESMOS checkpoints (sem treinar). NUNCA alimenta a fitness.

    CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 \
    python research/spifil_b0b4/agent6/tools/eval_fixed_attention.py \
        --checkpoint-list research/spifil_b0b4/agent6/lists/remote_sensing.txt \
        --experiments E0,E1,E2,E3,E4,E5,E6,E7,E8 --split validation --out <dir>/val.jsonl

Protocolo (adendo + R1/R2 + auditoria CHECK gate B-attn):
* dados = ``config_resolved.yaml`` da propria busca do checkpoint (mesmo split/pct/resolucao);
* embedding = cabeca original (>= 1280 d), com o operador fixo inserido (``fixed_attention``);
* treino do SVM = subconjunto ESTRATIFICADO ``max_train`` da busca (regua ``ORIG_sub<max_train>``,
  diferente do ORIG historico com o treino inteiro);
* PCA/SVD/z-score do alinhamento: ajustados SO em ``--fit-n-img`` imagens de treino SORTEADAS
  uniformemente (seed fixa, sem rotulo; C1) -- indices gravados na linha;
* SVM = ``core.svm.fit_svm`` linear, C=100, SEM scaler;
* por linha: ``sample_id``, ``y`` e ``pred`` de cada imagem num ``.npz`` lateral com md5 (C2),
  versoes de torch/sklearn, host e UUID/PCI da GPU (C6);
* escolha (melhor de E0–E5, depois TTA) SO pela validacao (``--split validation``);
* ``--split test`` so relata uma configuracao ja congelada (``--frozen-from``): linhas marcadas
  ``split_eval=test, used_for_selection=false, reporting_only=true``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from ga_search.modehb import fixed_attention as FA  # noqa: E402

TTA_GEOMETRY = {"flowers": "flip_scale", "food101": "flip_scale"}   # demais: rot_flip


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_info(device: torch.device) -> dict:
    """Ambiente gravado em CADA linha (C6)."""
    import sklearn
    info = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "code_md5": {"fixed_attention.py": md5_file(FA.__file__),
                         "eval_fixed_attention.py": md5_file(__file__)}}
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        info.update(gpu_name=p.name, gpu_total_gb=round(p.total_memory / 2 ** 30, 1),
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        uuid = getattr(p, "uuid", None)
        info["gpu_uuid"] = f"GPU-{uuid}" if uuid is not None and not str(uuid).startswith("GPU-") else (
            str(uuid) if uuid is not None else None)
        # fallback / PCI: nvidia-smi na mesma ordem fisica (exige CUDA_DEVICE_ORDER=PCI_BUS_ID)
        try:
            vis = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
            phys = vis[idx].strip() if idx < len(vis) and vis[idx].strip() else str(idx)
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20).stdout
            for line in q.splitlines():
                i, u, b = [t.strip() for t in line.split(",")]
                if i == phys or u == phys:
                    info.setdefault("gpu_uuid", u)
                    info["gpu_uuid_nvsmi"], info["gpu_pci_bus_id"], info["gpu_physical_index"] = u, b, i
        except Exception as e:                       # noqa: BLE001 -- so diagnostico
            info["gpu_query_error"] = repr(e)
    return info


def _data(ckpt: Path, split: str, num_workers: int, cache: Path):
    from ga_search.progressive_data import load_data
    from ga_search.modehb.svm_on_demand import _test_loader
    state = ckpt.parents[3]                       # .../state/seed42
    cfg = yaml.safe_load((state / "config_resolved.yaml").read_text())
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.update(num_workers=num_workers, seed=cfg["seed"])
    tr, va, _, ident = load_data(d, cache)
    ev = _test_loader(d) if split == "test" else va
    return cfg, tr, ev, ident


def _tensors(loader, idx=None):
    """(imagens, rotulos, sample_ids) de um loader em ordem fixa; ``idx`` = subconjunto."""
    if idx is None:
        xs, ys, ids = [], [], []
        for x, y, *rest in loader:
            xs.append(x), ys.append(torch.as_tensor(y))
            ids += [str(n) for n in rest[0]] if rest else [None] * len(x)
        return torch.cat(xs), torch.cat(ys).numpy(), ids
    from torch.utils.data import DataLoader, Subset
    sub = DataLoader(Subset(loader.dataset, [int(i) for i in idx]), batch_size=loader.batch_size,
                     num_workers=loader.num_workers, shuffle=False)
    return _tensors(sub)


def _svm_train_idx(loader, cfg, override):
    """Mesmo subconjunto ESTRATIFICADO do treino que a fitness da busca usa
    (``run_search._stratified_indices``, max_train da config, seed da busca)."""
    from ga_search.modehb.run_search import _stratified_indices
    ys = np.asarray([int(s.label) for s in loader.dataset.samples])
    mx = override if override is not None else int(
        ((cfg.get("eval") or {}).get("svm_fitness") or {}).get("max_train", 10000))
    return _stratified_indices(ys, mx, int(cfg["seed"])), mx, len(ys)


def _svm(Xtr, ytr, Xev, yev):
    from core.metrics import compute_metrics
    from core.svm import fit_svm, svm_protocol
    labels = np.unique(np.concatenate([ytr, yev]))
    remap = {int(c): i for i, c in enumerate(labels)}
    ytr0, yev0 = np.vectorize(remap.get)(ytr), np.vectorize(remap.get)(yev)
    t0 = time.perf_counter()
    clf = fit_svm(Xtr, ytr0, C=100.0, scaler=False, tag="SVM_ATTN6")
    pred0 = clf.predict(Xev)
    m = compute_metrics(yev0, pred0, num_classes=len(labels))
    pred = labels[np.asarray(pred0, dtype=int)]                   # de volta aos rotulos originais
    return {"kappa": float(m["kappa"]), "macro_f1": float(m["f1"]), "svm_fit_s": time.perf_counter() - t0,
            "svm_protocol": svm_protocol(clf)}, pred


def _save_preds(pred_dir: Path, ck_path: Path, exp: str, split: str, ids, y, pred) -> dict:
    """npz por linha com sample_id, y, pred (bootstrap pareado sem refazer GPU; C2)."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(str(ck_path).encode()).hexdigest()[:10]
    f = pred_dir / f"{tag}_{exp}_{split}.npz"
    np.savez(f, sample_id=np.asarray(ids, dtype=str), y=np.asarray(y), pred=np.asarray(pred))
    return {"preds_file": str(f), "preds_md5": md5_file(f), "preds_n": int(len(y))}


def run_one(enc, head, genes, Xtr, ytr, ids_tr, Xev, yev, *, tta, geometry, tta_agg, bs, device,
            fit_n_img, fit_n_px, fit_seed, mem_budget):
    emb, info, notes = FA.build_embedder(enc, head, dict(genes), Xtr[:1].to(device), batch_size=bs,
                                         mem_budget_bytes=mem_budget,
                                         n_fit_samples=fit_n_img * fit_n_px)
    t0 = time.perf_counter()
    emb.fit(Xtr, bs=bs, n_img=fit_n_img, n_px=fit_n_px, seed=fit_seed, device=device)
    fit_s = time.perf_counter() - t0
    fit_idx = list(getattr(emb, "fit_indices", []))
    fit_ids = [ids_tr[i] for i in fit_idx]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    ftr = emb.embed(Xtr, bs=bs, device=device, tta=tta, geometry=geometry, tta_agg=tta_agg).numpy()
    fev = emb.embed(Xev, bs=bs, device=device, tta=tta, geometry=geometry, tta_agg=tta_agg).numpy()
    emb_s = time.perf_counter() - t0
    res, pred = _svm(ftr, ytr, fev, yev)
    res.update(embed_dim=int(ftr.shape[1]), finite=bool(np.isfinite(ftr).all() and np.isfinite(fev).all()),
               attn_fit_s=fit_s, embed_s=emb_s, ms_per_image=1e3 * emb_s / (len(ftr) + len(fev)),
               genes=info["genes"], cost=info["cost"], repair_notes=notes,
               align_degenerate=bool(info["cost"].get("align_degenerate", False)),
               fitted_statistics=emb.fitted_statistics, tta=tta, tta_geometry=geometry,
               tta_agg=tta_agg, tta_cost=FA.tta_cost(tta, geometry),
               attn_trainable_params=int(sum(p.numel() for p in emb.parameters())),
               align_fit_seed=fit_seed if fit_idx else None,
               align_fit_indices=fit_idx,
               align_fit_ids_md5=hashlib.md5("\n".join(map(str, fit_ids)).encode()).hexdigest() if fit_ids else None,
               align_fit_n_classes=int(len(np.unique(ytr[fit_idx]))) if fit_idx else None)  # so diagnostico
    if device.type == "cuda":
        res["peak_cuda_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    return res, pred


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--checkpoint-list", default=None, help="arquivo com 1 checkpoint por linha")
    ap.add_argument("--experiments", default="E0,E1,E2,E3,E4,E5")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--plumbing-check", action="store_true",
                    help="exercita o caminho --split test com o loader de VALIDACAO (nunca le o teste)")
    ap.add_argument("--frozen-from", default=None,
                    help="jsonl de validacao: com --split test so relata o vencedor congelado")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--fit-n-img", type=int, default=160)
    ap.add_argument("--fit-n-px", type=int, default=256)
    ap.add_argument("--fit-seed", type=int, default=0, help="seed do sorteio uniforme do ajuste (C1)")
    ap.add_argument("--tta-agg", choices=FA.TTA_AGGS, default="mean")
    ap.add_argument("--mem-budget-gb", type=float, default=8.0)
    ap.add_argument("--svm-max-train", type=int, default=None,
                    help="None = max_train da config da busca (subconjunto estratificado, seed da busca)")
    ap.add_argument("--max-eval", type=int, default=None, help="SO para smoke (registrado)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 declarado (sem TF32)
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = out.parent / "preds"
    cache = Path(args.cache or out.parent / "data_cache")
    env = env_info(device)
    print(f"[attn6] env {json.dumps(env, default=str)}", flush=True)
    frozen = {}
    if args.split == "test":
        if not args.frozen_from:
            ap.error("--split test exige --frozen-from (selecao feita so na validacao)")
        for line in Path(args.frozen_from).read_text().splitlines():
            r = json.loads(line)
            if r.get("selected_by_validation"):
                frozen[r["checkpoint"]] = r
    if args.checkpoint_list:
        args.checkpoint += [l.strip() for l in Path(args.checkpoint_list).read_text().splitlines()
                            if l.strip() and not l.startswith("#")]
    data_mem: dict = {}
    split_tag = "validation_plumbing" if args.plumbing_check else args.split
    done: dict = {}                                   # retomada idempotente
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                              # linha truncada por queda
            if r.get("split_eval") != split_tag:
                continue
            key = "SELECTED" if r.get("selected_by_validation") else r["exp"]
            done.setdefault(r["checkpoint"], {})[key] = r
    for ck_path in args.checkpoint:
        if "SELECTED" in done.get(str(Path(ck_path)), {}) and args.split == "validation":
            print(f"[attn6] {ck_path}: completo (linha selected_by_validation); pulando", flush=True)
            continue
        if args.split == "test" and done.get(str(Path(ck_path))):
            print(f"[attn6] {ck_path}: teste ja relatado; pulando", flush=True)
            continue
        prev = done.get(str(Path(ck_path)), {})
        ck_path = Path(ck_path)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        enc, head = ck["encoder"].to(device).eval(), ck["head"].to(device).eval()
        load_split = "validation" if args.plumbing_check else args.split
        cfg, tr, ev, ident = _data(ck_path, load_split, args.num_workers, cache)
        tr_idx, mx, n_train_total = _svm_train_idx(tr, cfg, args.svm_max_train)
        dkey = (ident["sha256"], load_split, mx)
        if dkey not in data_mem:                      # 1 decodificacao por dataset no job
            data_mem.clear()
            data_mem[dkey] = (*_tensors(tr, tr_idx), *_tensors(ev))
        Xtr, ytr, ids_tr, Xev, yev, ids_ev = data_mem[dkey]
        if args.max_eval:
            Xev, yev, ids_ev = Xev[: args.max_eval], yev[: args.max_eval], ids_ev[: args.max_eval]
        dataset = cfg["data"]["dataset"]
        geom = TTA_GEOMETRY.get(dataset, "rot_flip")
        ruler = f"ORIG_sub{mx}" if n_train_total > mx else "ORIG_full"
        base = {"checkpoint": str(ck_path), "checkpoint_md5": md5_file(ck_path), "dataset": dataset,
                "cand_id": ck.get("cand_id"), "fidelity": ck.get("fidelity"), "split_fit": "train",
                "split_eval": split_tag, "data_sha256": ident["sha256"],
                "n_train": int(len(ytr)), "n_train_total": int(n_train_total), "n_eval": int(len(yev)),
                "svm_max_train": mx, "ruler": ruler, "selection_ruler_of_checkpoint": "cov_head (busca)",
                "smoke_subsample": bool(args.max_eval), "precision": "fp32",
                "svm": "core.svm.fit_svm linear C=100 scaler=False",
                "used_for_selection": args.split == "validation" and not args.plumbing_check,
                "never_feeds_fitness": True, "device": str(device), "env": env}
        todo = []
        if args.split == "test":
            fr = frozen.get(str(ck_path))
            if fr is None:
                print(f"[attn6] sem configuracao congelada para {ck_path}; pulando", flush=True)
                continue
            # relato: E0 (controle) + vencedor sem TTA + vencedor congelado (mesmo protocolo)
            todo = [("E0", FA.E_PRESETS["E0"], "none")]
            if fr["genes"].get("attention_type", "none") != "none":
                todo.append((fr.get("base_exp", fr["exp"]) + "_noTTA", fr["genes"], "none"))
            if fr["tta"] != "none" or fr["genes"].get("attention_type", "none") != "none":
                todo.append((fr["exp"] + "_frozen", fr["genes"], fr["tta"]))
        else:
            for e in exps:
                if e in FA.E_PRESETS:
                    todo.append((e, FA.E_PRESETS[e], "none"))
        kw = dict(geometry=geom, tta_agg=args.tta_agg, bs=args.bs, device=device,
                  fit_n_img=args.fit_n_img, fit_n_px=args.fit_n_px, fit_seed=args.fit_seed,
                  mem_budget=args.mem_budget_gb * 2 ** 30)
        rows = []
        with out.open("a") as fh:
            def emit(e, r, pred, extra=None):
                row = {**base, "exp": e, **(extra or {}), **r,
                       **_save_preds(pred_dir, ck_path, e, split_tag, ids_ev, yev, pred)}
                if args.split == "test":
                    row.update(used_for_selection=False, reporting_only=True)
                rows.append(row)
                fh.write(json.dumps(row, default=str) + "\n"), fh.flush()
                print(f"[attn6] {dataset} {ck.get('cand_id')} {e} tta={r['tta']} kappa={r['kappa']:.4f} "
                      f"{r['ms_per_image']:.2f} ms/img notes={r['repair_notes']}", flush=True)
                return row
            for e, genes, tta in todo:
                if e in prev:
                    rows.append(prev[e])
                    continue
                r, pred = run_one(enc, head, genes, Xtr, ytr, ids_tr, Xev, yev, tta=tta, **kw)
                emit(e, r, pred)
            # E6–E8: melhor de E0–E5 pela VALIDACAO, com TTA
            tta_exps = [e for e in exps if e in FA.E_TTA]
            if args.split == "validation" and rows:
                best = max(rows, key=lambda r: (r["kappa"], -FA.ATTENTION_TYPES.index(
                    r["genes"]["attention_type"])))
                sel = dict(best)
                for e in tta_exps:
                    if e in prev and prev[e].get("base_exp") == best["exp"]:
                        r = prev[e]
                        if r["kappa"] > sel["kappa"]:
                            sel = dict(r)
                        continue
                    r, pred = run_one(enc, head, best["genes"], Xtr, ytr, ids_tr, Xev, yev,
                                      tta=FA.E_TTA[e], **kw)
                    row = emit(e, r, pred, {"base_exp": best["exp"]})
                    if r["kappa"] > sel["kappa"]:
                        sel = dict(row)
                fh.write(json.dumps({**base, "exp": sel["exp"], "base_exp": sel.get("base_exp", sel["exp"]),
                                     "genes": sel["genes"], "tta": sel["tta"],
                                     "kappa": sel["kappa"], "selected_by_validation": True},
                                    default=str) + "\n")


if __name__ == "__main__":
    main()
