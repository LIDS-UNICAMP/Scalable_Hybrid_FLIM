"""Compatibilidade dos checkpoints entre versoes de torch (auditoria CHECK C7).

Os ``encoder.pt`` sao pickles de modulo inteiro (``weights_only=False``) salvos com torch 2.7.1 na
Jaci; o athena roda torch 2.2.2.  Para CADA checkpoint da lista: carrega, roda E0 =
``head(encoder(x))`` numa entrada sintetica FIXA (seed 0, 4 x 3 x 224 x 224 em [0,1]) em fp32
com TF32 desligado e grava (``--write-ref``) ou compara (``--compare``) com a referencia.
Criterio: cos(emb, ref) >= 0.9999 por imagem e erro relativo L2 <= 1e-3; finito; dimensao igual.
``--expect-gpu-uuid``: confere no nvidia-smi que ESTE processo esta na GPU esperada.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # ga_search (pickles de modulo inteiro)


def key(r):
    return f"{r['dataset']}|{r['kind']}|{r['stratum']}|{r['cand']}|{r['fid']}"


def gpu_of_this_process() -> str | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
        for line in q.splitlines():
            pid, uuid = [t.strip() for t in line.split(",")]
            if int(pid) == os.getpid():
                return uuid
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", default=None, help="substitui o prefixo 'artifacts/' (ex. staging do athena)")
    ap.add_argument("--device", default="cpu")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-ref")
    g.add_argument("--compare")
    ap.add_argument("--expect-gpu-uuid", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cos-min", type=float, default=0.9999)
    ap.add_argument("--rel-max", type=float, default=1e-3)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    x = torch.rand(4, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    rows = list(csv.DictReader(open(args.csv)))
    ref = dict(np.load(args.compare)) if args.compare else {}
    embs, res, bad = {}, [], 0
    uuid_seen = None
    for r in rows:
        ck = r["checkpoint"]
        path = Path(args.root) / ck.removeprefix("artifacts/") if args.root else Path(ck)
        rec = {"key": key(r), "path": str(path), "torch": torch.__version__, "host": platform.node()}
        try:
            c = torch.load(path, map_location=dev, weights_only=False)
            enc, head = c["encoder"].to(dev).eval(), c["head"].to(dev).eval()
            with torch.no_grad():
                e = head(enc(x.to(dev))).float().cpu().numpy()
            if dev.type == "cuda" and uuid_seen is None:
                uuid_seen = gpu_of_this_process()
            rec.update(dim=int(e.shape[1]), finite=bool(np.isfinite(e).all()))
            embs[key(r)] = e
            if args.compare:
                rr = ref.get(key(r))
                if rr is None:
                    rec.update(ok=False, why="sem referencia")
                else:
                    cos = (e * rr).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(rr, axis=1) + 1e-12)
                    rel = float(np.linalg.norm(e - rr) / (np.linalg.norm(rr) + 1e-12))
                    rec.update(cos_min=float(cos.min()), rel_l2=rel,
                               ok=bool(rec["finite"] and e.shape == rr.shape and cos.min() >= args.cos_min
                                       and rel <= args.rel_max))
            else:
                rec["ok"] = rec["finite"]
        except Exception as ex:                      # noqa: BLE001
            rec.update(ok=False, why=repr(ex))
        bad += not rec["ok"]
        res.append(rec)
        print(json.dumps(rec), flush=True)
    if args.write_ref:
        np.savez(args.write_ref, **embs)
    summary = {"n": len(rows), "falhas": bad, "torch": torch.__version__, "device": str(dev),
               "gpu_uuid_do_processo": uuid_seen, "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER")}
    if args.expect_gpu_uuid:
        summary["gpu_ok"] = bool(uuid_seen and uuid_seen.startswith(args.expect_gpu_uuid))
        bad += not summary["gpu_ok"]
    print(json.dumps({"resumo": summary}), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in res + [{"resumo": summary}]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
