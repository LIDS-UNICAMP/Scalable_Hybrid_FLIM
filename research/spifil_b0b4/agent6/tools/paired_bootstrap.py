"""Bootstrap PAREADO de dk = kappa(Ek) - kappa(E0) a partir dos preds/*.npz (auditoria CHECK C2).

Mesmo checkpoint, mesmas imagens, mesmo SVM: reamostra as imagens (com reposicao) B vezes e
recalcula os dois kappas na mesma amostra.  ``--exclude-ids`` (1 sample_id por linha) faz a
sensibilidade sem imagens suspeitas (ex. recapturas ``veredito_visual=mesmo`` do A1c).

    python paired_bootstrap.py --jsonl out/B-attn-E0E8-eggs/val.jsonl --out dk_eggs.csv \
        [--exclude-ids recapturas_eggs.txt] [--B 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-ids", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    excl = set()
    if a.exclude_ids:
        excl = {l.strip() for l in Path(a.exclude_ids).read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "preds_file" in r]
    by: dict = {}
    for r in rows:
        by.setdefault((r["checkpoint"], r["split_eval"]), {})[r["exp"]] = r
    out = []
    rng = np.random.default_rng(a.seed)
    for (ck, split), exps in sorted(by.items()):
        if "E0" not in exps:
            continue
        z0 = np.load(exps["E0"]["preds_file"])
        keep = np.array([s not in excl for s in z0["sample_id"]])
        y, p0 = z0["y"][keep], z0["pred"][keep]
        n = len(y)
        idx = rng.integers(0, n, size=(a.B, n))
        for e, r in sorted(exps.items()):
            if e == "E0":
                continue
            z = np.load(r["preds_file"])
            assert (z["sample_id"] == z0["sample_id"]).all(), "imagens nao pareadas"
            pk = z["pred"][keep]
            d = cohen_kappa_score(y, pk) - cohen_kappa_score(y, p0)
            bs = np.array([cohen_kappa_score(y[i], pk[i]) - cohen_kappa_score(y[i], p0[i]) for i in idx])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append({"checkpoint": ck, "split": split, "exp": e, "n": n, "n_excluidas": int((~keep).sum()),
                        "dk": d, "ic95_lo": lo, "ic95_hi": hi, "ic_acima_de_0": bool(lo > 0)})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["vazio"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} comparacoes -> {a.out}")


if __name__ == "__main__":
    main()
