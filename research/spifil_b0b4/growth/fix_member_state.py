"""Correção do growth_state.json (hotfix member_fid): membro da faixa promovido com f_high quando f_low foi MELHOR
pela validação (ORIG-val) -> o membro passa a ser o f_low (checkpoint, sha, kappa, alpha, fid/selected_fid).

* Lê só result.json de VALIDAÇÃO (nodes/<nó>/f<fl>/ e f<fh>/); o teste nunca entra.
* Recusa rodar com o controlador vivo (flock em <out>/controller.lock).
* Arquiva o estado antes (growth_state.json.bak_memberfix_<ts>), registra cada troca em member_fix_log.jsonl e no
  próprio membro (``member_fix``). Idempotente: membro que já aponta para o checkpoint de f_low é pulado.
* NÃO refaz filhos: quem já cresceu do f_high continua na faixa seguinte (o pai mantém o node_id).
* Enfileira o novo checkpoint no canal de teste (test_queue.jsonl, never_feeds_fitness).

    python research/spifil_b0b4/growth/fix_member_state.py --out <dir> [--out <dir> ...]           # dry-run
    python research/spifil_b0b4/growth/fix_member_state.py --out <dir> --apply
    python research/spifil_b0b4/growth/fix_member_state.py --glob 'artifacts/hmodehb_b0b4/growth_v3/*/*/seed42' --apply
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

FIELDS = ("kappa_val_orig", "checkpoint", "checkpoint_sha256", "params", "flops_2mac", "theta")


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _atomic_write(p: Path, obj) -> None:
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def plan(out: Path) -> tuple:
    st = _read(out / "growth_state.json")
    if st is None:
        return None, [], "sem growth_state.json"
    g = (st.get("config_full") or {}).get("growth") or {}
    fl, fh = int(g.get("fid_low", 15)), int(g.get("fid_high", 45))
    if fl == fh:
        return st, [], f"fid_low == fid_high ({fl}): nada a corrigir"
    swaps = []
    for band, mems in (st.get("members") or {}).items():
        for i, m in enumerate(mems):
            if m.get("adopted_as_parent") or int(m.get("fid", 0) or 0) == 0:
                continue                                   # sementes e adotados por saturação
            lo = _read(out / "nodes" / m["node_id"] / f"f{fl}" / "result.json")
            hi = _read(out / "nodes" / m["node_id"] / f"f{fh}" / "result.json")
            if not lo or lo.get("status") != "ok" or lo.get("kappa_val_orig") is None:
                continue
            if m.get("checkpoint") == lo.get("checkpoint"):
                continue                                   # já é o f_low (idempotente / retomada do fix6)
            k_lo = float(lo["kappa_val_orig"])
            k_hi = (float(hi["kappa_val_orig"]) if hi and hi.get("status") == "ok" and hi.get("kappa_val_orig")
                    is not None else None)
            k_mem = float(m["kappa_val_orig"])
            # regra do fix6: f_high só perde se kappa_val(f_high) < kappa_val(f_low) (empate fica com f_high)
            if k_hi is not None and not (k_hi < k_lo):
                continue
            if k_hi is None and not (k_mem < k_lo):
                continue
            if lo.get("band") is not None and lo["band"] != band:
                swaps.append({"band": band, "index": i, "node_id": m["node_id"], "skip": f"f_low na faixa {lo['band']}"})
                continue
            swaps.append({"band": band, "index": i, "node_id": m["node_id"], "kappa_member": k_mem,
                          "kappa_f_high": k_hi, "kappa_f_low": k_lo, "wise_alpha_f_low": lo.get("wise_alpha"),
                          "old": {k: m.get(k) for k in FIELDS + ("fid", "selected_fid")},
                          "new": {**{k: lo.get(k) for k in FIELDS if lo.get(k) is not None}, "fid": fl,
                                  "selected_fid": fl}})
    return st, swaps, None


def apply(out: Path, st: dict, swaps: list) -> int:
    real = [s for s in swaps if "skip" not in s]
    if not real:
        return 0
    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(out / "growth_state.json", out / f"growth_state.json.bak_memberfix_{ts}")
    for s in real:
        m = st["members"][s["band"]][s["index"]]
        m.update(s["new"])
        m["member_fix"] = {"time": ts, "rule": "hotfix member_fid: f_low > f_high pela ORIG-val",
                           "old": s["old"], "kappa_f_low": s["kappa_f_low"], "kappa_f_high": s["kappa_f_high"]}
    st["updated"] = time.time()
    st.setdefault("member_fix_log", []).append({"time": ts, "n": len(real), "nodes": [s["node_id"] for s in real]})
    _atomic_write(out / "growth_state.json", st)
    with (out / "member_fix_log.jsonl").open("a") as f:
        for s in real:
            f.write(json.dumps({**s, "time": ts, "used_for_selection": "validation_only"}, default=str) + "\n")
    with (out / "test_queue.jsonl").open("a") as f:      # canal de teste (never_feeds_fitness)
        for s in real:
            m = st["members"][s["band"]][s["index"]]
            f.write(json.dumps({"node_id": m["node_id"], "checkpoint": m["checkpoint"],
                                "checkpoint_sha256": m["checkpoint_sha256"], "band": m["band"], "fid": m["fid"],
                                "never_feeds_fitness": True}) + "\n")
    return len(real)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", action="append", default=[])
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--apply", action="store_true", help="sem isto: dry-run (só lista)")
    a = ap.parse_args(argv)
    outs = [Path(p) for p in a.out] + [Path(p) for gpat in a.glob for p in sorted(glob.glob(gpat))]
    rc = 0
    for out in outs:
        lockf = (out / "controller.lock")
        fh = None
        try:
            if lockf.exists():
                fh = lockf.open("a")
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # controlador vivo -> recusa
        except BlockingIOError:
            print(f"[fix] {out}: controlador VIVO (lock); pare-o antes. Pulado.")
            rc = 3
            continue
        try:
            st, swaps, why = plan(out)
            if why:
                print(f"[fix] {out}: {why}")
                continue
            if not swaps:
                print(f"[fix] {out}: nada a trocar")
                continue
            for s in swaps:
                if "skip" in s:
                    print(f"[fix] {out}: {s['band']} {s['node_id']}: PULADO ({s['skip']})")
                else:
                    print(f"[fix] {out}: {s['band']} {s['node_id']}: membro {s['kappa_member']:.4f} (f_high "
                          f"{s['kappa_f_high']}) -> f_low {s['kappa_f_low']:.4f}  ckpt {s['old']['checkpoint']} -> "
                          f"{s['new'].get('checkpoint')}")
            if a.apply:
                n = apply(out, st, swaps)
                print(f"[fix] {out}: {n} troca(s) aplicada(s); estado arquivado em growth_state.json.bak_memberfix_*")
            else:
                print(f"[fix] {out}: DRY-RUN (use --apply)")
        finally:
            if fh is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
