# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Avaliação avulsa de UM nó da cadeia de crescimento que morreu por OOM de contenção (failed, reason "oom",
sem checkpoint), sem reabrir a geração.

* Usa o código do clone DONO do --out (o primeiro ancestral com ga_search/hmodehb/growth_chain.py).
* Refaz o nó pelo próprio GrowthChain._job (mesma cand_key, ops, theta, genes, fid, pai e fitness_sig): a
  cand_key é recalculada do estado e tem de bater com a gravada, senão aborta.
* Arquiva o result.json antigo (result.json.oom_<ts>) e grava o novo no lugar, com retry_of=<ts>. NÃO toca
  growth_state.json, membros nem pais. No lineage.jsonl só ACRESCENTA um registro novo (mesma cand_key,
  retry_of=<ts>): quem lê o lineage usa o ÚLTIMO registro de cada cand_key.
* Geração fechada: roda mesmo com a cadeia viva (ela nunca revisita gerações fechadas).
  Geração aberta: só com a cadeia parada (segura o controller.lock); no resume a cadeia adota o resultado novo.

    python research/spifil_b0b4/growth/retry_oom_node.py --out <out> --node B3g0c00 --fid 15 --dry-run
    python research/spifil_b0b4/growth/retry_oom_node.py --out <out> --node B3g0c00 --fid 15
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def _record(G, I, ch, out, ndir, r, parent, ops, fid, g, ts, closed, code_version=None):
    """retry_of (+ code_version) no result.json + registro NOVO no lineage.jsonl. Append direto (não ch._lineage):
    o growth/code descarta cand_key repetida e não pode ser mudado."""
    r["retry_of"] = ts
    if code_version:
        r["code_version"] = code_version
    I.atomic_write_json(ndir / "result.json", G._jsonable(r))
    decision, ok = G.decide(r, g["delta"], g.get("require_improvement", False))
    lin = out / "lineage.jsonl"
    fam = None
    for line in (lin.read_text().splitlines() if lin.exists() else []):
        with contextlib.suppress(Exception):
            x = json.loads(line)
            fam = x.get("family") if x.get("cand_key") == r["cand_key"] else fam
    rec = ch._lin_rec(r, parent, ops, fid, ok, final=(int(g["fid_low"]) == int(g["fid_high"]))
                      or decision == "adopted_as_parent", decision=decision, family=fam)
    rec["retry_of"] = ts
    if code_version:
        rec["code_version"] = code_version
    if closed:
        rec["used_for_selection"] = "never (avaliação avulsa: geração já fechada)"
    with lin.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--fid", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    out = Path(os.path.abspath(a.out))     # sem resolve(): growth_v3/code/artifacts é symlink para growth/code
    root = next((p for p in out.parents if (p / "ga_search/hmodehb/growth_chain.py").is_file()), None)
    if root is None:
        sys.exit(f"nenhum clone com ga_search/hmodehb/growth_chain.py acima de {out}")
    os.chdir(root)                         # checkpoints e output_dir do estado são relativos ao clone
    sys.path.insert(0, str(root))          # o código do clone dono do --out, não o deste arquivo
    from ga_search.hmodehb import growth_chain as G
    from ga_search.hmodehb import integrity as I
    gc = Path(G.__file__)                  # versão do código que treina o nó (pode diferir da dos irmãos)
    code_version = {"growth_chain_md5": hashlib.md5(gc.read_bytes()).hexdigest(), "path": str(gc),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(gc.stat().st_mtime))}

    st = I.read_json(out / "growth_state.json")
    g, cfg = st["config_full"]["growth"], st["config_full"]["run"]
    if set(g) - set(G.GROWTH_DEFAULTS):    # o run usou chaves que este código não conhece: clone errado
        sys.exit(f"{G.__file__} não conhece {sorted(set(g) - set(G.GROWTH_DEFAULTS))}: passe o --out pelo clone dono")
    if set(G.GROWTH_DEFAULTS) - set(g):
        print(f"[aviso] código mais novo que o run: {sorted(set(G.GROWTH_DEFAULTS) - set(g))}", flush=True)
    ndir = out / "nodes" / a.node / f"f{a.fid}"
    old = I.read_json(ndir / "result.json")
    if not old or old.get("status") != "failed" or old.get("reason") != "oom":
        sys.exit(f"{ndir}/result.json não é failed/oom: {old and (old.get('status'), old.get('reason'))}")
    if a.fid != int(g["fid_low"]):
        sys.exit(f"fid {a.fid} != fid_low {g['fid_low']}: promoção (retomada) fora do escopo")
    target, gen, i = re.fullmatch(r"(B\d)g(\d+)c(\d+)", a.node).groups()
    band = G.BANDS[G.BANDS.index(target) - 1]
    tag = f"{band}:{gen}"
    parent = next(m for m in st["members"][band] if m["node_id"] == old["parent"])
    seed = G._h({"tag": tag, "i": int(i), "parent": parent["checkpoint_sha256"]}) % (2 ** 31 - 1)
    origin = {"kind": "inherit", "parent_sha256": parent["checkpoint_sha256"], "protect": g["protect"],
              "init": g["init_method"], "init_config": g.get("init_config") or {}}
    genes = old.get("genes") or {}
    if genes:
        origin["genes"] = genes
    ckey = I.cand_key(I.lineage_key({"ops": old["ops"], "theta": old["theta"]}, seed, origin, st["fitness_sig"]),
                      a.fid)
    closed = tag in st["gens_done"]
    print(json.dumps({"code": G.__file__, "node": a.node, "fid": a.fid, "gen": tag,
                      "geracao": "fechada" if closed else "aberta", "cand_key": ckey,
                      "cand_key_ok": ckey == old["cand_key"], "parent": parent["node_id"],
                      "ops": "+".join(o["op"] for o in old["ops"]), "genes": genes, "seed": seed,
                      "fitness_sig": st["fitness_sig"], "code_version": code_version}, ensure_ascii=False), flush=True)
    if ckey != old["cand_key"]:
        sys.exit(f"cand_key recalculada {ckey} != gravada {old['cand_key']}: abortado")
    if a.dry_run:
        return 0

    with (contextlib.nullcontext() if closed else I.ControllerLock(out)):   # aberta: cadeia tem de estar parada
        gev = G.GrowthEvaluator(cfg, out / "eval", g.get("debug_train"), g.get("debug_val"),
                                **({"v3": g.get("v3")} if "v3" in g else {}))
        if gev.fitness_sig != st["fitness_sig"]:
            sys.exit(f"fitness_sig {gev.fitness_sig} != estado {st['fitness_sig']}: régua mudou, abortado")
        ch = G.GrowthChain(cfg, g, gev, out)
        ts = time.strftime('%Y%m%d_%H%M%S')
        (ndir / "result.json").rename(ndir / f"result.json.oom_{ts}")
        r = ch._job(parent, old["ops"], old["theta"], a.fid, a.node, seed, ckey, **({"genes": genes} if genes else {}))
        _record(G, I, ch, out, ndir, r, parent, old["ops"], a.fid, g, ts, closed, code_version)
    print(json.dumps({k: r.get(k) for k in ("status", "reason", "kappa_val_orig", "delta_kappa", "band", "params")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
