# lista o que a Jaci tem e ainda NAO tem teste completo (100%: gap+so2+so2_tta8; top-20: % com tta8)
import glob, json, os, re, sys, collections
from pathlib import Path
HERE = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/readout_x")
MIR = Path("/dados/home/moliveira/exps/jaci_results_mirror/readout_x_results")
JA, AA = "/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/", "/dados/home/moliveira/artifacts/"
GROW = "/dados/home/moliveira/exps/growth/code/artifacts/"
TTA, H3 = "so2_4096_pca_tta8", {"gap1280", "so2_4096_pca", "so2_4096_pca_tta8"}
PCTS = [1, 5, 10, 25, 50, 75, 100]
inv = json.load(open(sys.argv[1]))
heads, pct = collections.defaultdict(set), collections.defaultdict(set)
md5h = collections.defaultdict(set)
files = [*MIR.glob("*.jsonl"), *(HERE / "results").glob("*.jsonl"), *(HERE / "results/all_untested_parts").glob("*.jsonl")]
for f in files:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        i = r.get("backbone_id", ""); m = re.search(r"_p(\d+)$", i)
        if "pct" in r or m:
            p = int(r.get("pct") or m.group(1)); base = re.sub(r"_p\d+$", "", i)
            if r.get("head") == TTA: pct[base].add(p)
        else:
            heads[i].add(r.get("head")); md5h[r.get("md5")].add(r.get("head"))
old = set()
for l in open("/dados/home/moliveira/Scalable_Hybrid_FLIM/ga_search/modehb/reports/test_svm_C100.jsonl"):
    try: r = json.loads(l); old.add((r["run"].rstrip("/").split("/")[-1], r["dataset"], r["cand_id"]))
    except Exception: pass
# top-20 por base (fonte ga) sobre Jaci U athena (modehb_*), por kval
ath = {}
for f in glob.glob(AA + "modehb_*/*/state/seed42/candidates/c*/f*/result.json"):
    p = Path(f); run, ds, c, fd = p.parts[-8], p.parts[-7], p.parts[-3], p.parts[-2]
    if not fd[1:].isdigit() or not (p.parent / "encoder.pt").is_file(): continue
    try: r = json.loads(p.read_text())
    except Exception: continue
    if r.get("status") != "ok": continue
    k = (run, ds, c)
    if k not in ath or int(fd[1:]) > ath[k][0]: ath[k] = (int(fd[1:]), r.get("fitness_kappa") or 0)
allm = {(x["run"], x["ds"], x["cand"]): (x["fid"], x["kval"]) for x in inv if x["kind"] == "modehb"}
for k, v in ath.items():
    if k not in allm or v[0] > allm[k][0]: allm[k] = v
top = set()
for ds in {k[1] for k in allm}:
    q = sorted(((v[1], k) for k, v in allm.items() if k[1] == ds and k not in old), reverse=True)[:20]
    top |= {k for _, k in q}
rows = []
for x in inv:
    if x["kind"] != "modehb": continue
    k = (x["run"], x["ds"], x["cand"])
    if allm[k][0] != x["fid"]: continue      # athena tem fidelidade maior: o runner usa a de la
    x["old"] = k in old
    x["h"] = sorted(h for h in heads[x["id"]] if h in H3)
    x["c100"] = H3 <= heads[x["id"]]
    x["top"] = k in top
    x["cpct"] = set(PCTS) <= pct[x["id"]]
    x["pend"] = not x["old"] and (not x["c100"] or (x["top"] and not x["cpct"]))
    dst = AA + x["ckpt"][len(JA):]
    x["dst"] = dst; x["here"] = os.path.isfile(dst) and os.path.getsize(dst) == x["size"]
    rows.append(x)
for x in inv:     # growth_v3_jaci: f mais alto ok por no (como o runner); hybrid: id por md5 nao disponivel -> pelo ckpt
    if x["kind"] == "growth":
        if x.get("status") != "ok" or any(y["kind"] == "growth" and y["run"] == x["run"] and y["ds"] == x["ds"] and y["cand"] == x["cand"]
                                         and y.get("status") == "ok" and y["fid"] > x["fid"] for y in inv): continue
        x["c100"] = H3 <= heads[x["id"]]; x["old"] = False; x["top"] = False; x["cpct"] = True
        x["pend"] = not x["c100"]; x["dst"] = AA + x["ckpt"][len(JA):]   # real; exps/growth/code/... tem symlink
        x["here"] = os.path.isfile(x["dst"]) and os.path.getsize(x["dst"]) == x["size"]; rows.append(x)
    elif x["kind"] == "hybrid":
        x["dst"] = AA + x["ckpt"][len(JA):]; x["here"] = os.path.isfile(x["dst"]) and os.path.getsize(x["dst"]) == x["size"]
        x["old"] = False; x["top"] = False; x["cpct"] = True
        if x["run"] == "disc_bp01w64c1280":
            m = re.search(r"/(\w+?)_split(\d)_pct100/", x["ckpt"]); x["id"] = f"OCTAE_{m.group(1)}_s{m.group(2)}_pct100"
            x["c100"] = H3 <= heads[x["id"]]; x["pend"] = not x["c100"]
        else: x["c100"] = True; x["pend"] = False      # ssl_pequeno: 11/11 com tta8 na Jaci
        rows.append(x)
json.dump(rows, open(sys.argv[2], "w"))
c = collections.Counter()
for x in rows:
    b = x["ds"] + ("/" + x["kind"] if x["kind"] != "modehb" else "")
    c[(b, "total")] += 1; c[(b, "old_C100")] += x["old"]; c[(b, "c100")] += x["c100"] and not x["old"]
    c[(b, "pend")] += x["pend"]; c[(b, "pend_sem_100")] += x["pend"] and not x["c100"]
    c[(b, "pend_so_tta8")] += x["pend"] and not x["c100"] and {"gap1280","so2_4096_pca"} <= set(x.get("h", []))
    c[(b, "pend_so_pct")] += x["pend"] and x["c100"]; c[(b, "pend_nao_esta_no_athena")] += x["pend"] and not x["here"]
    c[(b, "GB_copiar")] += (x["size"] / 1e9) if x["pend"] and not x["here"] else 0
for b in sorted({k[0] for k in c}):
    print(f"{b:15s}", {k[1]: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k[0] == b})
print("TOTAL pend", sum(x["pend"] for x in rows), "copiar", sum(x["pend"] and not x["here"] for x in rows),
      "GB", round(sum(x["size"] for x in rows if x["pend"] and not x["here"]) / 1e9, 2))
