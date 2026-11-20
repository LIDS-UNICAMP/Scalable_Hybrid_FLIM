# roda NA JACI (somente leitura): inventario dos checkpoints testaveis
import glob, json, os
from pathlib import Path
A = "/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts"
B = ["flowers","remote_sensing","ham10000","aid","coconut_trees","eggs","larvae","protozoan","eggs_raw","larvae_raw","protozoan_raw"]
per = {}
for f in glob.glob(f"{A}/modehb_*/*/state/seed42/candidates/c*/f*/result.json"):
    p = Path(f); run, ds, c, fd = p.parts[-8], p.parts[-7], p.parts[-3], p.parts[-2]
    if ds not in B or not fd[1:].isdigit() or not (p.parent/"encoder.pt").is_file(): continue
    try: r = json.loads(p.read_text())
    except Exception: continue
    if r.get("status") != "ok": continue
    fid = int(fd[1:]); k = (run, ds, c)
    if k not in per or fid > per[k]["fid"]:
        ck = p.parent/"encoder.pt"
        per[k] = dict(kind="modehb", run=run, ds=ds, cand=c, fid=fid, kval=r.get("fitness_kappa") or 0,
                      id=f"{ds}_{run.replace('modehb_','')}_{c}_f{fid}", ckpt=str(ck), size=ck.stat().st_size,
                      mtime=ck.stat().st_mtime)
out = list(per.values())
for st in glob.glob(f"{A}/hmodehb_b0b4/growth_v3_jaci/*/*/seed42/growth_state.json"):
    sd = Path(st).parent
    for nd in sorted((sd/"nodes").iterdir()):
        for q in sorted(nd.glob("f*/backbone.pt")):
            try: r = json.loads((q.parent/"result.json").read_text())
            except Exception: r = {}
            out.append(dict(kind="growth", run="growth_v3_jaci/"+sd.parts[-3], ds=sd.parts[-2], cand=nd.name, fid=int(q.parent.name[1:]),
                            kval=r.get("kappa_val_orig") or 0, status=r.get("status"), id=f"jaci_v3ae_{sd.parts[-2]}_{nd.name}_{q.parent.name}",
                            ckpt=str(q), size=q.stat().st_size, mtime=q.stat().st_mtime))
for q in glob.glob(f"{A}/spifil_growth/disc_bp01w64c1280/*_split?_pct100/L4/checkpoints/best_kappa.ckpt") + \
         glob.glob(f"{A}/external_ssl_pequeno/*_split1_pct100/*/checkpoints/best_kappa.ckpt"):
    q = Path(q); out.append(dict(kind="hybrid", run=q.parts[-5], ds=q.parts[-4].split("_split")[0], cand=q.parts[-3], fid=0, kval=0,
                                 id="", ckpt=str(q), size=q.stat().st_size, mtime=q.stat().st_mtime))
print(json.dumps(out))
