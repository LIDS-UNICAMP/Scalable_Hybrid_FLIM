# Migracao Jaci -> athena (19/11): inventario (so leitura na Jaci) -> pendentes -> rsync dos checkpoints que faltam.
# Uso: python mig.py [--no-copy]. Saidas: inv_jaci.json, rows.json, lista_pendentes.tsv, copy_<hora>.list
import json, os, subprocess, sys, time, random, hashlib
from pathlib import Path
D = Path(__file__).resolve().parent
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
JA, AA = "/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/", "/dados/home/moliveira/artifacts/"
LOG = "/exps/moliveira/queues/rsync_contador.log"
def log(msg):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%F %T')} {msg}\n")
def sh(cmd, **k): return subprocess.run(cmd, shell=True, check=True, text=True, **k)
# 1. espelho de resultados da Jaci (mesmo rsync do tmux sync_jaci_mirror)
sh("rsync -a --include='*.jsonl' --include='*.md' --exclude='*' jaci:/dados/home/moliveira/Scalable_Hybrid_FLIM/research/"
   "spifil_b0b4/readout_x/results/ /dados/home/moliveira/exps/jaci_results_mirror/readout_x_results/")
log("rsync migracao: jaci->athena espelho readout_x results (jsonl/md)")
# 2. inventario + pendentes
sh(f"ssh -o BatchMode=yes jaci 'nice -n 19 python3 -' < {D}/inv_jaci.py > {D}/inv_jaci.json")
print(sh(f"{PY} {D}/pend.py {D}/inv_jaci.json {D}/rows.json", capture_output=True).stdout, flush=True)
rows = json.load(open(D / "rows.json"))
with open(D / "lista_pendentes.tsv", "w") as f:
    f.write("base\tkind\tid\tkval\theads_100\ttop20\tpct_ok\tno_athena\tckpt_jaci\n")
    for x in sorted(rows, key=lambda x: (x["ds"], -float(x["kval"] or 0))):
        if x.get("pend"): f.write(f"{x['ds']}\t{x['kind']}\t{x['id']}\t{float(x['kval'] or 0):.4f}\t{','.join(x.get('h', []))}"
                                  f"\t{x['top']}\t{x['cpct']}\t{x['here']}\t{x['ckpt']}\n")
if "--no-copy" in sys.argv: raise SystemExit
# 3. copia: pendentes que nao estao no athena; bases intercaladas, melhor kval primeiro
todo = [x for x in rows if x.get("pend") and not x["here"] and x["kind"] in ("modehb", "growth")]
by = {}
for x in sorted(todo, key=lambda x: -float(x["kval"] or 0)): by.setdefault(x["ds"], []).append(x)
order = []
while any(by.values()):
    for b in list(by):
        if by[b]: order.append(by[b].pop(0))
def files(x):
    rel = Path(x["ckpt"][len(JA):])
    if x["kind"] == "modehb":   # .../state/seed42/candidates/cX/fN/encoder.pt
        st = rel.parents[3]
        return [st / "config_resolved.yaml", st / "search_state.json", rel, rel.parent / "result.json"]
    sd = rel.parents[3]          # .../seed42/nodes/<no>/fN/backbone.pt
    return [sd / "growth_state.json", rel, rel.parent / "result.json", rel.parent / "growth_spec.json"]
# growth_v3_jaci: real em artifacts/hmodehb_b0b4 (= caminho da Jaci pelo symlink do repo); o runner varre exps/growth/code
link = Path("/dados/home/moliveira/exps/growth/code/artifacts/hmodehb_b0b4/growth_v3_jaci")
if any(x["kind"] == "growth" for x in order) and not link.exists():
    Path(AA + "hmodehb_b0b4/growth_v3_jaci").mkdir(parents=True, exist_ok=True)
    link.symlink_to(AA + "hmodehb_b0b4/growth_v3_jaci")
seen, gb = set(), 0.0
for k in range(0, len(order), 100):
    chunk = order[k:k + 100]; lst = D / f"copy_{time.strftime('%H%M%S')}_{k:04d}.list"
    fs = [str(f) for x in chunk for f in files(x) if not (str(f) in seen or seen.add(str(f)))]
    lst.write_text("\n".join(fs) + "\n")
    t = time.time()
    r = subprocess.run(f"rsync -a --ignore-existing --files-from={lst} jaci:{JA} {AA}", shell=True, text=True,
                       capture_output=True)
    g = sum(x["size"] for x in chunk) / 1e9; gb += g
    log(f"rsync migracao Jaci->athena: {len(chunk)} checkpoints ({g:.2f} GB, maior fid + result.json + config/search_state) "
        f"-> {AA} (lista {lst.name}) rc={r.returncode}")
    print(f"[COPY] {k + len(chunk)}/{len(order)} {g:.2f} GB {time.time() - t:.0f}s rc={r.returncode} {r.stderr[-300:]}", flush=True)
print(f"[COPY] total {gb:.2f} GB", flush=True)
# 4. md5 de amostra
smp = random.sample(order, min(15, len(order)))
rem = sh("ssh -o BatchMode=yes jaci 'md5sum " + " ".join(x["ckpt"] for x in smp) + "'", capture_output=True).stdout.split()
rm = dict(zip(rem[1::2], rem[0::2])); bad = 0
for x in smp:
    with open(AA + x["ckpt"][len(JA):], "rb") as fh: m = hashlib.file_digest(fh, "md5").hexdigest()
    bad += m != rm.get(x["ckpt"])
print(f"[MD5] amostra {len(smp)}: {len(smp) - bad} iguais, {bad} diferentes", flush=True)
