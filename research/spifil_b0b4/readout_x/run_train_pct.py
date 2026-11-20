"""run_train_pct.py (cópia de run_all_untested.py): teste com a SVM ajustada em p% do treino (encoder congelado), para os top-N por VAL de cada base.
run_all_untested.py -- testa (SVM original C=100; gap1280 + so2_4096_pca, sem TTA) TODOS os modelos do GA
sem teste ainda, na CPU. Um checkpoint por modelo: a MAIOR fidelidade. Ordem: bases intercaladas
(flowers, RS, ham10000, aid, coconut), maior fitness_kappa de VAL primeiro. Teste nunca alimenta seleção.
Saída: results/all_untested_test.jsonl (append, idempotente pelo backbone_id)."""
import json, glob, os, subprocess, sys, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
B = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
ORIENTED = {"flowers"}  # orientação natural: TTA certo = identidade + flip horizontal (--tta2h), além do TTA8 para comparar
ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--gpu-jobs", type=int, default=20); ap.add_argument("--top", type=int, default=0); ap.add_argument("--tta8", action="store_true"); ap.add_argument("--skip-top", type=int, default=0); ap.add_argument("--bases", nargs="*", default=None); ap.add_argument("--cpu-only", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--pcts", nargs="*", type=int, default=[1, 5, 10, 25, 50, 75])
ap.add_argument("--out", default=str(HERE / "results" / "train_pct_test.jsonl")); a = ap.parse_args()
if a.bases: B = a.bases
a.out = str(Path(a.out) if Path(a.out).is_absolute() else (HERE / a.out))
tested = set()
for f in [a.out] if Path(a.out).is_file() else []:
    for l in open(f):
        try:
            _r = json.loads(l)
            if (not a.tta8) or _r.get("head") == "so2_4096_pca_tta8": tested.add(("id", _r["backbone_id"]))
        except Exception: pass
per = {b: {} for b in B}
for f in glob.glob(str(REPO / "artifacts/modehb_*/*/state/seed42/candidates/c*/f*/result.json")):
    p = Path(f); run, ds, c, fid = p.parts[-8], p.parts[-7], p.parts[-3], int(p.parts[-2][1:])
    if ds not in B or not (p.parent / "encoder.pt").is_file() or (run, ds, c) in tested: continue
    try: r = json.loads(p.read_text())
    except Exception: continue
    if r.get("status") != "ok": continue
    k = (run, c)
    if k not in per[ds] or fid > per[ds][k][0]:
        per[ds][k] = (fid, r.get("fitness_kappa") or 0, str(p.parent / "encoder.pt"))
queues = {b: sorted([(run, c, *v) for (run, c), v in per[b].items()], key=lambda x: -x[3]) for b in B}
if a.top: queues = {b: q[:a.top] for b, q in queues.items()}
if a.skip_top: queues = {b: q[a.skip_top:] for b, q in queues.items()}
todo = []
while any(queues.values()):
    for b in B:
        if queues[b]:
            run, c, fid, kv, ck = queues[b].pop(0); i0 = f"{b}_{run.replace('modehb_','')}_{c}_f{fid}"
            for pc in a.pcts:
                i = f"{i0}_p{pc}"
                if ("id", i) not in tested: todo.append((b, i, ck, pc))
if a.reverse: todo.reverse()
# pula modelos já testados no athena (md5 do encoder.pt; lista em results/athena_tested_md5.txt)
import hashlib
_skip = set()
_f = HERE / "results" / "athena_tested_md5.txt"
if _f.is_file():
    _skip = {l.split()[0] for l in _f.read_text().splitlines() if l.strip()}
if _skip:
    def _md5(pth):
        h = hashlib.md5()
        with open(pth, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""): h.update(ch)
        return h.hexdigest()
    n0 = len(todo); todo = todo
    print(f"[RUNNER] pulados {n0 - len(todo)} já testados no athena (md5)", flush=True)
logs = HERE / "results" / "all_untested_logs"; logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
           OPENBLAS_NUM_THREADS=str(a.threads), PYTHONWARNINGS="ignore")
print(f"[RUNNER] {len(todo)} modelos; por base: " + str({b: sum(1 for t in todo if t[0] == b) for b in B}), flush=True)
run = []
while todo or run:
    import datetime as _dt
    _n = _dt.datetime.now(); gpu_ok = (not a.cpu_only) and not (_n.day >= 19 and _n.hour * 60 + _n.minute >= 8 * 60 + 20)
    dev, cap = ("cuda", a.gpu_jobs) if gpu_ok else ("cpu", a.jobs)
    while todo and len(run) < cap:
        b, i, ck, pc = todo.pop(0)
        cmd = ["nice", "-n", "15", PY, str(HERE / "eval_backbone_head.py"), "--code-root", str(REPO),
               "--item", f"id={i},kind=modehb,ckpt={ck}", "--heads", "gap1280", "so2_4096_pca", *(["--tta8"] if a.tta8 else []), *(["--tta2h"] if (a.tta8 and b in ORIENTED) else []), "--train-pct", str(pc), "--device", dev,
               "--batch-size", "16", "--num-workers", "4" if dev == "cuda" else "2", "--data-cache", str(REPO / "artifacts/readout_x_cache"),
               "--out", a.out, "--table-md", str(logs / "_tabela_bruta.md")]
        fh = open(logs / f"{i}.log", "w")
        run.append((subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT), i, fh, time.time()))
    time.sleep(10)
    for t in list(run):
        if t[0].poll() is not None:
            t[2].close(); run.remove(t)
            print(f"[END] {time.strftime('%d/%m %H:%M')} {t[1]} rc={t[0].returncode} {(time.time()-t[3])/60:.1f} min", flush=True)
print("[RUNNER] fim", flush=True)
