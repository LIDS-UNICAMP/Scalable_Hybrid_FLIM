"""run_ga_best.py -- roda eval_backbone_head.py (modehb encoder.pt) nos modelos de ga_best_list.json.

Uma subprocesso por modelo, ate --jobs em paralelo, na ordem de prioridade das bases. Cabecas:
gap1280 (oficial), so2_4096_pca e so2_4096_pca_tta8 (--tta8). CPU: CUDA_VISIBLE_DEVICES="",
nice 15, OMP/MKL = --threads. Saida: results/ga_best_test.jsonl (append, retomada idempotente).
  python run_ga_best.py --jobs 8 --threads 4
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
ORDER = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results" / "ga_best_test.jsonl"))
    ap.add_argument("--cache", default=str(REPO / "artifacts" / "readout_x_cache"))
    ap.add_argument("--logs", default=str(HERE / "results" / "ga_best_logs"))
    ap.add_argument("--bases", nargs="*", default=ORDER)
    ap.add_argument("--list", default=str(HERE / "ga_best_list.json"))
    a = ap.parse_args()
    L = json.loads(Path(a.list).read_text())
    todo = [(ds, m) for ds in a.bases for m in L["bases"].get(ds, {}).get("models", [])]
    done_ids = set()
    if Path(a.out).is_file():
        heads = {}
        for l in Path(a.out).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                heads.setdefault(r["backbone_id"], set()).add(r["head"])
        done_ids = {b for b, h in heads.items() if {"gap1280", "so2_4096_pca", "so2_4096_pca_tta8"} <= h}
    todo = [t for t in todo if t[1]["id"] not in done_ids]
    Path(a.logs).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
               PYTHONWARNINGS="ignore")
    run = []
    print(f"[RUNNER] {len(todo)} modelos, jobs={a.jobs}", flush=True)
    while todo or run:
        while todo and len(run) < a.jobs:
            ds, m = todo.pop(0)
            cmd = ["nice", "-n", "15", PY, str(HERE / "eval_backbone_head.py"), "--code-root", str(REPO),
                   "--item", f"id={m['id']},kind=modehb,ckpt={REPO / m['ckpt']}",
                   "--heads", "gap1280", "so2_4096_pca", "--tta8", "--device", "cpu",
                   "--batch-size", "16", "--num-workers", "2", "--data-cache", a.cache,
                   "--out", a.out, "--table-md", str(Path(a.logs) / "_tabela_bruta.md")]
            fh = open(Path(a.logs) / f"{m['id']}.log", "w")
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            run.append((p, ds, m, fh, time.time()))
            print(f"[START] {time.strftime('%H:%M:%S')} {m['id']}", flush=True)
        time.sleep(10)
        for t in list(run):
            p, ds, m, fh, t0 = t
            if p.poll() is not None:
                fh.close()
                run.remove(t)
                print(f"[END] {time.strftime('%H:%M:%S')} {m['id']} rc={p.returncode} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
                left = [x for x in todo if x[0] == ds] + [x for x in run if x[1] == ds]
                if not left:
                    print(f"[BASE_DONE] {ds}", flush=True)
    print("[RUNNER] fim", flush=True)


if __name__ == "__main__":
    main()
