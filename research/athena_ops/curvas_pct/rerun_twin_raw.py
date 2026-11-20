# Re-roda SO o gemeo de protozoan_raw: o Evaluator._load_data (num_workers 64, 26k imagens) estourou
# "Too many open files" -> mesma chamada com torch.multiprocessing em 'file_system'. GPU 2/3 pela memoria real.
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); D = Path(__file__).parent
src = open(D / "run_curvas_athena.py").read()
ns = {"__file__": str(D / "run_curvas_athena.py")}
exec(src[:src.index("# protozoan melhor:")], ns)            # constantes, JOBS, NOTE, merge (sem rodar nada)
i, kind, ck, init, code, _n = [j for j in ns["JOBS"] if j[0] == "curva_protozoan_raw_spifil_sem_bp"][0]
RX, PY = ns["RX"], ns["PY"]
argv = [str(RX / "eval_backbone_head.py"), "--code-root", code, "--item", f"id={i},kind={kind},ckpt={ck},init=spifil,variant=spifil_init",
        "--heads", "gap1280", "so2_4096_pca", "--tta8", "--train-pcts", "1", "5", "10", "25", "50", "75", "100",
        "--device", "cuda", "--batch-size", "32", "--num-workers", "4", "--data-cache", str(ns["REPO"] / "artifacts/readout_x_cache"),
        "--out", str(D / "parts" / f"{i}.jsonl"), "--table-md", os.devnull]
wrap = ("import sys, runpy, torch.multiprocessing as m; m.set_sharing_strategy('file_system'); "
        f"sys.argv = {argv!r}; sys.path.insert(0, {str(RX)!r}); runpy.run_path({argv[0]!r}, run_name='__main__')")
while True:
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    fr = {int(a): int(b) / 1024 for a, b in (l.split(", ") for l in o.strip().splitlines())}
    g = max((2, 3), key=lambda k: fr[k])                 # NUNCA 0/1
    if fr[g] >= 11.0: break
    time.sleep(2)
t = time.time(); print(f"[START] {i} gpu{g} (free {fr[g]:.1f} GB)", flush=True)
env = dict(os.environ, READOUT_SO2_DEVICE="cuda", LD_LIBRARY_PATH=ns["LIB"], CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4",
           PYTHONWARNINGS="ignore", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUDA_VISIBLE_DEVICES=str(g))
rc = subprocess.run(["nice", "-n", "10", PY, "-c", wrap], cwd=ns["REPO"], env=env,
                    stdout=open(D / "logs" / f"{i}.log", "a"), stderr=subprocess.STDOUT).returncode
part = D / "parts" / f"{i}.jsonl"
rows = [json.loads(l) for l in open(part)] if part.is_file() else []
for r in rows: r.update(nota=ns["NOTE"][i], protocolo=ns["PROT"])
ns["merge"](rows)
print(f"[END] {i} rc={rc} {len(rows)} linhas {(time.time() - t) / 60:.1f} min", flush=True)
