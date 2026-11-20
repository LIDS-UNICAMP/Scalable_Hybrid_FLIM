# Vigia da passada FINAL Jaci -> athena: dispara 08:35 BRT (11:35 UTC) de 19/11, depois que a Jaci solta a GPU.
# 1) mig.py: re-sincroniza o espelho, refaz a lista de pendentes e copia os checkpoints que faltarem;
# 2) OCTAE (oct_ae parasitos, fora do runner): o que nao tiver tta8 roda aqui, so GPU 2/3;
# 3) relanca o runner (mata SO o PID do runner; os testes vivos sao adotados), mesmos argumentos.
import datetime as dt, json, os, re, signal, subprocess, time
from pathlib import Path
D = Path(__file__).resolve().parent
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"; LIB = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib"
RX = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/readout_x")
MIR = Path("/dados/home/moliveira/exps/jaci_results_mirror/readout_x_results")
T = dt.datetime(2026, 11, 19, 11, 35, tzinfo=dt.timezone.utc)
def say(m): print(f"[VIGIA] {time.strftime('%d/%m %H:%M:%S')} UTC {m}", flush=True)
say(f"armado; dispara {T:%d/%m %H:%M} UTC (08:35 BRT)")
while dt.datetime.now(dt.timezone.utc) < T: time.sleep(30)
say("passada final: mig.py")
subprocess.run([PY, "-u", str(D / "mig.py")], cwd=D)
# OCTAE: ids sem tta8 na Jaci nem aqui
heads = {}
for f in [*MIR.glob("*.jsonl"), *(RX / "results").glob("*.jsonl")]:
    for l in open(f):
        try: r = json.loads(l); heads.setdefault(r["backbone_id"], set()).add(r["head"])
        except Exception: pass
A = "/dados/home/moliveira/artifacts/spifil_growth/disc_bp01w64c1280"
items = []
for ds in ("eggs", "larvae", "protozoan"):
    for s in (1, 2, 3):
        i = f"OCTAE_{ds}_s{s}_pct100"
        if "so2_4096_pca_tta8" in heads.get(i, set()): continue
        ck = f"{A}/{ds}_split{s}_pct100/L4/checkpoints/best_kappa.ckpt"
        if (ds, s) == ("protozoan", 3):   # a Jaci re-treinou este (10/11); o do athena e outro: usa o da Jaci, sem sobrescrever
            jd = f"{A}/protozoan_split3_pct100/L4/checkpoints_jaci"
            subprocess.run(f"mkdir -p {jd} && rsync -a --ignore-existing jaci:/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/"
                           f"spifil_growth/disc_bp01w64c1280/protozoan_split3_pct100/L4/checkpoints/best_kappa.ckpt {jd}/", shell=True)
            with open("/exps/moliveira/queues/rsync_contador.log", "a") as fh:
                fh.write(f"{time.strftime('%F %T')} rsync migracao: OCTAE protozoan s3 da Jaci -> {jd}\n")
            ck = f"{jd}/best_kappa.ckpt"
        items += ["--item", f"id={i},kind=hybrid,ckpt={ck}"]
if items:
    free = {int(i): int(f) for i, f in (l.split(", ") for l in subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines())}
    g = max((2, 3), key=lambda k: free.get(k, 0))      # so GPU 2/3
    cmd = ["nice", "-n", "15", PY, "-u", str(RX / "eval_backbone_head.py"), *items, "--heads", "gap1280", "so2_4096_pca", "--tta8",
           "--device", "cuda", "--batch-size", "16", "--num-workers", "8",
           "--out", str(RX / "results/oct_ae_parasitos_test.jsonl"), "--table-md", str(RX / "results/oct_ae_parasitos_test.md")]
    env = dict(os.environ, READOUT_SO2_DEVICE="cuda", LD_LIBRARY_PATH=LIB, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="8")
    subprocess.Popen(cmd, cwd="/dados/home/moliveira/Scalable_Hybrid_FLIM", env=env, start_new_session=True,
                     stdout=open(RX / "results/oct_ae_parasitos_test.log", "a"), stderr=subprocess.STDOUT)
    say(f"OCTAE: {len(items) // 2} na GPU {g}")
else: say("OCTAE: nada pendente")
# relanca o runner
pids = [p for p in os.listdir("/proc") if p.isdigit() and os.stat(f"/proc/{p}").st_uid == os.getuid()]
def argv(p):
    try: return open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
    except OSError: return []
run = [p for p in pids if any(x.endswith(b"run_all_untested.py") for x in argv(p)) and b"--dry-run" not in argv(p)
       and argv(p)[0].endswith(b"python")]
args = [x.decode() for x in argv(run[0])[argv(run[0]).index(next(x for x in argv(run[0]) if x.endswith(b"run_all_untested.py"))) + 1:] if x] \
    if run else ["--gpus", "2,3", "--parasitas", "--variant", "spifil_init", "--train-pct", "--peak-gb", "9"]
if "--gpus" in args: args[args.index("--gpus") + 1] = "2,3"   # so GPU 2/3, sempre
for p in run:
    os.kill(int(p), signal.SIGTERM); say(f"runner {p} morto (so ele; testes vivos ficam)")
time.sleep(5)
cmd = (f"cd {RX} && READOUT_SO2_DEVICE=cuda LD_LIBRARY_PATH={LIB} {PY} -u run_all_untested.py {' '.join(args)} "
       f"2>&1 | tee -a results/all_untested_runner.log")
subprocess.run(["tmux", "new-session", "-d", "-s", "tests_final_0835", cmd])
say(f"runner relancado no tmux tests_final_0835: {' '.join(args)}")
