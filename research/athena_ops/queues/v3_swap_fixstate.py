"""Troca entre nos (ordem Jaci): para o controlador v3 no fim do no atual, roda fix_member_state (dry-run + --apply),
acrescenta --allow-drift growth.v3 e relanca o MESMO job na MESMA GPU, com o codigo novo (member_fid + ae_diverge)."""
import fcntl, glob, json, os, re, subprocess, sys, time
jid = sys.argv[1]
Q, R, L = "/exps/moliveira/queues/gpu3_queue.json", "/exps/moliveira/queues/gpu3_runs", "/exps/moliveira/queues/gpu_launcher.log"
CODE = "/dados/home/moliveira/exps/growth_v3/code"; PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
log = lambda m: open(L, "a").write(time.strftime("%F_%T") + f" {jid}: {m}\n")
j = next(x for x in json.load(open(Q))["jobs"] if x["id"] == jid)
rel = re.findall(r"--out (\S+)", j["cmd"])[-1]; out = f"{CODE}/{rel}"
g = j.get("gpu", 3); sess = f"q{g}_{jid}"
n0 = len(glob.glob(out + "/nodes/B*/*/result.json"))
while len(glob.glob(out + "/nodes/B*/*/result.json")) == n0:
    time.sleep(10)
pane = sh(f"tmux list-panes -t '={sess}' -F '#{{pane_pid}}'")
for p in sh(f"pstree -p {pane} | grep -oE 'python\\(([0-9]+)\\)' | grep -oE '[0-9]+'").split():
    if "growth_chain" in sh(f"ps -o args= -p {p}"):
        sh(f"kill {p}")
time.sleep(15)
sh(f"tmux kill-session -t '={sess}'")
env = "LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_VISIBLE_DEVICES="
dry = sh(f"cd {CODE} && {env} {PY} research/spifil_b0b4/growth/fix_member_state.py --out {rel} 2>&1")
open(f"/exps/moliveira/queues/fixstate_{jid}.dry.txt", "w").write(dry)
app = sh(f"cd {CODE} && {env} {PY} research/spifil_b0b4/growth/fix_member_state.py --out {rel} --apply 2>&1")
open(f"/exps/moliveira/queues/fixstate_{jid}.apply.txt", "w").write(app)
rc = f"{R}/{jid}.rc"
if os.path.exists(rc):
    os.rename(rc, rc + ".fixstate_" + time.strftime("%H%M%S"))
lk = open(Q + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
q = json.load(open(Q))
for x in q["jobs"]:
    if x["id"] == jid:
        if "--allow-drift growth.v3 " not in x["cmd"]:
            i = x["cmd"].rfind(" --out "); x["cmd"] = x["cmd"][:i] + " --allow-drift growth.v3" + x["cmd"][i:]
        x["status"] = "queued"; [x.pop(k, None) for k in ("rc", "ended", "started", "started_ts")]
json.dump(q, open(Q + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(Q + ".tmp", Q)
fcntl.flock(lk, fcntl.LOCK_UN)
subprocess.run(["flock", Q + ".lock", "python3", "/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad/force_launch.py", f"{jid}:{g}"])
log(f"troca entre nos: fix_member_state aplicado (ver fixstate_{jid}.*.txt) e relancado na GPU {g} com member_fid + ae_diverge (mesmo --out)")
