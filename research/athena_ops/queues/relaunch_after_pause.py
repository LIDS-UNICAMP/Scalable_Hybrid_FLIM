"""Ordem Jaci 18/11: quando o job estiver "pausado", acrescenta os --gset pedidos antes do ultimo --out (parte principal,
nao o smoke), volta para queued e relanca na mesma GPU via force_launch. Uso: <job_id> <gpu> "<--gset ...>" """
import fcntl, json, os, subprocess, sys, time
jid, g, extra = sys.argv[1], sys.argv[2], sys.argv[3]
Q, L, R = "/exps/moliveira/queues/gpu3_queue.json", "/exps/moliveira/queues/gpu_launcher.log", "/exps/moliveira/queues/gpu3_runs"
while next(x for x in json.load(open(Q))["jobs"] if x["id"] == jid)["status"] != "pausado":
    time.sleep(20)
time.sleep(20)
rc = f"{R}/{jid}.rc"
if os.path.exists(rc):
    os.rename(rc, rc + ".pausa_" + time.strftime("%H%M%S"))
lk = open(Q + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
q = json.load(open(Q))
for x in q["jobs"]:
    if x["id"] == jid:
        if extra not in x["cmd"]:
            i = x["cmd"].rfind(" --out "); x["cmd"] = x["cmd"][:i] + " " + extra + x["cmd"][i:]
        x["status"] = "queued"; [x.pop(k, None) for k in ("rc", "ended", "started", "started_ts")]
        x["nota"] += f" | {time.strftime('%F %T')} relancado com {extra} (Jaci)"
json.dump(q, open(Q + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(Q + ".tmp", Q)
fcntl.flock(lk, fcntl.LOCK_UN)
subprocess.run(["flock", Q + ".lock", "python3", "/tmp/claude-1033/-dados-home-moliveira-Scalable-Hybrid-FLIM/990a33ea-4327-4228-ac99-2136fb213621/scratchpad/force_launch.py", f"{jid}:{g}"])
open(L, "a").write(time.strftime("%F_%T") + f" {jid}: RELANCADO apos a pausa na GPU {g} com {extra} (Jaci)\n")
