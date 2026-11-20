"""Para um job da fila ENTRE unidades (ordem do usuario 18/11 19:30): espera a proxima unidade terminar e mata so os
processos python do tmux do job (PID exato), marcando "pausado" com o motivo. Unidade terminada =
  log:<arquivo>:<regex>  -> nova linha que casa com a regex
  jobs:<search_state.json> -> cresce o numero de jobs
Uso: python stop_between.py <job_id> <unidade> "<motivo>" """
import fcntl, json, os, re, subprocess, sys, time
jid, unit, motivo = sys.argv[1], sys.argv[2], sys.argv[3]
Q, L = "/exps/moliveira/queues/gpu3_queue.json", "/exps/moliveira/queues/gpu_launcher.log"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
kind, rest = unit.split(":", 1)
if kind == "log":
    path, rx = rest.split(":", 1)
    count = lambda: sum(1 for l in open(path, errors="ignore") if re.search(rx, l))
else:
    path = rest
    count = lambda: len(json.load(open(path))["jobs"])
n0 = count()
while count() == n0:
    time.sleep(20)
j = next(x for x in json.load(open(Q))["jobs"] if x["id"] == jid)
sess = f"q{j.get('gpu', 3)}_{jid}"
lk = open(Q + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
q = json.load(open(Q))
for x in q["jobs"]:
    if x["id"] == jid:
        x["status"] = "pausado"; x["nota"] = x.get("nota", "") + f" | {time.strftime('%F %T')} PAUSADO entre unidades: {motivo}"
json.dump(q, open(Q + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(Q + ".tmp", Q)
fcntl.flock(lk, fcntl.LOCK_UN)
pane = sh(f"tmux list-panes -t '={sess}' -F '#{{pane_pid}}'")
pids = sh(f"pstree -p {pane} | grep -oE 'python\\(([0-9]+)\\)' | grep -oE '[0-9]+'").split() if pane else []
for p in pids:
    sh(f"kill {p}")
time.sleep(15)
sh(f"tmux kill-session -t '={sess}'")
open(L, "a").write(time.strftime("%F_%T") + f" {jid}: PAUSADO entre unidades (pid {','.join(pids)}; {unit.split(':')[0]} "
                   f"{n0}->{count()}): {motivo}\n")
