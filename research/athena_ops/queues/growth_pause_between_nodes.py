"""Pausa ENTRE nos (ordem Jaci, 18/11): espera o proximo result.json do crescimento, marca o job "pausado" na fila
(com o motivo) e encerra so o processo growth_chain daquele tmux (PID exato). Estado e arquivos ficam; retoma relancando
o mesmo job (mesmo --out). Uso: python growth_pause_between_nodes.py <job_id> <code_dir> "<motivo>" """
import fcntl, glob, json, os, re, subprocess, sys, time
jid, code, motivo = sys.argv[1], sys.argv[2], sys.argv[3]
Q, L = "/exps/moliveira/queues/gpu3_queue.json", "/exps/moliveira/queues/gpu_launcher.log"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
j = next(x for x in json.load(open(Q))["jobs"] if x["id"] == jid)
out = code + "/" + re.findall(r"--out (\S+)", j["cmd"])[-1]
sess = f"q{j.get('gpu', 3)}_{jid}"
n0 = len(glob.glob(out + "/nodes/B*/*/result.json"))
while len(glob.glob(out + "/nodes/B*/*/result.json")) == n0:
    time.sleep(10)
lk = open(Q + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
q = json.load(open(Q))
for x in q["jobs"]:
    if x["id"] == jid:
        x["status"] = "pausado"; x["nota"] = x.get("nota", "") + f" | {time.strftime('%F %T')} PAUSADO entre nos: {motivo}"
json.dump(q, open(Q + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(Q + ".tmp", Q)
fcntl.flock(lk, fcntl.LOCK_UN)
pane = sh(f"tmux list-panes -t '={sess}' -F '#{{pane_pid}}'")
pids = [p for p in sh(f"pstree -p {pane} | grep -oE 'python\\(([0-9]+)\\)' | grep -oE '[0-9]+'").split()
        if "growth_chain" in sh(f"ps -o args= -p {p}")]
for p in pids:
    sh(f"kill {p}")
time.sleep(15)
sh(f"tmux kill-session -t '={sess}'")
open(L, "a").write(time.strftime("%F_%T") + f" {jid}: PAUSADO entre nos (pid {','.join(pids)}; {n0 + 1} nos fechados; "
                   f"estado preservado em {out}): {motivo}\n")
