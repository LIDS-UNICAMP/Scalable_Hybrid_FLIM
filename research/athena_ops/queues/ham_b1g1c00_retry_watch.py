"""Ordem Jaci 18/11: quando a geracao B0:1 do ham10000_fix5_nossl fechar (gens_done), poe na fila a avaliacao avulsa
do no B1g1c00 (morto por OOM de contencao) com o codigo do clone que rodou a cadeia (growth/code). O resultado vira dado."""
import fcntl, json, os, time
S = "/dados/home/moliveira/exps/growth/code/artifacts/hmodehb_b0b4/growth/athena_gpu3/ham10000_fix5_nossl/seed42/growth_state.json"
Q, L = "/exps/moliveira/queues/gpu3_queue.json", "/exps/moliveira/queues/gpu_launcher.log"
while "B0:1" not in json.load(open(S)).get("gens_done", []):
    time.sleep(60)
cmd = ("cd /dados/home/moliveira/exps/growth/code && CUDA_VISIBLE_DEVICES=3 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -u "
       "/dados/home/moliveira/exps/growth_v3/code/research/spifil_b0b4/growth/retry_oom_node.py --out "
       "/dados/home/moliveira/exps/growth/code/artifacts/hmodehb_b0b4/growth/athena_gpu3/ham10000_fix5_nossl/seed42 "
       "--node B1g1c00 --fid 15 2>&1 | tee -a artifacts/hmodehb_b0b4/growth/logs/retry_oom_ham10000_B1g1c00.log; exit ${PIPESTATUS[0]}")
lk = open(Q + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
q = json.load(open(Q))
if not any(j["id"] == "retry_oom_ham10000_B1g1c00" for j in q["jobs"]):
    q["jobs"].append({"id": "retry_oom_ham10000_B1g1c00", "cmd": cmd, "peak_gb": 21, "gpus": [2, 3], "deps": [],
                      "status": "queued", "nota": f"{time.strftime('%F %T')}: avaliacao avulsa do no morto por OOM; B0:1 ja fechada -> vira dado (Jaci)"})
json.dump(q, open(Q + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(Q + ".tmp", Q)
fcntl.flock(lk, fcntl.LOCK_UN)
open(L, "a").write(time.strftime("%F_%T") + " retry_oom_ham10000_B1g1c00: B0:1 fechada -> avaliacao avulsa na fila (Jaci)\n")
