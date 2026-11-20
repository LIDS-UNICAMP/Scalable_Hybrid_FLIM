"""Escalonador do athena (ordem do usuario, 17/11 18h): GPUs 2 e 3, backfill agressivo (first-fit decrescente).

Fila: /exps/moliveira/queues/gpu3_queue.json (editavel a quente; lida a cada 60 s)
  job: id, cmd, peak_gb, status (queued|running|done|failed|external), deps, group/max_parallel,
       gpus [3] (onde pode rodar), est_min (duracao estimada), reserve_gb + reserve_pattern (externos/growth).
Folga(GPU) = livre real - sum(reserva dos jobs vivos naquela GPU); reserva = max(0, pico - uso atual).
Pico = maior uso medido do job (peak_measured_gb) ou peak_gb da fila; +2 GB de margem para lancar.
Regra: 1o da fila (pela ordem) se couber; senao o MAIOR que couber, depois o proximo, ate a folga acabar.
Anti-fome: se folga + memoria dos jobs que terminam em <= 15 min (est_min) >= pico do 1o, nao lanca backfill nela.
Util < 50% por 5 min: lanca o que couber no livre real, ignorando reservas de growth em fase de CPU.
Cada job roda no tmux q<gpu>_<id>; o fim grava <id>.rc em gpu3_runs/. Log de cada decisao em gpu_launcher.log.
PNAS-CS e monitores da GPU 2 nao estao na fila: so ocupam memoria real. GPU 0/1: nunca.
Edicao manual da fila: flock /exps/moliveira/queues/gpu3_queue.json.lock <comando> (senao o ciclo regrava por cima).
Teste: python gpu_launcher.py --dry-run (1 ciclo, so imprime, nao lanca nem grava).
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time

Q = "/exps/moliveira/queues/gpu3_queue.json"
RUNS = "/exps/moliveira/queues/gpu3_runs"
LOG = "/exps/moliveira/queues/gpu_launcher.log"
GPUS, PERIOD, MARGIN, LOWUTIL_S = (2, 3), 60, 4.0, 300   # 18/11 21:2x (usuario via Jaci): margem 4 GB sobre a memoria REAL
DRY = "--dry-run" in sys.argv
os.makedirs(RUNS, exist_ok=True)


def log(msg):
    if DRY:
        print(msg)
        return
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F_%T')} {msg}\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_state():
    st = {}
    for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu "
                   "--format=csv,noheader,nounits").splitlines():
        i, uuid, used, total, util = [x.strip() for x in line.split(",")]
        st[int(i)] = {"uuid": uuid, "free": (int(total) - int(used)) / 1024, "util": int(util), "procs": {}}
    by_uuid = {v["uuid"]: k for k, v in st.items()}
    for line in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits").splitlines():
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        if uuid in by_uuid:
            st[by_uuid[uuid]]["procs"][pid] = int(mem) / 1024
    return st


def sess(j):
    return f"q{j.get('gpu', 3)}_{j['id']}"


def job_pids(j):
    if j.get("reserve_pattern"):
        return set(sh(f"pgrep -u moliveira -f '{j['reserve_pattern']}'").split())
    pane = sh(f"tmux list-panes -t '={sess(j)}' -F '#{{pane_pid}}' 2>/dev/null")
    if not pane:
        return set()
    return set(sh(f"pstree -p {pane} | grep -oE '\\(([0-9]+)\\)' | tr -d '()'").split())


def cpu_phase(j, used):
    """growth em fase de CPU: usa < 5 GB de GPU e o log nao mexe ha > 5 min -> reserva zero."""
    if not j.get("reserve_pattern") or used >= 5:
        return False
    log = j.get("log")
    if not log:
        m = re.search(r">> (\S+\.log)", j.get("cmd", ""))
        log = m and (m.group(1) if m.group(1).startswith("/") else "/dados/home/moliveira/exps/growth/code/" + m.group(1))
    return bool(log) and os.path.exists(log) and time.time() - os.path.getmtime(log) > 300


def peak(j):
    m = float(j.get("peak_measured_gb") or 0)
    if j.get("peak_from_measured") and m and time.time() - float(j.get("started_ts") or time.time()) > 900:
        return 1.1 * m                                       # Jaci 18/11: reserva = maior pico medido do job x 1,1
    return float(j.get("reserve_gb") or max(m, float(j["peak_gb"])))


def mg(j):
    """18/11 (Jaci): margem 2 GB para job pequeno (pico <= 10 GB), senao MARGIN."""
    return 2.0 if peak(j) <= 10 else MARGIN


def load():
    with open(Q) as f:
        return json.load(f)


def save(q):
    if DRY:
        return
    tmp = Q + ".tmp"
    with open(tmp, "w") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
    os.replace(tmp, Q)


def launch(j, g, folga, motivo):
    log(f"LANCA {j['id']} na GPU {g} ({motivo}; pico {peak(j):.1f} GB; folga {folga:.1f} GB)")
    j["gpu"] = g
    if DRY:
        j["status"] = "running"
        return
    rc = f"{RUNS}/{j['id']}.rc"
    if os.path.exists(rc):
        os.rename(rc, rc + ".old_" + time.strftime("%H%M%S"))
    cmd = (f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={g} OPENBLAS_NUM_THREADS=2 "
           f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib; "
           f"( {j['cmd'].replace('CUDA_VISIBLE_DEVICES=3', f'CUDA_VISIBLE_DEVICES={g}')} ) "
           f">> {RUNS}/{j['id']}.log 2>&1; echo $? > {rc}")
    subprocess.run(["tmux", "new-session", "-d", "-s", sess(j), cmd])
    j["status"], j["started"], j["started_ts"] = "running", time.strftime("%F_%T"), time.time()


low_since, last_wait = {g: None for g in GPUS}, {}
log("INICIO escalonador (GPUs 2 e 3; first-fit decrescente; anti-fome 15 min; util<50% por 5 min)"
    + (" [DRY-RUN]" if DRY else ""))
while True:
    lockf = open(Q + ".lock", "w")                       # quem edita a fila usa o mesmo flock (gpu3_queue.json.lock)
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        q, ses, st = load(), set(sh("tmux ls -F '#{session_name}' 2>/dev/null").split()), gpu_state()
        for j in q["jobs"]:                                   # fecha os que terminaram
            if j["status"] != "running":
                continue
            rc_file = f"{RUNS}/{j['id']}.rc"
            if os.path.exists(rc_file):
                rc = open(rc_file).read().strip()
                j["status"], j["rc"], j["ended"] = ("done" if rc == "0" else "failed"), rc, time.strftime("%F_%T")
                log(("FIM" if rc == "0" else "ALERTA FALHOU") + f" {j['id']} rc={rc}")
            elif sess(j) not in ses:
                j["status"] = "failed"
                log(f"ALERTA {j['id']}: sessao {sess(j)} sumiu sem rc")
        done = {j["id"] for j in q["jobs"] if j["status"] == "done"}
        folga = {g: st[g]["free"] for g in GPUS}
        reserv, ending = {g: 0.0 for g in GPUS}, {g: 0.0 for g in GPUS}
        for j in q["jobs"]:                                   # reservas + pico medido dos vivos
            g = j.get("gpu", 3)
            if j["status"] not in ("running", "external") or g not in folga:
                continue
            used = sum(st[g]["procs"].get(p, 0) for p in job_pids(j))
            if j["status"] == "external" and not used:
                continue
            if j["status"] == "running" and used > float(j.get("peak_measured_gb") or 0):
                j["peak_measured_gb"] = round(used, 2)
            young = j["status"] == "running" and time.time() - float(j.get("started_ts") or 0) <= 900
            # 18/11 21:2x (usuario via Jaci): backfill pela memoria REAL; reserva so para job com <= 15 min (ainda subindo)
            r = max(0.0, peak(j) - used) if young and not (j.get("reserve_off") or cpu_phase(j, used)) else 0.0
            folga[g] -= r
            if j.get("reserve_pattern"):
                reserv[g] += r
            if j.get("est_min") and j.get("started_ts") and \
                    j["started_ts"] + 60 * float(j["est_min"]) - time.time() <= 900:
                ending[g] += used + r
        running_grp = {}
        for j in q["jobs"]:
            if j["status"] == "running" and j.get("group"):
                running_grp[j["group"]] = running_grp.get(j["group"], 0) + 1

        def fits_group(j):
            return running_grp.get(j.get("group"), 0) < int(j.get("max_parallel", 1))

        ready = [j for j in q["jobs"] if j["status"] == "queued" and set(j.get("deps", [])) <= done]
        for g in GPUS:
            util = st[g]["util"]
            low_since[g] = (low_since[g] or time.time()) if util < 50 else None
            lowutil = low_since[g] is not None and time.time() - low_since[g] >= LOWUTIL_S
            cand = [j for j in ready if g in j.get("gpus", [3]) and fits_group(j)]
            if not cand:
                continue
            f = folga[g] + (reserv[g] if lowutil else 0.0)
            head = cand[0]
            if peak(head) + mg(head) <= f:
                launch(head, g, f, "1o da fila")
            elif f + ending[g] >= peak(head) + mg(head):
                msg = (f"ESPERA GPU {g}: anti-fome, {head['id']} ({peak(head):.0f} GB) entra quando os que "
                       f"terminam em 15 min sairem (folga {f:.1f} + saindo {ending[g]:.1f})")
                if last_wait.get(g) != msg:
                    log(msg)
                    last_wait[g] = msg
                continue
            for j in [head] + sorted(cand[1:], key=peak, reverse=True):   # backfill: o maior que cabe primeiro
                if j["status"] == "running":
                    f -= peak(j) + mg(j)
                    ready.remove(j)
                    running_grp[j.get("group")] = running_grp.get(j.get("group"), 0) + 1
                    continue
                if fits_group(j) and peak(j) + mg(j) <= f:
                    launch(j, g, f, "backfill" + (" util<50% 5 min" if lowutil else ""))
                    f -= peak(j) + mg(j)
                    ready.remove(j)
                    running_grp[j.get("group")] = running_grp.get(j.get("group"), 0) + 1
            left = [j for j in cand if j in ready]
            if left:
                msg = (f"ESPERA GPU {g}: folga {f:.1f} GB (livre {st[g]['free']:.1f}, util {util}%); "
                       f"menor da fila {min(peak(j) for j in left):.0f}+{MARGIN:.0f} GB")
                if last_wait.get(g, "")[:40] != msg[:40]:
                    log(msg)
                    last_wait[g] = msg
        save(q)
    except Exception as e:                                    # nunca morre por um erro de leitura
        log(f"ERRO escalonador: {e!r}")
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    if DRY:
        break
    time.sleep(PERIOD)
