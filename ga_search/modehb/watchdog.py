"""Watchdog for the MO-DEHB searches on this host (reporting + self-healing of the monitor only).

Every POLL seconds:
  * new failed jobs in any search_state.json under artifacts/modehb_*/ and artifacts/pnas_cs/ -> ALERT
  * OOM retries in the search logs -> INFO
  * a search tmux session that disappeared (retry loop exhausted / crash) -> ALERT
  * the test-SVM monitor (tmux modehb_monitor) died -> restart it + ALERT
  * GPU memory / utilisation snapshot; GPU idle (<10% for 3 polls) -> ALERT
Alerts go to ga_search/modehb/reports/watchdog_alerts.log (one line each), status to watchdog_status.json.
Run: nice -n 10 python -m ga_search.modehb.watchdog   (inside tmux modehb_watchdog)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

POLL = 120
REPO = Path(__file__).resolve().parents[2]
ALERTS = REPO / "ga_search/modehb/reports/watchdog_alerts.log"
STATUS = REPO / "ga_search/modehb/reports/watchdog_status.json"
STATE_GLOBS = ["artifacts/modehb_hybrid/*/state/seed42/search_state.json",
               "artifacts/modehb_lemonade/*/state/seed42/search_state.json",
               "artifacts/modehb_seeded/*/state/seed42/search_state.json",
               "artifacts/modehb_lemonade_seeded/*/state/seed42/search_state.json",
               "artifacts/pnas_cs/*/state/seed42/search_state.json"]
MONITOR_CMD = ("cd {repo}; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= MONITOR_SKIP_DS=eggs,larvae,protozoan,eggs_raw,larvae_raw,protozoan_raw; nice -n 5 "
               "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.monitor_report "
               ">> ga_search/modehb/reports/monitor.log 2>&1")


def alert(msg: str):
    line = f"{time.strftime('%d/%m %H:%M:%S')} {msg}"
    print(line, flush=True)
    with ALERTS.open("a") as f:
        f.write(line + "\n")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def sessions() -> set:
    return {l.split(":")[0] for l in sh("tmux ls 2>/dev/null").splitlines() if ":" in l}


def gpus():
    """[(index, used_gb, total_gb, util)] for the GPUs we own (env WATCH_GPUS="2,3"; default: all)."""
    import os
    out = sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    want = {int(x) for x in os.environ.get("WATCH_GPUS", "").split(",") if x.strip()}
    res = []
    for line in out.splitlines():
        try:
            i, used, total, util = [float(x) for x in line.split(",")]
        except ValueError:
            continue
        if not want or int(i) in want:
            res.append((int(i), used / 1024, total / 1024, util))
    return res


def main():
    seen_fail, seen_oom, prev_sessions = {}, None, None
    util_hist, low_alerted = {}, {}
    alert("watchdog started")
    while True:
        status = {"time": time.time(), "searches": {}}
        for g in STATE_GLOBS:
            for st in REPO.glob(g):
                try:
                    s = json.loads(st.read_text())
                except Exception:
                    continue
                jobs = s.get("jobs", [])
                fails = [j for j in jobs if j.get("status") != "ok"]
                key = str(st.relative_to(REPO))
                status["searches"][key] = {"jobs": len(jobs), "failed": len(fails), "bracket": s.get("bracket"),
                                           "gpu_hours": round(s.get("gpu_hours", 0.0), 2), "done": s.get("done")}
                if key in seen_fail and len(fails) > seen_fail[key]:
                    for j in fails[seen_fail[key]:]:
                        alert(f"ALERT failed job {key} {j.get('cand_id')} f{j.get('fidelity')}: "
                              f"{str(j.get('result', {}).get('error', ''))[:200]}")
                seen_fail[key] = len(fails)
        n_oom = int(sh("cat artifacts/modehb_*/*/logs/*.log artifacts/pnas_cs/*/logs/*.log 2>/dev/null | grep -c 'OOM on job'") or 0)
        if seen_oom is not None and n_oom > seen_oom:
            alert(f"INFO OOM retries total {seen_oom} -> {n_oom}")
        seen_oom = n_oom
        cur = sessions()
        watched = {s for s in cur if s.startswith(("modehb_", "queue_", "seedtrain_", "pnas_", "seedq_", "migr_", "g3_"))}
        if prev_sessions is not None:
            for gone in sorted(prev_sessions - cur):
                alert(f"ALERT tmux session ended: {gone}")
        prev_sessions = watched
        if "modehb_monitor" not in cur:
            import os
            shard = os.environ.get("WATCHDOG_MONITOR_SHARD")      # e.g. "0/2" when a 2nd monitor shard runs
            cmd = MONITOR_CMD.format(repo=REPO)
            if shard:
                cmd = cmd.replace("nice -n 5", f"MONITOR_SHARD={shard} MONITOR_COV=top nice -n 5", 1)
            subprocess.run(["tmux", "new-session", "-d", "-s", "modehb_monitor", "bash -c '" + cmd + "'"])
            alert("ALERT modehb_monitor was down -> restarted")
        utils = {}
        for i, used, total, util in gpus():
            status.setdefault("gpus", {})[str(i)] = {"used_gb": round(used, 1), "total_gb": round(total, 1), "util": util}
            hist = util_hist.setdefault(i, [])
            hist.append(util)
            del hist[:-8]                        # last 8 polls = 16 min
            if len(hist) == 8 and sum(hist) / 8 < 30 and not low_alerted.get(i):
                alert(f"ALERT GPU {i} underused: mean util {sum(hist) / 8:.0f}% over 16 min ({used:.1f}/{total:.1f} GB)")
                low_alerted[i] = True
            elif len(hist) == 8 and sum(hist) / 8 >= 30:
                low_alerted[i] = False
        if "gpus" in status:
            first = next(iter(status["gpus"].values()))
            status["gpu"] = first                 # backwards-compatible single-GPU field
        status["sessions"] = sorted(watched)
        STATUS.write_text(json.dumps(status, indent=1))
        time.sleep(POLL)


if __name__ == "__main__":
    main()
