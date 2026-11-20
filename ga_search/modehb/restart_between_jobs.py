"""Restart a running search BETWEEN jobs so it picks up new code without losing a job in progress.

Waits until search_state.json gains a job (the current evaluation finished and was saved), then sends
SIGTERM to the search python; its retry loop (`for i in 1..5`) relaunches the same command, which
replays the state (resume) with the new code. Optional --max-wait-min kills anyway after that long.

    python -m ga_search.modehb.restart_between_jobs --state <.../search_state.json> --pid <python pid>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def n_jobs(p: Path) -> int:
    try:
        return len(json.loads(p.read_text()).get("jobs", []))
    except Exception:
        return -1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--max-wait-min", type=float, default=None)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args(argv)
    st, t0 = Path(a.state), time.time()
    n0 = n_jobs(st)
    print(f"[restart] {st} jobs={n0} pid={a.pid}", flush=True)
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print("[restart] process already gone", flush=True)
            return
        n = n_jobs(st)
        timeout = a.max_wait_min is not None and time.time() - t0 > a.max_wait_min * 60
        if n > n0 or timeout:
            os.kill(a.pid, signal.SIGTERM)
            print(f"[restart] {'timeout' if timeout else 'job saved'} jobs={n}: SIGTERM {a.pid}", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
