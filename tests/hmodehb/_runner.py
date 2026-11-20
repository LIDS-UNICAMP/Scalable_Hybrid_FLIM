"""Processo controlador usado pelos testes de kill -9 / dois controladores / adocao.

    python tests/hmodehb/_runner.py OUT [--arm A] [--slow S] [--real] [--set a.b=V ...]

``H4_CRASH=k:fase`` mata o PROPRIO processo com SIGKILL (kill -9 real) num ponto exato:
  start | before_result | after_result   (dentro do avaliador; after_result = result.json ja gravado)
  after_hjob                             (hjobs/<k>.json gravado, antes do _save do estado)
  after_save                             (estado salvo com k+1 jobs)
Cada chamada real do avaliador e anotada em OUT/../calls.log (contagem de re-treinos).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _helpers as H  # noqa: E402

from ga_search.hmodehb import integrity as I  # noqa: E402
from ga_search.hmodehb import hsearch as HS  # noqa: E402
from ga_search.hmodehb import run_hsearch as R  # noqa: E402

REAL_OV = ["data.synthetic=true", "data.image_size=32", "data.synthetic_train_samples=16",
           "data.synthetic_val_samples=12", "data.batch_size=4", "data.num_workers=0", "data.ram_cache=true",
           "train.max_steps_per_epoch=1", "search.fidelities=[1,2]", "search.max_brackets=1", "search.n_classes=2",
           "search.initial_size=5", "search.stratified.calibration=false", "search.stratified.extension_fidelities=[]",
           "search.stratified.memory_budget_gb=null", "search.spifil_images_choices=[1,2,4]"]


def real_cfg(out, overrides=()):
    cfg = R.load_hconfig("A", "remote_sensing", 42, ["device=cpu", "wandb.enabled=false", *REAL_OV,
                                                     f"output_dir={out}", *overrides])
    cfg["train"].pop("max_images_per_epoch", None)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--slow", type=float, default=0.0)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--allow-drift", action="append", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    calls = out.parent / "calls.log"
    crash = os.environ.get("H4_CRASH", "")
    ck, _, phase = crash.partition(":")

    real_write = I.atomic_write_json

    def write(path, value):                  # after_hjob: morre entre hjobs/<k>.json e o _save
        real_write(path, value)
        if phase == "after_hjob" and Path(path).name == f"{int(ck):06d}.json" and Path(path).parent.name == "hjobs":
            os.kill(os.getpid(), signal.SIGKILL)
    I.atomic_write_json = write

    real_save = HS.HMODEHB._save

    def save(self):
        real_save(self)
        if phase == "after_save" and len(self.state["jobs"]) == int(ck) + 1 and \
                len(self.state["jobs"]) >= len(getattr(self, "_replay", [])):
            os.kill(os.getpid(), signal.SIGKILL)
    HS.HMODEHB._save = save

    if a.real:
        from ga_search.hmodehb.hevaluate import HEvaluator
        cfg = real_cfg(out, a.set)
        ev = HEvaluator(cfg, out)
        cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(ev.spifil_pool.items())]
        base_call = ev._evaluate

        def logged(genome, fidelity, ctx, cand_dir, run):
            with calls.open("a") as f:
                f.write(json.dumps([ctx["job_index"], ctx["cand_id"], float(fidelity)]) + "\n")
            return base_call(genome, fidelity, ctx, cand_dir, run)
        ev._evaluate = logged
        base_full = ev.__call__

        class _Ev:                            # after_result: result.json gravado pelo Evaluator.__call__
            fitness_spec, fitness_sig, split_hash = ev.fitness_spec, ev.fitness_sig, ev.split_hash

            def __call__(self, g, fid, ctx):
                r = base_full(g, fid, ctx)
                if phase == "after_result" and int(ctx["job_index"]) == int(ck):
                    os.kill(os.getpid(), signal.SIGKILL)
                return r
        evaluator = _Ev()
    else:
        cfg = H.search_cfg(a.arm, overrides=[f"output_dir={out}", *a.set])
        fake = H.FakeEval(out)
        base = fake.__call__

        class _Ev:
            fitness_spec, fitness_sig = fake.fitness_spec, fake.fitness_sig

            def __call__(self, g, fid, ctx):
                with calls.open("a") as f:
                    f.write(json.dumps([ctx["job_index"], ctx["cand_id"], float(fid)]) + "\n")
                if a.slow:
                    time.sleep(a.slow)
                return base(g, fid, ctx)
        evaluator = _Ev()
    allow = [k for x in a.allow_drift for k in x.split(",") if k]
    s = R.build(cfg, evaluator, allow_drift=allow)
    try:
        st = s.run(resume=s.state_path.exists())
    except I.ControllerLocked as exc:
        print(f"LOCKED {exc}", flush=True)
        sys.exit(3)
    except I.ConfigDrift as exc:
        print(f"DRIFT {exc}", flush=True)
        sys.exit(4)
    R.write_outputs(s, st, out)
    print(json.dumps({"n_jobs": len(st["jobs"]), "fp": H.digest(H.fingerprint(st))}), flush=True)


if __name__ == "__main__":
    main()
