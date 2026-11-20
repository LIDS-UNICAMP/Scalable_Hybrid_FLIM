"""End-to-end em CPU: kill -9 real (processo controlador separado) em pontos exatos e em instantes
aleatorios, dois controladores, e uma mini-busca REAL (HEvaluator: SPiFiL -> AE -> LeJEPA -> WiSE ->
SVM ORIG) com fidelidade 1, adocao no resume, NaN injetado e kill -9 depois do result.json."""
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import _helpers as H

RUNNER = Path(__file__).with_name("_runner.py")
PY = sys.executable


def _env(crash=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "PYTHONPATH": str(H.ROOT)}
    env.pop("H4_CRASH", None)
    if crash:
        env["H4_CRASH"] = crash
    return env


def _runner(out, *args, crash=None, timeout=600):
    return subprocess.run([PY, str(RUNNER), str(out), *args], cwd=H.ROOT, env=_env(crash), capture_output=True,
                          text=True, timeout=timeout)


def _fp(proc):
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _calls(out):
    p = Path(out).parent / "calls.log"
    return [tuple(json.loads(x)) for x in p.read_text().splitlines()] if p.exists() else []


def _paid_unique(out):
    st = json.loads(Path(out, "search_state.json").read_text())
    paid = [j for j in st["jobs"] if not j["cached"]]
    assert len({(j["cand_id"], j["fidelity"], j["role"]) for j in paid}) == len(paid)
    assert len({j["cand_id"] for j in st["jobs"]}) == st["n_candidates"]
    return st


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref") / "run"
    return _fp(_runner(out, "--arm", "B"))


# ---------------------------------------------------------------- kill -9 em pontos exatos
@pytest.mark.parametrize("k,phase,retrained", [(3, "start", True), (6, "before_result", True),
                                               (9, "after_result", False), (12, "after_hjob", False),
                                               (15, "after_save", False), (30, "after_result", False)])
def test_kill9_points_resume_identical(tmp_path, reference, k, phase, retrained):
    out = tmp_path / "run"
    p1 = _runner(out, "--arm", "B", crash=f"{k}:{phase}")
    assert p1.returncode == -signal.SIGKILL, p1.stderr[-2000:]
    disk = json.loads(Path(out, "search_state.json").read_text())
    assert len(disk["jobs"]) == (k + 1 if phase == "after_save" else k)       # estado nunca parcial
    got = _fp(_runner(out, "--arm", "B"))
    assert got == reference                                                   # mesma sequencia, F, chaves, ids
    n = Counter(c[0] for c in _calls(out))
    assert n[k] == (2 if retrained else 1)                                    # adotado = sem re-treino
    assert all(v == 1 for j, v in n.items() if j != k)
    st = _paid_unique(out)
    if phase in ("after_result", "after_hjob"):
        assert st["jobs"][k]["result"]["adopted"]["job"] == k
    assert not list(Path(out).rglob("*.tmp*"))


# ---------------------------------------------------------------- kill -9 em instantes aleatorios
def test_kill9_random_rounds(tmp_path, reference):
    """12 kill -9 externos em instantes aleatorios (inclusive durante import, replay, avaliacao e _save),
    depois uma execucao ate o fim: mesma sequencia que a execucao continua."""
    out = tmp_path / "run"
    rnd = random.Random(1234)
    kills, done = 0, None
    for r in range(13):
        p = subprocess.Popen([PY, str(RUNNER), str(out), "--arm", "B", "--slow", "0.3"], cwd=H.ROOT, env=_env(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            so, se = p.communicate(timeout=None if r == 12 else rnd.uniform(1.0, 7.0))
            assert p.returncode == 0, se[-3000:]
            done = json.loads(so.strip().splitlines()[-1])
            break
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGKILL)
            p.communicate()
            kills += 1
    assert done == reference and kills >= 10
    _paid_unique(out)
    n = Counter(c[0] for c in _calls(out))
    assert sum(v - 1 for v in n.values()) <= kills                            # so o job em voo de cada kill refeito


# ---------------------------------------------------------------- dois controladores
def test_two_controllers_second_exits(tmp_path, reference):
    out = tmp_path / "run"
    a = subprocess.Popen([PY, str(RUNNER), str(out), "--arm", "B", "--slow", "0.1"], cwd=H.ROOT, env=_env(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while not Path(out, "search_state.json").exists() and time.time() - t0 < 120:
        time.sleep(0.2)
    assert a.poll() is None
    b = _runner(out, "--arm", "B")
    assert b.returncode == 3 and "LOCKED" in b.stdout and str(a.pid) in b.stdout
    so, se = a.communicate(timeout=600)
    assert a.returncode == 0, se[-3000:]
    assert json.loads(so.strip().splitlines()[-1]) == reference
    assert all(v == 1 for v in Counter(c[0] for c in _calls(out)).values())


def test_config_drift_cli(tmp_path):
    out = tmp_path / "run"
    _fp(_runner(out, "--set", "search.max_brackets=1"))
    p = _runner(out, "--set", "search.max_brackets=1", "--set", "eval.svm_fitness.wise_alphas=[0.0]")
    assert p.returncode == 4 and "DRIFT" in p.stdout and "wise_alphas" in p.stdout
    p = _runner(out, "--set", "search.max_brackets=2", "--allow-drift", "search.max_brackets")
    assert p.returncode == 0
    assert json.loads(Path(out, "search_state.json").read_text())["drift_log"][0]["keys"] == ["search.max_brackets"]


# ---------------------------------------------------------------- mini-busca REAL em CPU (fidelidade 1)
@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    import _runner as RN
    from ga_search.hmodehb import run_hsearch as R
    from ga_search.hmodehb.hevaluate import HEvaluator
    from ga_search.modehb import supervised_fitness as SF
    out = tmp_path_factory.mktemp("real") / "run"
    cfg = RN.real_cfg(out)
    ev = HEvaluator(cfg, out)
    cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(ev.spifil_pool.items())]
    svm_calls = []
    real_svm = SF.svm_fitness

    def spy(*a, **k):
        svm_calls.append({"d": int(np.asarray(a[2]).reshape(len(a[2]), -1).shape[1]), **k})
        return real_svm(*a, **k)
    SF.svm_fitness = spy
    try:
        s = R.build(cfg, ev)
        st = s.run()
        R.write_outputs(s, st, Path(out))
    finally:
        SF.svm_fitness = real_svm
    return out, cfg, ev, s, st, svm_calls


def test_real_e2e_zero_failed_v2(real_run):
    out, cfg, ev, s, st, svm_calls = real_run
    assert st["done"] and len(st["jobs"]) >= 6 and all(j["status"] == "ok" for j in st["jobs"])
    assert {j["fidelity"] for j in st["jobs"]} == {1.0, 2.0}
    assert all(c["C"] == 100.0 and c["scaler"] == "none" and c["whiten"] == "none" and c["kappa_weight"] == 1.0
               and c["d"] >= 1280 for c in svm_calls) and svm_calls
    for j in st["jobs"]:
        r = j["result"]
        assert r["fitness_version"] == "fitness_v2" and r["embed_dim"] >= 1280 and r["collapse_multiplier"] == 1.0
        assert r["quality"] == min(max(r["kappa_val_orig"], 0.0), 1.0) and "aux" in r
        assert r["params"] <= 5_500_000 and j["bk"] == r["bk"]
        assert r["fitness_sig"] == st["fitness_sig"] == ev.fitness_sig and r["cand_key"] == j["cand_key"]
        from ga_search.hmodehb import integrity as I
        assert I.file_sha256(r["checkpoint"]) == r["checkpoint_sha256"]
        assert json.loads(Path(r["checkpoint"]).with_name("result.json").read_text())["cand_key"] == j["cand_key"]
    assert st["fitness_spec"]["val_sha256"] == ev.split_hash and st["fitness_spec"]["n_val"] == 12
    # A1b: TODA decisao (5 alphas do WiSE por job + kappa final) pela regua ORIG-val gravada
    from ga_search.hmodehb import fitness_v2 as FV
    want = FV.ruler_sig(st["fitness_spec"])
    paid = [j for j in st["jobs"] if not j["cached"]]
    assert all(j["result"]["decision_rulers"] == [want] for j in st["jobs"])
    assert all(j["result"]["n_svm_decisions"] == len(cfg["eval"]["svm_fitness"]["wise_alphas"]) for j in paid)
    assert len(svm_calls) == sum(j["result"]["n_svm_decisions"] for j in paid)
    for j in paid:                                     # alpha = argmax do kappa ORIG-val
        wk = j["result"]["aux"]["wise_kappas"]
        assert wk[str(j["result"]["wise_alpha"])] == max(wk.values()) == j["result"]["kappa_val_orig"]
    best = json.loads(Path(out, "best_by_bk.json").read_text())
    assert best["fitness_sig"] == st["fitness_sig"] and set(best["best"]) <= {"B0", "B1", "B2", "B3", "B4"}


def test_real_checkpoint_reloads_bitwise(real_run):
    import torch
    out, cfg, ev, s, st, _ = real_run
    j = st["jobs"][0]
    ck = torch.load(j["result"]["checkpoint"], map_location="cpu", weights_only=False)
    enc, head = ck["encoder"].eval(), ck["head"].eval()
    x = ev.val_x[:4]
    with torch.no_grad():
        a, b = head(enc(x)), head(enc(x))
    assert a.shape[1] >= 1280 and torch.equal(a, b) and ck["cand_id"] == j["cand_id"]


def test_real_resume_adopts_without_retraining(real_run, monkeypatch):
    from ga_search.hmodehb import run_hsearch as R
    out, cfg, ev, s, st, _ = real_run
    p = Path(out, "search_state.json")
    full = json.loads(p.read_text())
    k = len(full["jobs"]) - 1
    p.write_text(json.dumps({**full, "jobs": full["jobs"][:k], "done": False}))
    calls = []
    monkeypatch.setattr(ev, "_evaluate", lambda *a, **kw: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    s2 = R.build(cfg, ev)
    st2 = s2.run(resume=True)
    assert calls == [] and st2["jobs"][k]["result"]["adopted"]["job"] == k
    assert H.fingerprint(st2) == H.fingerprint(full)


def test_real_nan_in_ssl_fails_and_search_continues(tmp_path):
    import _runner as RN
    from ga_search.hmodehb import run_hsearch as R
    from ga_search.hmodehb.hevaluate import HEvaluator
    from ga_search.modehb import ssl_trainers as SSL
    out = tmp_path / "run"
    cfg = RN.real_cfg(out, ["search.initial_size=3"])
    ev = HEvaluator(cfg, out)
    cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(ev.spifil_pool.items())]
    real = SSL.train_global_ssl
    n = {"i": 0}

    def nan_once(*a, **k):
        log = real(*a, **k)
        n["i"] += 1
        return {**log, "final_loss_total": float("nan")} if n["i"] == 2 else log
    SSL.train_global_ssl = nan_once
    try:
        st = R.build(cfg, ev).run()
    finally:
        SSL.train_global_ssl = real
    failed = [j for j in st["jobs"] if j["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["result"]["reason"] == "nan" and failed[0]["job"] == 1
    assert st["done"] and sum(j["status"] == "ok" for j in st["jobs"]) >= 3
    assert (Path(out) / "candidates" / failed[0]["cand_id"] / "f1" / "error.txt").exists()


def test_real_kill9_after_result_adopted(tmp_path):
    out = tmp_path / "run"
    p1 = _runner(out, "--real", crash="2:after_result", timeout=900)
    assert p1.returncode == -signal.SIGKILL, p1.stderr[-3000:]
    st1 = json.loads(Path(out, "search_state.json").read_text())
    assert len(st1["jobs"]) == 2
    p2 = _runner(out, "--real", timeout=900)
    assert p2.returncode == 0, p2.stderr[-3000:]
    st = _paid_unique(out)
    n = Counter(c[0] for c in _calls(out))
    assert n[2] == 1 and all(v == 1 for v in n.values())                      # job 2 nao re-treinado
    assert st["jobs"][2]["result"]["adopted"]["job"] == 2
    # "op" do ultimo job entra no disco so no _save seguinte (mesmo atraso do MODEHB historico)
    strip = lambda js: [x[:-1] for x in H.fingerprint({"jobs": js})]
    assert strip(st["jobs"][:2]) == strip(st1["jobs"])
    assert all(j["status"] == "ok" for j in st["jobs"])


def test_real_cov_in_any_decision_fails(tmp_path, monkeypatch):
    """Se o WiSE/fitness voltar a ler cov_head (o bug medido pelo A1b), a busca para no 1o job."""
    import _runner as RN
    from ga_search.hmodehb import fitness_v2 as FV
    from ga_search.hmodehb import run_hsearch as R
    from ga_search.hmodehb.hevaluate import HEvaluator
    out = tmp_path / "run"
    cfg = RN.real_cfg(out, ["search.initial_size=2"])
    ev = HEvaluator(cfg, out)
    cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(ev.spifil_pool.items())]
    monkeypatch.setattr(ev, "_svm_features", lambda enc, head: ev._embed_cov(enc, head))
    with pytest.raises(FV.RulerViolation, match="regua"):
        R.build(cfg, ev).run()
    assert not Path(out, "hjobs", "000000.json").exists()
