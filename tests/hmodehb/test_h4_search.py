"""Controller H-MODEHB em processo (avaliador sintetico): estado, ids, cand_key x cache, resume,
regua, deriva, OOM / NaN / CUDA fatal / ancestral ausente, adocao."""
import json
from collections import Counter
from pathlib import Path

import pytest

import _helpers as H
from ga_search.hmodehb import bk as BK
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import integrity as I
from ga_search.hmodehb import run_hsearch as R
from ga_search.modehb import modehb as M


def _run(out, arm="A", ev=None, overrides=(), allow=(), resume=None, **kw):
    cfg = H.search_cfg(arm, overrides=[f"output_dir={out}", *overrides])
    ev = ev or H.FakeEval(out, **kw)
    s = R.build(cfg, ev, allow_drift=allow)
    st = s.run(resume=s.state_path.exists() if resume is None else resume)
    return s, st, ev


@pytest.fixture(scope="module")
def ref_a(tmp_path_factory):
    out = tmp_path_factory.mktemp("refA")
    return _run(out)


@pytest.fixture(scope="module")
def ref_b(tmp_path_factory):
    out = tmp_path_factory.mktemp("refB")
    return _run(out, arm="B")


def test_state_ids_and_subpops(ref_a):
    s, st, _ = ref_a
    assert st["kind"] == "h4_A" and st["arm"] == "A" and st["id_prefix"] == "h4-A-"
    assert st["fitness_sig"] == H.FakeEval.fitness_sig and st["fitness_spec"] == H.SPEC
    assert st["config_sha"] == I.config_sha(st["config_full"]) == st["config_sha_initial"]
    for k in ("search", "eval", "train", "data", "hmodehb"):                  # config COMPLETO no estado
        assert k in st["config_full"]
    assert st["bk_bands"] == BK.bands_dict() and st["param_cap"] == 5_500_000 and st["drift_log"] == []
    assert all(j["cand_id"].startswith("h4-A-c") and len(j["cand_id"]) == len("h4-A-c00001") for j in st["jobs"])
    assert set(Counter(j["bk"] for j in st["jobs"])) == set(BK.BK_NAMES)          # 5 subpopulacoes presentes
    assert all(j["status"] == "ok" for j in st["jobs"])
    assert all(j["result"]["params"] <= 5_500_000 and BK.bk_of(j["result"]["params"]) == j["bk"] for j in st["jobs"])
    assert all(j["fitness_sig"] == st["fitness_sig"] and j["config_sha"] == st["config_sha"] for j in st["jobs"])
    paid = [j for j in st["jobs"] if not j["cached"]]                         # cache = registro sem custo
    assert len({(j["cand_id"], j["fidelity"], j["role"]) for j in paid}) == len(paid)
    assert len({j["cand_key"] for j in paid}) == len(paid)
    assert all(j["cand_key"] == st["jobs"][j["result"]["cached_from"]]["cand_key"] for j in st["jobs"] if j["cached"])
    assert all(len(list(Path(s.out, "hjobs").glob(f"{j['job']:06d}.json"))) == 1 for j in st["jobs"] if not j["cached"])


def test_deterministic(tmp_path, ref_a):
    _, st, _ = _run(tmp_path)
    assert H.fingerprint(st) == H.fingerprint(ref_a[1])


def test_promotion_keeps_lineage_new_cand_key(ref_a):
    st = ref_a[1]
    promos = [j for j in st["jobs"] if (j.get("op") or {}).get("type") in ("promotion", "extension")]
    assert promos
    for p in promos:
        low = [j for j in st["jobs"] if j["cand_id"] == p["cand_id"] and j["fidelity"] < p["fidelity"]]
        assert low and all(j["lineage_key"] == p["lineage_key"] and j["cand_key"] != p["cand_key"] for j in low)
        assert p["resumed_from"] is not None


def test_inherited_child_never_gets_cached_result(tmp_path):
    """Regressao da colisao heranca x cache (desenho 2.1): mesma arquitetura, pesos herdados -> outro job."""
    s, st, ev = _run(tmp_path, overrides=["search.max_brackets=1"])
    s._replay = []
    src = st["jobs"][0]
    inherit = {"checkpoint": src["result"]["checkpoint"], "n_blocks": 1, "parent_cand_id": src["cand_id"],
               "parent_fidelity": src["fidelity"], "parent_job": src["job"]}
    n_calls = len(ev.calls)
    rec = s._evaluate(src["genome"], src["fidelity"], parent_id=src["cand_id"], generation=9, rung=0,
                      inherit=inherit)
    assert len(ev.calls) == n_calls + 1 and not rec["cached"]                  # treinou de verdade
    assert rec["cand_id"] != src["cand_id"] and rec["cand_key"] != src["cand_key"]
    assert rec["arch_hash"] == src["arch_hash"] and rec["inherit"]["parent_cand_id"] == src["cand_id"]
    assert rec["weight_origin"]["parent_ckpt_sha256"] == src["result"]["checkpoint_sha256"]
    # o mesmo filho herdado de novo: AGORA sim e cache (mesma cand_key)
    again = s._evaluate(src["genome"], src["fidelity"], parent_id=src["cand_id"], generation=9, rung=0,
                        inherit=inherit)
    assert again["cached"] and again["cand_key"] == rec["cand_key"] and len(ev.calls) == n_calls + 1
    # e o MODEHB historico (intocado) teria descartado a heranca: resultado em cache do pai
    base = M.MODEHB(s.cfg, lambda g, f, c: {"status": "ok", "quality": 0.5, "flops": 1e8, "checkpoint": "x"},
                    tmp_path / "base")
    base.state, base._replay = base._fresh_state(), []
    first = base._evaluate(src["genome"], src["fidelity"], parent_id=None, generation=0, rung=0)
    second = base._evaluate(src["genome"], src["fidelity"], parent_id=None, generation=0, rung=0, inherit=inherit)
    assert second["cached"] and "inherit" not in second and second["cand_id"] == first["cand_id"]


def test_arm_b_inheritance_keys(ref_b):
    st = ref_b[1]
    inh = [j for j in st["jobs"] if j["weight_origin"]["kind"] == "inherit"]
    assert inh and all(j["cand_id"].startswith("h4-B-") for j in st["jobs"])
    by_cand = {(j["cand_id"], j["fidelity"]): j for j in st["jobs"]}
    for j in inh:
        o = j["weight_origin"]
        parent = by_cand[(o["parent_cand_id"], o["parent_fidelity"])]
        assert o["parent_ckpt_sha256"] == parent["result"]["checkpoint_sha256"] is not None
        assert j["cand_key"] != parent["cand_key"]


# ---------------------------------------------------------------- resume / regua / deriva
def test_resume_replays_identically_and_extends(tmp_path, ref_a):
    _, st1, _ = _run(tmp_path, overrides=["search.max_brackets=1"])
    n1 = len(st1["jobs"])
    with pytest.raises(I.ConfigDrift, match="search.max_brackets"):
        _run(tmp_path)                                                       # 1 -> 2 brackets sem permissao
    s, st, ev = _run(tmp_path, allow=["search.max_brackets"])
    assert H.fingerprint(st) == H.fingerprint(ref_a[1])                       # replay sem ReplayMismatch
    assert len(st["drift_log"]) == 1 and st["drift_log"][0]["keys"] == ["search.max_brackets"]
    assert st["config_sha_initial"] != st["config_sha"]
    assert ev.calls and min(k for k, _, _ in ev.calls) >= n1                  # nada do 1o processo re-treinado
    disk = json.loads(Path(tmp_path, "search_state.json").read_text())
    assert len(disk["jobs"]) == len(st["jobs"]) and disk["drift_log"] == st["drift_log"]


def test_resume_refuses_other_ruler(tmp_path):
    _run(tmp_path, overrides=["search.max_brackets=1"])
    other = H.FakeEval(tmp_path)
    other.fitness_spec = {**H.SPEC, "C": 0.1}
    cfg = H.search_cfg("A", overrides=[f"output_dir={tmp_path}", "search.max_brackets=1"])
    s = R.build(cfg, other)
    with pytest.raises(FV.FitnessSigMismatch):
        s.run(resume=True)


def test_resume_refuses_eval_drift(tmp_path):
    _run(tmp_path, overrides=["search.max_brackets=1"])
    with pytest.raises((I.ConfigDrift, FV.FitnessConfigError)):
        _run(tmp_path, overrides=["search.max_brackets=1", "eval.svm_fitness.max_train=10"])
    with pytest.raises(I.ConfigDrift, match="train.lr_new"):
        _run(tmp_path, overrides=["search.max_brackets=1", "train.lr_new=0.5"])


def test_replay_mismatch_on_tampered_key(tmp_path):
    _run(tmp_path, overrides=["search.max_brackets=1"])
    p = Path(tmp_path, "search_state.json")
    st = json.loads(p.read_text())
    st["jobs"][2]["cand_key"] = "tampered"
    p.write_text(json.dumps(st))
    with pytest.raises(M.ReplayMismatch):
        _run(tmp_path, overrides=["search.max_brackets=1"])


def test_fresh_run_refuses_existing_state(tmp_path):
    _run(tmp_path, overrides=["search.max_brackets=1"])
    with pytest.raises(FileExistsError):
        _run(tmp_path, overrides=["search.max_brackets=1"], resume=False)


# ---------------------------------------------------------------- falhas injetadas
def test_oom_retried_then_ok(tmp_path):
    s, st, ev = _run(tmp_path, overrides=["search.max_brackets=1", "hmodehb.oom.sleep_s=0"], oom_times=2)
    assert s.oom_retries == 4
    assert all(j["status"] == "ok" and j["result"]["oom_retries"] == 2 for j in st["jobs"] if not j["cached"])


def test_oom_exhausted_fails_without_loop(tmp_path):
    s, st, ev = _run(tmp_path, overrides=["search.max_brackets=1", "hmodehb.oom.retries=2",
                                          "hmodehb.oom.sleep_s=0"], oom_times=99)
    failed = [j for j in st["jobs"] if j["status"] == "failed"]
    assert failed and all(j["result"]["reason"] == "oom" and j["result"]["oom_retries"] == 2 for j in failed)
    assert all(n == 3 for n in Counter(k for k, _, _ in ev.calls).values())   # 1 + 2 tentativas, nunca mais
    assert all(j["F"] == [1.0, 1.0] for j in failed)


def test_nan_fails_and_search_continues(tmp_path, ref_a):
    s, st, ev = _run(tmp_path, nan_jobs={3})
    j = st["jobs"][3]
    assert j["status"] == "failed" and j["result"]["reason"] == "nan" and j["F"] == [1.0, 1.0]
    assert len(st["jobs"]) > 10 and sum(x["status"] == "failed" for x in st["jobs"]) == 1
    assert not any(x["cand_id"] == j["cand_id"] and x["fidelity"] > j["fidelity"] for x in st["jobs"])


def test_fatal_cuda_unrecorded_then_resume(tmp_path, ref_a):
    ev = H.FakeEval(tmp_path, fatal_jobs={4})
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _run(tmp_path, ev=ev)
    st = json.loads(Path(tmp_path, "search_state.json").read_text())
    assert len(st["jobs"]) == 4                                                # nada parcial do job 4
    assert json.loads(Path(tmp_path, "pending_job.json").read_text())["job"] == 4
    _, st2, _ = _run(tmp_path, ev=ev)
    assert H.fingerprint(st2) == H.fingerprint(ref_a[1])


def test_missing_ancestor_reason(tmp_path):
    ev = H.FakeEval(tmp_path, missing_jobs=set(range(200)))
    s, st, _ = _run(tmp_path, arm="B", ev=ev)
    inh = [j for j in st["jobs"] if j.get("inherit")]
    assert inh and all(j["status"] == "failed" and j["result"]["reason"] == "missing_ancestor" for j in inh)
    assert st["done"]


def test_param_cap_hard(tmp_path):
    s, st, ev = _run(tmp_path, overrides=["search.max_brackets=1"])
    s._replay = []
    big = M.repair({**st["jobs"][0]["genome"], "embed_dim": 2560}, **s.repair_kw)
    s.param_fn = lambda g: 6_000_000
    n = len(ev.calls)
    rec = s._evaluate(big, 5.0, parent_id=None, generation=9, rung=0)
    assert rec["status"] == "failed" and rec["result"]["reason"] == "param_cap" and len(ev.calls) == n
    s.param_fn = lambda g: 100_000                                           # estimado ok, medido acima
    ev2 = s.evaluate_fn
    s.evaluate_fn = lambda g, f, c: {**ev2(g, f, c), "params": 5_500_001}
    rec = s._evaluate({**big, "embed_dim": 1536}, 5.0, parent_id=None, generation=9, rung=0)
    assert rec["status"] == "failed" and rec["result"]["infeasible"] == "param_cap" and rec["bk"] is None


# ---------------------------------------------------------------- adocao (crash entre gravar e _save)
@pytest.mark.parametrize("source", ["hjobs", "candidate"])
def test_adoption_without_retraining(tmp_path, ref_a, source):
    _, st, ev = _run(tmp_path)
    p = Path(tmp_path, "search_state.json")
    full = json.loads(p.read_text())
    k = 7
    cut = {**full, "jobs": full["jobs"][:k], "done": False}
    p.write_text(json.dumps(cut))                                            # morreu antes do _save do job k
    j = full["jobs"][k]
    if source == "candidate":                                                # ... e antes do hjobs/<k>.json
        Path(tmp_path, "hjobs", f"{k:06d}.json").unlink()
    later = sorted(Path(tmp_path, "hjobs").glob("*.json"))
    for f in later:
        if int(f.stem) > k:
            f.unlink()
    for x in full["jobs"][k + 1:]:                                           # jobs seguintes nunca rodaram
        d = Path(tmp_path, "candidates", x["cand_id"], f"f{int(x['fidelity'])}")
        if (d / "result.json").exists() and not (x["cand_id"] == j["cand_id"] and x["fidelity"] == j["fidelity"]):
            (d / "result.json").unlink()
    ev2 = H.FakeEval(tmp_path)
    _, st2, _ = _run(tmp_path, ev=ev2)
    assert H.fingerprint(st2) == H.fingerprint(ref_a[1])
    assert k not in [c[0] for c in ev2.calls]                                 # job k adotado, nao re-treinado
    assert st2["jobs"][k]["result"]["adopted"]["job"] == k
    assert ("hjobs" in st2["jobs"][k]["result"]["adopted"]["from"]) == (source == "hjobs")


def test_adoption_refuses_corrupted_checkpoint(tmp_path):
    _, st, _ = _run(tmp_path, overrides=["search.max_brackets=1"])
    p = Path(tmp_path, "search_state.json")
    full = json.loads(p.read_text())
    k = max(j["job"] for j in full["jobs"] if not j["cached"])
    p.write_text(json.dumps({**full, "jobs": full["jobs"][:k], "done": False}))
    Path(full["jobs"][k]["result"]["checkpoint"]).write_bytes(b"truncated")
    ev2 = H.FakeEval(tmp_path)
    _, st2, _ = _run(tmp_path, ev=ev2, overrides=["search.max_brackets=1"])
    assert [c[0] for c in ev2.calls] == [k] and "adopted" not in st2["jobs"][k]["result"]


def test_decision_with_other_ruler_stops_search_unrecorded(tmp_path):
    """Qualquer decisao (aqui: o job 2 decidido por cov_head+scaler) para a busca sem gravar nada."""
    from ga_search.hmodehb import fitness_v2 as FV2
    ev = H.FakeEval(tmp_path)
    base = ev.__call__

    class _Cov:
        fitness_spec, fitness_sig = ev.fitness_spec, ev.fitness_sig

        def __call__(self, g, f, ctx):
            r = base(g, f, ctx)
            if ctx["job_index"] == 2:
                r["decision_rulers"] = [FV2.ruler_sig({**H.SPEC, "readout": "cov_head", "scaler": "standard"})]
            return r
    with pytest.raises(FV2.RulerViolation):
        _run(tmp_path, ev=_Cov())
    st = json.loads(Path(tmp_path, "search_state.json").read_text())
    assert len(st["jobs"]) == 2 and not Path(tmp_path, "hjobs", "000002.json").exists()


def test_all_decisions_read_orig_quality(ref_a):
    """Selecao/promocao/calibracao/extensao leem F/quality: F[0] = 1 - clip(kappa ORIG) em todo job."""
    st = ref_a[1]
    for j in st["jobs"]:
        r = j["result"]
        assert r["decision_rulers"] == [FV.ruler_sig(st["fitness_spec"])]
        assert j["F"][0] == pytest.approx(1 - min(max(r["kappa_val_orig"], 0), 1), abs=1e-12)
    cal = st["calibration"]
    for b, c in cal.items():
        path = {float(k): v for k, v in c["quality_by_epochs"].items()}
        js = {float(j["fidelity"]): j for j in st["jobs"] if j["cand_id"] == c["cand_id"] and not j["cached"]}
        assert all(path[f] == js[f]["result"]["kappa_val_orig"] for f in path)
