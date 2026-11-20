"""OOM on a shared GPU is retried (not logged as a failed candidate); fatal CUDA errors crash."""
import pytest

from ga_search.modehb import modehb as M


def _search(fn):
    s = M.MODEHB.__new__(M.MODEHB)
    s.state, s._replay, s.evaluate_fn = {"jobs": []}, [], fn
    return s


def test_oom_is_retried_then_ok(monkeypatch):
    monkeypatch.setattr(M, "OOM_RETRY_SLEEP_S", 0)
    calls = []

    def fn(g, f, ctx):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 980.00 MiB")
        return {"quality": 0.5}
    r = _search(fn)._call({}, 5.0, {"role": "search"})
    assert r["status"] == "ok" and len(calls) == 3


def test_persistent_oom_fails_after_retries(monkeypatch):
    monkeypatch.setattr(M, "OOM_RETRY_SLEEP_S", 0)
    calls = []

    def fn(g, f, ctx):
        calls.append(1)
        raise RuntimeError("InfeasibleCandidate: SPiFiL fit failed: OutOfMemoryError: CUDA out of memory")
    r = _search(fn)._call({}, 5.0, {"role": "search"})
    assert r["status"] == "failed" and r["oom_retries"] == M.OOM_RETRIES and len(calls) == M.OOM_RETRIES + 1


def test_other_errors_fail_once_and_fatal_cuda_raises():
    calls = []

    def bad(g, f, ctx):
        calls.append(1)
        raise ValueError("shape mismatch")
    assert _search(bad)._call({}, 5.0, {"role": "search"})["status"] == "failed" and len(calls) == 1

    def fatal(g, f, ctx):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError):
        _search(fatal)._call({}, 5.0, {"role": "search"})


def test_monitor_retries_errored_keys(tmp_path, monkeypatch):
    import json
    from ga_search.modehb import monitor_report as MR
    f = tmp_path / "t.jsonl"
    rows = [{"key": "a", "test_kappa": 0.5}, {"key": "b", "error": "x"},
            {"key": "c", "error": "x"}, {"key": "c", "error": "x"}, {"key": "c", "error": "x"}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(MR, "OUT_JSONL", f)
    assert MR.done_keys() == {"a", "c"}      # b errored once -> retried; c gave up after 3


def test_monitor_fit_sig_by_directory():
    from ga_search.modehb import monitor_report as MR
    assert MR._fit_sig({"config": {}}, "artifacts/modehb_hybrid_C01_final") == "C0.1_kw0.75"
    assert MR._fit_sig({"config": {}}, "artifacts/modehb_hybrid") == "C100_kw1"
    assert MR._fit_sig({"config": {}}, "artifacts/modehb_seeded") == "C100_kw1"
    assert MR._fit_sig({"config": {"eval": {"svm_fitness": {"C": 100, "kappa_weight": 1.0}}}}, "x_C01_y") == "C100_kw1"


def test_monitor_seed43_gets_own_key(tmp_path, monkeypatch):
    import json
    from ga_search.modehb import monitor_report as MR
    root = tmp_path / "artifacts" / "modehb_hybrid"
    for seed in ("seed42", "seed43"):
        d = root / "flowers" / "state" / seed
        d.mkdir(parents=True)
        (d / "search_state.json").write_text(json.dumps({"config": {}, "jobs": [
            {"status": "ok", "role": "search", "cand_id": "c00001", "fidelity": 5.0, "result": {"checkpoint": "x"}}]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MR, "ROOT_GLOBS", ["artifacts/modehb_hybrid"])
    monkeypatch.setattr(MR, "OUT_JSONL", tmp_path / "t.jsonl")
    keys = {t[8] for t in MR.pending()}
    assert keys == {"hybrid|flowers|C100_kw1|c00001|5", "hybrid_seed43|flowers|C100_kw1|c00001|5"}


def test_cpu_thread_cap(monkeypatch):
    import torch
    from ga_search.modehb import run_search as RS
    before = torch.get_num_threads()
    monkeypatch.setenv("MODEHB_THREADS", "3")
    try:
        assert RS._cap_cpu_threads() == 3 and torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)


def test_cudnn_engine_error_is_retried_like_oom(monkeypatch):
    monkeypatch.setattr(M, "OOM_RETRY_SLEEP_S", 0)
    calls = []

    def fn(g, f, ctx):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("GET was unable to find an engine to execute this computation")
        return {"quality": 0.5}
    r = _search(fn)._call({}, 5.0, {"role": "search"})
    assert r["status"] == "ok" and len(calls) == 2


def test_monitor_priority_and_shards(tmp_path, monkeypatch):
    import json
    from ga_search.modehb import monitor_report as MR
    d = tmp_path / "artifacts" / "modehb_hybrid" / "aid" / "state" / "seed42"
    d.mkdir(parents=True)
    jobs = [{"status": "ok", "role": "search", "cand_id": f"c{i:05d}", "fidelity": 5.0, "stratum": st,
             "result": {"checkpoint": "x", "fitness_kappa": q}}
            for i, (st, q) in enumerate([("tiny", 0.5), ("tiny", 0.7), ("huge", 0.4), ("huge", 0.3)])]
    (d / "search_state.json").write_text(json.dumps({"config": {}, "jobs": jobs}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MR, "ROOT_GLOBS", ["artifacts/modehb_hybrid"])
    monkeypatch.setattr(MR, "OUT_JSONL", tmp_path / "t.jsonl")
    order = [t[7]["cand_id"] for t in MR.pending()]
    assert set(order[:2]) == {"c00001", "c00002"}            # best of each stratum first
    monkeypatch.setenv("MONITOR_SHARD", "0/2")
    a = {t[8] for t in MR.pending()}
    monkeypatch.setenv("MONITOR_SHARD", "1/2")
    b = {t[8] for t in MR.pending()}
    assert a.isdisjoint(b) and len(a | b) == 4
