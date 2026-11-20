"""growth_fix5: estimador de pico de memória por plano, filtro infeasible_mem no planner, falha rápida de OOM de
capacidade (sem os 10 retries). Só CPU."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_growth_fix4 import _tiny  # noqa: E402

SMOKE = ROOT / "artifacts/hmodehb_b0b4/growth_smoke/jaci_gpu0/remote_sensing/nodes"
ATHENA = dict(batch=64, views=4, amp_bf16=True, ae_amp_bf16=True, ssl=True, grad_checkpoint=True, max_gb=None)
# mensagem real do athena (torch 2.2), filho que não cabe nem na placa vazia
OOM_CAP = ("CUDA out of memory. Tried to allocate 21.77 GiB. GPU 0 has a total capacity of 47.54 GiB of which "
           "21.35 GiB is free. Including non-PyTorch memory, this process has 25.46 GiB memory in use. Of the "
           "allocated memory 24.10 GiB is allocated by PyTorch")
OOM_CONT = ("CUDA out of memory. Tried to allocate 9.00 GiB. GPU 0 has a total capacity of 47.54 GiB of which "
            "3.10 GiB is free. Process 777 has 30.00 GiB memory in use. Including non-PyTorch memory, this process "
            "has 14.20 GiB memory in use.")


def _rs(name):
    p = SMOKE / name / "backbone.pt"
    if not p.exists():
        pytest.skip(f"checkpoint de semente ausente: {p}")
    return G.BB_load(p)[0]


def _widen(parent, idx, w):
    c = int(parent.spec.layers()[idx[0]].channels)
    return [G._widen_op(MO.start(parent), idx, w / c, 1)]


# ───────────────────────────────────────────────────────────────── (1) estimador
def test_estimator_monotone():
    m = _tiny(widths=(12, 24, 24), res=64)
    base = G.estimate_peak_gb(m, batch=8, views=4)
    assert base["ssl_gb"] > 0 and base["ae_gb"] > 0
    assert G.estimate_peak_gb(m, batch=16, views=4)["ssl_gb"] > base["ssl_gb"]
    assert G.estimate_peak_gb(m, batch=8, views=8)["ssl_gb"] > base["ssl_gb"]
    assert G.estimate_peak_gb(m, batch=8, views=4, amp_bf16=False)["ssl_gb"] > base["ssl_gb"]
    prev = base["peak_gb"]
    for w in (24, 48, 96):
        r = G.realize_plan(m, _widen(m, [0], w), None)
        e = G.estimate_peak_gb(r.child, batch=8, views=4)
        assert e["peak_gb"] > prev
        prev = e["peak_gb"]
    big = _tiny(widths=(12, 24, 24), res=128)
    assert G.estimate_peak_gb(big, batch=8, views=4)["ssl_gb"] > base["ssl_gb"]
    assert G.estimate_peak_gb(m, batch=8, views=4, grad_checkpoint=False)["ssl_gb"] > base["ssl_gb"]
    assert G.estimate_peak_gb(m, batch=8, ssl=False)["ssl_gb"] == 0.0


@pytest.mark.parametrize("w,lo,hi", [(376, 45.3, 47.2), (295, 40.6, 40.6)])
def test_estimator_calibration_rs_c00306(w, lo, hi):
    """B1 RS (c00306, widen nos blocos 0/1): medido 45,3–47,2 GB (376) e 40,6 GB (295); ±20%."""
    p = _rs("seed01")
    e = G.plan_memory(p, _widen(p, [0, 1], w), ATHENA)
    assert 0.8 * lo <= e["peak_gb"] <= 1.2 * hi, e


def test_estimator_oom_point_exceeds_athena():
    """Planner novo: B1g0c00 RS = c00306 widen 0/1 -> 455 (OOM 'Tried to allocate 21.77 GiB'): > 47,3 GB."""
    p = _rs("seed01")
    e = G.plan_memory(p, _widen(p, [0, 1], 455), ATHENA)
    assert e["peak_gb"] > 47.3, e


@pytest.mark.skip(reason="B1g0c03 (pai tiny, 14,3 GB) e flowers B1g0c00 (33,6 GB): os genomas exatos desses nós "
                         "do athena não estão nesta máquina; trazer growth_spec.json/backbone.pt para calibrar")
def test_estimator_calibration_athena_tiny_flowers():
    pass


# ───────────────────────────────────────────────────────────────── (2) filtro no planner
def test_resolve_max_gpu_gb_config_and_cpu():
    g = copy.deepcopy(G.GROWTH_DEFAULTS)
    g["mem"] = {**g["mem"], "max_gpu_gb": 30.0}
    assert G.resolve_max_gpu_gb(g, torch.device("cpu")) == 30.0
    if not torch.cuda.is_available():
        assert G.resolve_max_gpu_gb(G.GROWTH_DEFAULTS, torch.device("cpu")) is None


def test_mem_ctx_from_cfg():
    cfg = {"data": {"batch_size": 64}, "train": {"amp_dtype": "bf16"}}
    g = copy.deepcopy(G.GROWTH_DEFAULTS)
    g["mem"] = {**g["mem"], "max_gpu_gb": 45.0}
    c = G.mem_ctx_from(cfg, g)
    assert c == {"batch": 64, "views": 4, "amp_bf16": True, "ae_amp_bf16": True, "ssl": True,
                 "grad_checkpoint": True, "max_gb": 45.0}


def test_planner_marks_infeasible_and_replaces():
    a = _tiny(widths=(4, 8, 8), res=64)
    th = [list(np.random.default_rng(0).uniform(0, 1, 4)) for _ in range(4)]
    ctx = dict(batch=64, views=4, amp_bf16=True, ssl=True, grad_checkpoint=True, max_gb=None)
    free = G.plan_children([({"node_id": "a"}, a)], "B1", 4, th, list(range(4)), G.GROWTH_DEFAULTS, mem_ctx=ctx)
    peaks = sorted(G.plan_memory(a, p["ops"], ctx)["peak_gb"] for p in free if p["ops"])
    assert len(peaks) >= 2 and peaks[0] < peaks[-1]
    ctx["max_gb"] = (peaks[0] + peaks[-1]) / 2              # corta os planos de pico alto
    pl = G.plan_children([({"node_id": "a"}, a)], "B1", 4, th, list(range(4)), G.GROWTH_DEFAULTS, mem_ctx=ctx)
    inf = [x for p in pl for x in p["infeasible_mem"]]
    assert inf and all("infeasible_mem" in x["reason"] and x["mem_estimate"]["peak_gb"] > ctx["max_gb"] for x in inf)
    for p in pl:
        if p["ops"] is not None:
            assert p["mem_estimate"]["peak_gb"] <= ctx["max_gb"]
            assert G.band_of(G._params(G.realize_plan(a, p["ops"], None))) == "B1"
        else:
            assert "infeasible_mem" in p["reason"] or "plano repetido" in p["reason"]
    ctx["max_gb"] = 0.001                                    # nada cabe -> sem plano, motivo infeasible_mem
    pl = G.plan_children([({"node_id": "a"}, a)], "B1", 2, th[:2], [0, 1], G.GROWTH_DEFAULTS, mem_ctx=ctx)
    assert pl[0]["ops"] is None and "infeasible_mem" in pl[0]["reason"]


def test_planner_without_ctx_unchanged():
    a = _tiny(widths=(4, 8, 8))
    th = [[0.5] * 4] * 3
    p0 = G.plan_children([({"node_id": "a"}, a)], "B1", 3, th, [0, 1, 2], G.GROWTH_DEFAULTS)
    p1 = G.plan_children([({"node_id": "a"}, a)], "B1", 3, th, [0, 1, 2], G.GROWTH_DEFAULTS,
                         mem_ctx=dict(ATHENA))                 # max_gb None -> filtro desligado
    assert [p["ops"] for p in p0] == [p["ops"] for p in p1]


# ───────────────────────────────────────────────────────────────── (3) falha rápida de OOM
def test_oom_capacity_classifier():
    cap, d = G.oom_is_capacity(OOM_CAP)
    assert cap and abs(d["requested"] - 21.77) < 1e-6 and abs(d["this_process"] - 25.46) < 1e-6
    cap, d = G.oom_is_capacity(OOM_CONT)
    assert not cap and d["others"] > 20
    assert G.oom_is_capacity("CUDA out of memory.")[0] is False         # sem números: contenção (retries)
    # pedido > livre com a GPU só deste processo
    alone = ("Tried to allocate 10.00 GiB. GPU 0 has a total capacity of 47.54 GiB of which 8.00 GiB is free. "
             "Including non-PyTorch memory, this process has 39.30 GiB memory in use.")
    assert G.oom_is_capacity(alone, margin_gib=0.0)[0]


def _chain(tmp_path, exc_msg, retries=3):
    ch = object.__new__(G.GrowthChain)
    g = copy.deepcopy(G.GROWTH_DEFAULTS)
    g["oom"] = {"retries": retries, "sleep_s": 0}
    ch.g, ch.out = g, tmp_path
    ch.gev = SimpleNamespace(fitness_sig="sig", device=torch.device("cpu"))
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError(exc_msg)
    ch._run_job = boom
    return ch, calls


def test_job_fast_fail_capacity(tmp_path):
    ch, calls = _chain(tmp_path, OOM_CAP)
    r = ch._job({"node_id": "p"}, [], [0.5] * 4, 15, "B1g0c00", 1, "ck")
    assert len(calls) == 1 and r["status"] == "failed" and r["reason"] == "oom_capacidade"
    assert G.decide(r, 0.0, False) == ("failed", False)
    assert (tmp_path / "nodes/B1g0c00/f15/result.json").exists()


def test_job_contention_keeps_retries(tmp_path):
    ch, calls = _chain(tmp_path, OOM_CONT, retries=3)
    r = ch._job({"node_id": "p"}, [], [0.5] * 4, 15, "B1g0c01", 1, "ck")
    assert len(calls) == 4 and r["reason"] == "oom"


# ───────────────────────────────────────────────────────── oom.retry_failed (resume re-roda só reason "oom")


def _failed(tmp_path, node, reason):
    p = tmp_path / "nodes" / node / "f15" / "result.json"
    G.I.atomic_write_json(p, {"status": "failed", "reason": reason, "cand_key": "ck", "fitness_sig": "sig"})
    return p


@pytest.mark.parametrize("flag,reason,reran", [(False, "oom", False), (True, "oom", True),
                                               (True, "oom_capacidade", False), (True, "nan", False)])
def test_job_retry_failed_oom(tmp_path, flag, reason, reran):
    ch, calls = _chain(tmp_path, OOM_CONT, retries=0)
    ch.g["oom"]["retry_failed"] = flag
    ch._run_job = lambda *a, **k: calls.append(1) or {"status": "ok", "node_id": "B3g0c00"}
    p = _failed(tmp_path, "B3g0c00", reason)
    r = ch._job({"node_id": "p"}, [], [0.5] * 4, 15, "B3g0c00", 1, "ck")
    assert bool(calls) is reran and r.get("adopted", False) is (not reran)
    arch = list(p.parent.glob("result.json.oom_*"))
    assert len(arch) == int(reran) and (not reran or G.I.read_json(arch[0])["reason"] == "oom")


def _last_by_ckey(out):
    """Leitor do lineage: o ÚLTIMO registro de cada cand_key vale (retry de OOM acrescenta, nunca reescreve)."""
    import json
    return {r["cand_key"]: r for r in map(json.loads, (out / "lineage.jsonl").read_text().splitlines())}


def test_retry_appends_lineage_record(tmp_path):
    ch, calls = _chain(tmp_path, OOM_CONT, retries=0)
    ch.lineage_path, ch._lin_keys = tmp_path / "lineage.jsonl", set()
    par = {"node_id": "p", "checkpoint_sha256": "s", "kappa_val_orig": 0.5}
    r = ch._job(par, [], [0.5] * 4, 15, "B3g0c00", 1, "ck")                       # morre por OOM
    ch._lineage(ch._lin_rec(r, par, [], 15, False, final=False, decision="failed"))
    ch._run_job = lambda *a, **k: {"status": "ok", "node_id": "B3g0c00", "cand_key": "ck", "kappa_val_orig": 0.6}
    ch.g["oom"]["retry_failed"] = True
    r2 = ch._job(par, [], [0.5] * 4, 15, "B3g0c00", 1, "ck")
    ts = r2["retry_of"]
    assert (tmp_path / f"nodes/B3g0c00/f15/result.json.oom_{ts}").exists()
    assert G.I.read_json(tmp_path / "nodes/B3g0c00/f15/result.json")["retry_of"] == ts
    for _ in range(2):                    # o resume seguinte adota e NÃO duplica (dedup por cand_key + retry_of)
        ch._lineage(ch._lin_rec(r2, par, [], 15, True, final=False, decision="accepted"))
        ch._lin_keys = ch._lineage_keys()
    lines = (tmp_path / "lineage.jsonl").read_text().splitlines()
    assert len(lines) == 2
    last = _last_by_ckey(tmp_path)["ck"]
    assert last["retry_of"] == ts and last["status"] == "ok" and last["kappa_val_orig"] == 0.6


def test_retry_tool_record_bypasses_old_dedup(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location(
        "retry_oom_node", ROOT / "research/spifil_b0b4/growth/retry_oom_node.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ch, _ = _chain(tmp_path, OOM_CONT)
    par = {"node_id": "p", "checkpoint_sha256": "s", "kappa_val_orig": 0.5}
    (tmp_path / "lineage.jsonl").write_text(json.dumps({"cand_key": "ck", "status": "failed", "family": "fam"}) + "\n")
    ndir = tmp_path / "nodes/B1g1c00/f15"
    r = {"status": "ok", "node_id": "B1g1c00", "cand_key": "ck", "kappa_val_orig": 0.7, "delta_kappa": 0.2,
         "kappa_parent_rescored": 0.5, "wise_alpha": 0.5}
    cv = {"growth_chain_md5": "abc", "path": "/x/growth_chain.py", "mtime": "2026-11-18 02:02"}
    tool._record(G, G.I, ch, tmp_path, ndir, r, par, [], 15, ch.g, "20261118_210000", closed=True, code_version=cv)
    assert len((tmp_path / "lineage.jsonl").read_text().splitlines()) == 2
    last = _last_by_ckey(tmp_path)["ck"]
    assert last["retry_of"] == "20261118_210000" and last["family"] == "fam" and last["kappa_val_orig"] == 0.7
    assert last["used_for_selection"].startswith("never")
    assert G.I.read_json(ndir / "result.json")["retry_of"] == "20261118_210000"
    assert last["code_version"] == cv and G.I.read_json(ndir / "result.json")["code_version"] == cv
