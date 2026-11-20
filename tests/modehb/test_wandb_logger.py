"""Tests for ga_search.modehb.wandb_logger (WANDB_MODE=disabled/offline, HV mocked)."""
import json
import os

import numpy as np
import pytest

from ga_search.modehb import wandb_logger as wl


def _fake_hv2d(points, ref):
    """Exact 2-D minimization HV (test double for I9's implementation)."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def _fake_contrib(points, ref):
    pts = np.asarray(points, dtype=float)
    total = _fake_hv2d(pts, ref)
    return np.array([total - _fake_hv2d(np.delete(pts, i, axis=0), ref) for i in range(len(pts))])


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setattr(wl, "_hv_funcs", lambda: (_fake_hv2d, _fake_contrib))


GENOME = {"init": "flim", "ssl": "lejepa", "depth": 3, "operators": ["conv", "pool"],
          "experimental_group": "e2_hybrid", "reg": "sigreg"}


def _candidate(tmp_path, mode=None):
    return wl.start_candidate(GENOME, "eggs", 3, "parent123", 2, 7, "splithash", "archhash1234", "abc123",
                              mode=mode, alerts_path=tmp_path / "alerts.jsonl")


def test_candidate_job_type_tags_config(tmp_path):
    c = _candidate(tmp_path)
    kw = c.init_kwargs
    assert kw["job_type"] == "candidate" and kw["group"] == "eggs"
    assert kw["project"] == "progressive_mo_dehb" and kw["entity"] == "ophira-ai"
    for t in ["dataset:eggs", "fidelity:3", "init:flim", "ssl:lejepa", "depth:3", "operators:conv+pool",
              "group:e2_hybrid"]:
        assert t in kw["tags"]
    cfg = kw["config"]
    assert cfg["genome"] == GENOME and cfg["parent_id"] == "parent123" and cfg["generation"] == 2
    assert cfg["split_hash"] == "splithash" and cfg["git_commit"] == "abc123" and cfg["seed"] == 7
    c.log({"loss": 1.0}, step=0)
    c.log_manual_metrics(dataloader_wait=0.1, latency_per_img=0.002, flops=1e6, gpu_hours_cum=0.5)
    c.log_table("t", ["a"], [[1]])
    c.finish()


def test_candidate_offline_run_attributes(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    c = _candidate(tmp_path)
    assert c.run.job_type == "candidate"
    assert "init:flim" in c.run.tags and c.run.group == "eggs"
    assert c.run.config["arch_hash"] == "archhash1234"
    ckpt = tmp_path / "enc.ckpt"
    ckpt.write_bytes(b"x")
    art = c.log_checkpoint(ckpt, {"front_rank": 0})
    assert art.type == "encoder" and art.metadata["generation"] == 2 and art.metadata["front_rank"] == 0
    c.finish()


def test_alert_writes_jsonl(tmp_path):
    c = _candidate(tmp_path)
    c.alert("NaN loss", "loss became NaN at step 10", level="ERROR")
    lines = (tmp_path / "alerts.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["title"] == "NaN loss" and rec["level"] == "ERROR" and rec["generation"] == 2
    c.finish()


def test_main_hv_excludes_low_fidelity():
    ctrl = wl.ControllerRun(ref_point=[1.0, 1.0], max_fidelity=9)
    assert ctrl.init_kwargs["job_type"] == "controller"
    hi = [{"objectives": [0.5, 0.5], "fidelity": 9, "init": "flim", "ssl": "lejepa", "quality": 0.5, "flops": 0.5}]
    # a dominating low-fidelity point must NOT change the main HV
    lo = [{"objectives": [0.0, 0.0], "fidelity": 1, "init": "random"}]
    data = ctrl.log_generation(hi + lo, fidelity=9, generation=0, gpu_hours=1.0, wall_time=10.0, n_evals=2)
    assert data["hv/best_cumulative"] == pytest.approx(0.25)
    assert data["hv/population"] == pytest.approx(0.25)
    assert data[f"{wl.LOWFID_PREFIX}hv_population/fid-1"] == pytest.approx(1.0)
    assert "hv_by_init/flim" in data and "hv_by_init/random" not in data
    assert "pareto/quality_vs_flops" in data and "tables/hv_contribution" in data

    hi2 = [{"objectives": [0.2, 0.6], "fidelity": 9, "init": "random", "quality": 0.8, "flops": 0.6}]
    d2 = ctrl.log_generation(hi2, fidelity=9, generation=1, gpu_hours=3.0, wall_time=20.0, n_evals=4,
                             meta={"n_promotions": 2})
    assert d2["hv/best_cumulative"] == pytest.approx(0.25 + 0.3 * 0.4)
    assert d2["hv/marginal"] == pytest.approx(0.12)
    assert d2["promotion/delta_hv_per_promotion"] == pytest.approx(0.06)
    assert d2["stagnation/evals_since_last_gain"] == 0
    assert d2["cost/gpu_hours_per_hv_gain"] == pytest.approx(2.0 / 0.12)
    # low-fidelity-only generation: main HV frozen, stagnation grows
    d3 = ctrl.log_generation([{"objectives": [0.0, 0.0], "fidelity": 3}], fidelity=3, generation=2,
                             gpu_hours=4.0, wall_time=30.0, n_evals=6)
    assert d3["hv/best_cumulative"] == pytest.approx(0.37) and d3["hv/marginal"] == 0
    assert d3["stagnation/evals_since_last_gain"] == 2
    ctrl.finish()


def test_diversity_and_ci():
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert wl.front_spacing(front) == pytest.approx(0.0)
    assert wl.front_spread(front) == pytest.approx(np.sqrt(2))
    assert wl.pareto_mask(np.array([[0, 1], [1, 1], [1, 0]])).tolist() == [True, False, True]
    ci = wl.hv_ci([[0.1, 0.2, 0.3], [0.1, 0.4, 0.5, 0.9]])
    assert ci["n"] == 2 and len(ci["mean"]) == 3
    assert ci["lo"][1] < ci["mean"][1] < ci["hi"][1]


def test_external_eval_and_dashboard(tmp_path):
    ev = wl.ExternalEvalRun("parent123", dataset="eggs", local_path=tmp_path / "svm.jsonl")
    assert ev.init_kwargs["job_type"] == "external_eval" and ev.init_kwargs["config"]["parent_id"] == "parent123"
    ev.add_result(macro_f1=0.8, kappa=0.7, balanced_acc=0.75, per_class={"a": 0.9, "b": 0.6})
    ev.add_result(macro_f1=0.82, kappa=0.71, balanced_acc=0.76)
    assert len(ev.table.data) == 2
    assert len((tmp_path / "svm.jsonl").read_text().splitlines()) == 2
    ev.finish()
    paths = wl.export_dashboard_spec(tmp_path / "dash")
    spec = json.loads(open(paths["json"]).read())
    assert spec["name"] == wl.DASHBOARD_NAME and len(spec["sections"]) == 4
    assert "hv/best_cumulative" in open(paths["md"]).read()
