"""Glue between search.yaml (I11 schema) and run_search, and W&B step re-basing."""
from pathlib import Path

from ga_search.modehb import run_search as RS
from ga_search.modehb import modehb as M
import tempfile

CFG = Path(__file__).resolve().parents[2] / "ga_search" / "config" / "modehb" / "search.yaml"


def test_search_yaml_translates():
    c = RS.load_config(CFG, "remote_sensing", None, ["search.max_brackets=2", "data.ram_cache=false"])
    assert c["dataset_name"] == "remote_sensing" and c["data"]["dataset_config"]["name"] == "remote_sensing"
    assert c["output_dir"].endswith("artifacts/modehb/remote_sensing/state/seed42")
    assert c["search"]["fidelities"] == [5, 15, 45] and c["search"]["max_brackets"] == 2
    assert c["search"]["spifil_fraction"] == 0.75 and c["search"]["epsilon"] == 0.1
    assert c["train"]["ae_ssl_split"] == [1, 2] and c["data"]["batch_size"] == 64
    assert c["eval"]["n_views"] in (4, 8) and c["data"]["ram_cache"] is False
    assert c["wandb"] == {"enabled": True, "project": "progressive_mo_dehb", "entity": "ophira-ai",
                          "mode": None}


class _Rec:
    def __init__(self):
        self.steps = []

    def log(self, data, step=None):
        self.steps.append(step)


def test_monotonic_steps_across_phase_restarts():
    r = _Rec()
    m = RS._MonotonicRun(r)
    for s in (0, 1, 2, 2, 0, 1, 0, 0, 5):
        m.log({}, step=s)
    m.log_now({})
    assert r.steps == sorted(r.steps)
    assert r.steps[:4] == [0, 1, 2, 2] and r.steps[4] == 3


# ---- SPiFiL donor pool (spifil_images_per_class gene)
from types import SimpleNamespace as _NS


def _mk(prefix, n_per_class, tmp):
    out = []
    for c in range(3):
        for j in range(n_per_class):
            p = tmp / f"{prefix}_{c}_{j}.png"
            p.touch()
            out.append(_NS(name=p.stem, image_path=p, mask_path=None, label=c))
    return out


def test_search_yaml_has_spifil_images_choices():
    c = RS.load_config(CFG, "remote_sensing", None, [])
    assert c["search"]["spifil_images_choices"] == [1, 2, 4, 8, 16]


def test_spifil_pool_train_only_nested_and_k(tmp_path):
    train, val = _mk("tr", 20, tmp_path), _mk("va", 20, tmp_path)
    selected = [train[5], train[25], train[45]]          # one per class, as progressive_data
    pool = RS.build_spifil_pool(selected, train, 16, seed=42)
    train_names = {s.name for s in train}
    assert all(s.name in train_names for v in pool.values() for s in v)
    assert not any(s.name.startswith("va") for v in pool.values() for s in v)
    assert [pool[c][0] for c in range(3)] == selected     # k=1 == previous protocol
    prev = []
    for k in (1, 2, 4, 8, 16):
        sub = RS.spifil_subset(pool, k)
        assert len(sub) == 3 * k and len({s.name for s in sub}) == 3 * k
        assert all(sum(s.label == c for s in sub) == k for c in range(3))
        assert set(s.name for s in prev) <= set(s.name for s in sub)
        prev = sub
    # deterministic by seed (a SPiFiL candidate and its random twin see the same images)
    again = RS.build_spifil_pool(selected, list(reversed(train)), 16, seed=42)
    assert [s.name for s in RS.spifil_subset(again, 8)] == [s.name for s in RS.spifil_subset(pool, 8)]
    other = RS.build_spifil_pool(selected, train, 16, seed=7)
    assert [s.name for s in RS.spifil_subset(other, 8)] != [s.name for s in RS.spifil_subset(pool, 8)]
    # small class: returns what exists; val donor rejected
    small = train[:20] + train[25:27] + train[40:]           # class 1 has only 2 training images
    assert [sum(s.label == c for s in RS.spifil_subset(RS.build_spifil_pool(selected, small, 16, 42), 4))
            for c in range(3)] == [4, 2, 4]
    import pytest
    with pytest.raises(ValueError):
        RS.build_spifil_pool([val[0], train[25]], train, 4, 42)


def test_evaluator_spifil_samples_uses_genome_gene(tmp_path):
    train = _mk("tr", 10, tmp_path)
    ev = RS.Evaluator.__new__(RS.Evaluator)
    ev.spifil_k_max = 4
    ev.spifil_pool = RS.build_spifil_pool([train[0], train[10], train[20]], train, 4, 42)
    k, d = ev.spifil_samples({"init": "spifil", "spifil_images_per_class": 2})
    assert k == 2 and len(d) == 6
    assert ev.spifil_samples({"init": "random", "spifil_images_per_class": 2})[1] == d   # twin: same donors
    assert ev.spifil_samples({})[0] == 1                     # gene absent -> old protocol
    assert ev.spifil_samples({"spifil_images_per_class": 16})[0] == 4    # clipped to pool size


# ---- fitness_mode: svm_kappa (supervised validation kappa/F1 fitness, opt-in)
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

CFG_V3 = CFG.parent / "search_v3.yaml"


def test_fitness_mode_absent_is_label_free():
    c = RS.load_config(CFG, "remote_sensing", None, [])
    assert "fitness_mode" not in c["eval"] and "svm_fitness" not in c["eval"]


def test_search_v3_translates_svm_kappa():
    c = RS.load_config(CFG_V3, "remote_sensing", None, [])
    assert c["eval"]["fitness_mode"] == "svm_kappa"
    assert c["eval"]["svm_fitness"] == {"C": 100, "scaler": "standard", "max_train": 4000, "kappa_weight": 1.0,
                                        "solver": "libsvm", "readout": "cov_head", "cov_k": 64, "tta": 1,
                                        "whiten": "none", "wise_alphas": [0.0, 0.25, 0.5, 0.75, 1.0]}
    assert "embedding_eval" in c["eval"] and c["eval"]["collapse_dimensional_multiplier"] == 0.5


class _DS(torch.utils.data.Dataset):
    def __init__(self, n, n_cls=3, off=0):
        self.samples = [types.SimpleNamespace(name=f"s{off + i}", label=i % n_cls) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return torch.full((3, 8, 8), float(s.label)) + 0.01 * i, s.label, s.name


def _fake_load_data(cfg, out):
    tr, va = _DS(30), _DS(12, off=100)
    sel = [tr.samples[0], tr.samples[1], tr.samples[2]]
    return (DataLoader(tr, batch_size=4), DataLoader(va, batch_size=4), sel, {"sha256": "x"})


class _Calls:
    def __init__(self):
        self.args = None

    def __call__(self, tr, ytr, va, yva, **kw):
        self.args = (tr, ytr, va, yva, kw)
        return {"kappa": 0.8, "macro_f1": 0.6, "balanced_acc": 0.7, "quality": 0.75, "fit_s": 0.01,
                "n_train_used": len(ytr)}


def _evaluator(tmp_path, monkeypatch, mode, ram_cache=True, max_train=10000):
    import ga_search.progressive_data as PD
    monkeypatch.setattr(PD, "load_data", _fake_load_data)
    monkeypatch.setattr(RS, "build_spifil_pool", lambda sel, train, k, seed: {s.label: [s] for s in sel})
    cfg = RS.load_config(CFG_V3 if mode else CFG, "remote_sensing", None,
                         ["device=cpu", f"data.ram_cache={str(ram_cache).lower()}", "data.batch_size=4",
                          "data.val_max_samples=null", "wandb.enabled=false",
                          f"eval.svm_fitness.max_train={max_train}"])
    if not mode:
        assert "fitness_mode" not in cfg["eval"]
    return RS.Evaluator.__new__(RS.Evaluator), cfg


def _init(ev, cfg, tmp_path):
    ev.cfg, ev.out, ev.device, ev.seed = cfg, tmp_path, torch.device("cpu"), 0
    ev._load_data()
    return ev


def test_load_data_keeps_val_labels_and_caches_train(tmp_path, monkeypatch):
    ev, cfg = _evaluator(tmp_path, monkeypatch, True, ram_cache=True)
    _init(ev, cfg, tmp_path)
    assert ev.val_y.tolist() == [i % 3 for i in range(12)] and len(ev.val_x) == 12
    assert len(ev.svm_train_x) == 30 and ev.svm_train_y.tolist() == [i % 3 for i in range(30)]
    ev2, cfg2 = _evaluator(tmp_path, monkeypatch, True, ram_cache=False, max_train=9)
    _init(ev2, cfg2, tmp_path)
    assert len(ev2.svm_train_x) == 9 and np.bincount(ev2.svm_train_y.numpy()).tolist() == [3, 3, 3]
    assert torch.allclose(ev2.svm_train_x[:, 0, 0, 0].round(), ev2.svm_train_y.float())  # x/y aligned
    ev3, cfg3 = _evaluator(tmp_path, monkeypatch, False)
    _init(ev3, cfg3, tmp_path)
    assert not hasattr(ev3, "svm_train_x")                   # label-free: no train labels cached


def _run_fase_d(tmp_path, monkeypatch, mode):
    from ga_search.modehb import ssl_trainers
    ev, cfg = _evaluator(tmp_path, monkeypatch, mode)
    _init(ev, cfg, tmp_path)
    calls = _Calls()
    monkeypatch.setitem(sys.modules, "ga_search.modehb.supervised_fitness",
                        types.SimpleNamespace(svm_fitness=calls))
    enc = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    monkeypatch.setattr(RS.Evaluator, "_grow", lambda self, *a, **k: (enc, [], []))
    monkeypatch.setattr(ssl_trainers, "train_global_ssl", lambda *a, **k: {"history": [], "steps": 0})
    monkeypatch.setattr(RS.M, "arch_hash", lambda g: "h")
    monkeypatch.setattr(RS, "profile", lambda *a, **k: {"inference_macs": 1.0, "latency_ms_mean": 0.1})
    genome = {"seed": 0, "depth": 1, "embed_dim": 1280, "pooling": "gap", "ssl": "lejepa", "reg": "none",
              "init": "random"}
    ctx = {"cand_id": "c0", "generation": 0, "parent_id": None, "role": "x", "rung": 0, "front_rank": None}
    res = ev(genome, 5, ctx)
    return res, calls


def test_fase_d_label_free_unchanged(tmp_path, monkeypatch):
    res, calls = _run_fase_d(tmp_path, monkeypatch, False)
    assert calls.args is None and "fitness_mode" not in res and "fitness_kappa" not in res
    assert res["quality"] == res["quality_raw"] * res["collapse_multiplier"]


def test_fase_d_svm_kappa_quality_from_svm_fitness(tmp_path, monkeypatch):
    import json
    res, calls = _run_fase_d(tmp_path, monkeypatch, True)
    tr, ytr, va, yva, kw = calls.args
    assert yva.tolist() == [i % 3 for i in range(12)] and len(ytr) == 30
    assert tr.shape[0] == 30 and va.shape[0] == 12 and tr.shape[1] == va.shape[1]
    sv = (ev_cfg_svm := kw)   # svm_fitness kwargs come from eval.svm_fitness (+ seed)
    assert sv["seed"] == 0 and "C" in sv and "scaler" in sv
    assert res["quality_raw"] == 0.75 and res["quality"] == 0.75 * res["collapse_multiplier"]
    assert res["fitness_mode"] == "svm_kappa" and res["fitness_kappa"] == 0.8
    assert res["fitness_macro_f1"] == 0.6 and "quality_label_free" in res and "metrics" in res
    saved = json.loads((tmp_path / "candidates" / "c0" / "f5" / "result.json").read_text())
    assert saved["fitness_kappa"] == 0.8 and saved["fitness_balanced_acc"] == 0.7


def test_search_v3_spifil_only_operators_from_start_space():
    c = RS.load_config(CFG_V3, "remote_sensing", None, [])["search"]
    assert c["random_baseline"] is False and c["stage1_brackets"] == 0 and c["initial_size"] >= 16
    assert c["initial_operators"] == "sampled" and c["initial_pyramid_growth"] == 2.0
    ss = M.normalize_search_space(c["search_space"])
    assert ss == {"conv": ("conv", "dwsep", "inception"), "softhebb": (True, False), "kernel": (3, 5, 7),
                  "activation": ("relu", "gelu", "silu", "selu"),
                  "norm": ("none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"),
                  "pool": ("none", "max", "avg"), "residual": (True, False), "pooling": ("gap", "gem", "moments"),
                  "protect": ("full", "lowlr", "freeze"), "projector": (True, False),
                  "ae_schedule": ("decoder_first", "joint")}
    s = M.MODEHB(c, lambda g, f, ctx: {}, Path(tempfile.mkdtemp()))
    pop = s._initial()
    assert len(pop) >= 16 and all(g["init"] == "spifil" for g in pop)
    assert len({L["activation"] for g in pop for L in g["layers"]}) > 1


def test_search_yaml_v1_unchanged_keys():
    c = RS.load_config(CFG, "remote_sensing", None, [])["search"]
    for k in ("random_baseline", "search_space", "stage1_brackets", "initial_operators", "initial_size"):
        assert k not in c or c[k] == M.DEFAULTS[k]
