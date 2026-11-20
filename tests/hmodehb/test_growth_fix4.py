"""growth_fix4: (a) leitura fp32 determinística na SVM do growth, (b) adopted_as_parent + pai re-pontuado na
mesma leitura, (c) planner com n_children planos válidos (round-robin, ops variadas, pais tiny), ``ssl: none``."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402

PY = sys.executable
SYN = ["data.synthetic=true", "data.image_size=32", "data.synthetic_train_samples=16", "data.synthetic_val_samples=12",
       "data.batch_size=4", "data.num_workers=0", "data.ram_cache=true", "train.max_steps_per_epoch=1",
       "search.n_classes=2", "search.spifil_images_choices=[1,2]", "device=cpu", "wandb.enabled=false"]
REGISTRY = ROOT / "research/spifil_b0b4/agent1/seed_registry.csv"
EXACT_OPS = ([{"op": "deepen_residual_zero", "at": 2, "act": "silu", "kernel": 3}],
             [{"op": "add_skip_zero", "layer": 1}],
             [{"op": "deepen_residual_zero", "at": 1, "act": "gelu", "kernel": 3}, {"op": "add_skip_zero", "layer": 2}])


def _tiny(seed=0, widths=(12, 24, 24), res=32):
    L = [LayerSpec(channels=widths[0], kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=widths[1], kernel=3, conv="dwsep", norm="batch_dbn", activation="silu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=widths[2], kernel=3, conv="conv", norm="none", activation="gelu", residual="identity",
                   pool="none", flavor="spifil_ops", init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


@pytest.fixture(scope="module")
def gev(tmp_path_factory):
    out = tmp_path_factory.mktemp("gev")
    cfg = G.build_cfg("remote_sensing", 42, SYN, out)
    return G.GrowthEvaluator(cfg, out / "eval")


# ───────────────────────────────────────────────────────────────── (a) leitura fp32
def test_fp32_reading_context_restores_flags():
    ev = SimpleNamespace(embed_amp=True)
    c0, m0 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    with G.fp32_reading(ev):
        assert ev.embed_amp is False
        assert torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
    assert ev.embed_amp is True
    assert torch.backends.cudnn.allow_tf32 == c0 and torch.backends.cuda.matmul.allow_tf32 == m0


def test_run_search_embed_amp_switch_static():
    """run_search._embed: autocast só liga se ``embed_amp`` (padrão True, histórico) e device cuda."""
    src = (ROOT / "ga_search/modehb/run_search.py").read_text()
    assert 'amp = self.device.type == "cuda" and bool(getattr(self, "embed_amp", True))' in src
    gsrc = (ROOT / "ga_search/hmodehb/growth_chain.py").read_text()
    assert "ev.embed_amp = False" in gsrc and "with fp32_reading(self.ev):" in gsrc
    tsrc = (ROOT / "ga_search/hmodehb/growth_test_channel.py").read_text()
    assert "with fp32_reading(ev):" in tsrc


def test_evaluator_declares_precision(gev):
    assert gev.ev.embed_amp is False
    assert gev.fitness_spec["embed_precision"] == G.EMBED_PRECISION == gev.embed_precision
    from ga_search.hmodehb import fitness_v2 as FV
    assert gev.fitness_sig == FV.signature(gev.fitness_spec)


@pytest.mark.parametrize("ops", EXACT_OPS)
def test_alpha0_child_equals_parent_embedding_and_kappa_bitwise(gev, ops):
    """Com alpha=0 (âncora do morfismo exato), o embedding do filho == pai bit a bit na leitura do growth,
    e o kappa (C=100, sem scaler, cabeça 1280) é idêntico."""
    from torch import nn
    m = _tiny()
    res = G.realize_plan(m, ops, None)
    child = res.child.eval()
    anchor = {k: v.detach().clone() for k, v in child.state_dict().items()}
    trained = {k: (v + 0.1 * torch.randn_like(v)) if v.is_floating_point() else v for k, v in anchor.items()}
    G._blend_into(child, anchor, trained, 0.0)
    child.eval()
    ev = gev.ev
    with G.fp32_reading(ev):
        ep = ev._embed(nn.Sequential(*m.blocks), m.head, ev.val_x).numpy()
        ec = ev._embed(nn.Sequential(*child.blocks), child.head, ev.val_x).numpy()
    assert ep.shape[1] >= 1280 and ep.dtype == np.float32
    assert np.array_equal(ep, ec)
    gev.reset_stamps()
    sp, sc = gev.score(m), gev.score(child)
    assert sp["kappa"] == sc["kappa"]
    assert sp["embed_precision"] == sc["embed_precision"] == G.EMBED_PRECISION
    assert float(gev.ev.cfg["eval"]["svm_fitness"].get("C", 100.0)) == 100.0


def test_score_is_repeatable_bitwise(gev):
    m = _tiny(seed=3)
    gev.reset_stamps()
    assert gev.score(m)["kappa"] == gev.score(m)["kappa"]


# ───────────────────────────────────────────────────────────────── (b) aceitação
def _r(**kw):
    base = {"status": "ok", "kappa_val_orig": 0.60, "kappa_parent_rescored": 0.60, "wise_alpha": 0.25,
            "preservation_verdict": "exact", "parent_kappa_val_orig": 0.6747}
    base.update(kw)
    return base


def test_decide_rules():
    # caso flowers B1g0c00: alpha=0, kappa abaixo do pai por ruído -> é o pai, não é rejeitado
    assert G.decide(_r(wise_alpha=0.0, kappa_val_orig=0.6656, kappa_parent_rescored=0.6718), 0.0, False) == \
        ("adopted_as_parent", False)
    # caso RS B1g0c00: alpha=0 com kappa IDÊNTICO -> também adopted (não promove, não ocupa a faixa)
    assert G.decide(_r(wise_alpha=0.0, kappa_val_orig=0.5719, kappa_parent_rescored=0.5719), 0.0, False) == \
        ("adopted_as_parent", False)
    assert G.decide(_r(wise_alpha=0.0), 0.0, True) == ("adopted_as_parent", False)
    # aceitação real contra o pai RE-PONTUADO (não contra o kappa gravado no membro)
    assert G.decide(_r(kappa_val_orig=0.61, kappa_parent_rescored=0.60, parent_kappa_val_orig=0.70), 0.0, False) == \
        ("accepted", True)
    assert G.decide(_r(kappa_val_orig=0.59, kappa_parent_rescored=0.60, parent_kappa_val_orig=0.10), 0.0, False) == \
        ("rejected", False)
    assert G.decide(_r(kappa_val_orig=0.61, kappa_parent_rescored=0.60), 0.02, False) == ("rejected", False)
    assert G.decide({"status": "failed"}, 0.0, False) == ("failed", False)


# ───────────────────────────────────────────────────────────────── ssl: none (ablação)
def test_ssl_disabled_flag():
    assert not G.ssl_disabled(G.GROWTH_DEFAULTS)
    for v in ("none", "None", None, False, "off"):
        assert G.ssl_disabled({"ssl": v})
    assert not G.ssl_disabled({"ssl": "lejepa"})


def test_train_child_ssl_none_skips_global_ssl(gev):
    m = _tiny()
    res = G.realize_plan(m, [{"op": "widen", "widths": {"0": 16}, "seed": 1}], None)
    g = {**G.GROWTH_DEFAULTS, "ssl": "none", "wise_alphas": [0.0, 1.0]}
    model, out = G.train_child(gev, res, 1, g, 0, 0.5)
    assert out["ssl_skipped"] is True and out["ssl"] == "none" and out["ssl_steps"] == 0
    assert out["embed_precision"] == G.EMBED_PRECISION
    assert out["ae_blocks"] == [0, 1]


# ───────────────────────────────────────────────────────────────── (c) planner
def test_plan_children_synthetic_varied_and_round_robin():
    a, b = _tiny(widths=(4, 8, 8)), _tiny(widths=(32, 64, 64), seed=1)
    parents = [({"node_id": "a"}, a), ({"node_id": "b"}, b)]
    rng = np.random.default_rng(0)
    plans = G.plan_children(parents, "B1", 8, [list(rng.uniform(0, 1, 4)) for _ in range(8)], list(range(8)),
                            G.GROWTH_DEFAULTS)
    assert len(plans) == 8 and all(p["ops"] is not None for p in plans), [p.get("reason") for p in plans]
    assert [p["parent"]["node_id"] for p in plans[:4]] == ["a", "b", "a", "b"]
    assert len({p["family"] for p in plans}) >= 4
    for p in plans:
        r = G.realize_plan(parents[p["parent_index"]][1], p["ops"], None)
        assert G.band_of(G._params(r)) == "B1"
        gs = MO.finalize(r, torch.rand(2, 3, 32, 32))
        assert gs.preservation.verdict == "exact"


def test_plan_children_impossible_gives_reason():
    m = _tiny(widths=(64, 256, 256))           # já acima de B0 -> nenhum plano para B0 é possível
    pl = G.plan_children([({"node_id": "big"}, m)], "B0", 2, [[0.5] * 4] * 2, [0, 1], G.GROWTH_DEFAULTS)
    assert all(p["ops"] is None for p in pl)
    assert all("acima da faixa" in p["reason"] for p in pl)


def _registry_seeds(ds):
    rows = []
    if REGISTRY.exists():
        for r in csv.DictReader(open(REGISTRY)):
            if r["dataset"] == ds and r["regua"] == "ORIG" and r["faixa"] == "B0" and r.get("checkpoint") \
                    and r.get("utilizavel", "sim") == "sim" and (ROOT / r["checkpoint"]).exists():
                rows.append(ROOT / r["checkpoint"])
    return rows


@pytest.mark.parametrize("ds", ["remote_sensing", "flowers"])
def test_real_seed_genomes_each_parent_ge2_plans_b1(ds):
    """Genomas REAIS das sementes B0 do seed_registry (seed00/01; a seed02 'SPiFiL pequeno' só entra se cair em
    B0): cada pai gera >= 2 planos válidos e distintos para B1; juntos, 8 filhos válidos sem 'sem plano'."""
    paths = _registry_seeds(ds)
    if len(paths) < 2:
        pytest.skip(f"checkpoints de semente de {ds} ausentes")
    parents = []
    for i, p in enumerate(paths):
        m, _ = G.backbone_from_production(p, ds, 224)
        parents.append(({"node_id": f"seed{i:02d}"}, m))
    try:
        ext, _ = G.backbone_from_external_small(ds, 224, seed=42)
        if G.band_of(ext.spec.cost.params_total) == "B0":
            parents.append(({"node_id": f"seed{len(parents):02d}"}, ext))
    except Exception:
        pass
    rng = np.random.default_rng(1)
    th = [list(rng.uniform(0, 1, 4)) for _ in range(8)]
    tiny_chain = False
    for meta, m in parents:
        pl = G.plan_children([(meta, m)], "B1", 8, th, list(range(8)), G.GROWTH_DEFAULTS)
        ok = [p for p in pl if p["ops"] is not None]
        assert len(ok) >= 2, (meta, m.spec.cost.params_total, [p.get("reason") for p in pl])
        assert len({G._plan_shape(p["ops"]) for p in ok}) == len(ok)
        for p in ok:
            r = G.realize_plan(m, p["ops"], None)
            assert G.band_of(G._params(r)) == "B1"
            assert r.child.spec.head.embed_dim >= 1280
        if m.spec.cost.params_total < 50_000:
            tiny_chain |= any(sum(o["op"] == "widen" for o in p["ops"]) >= 2 for p in ok)
    if any(m.spec.cost.params_total < 50_000 for _, m in parents):
        assert tiny_chain                       # o tiny chega a B1 por widens encadeados
    pl = G.plan_children(parents, "B1", 8, th, list(range(8)), G.GROWTH_DEFAULTS)
    assert all(p["ops"] is not None for p in pl), [p.get("reason") for p in pl]
    assert len({p["parent"]["node_id"] for p in pl}) == len(parents)
    fams = {p["family"] for p in pl}
    assert len(fams) >= 3 and any(f.startswith("widen") or f == "head_widen" for f in fams) \
        and any(f.startswith("deepen") for f in fams), fams


# ───────────────────────────────────────────────────────────────── controlador: decisão, kappas, sem 'sem plano'
def test_controller_decisions_and_two_kappas(tmp_path):
    seed = Path(G.BB_save(_tiny(widths=(16, 32, 32)), tmp_path / "seed_b0.pt")["path"])
    out = tmp_path / "ctl"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PYTHONPATH": str(ROOT)}
    g = ["seeds.registry=null", "seeds.external_small=false", f"seeds.checkpoints=[{seed}]", "n_children=4",
         "n_parents=1", "generations=1", "fid_low=1", "fid_high=1", "eta=1", "end_band=B1",
         "spifil_images_per_class=1", "test_channel=false", "oom.sleep_s=0", "wise_alphas=[0.0,1.0]"]
    cmd = [PY, "-m", "ga_search.hmodehb.growth_chain", "--dataset", "remote_sensing", "--out", str(out),
           *sum((["--set", s] for s in SYN), []), *sum((["--gset", s] for s in g), [])]
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    st = json.loads((out / "growth_state.json").read_text())
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    kids = [r for r in lin if r.get("op") != "seed"]
    assert len(kids) == 4 and not any(r.get("decision") == "no_plan" for r in kids), kids
    seedk = next(r for r in lin if r.get("op") == "seed")["kappa_val_orig"]
    members_b1 = {m["node_id"] for m in st["members"]["B1"]}
    for r in kids:
        assert r["status"] == "ok", r
        assert r["embed_precision"] == G.EMBED_PRECISION
        assert r["kappa_parent_rescored"] == seedk            # mesma leitura fp32 -> pai idêntico ao da semente
        assert r["parent_kappa_val_orig"] == r["kappa_parent_rescored"]
        assert r["kappa_val_orig"] is not None and r["family"]
        if r["wise_alpha"] == 0.0:
            assert r["decision"] == "adopted_as_parent" and not r["accepted"]
            assert r["node_id"] not in members_b1
        else:
            assert r["decision"] == ("accepted" if r["kappa_val_orig"] >= r["kappa_parent_rescored"] else "rejected")
            assert (r["node_id"] in members_b1) == r["accepted"]
        res = json.loads((out / "nodes" / r["node_id"] / "f1" / "result.json").read_text())
        assert res["embed_precision"] == G.EMBED_PRECISION and res["kappa_parent_rescored"] == r["kappa_parent_rescored"]
