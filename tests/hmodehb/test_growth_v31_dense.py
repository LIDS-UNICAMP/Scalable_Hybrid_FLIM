"""growth v3.1 bloco 3: add_dense_layer (DenseNet, cat(H(x), x), SPiFiL em H, colunas zero no consumidor) e
deepen_rezero com SPiFiL (bloco gated do bloco 1)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones import morphisms_v3 as MV3  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_growth_v31_covgate import _rand_filters, _tiny, _x  # noqa: E402


@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("at", [1, 2])
def test_add_dense_layer_exact(flavor, at):
    m = _tiny(flavor)
    out = MV3.add_dense_layer(MO.start(m), at, 6, new_filters=_rand_filters)
    x = _x()
    with torch.no_grad():
        assert torch.allclose(out.child.eval()(x), m(x), atol=1e-6, rtol=0)
    assert float(out.child.blocks[at].conv.weight.abs().sum()) > 0          # H com filtros (não zero)
    gs = MO.finalize(out, x, data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
    assert out.child.spec.layers()[at].dense_out
    assert not any(bool(v.any()) for k, v in out.masks.items() if k.startswith(f"blocks.{at}."))


def test_two_dense_layers_chain_and_ckpt(tmp_path):
    m = _tiny()
    r = MV3.add_dense_layer(MO.start(m), 2, 4, new_filters=_rand_filters)
    r = MV3.add_dense_layer(r, 3, 4, new_filters=_rand_filters)                # DenseNet: a 2ª vê x + H1
    MO.finalize(r)
    x = _x()
    with torch.no_grad():
        assert torch.allclose(r.child.eval()(x), m(x), atol=1e-6, rtol=0)
    info = G.BB_save(r.child, tmp_path / "d.pt")
    m2, _ = G.BB_load(info["path"], info["file_sha256"])
    with torch.no_grad():
        assert torch.equal(m2(x), r.child(x))


def test_add_dense_layer_refuses_tied_consumer():
    m = _tiny()
    with pytest.raises(MO.MorphismError):
        MV3.add_dense_layer(MO.start(m), 3, 4)                              # fim: a cabeça leria a passagem
    g = MV3.add_gated_residual(MO.start(m), 2, norm="none", gate="scalar", tag="deepen_rezero")
    with pytest.raises(MO.MorphismError):
        MV3.add_dense_layer(g, 2, 4)


def test_planner_add_dense_layer_family():
    m = _tiny()
    g = G.load_growth_config(None, [])
    g["v3"] = {"ops": ["add_dense_layer"], "families": ["add_dense_layer", "widen"]}
    fams = [f for f in G.plan_families(m, g) if f[0] == "add_dense_layer"]
    assert fams
    ok = 0
    for f in fams:
        ops, why = G.plan_with_family(m, "B1", f, [0.5] * 4, 9, g)
        if ops is None:
            continue
        res = G.realize_plan(m, ops, None, g)
        gs = MO.finalize(res, _x(), data={"split": "synthetic"})
        assert gs.preservation.verdict == "exact" and ops[0]["op"] == "add_dense_layer"
        ok += 1
    assert ok >= 1


FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_B0 = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_B0.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v31_dense_rezero_flowers(tmp_path):
    import json
    from ga_search.hmodehb import fitness_v2 as FV
    out = tmp_path / "fl"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=1"]
    cfg = G.build_cfg("flowers", 42, ov, out)
    g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml",
                             ["seeds.top_k=1", "n_parents=1", "n_children=3", "generations=1", "fid_low=1",
                              "fid_high=1", "end_band=B1", "spifil_images_per_class=1", "debug_train=48",
                              "debug_val=32", "test_channel=false", "seeds.external_small=false",
                              f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}",
                              "v3.so2.seed_mode=fixed"])
    g["v3"]["genes"] = {"ssl": ["none", "lejepa"]}
    g["v3"]["ops"] = ["add_dense_layer", "deepen_rezero"]
    g["v3"]["families"] = ["add_dense_layer", "deepen_rezero"]
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"], v3=g["v3"])
    G.GrowthChain(cfg, g, gev, out).run()
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    assert res
    kinds = set()
    for r in res:
        assert r["status"] == "ok", r.get("error")
        FV.check_decisions(r, gev.fitness_spec)
        assert r["preservation_verdict"] == "exact"
        kinds.add(r["ops"][0]["op"])
        recs = [x for x in r["new_filter_init"] if x.get("target") in ("add_dense_layer", "deepen_rezero")]
        assert recs and all(x["n_fit"] > 0 or x["fallback"] for x in recs)      # SPiFiL (ou fallback registrado)
    assert kinds == {"add_dense_layer", "deepen_rezero"}
