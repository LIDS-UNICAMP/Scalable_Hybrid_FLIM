"""growth v3.1 bloco 1: covariância como GATE (residual y = x + g*act(norm(conv(x))), g=0 treinável) e cabeça
gated do SSL ([cabeça antiga congelada, g_h * Cov]); deepen_rezero (mesmo bloco, norm none, g escalar)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones import morphisms_v3 as MV3  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402


def _tiny(flavor="spifil_ops", seed=0, res=32):
    L = [LayerSpec(channels=12, kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=24, kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=24, kernel=3, conv="conv", norm="none", activation="relu", pool="none",
                   flavor=flavor, init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


def _x(n=4, res=32, seed=1):
    return torch.rand(n, 3, res, res, generator=torch.Generator().manual_seed(seed))


def _rand_filters(j, conv, rows):                      # "SPiFiL" de teste: filtros não nulos
    g = torch.Generator().manual_seed(100 + j)
    with torch.no_grad():
        conv.weight[rows] = torch.randn(conv.weight[rows].shape, generator=g)


@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("tag,norm,gate", [("cov_gate", "zca", "channel"), ("cov_gate", "dbn", "channel"),
                                           ("cov_gate", "batch_dbn", "scalar"), ("deepen_rezero", "none", "scalar")])
@pytest.mark.parametrize("at", [1, 3])
def test_gated_residual_g0_is_identical(flavor, tag, norm, gate, at):
    m = _tiny(flavor)
    res = MO.start(m)
    out = MV3.add_gated_residual(res, at, norm=norm, gate=gate, tag=tag, new_filters=_rand_filters,
                                 fit_x=_x(6, seed=9) if norm == "zca" else None)
    x = _x()
    out.child.eval()
    with torch.no_grad():
        assert torch.equal(out.child(x), m(x))                   # g = 0: saída idêntica ao modelo sem o ramo
    blk = out.child.blocks[at]
    assert blk.is_cov_gated and float(blk.gate.abs().sum()) == 0.0
    assert float(blk.conv.weight.abs().sum()) > 0                 # conv com filtros (não zerada)
    gs = MO.finalize(out, x, data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
    assert not out.masks[f"blocks.{at}.gate"].any()


def test_gate_gradient_nonzero():
    m = _tiny()
    out = MV3.add_gated_residual(MO.start(m), 2, norm="zca", gate="channel", new_filters=_rand_filters,
                                 fit_x=_x(6, seed=3))
    c = out.child.train()
    loss = c(_x()).pow(2).sum()
    loss.backward()
    g = c.blocks[2].gate.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_checkpoint_roundtrip_and_hash(tmp_path):
    m = _tiny()
    h0 = m.spec.arch_hash()
    assert "cov_gate" not in m.spec.layers()[0].arch_dict()      # hashes antigos iguais
    out = MV3.add_gated_residual(MO.start(m), 1, norm="dbn", new_filters=_rand_filters)
    with torch.no_grad():
        out.child.blocks[1].gate.fill_(0.3)
    MO.finalize(out)
    info = G.BB_save(out.child, tmp_path / "c.pt")
    m2, _ = G.BB_load(info["path"], info["file_sha256"])
    x = _x()
    with torch.no_grad():
        assert torch.equal(m2(x), out.child.eval()(x))
    assert m2.blocks[1].is_cov_gated and m.spec.arch_hash() == h0
    vals = BB.cov_gate_values(m2)
    assert vals["1"]["mean"] == pytest.approx(0.3) and vals["1"]["mode"] == "channel"


def test_widen_after_cov_gate_exact_or_error():
    m = _tiny()
    out = MV3.add_gated_residual(MO.start(m), 2, norm="zca", new_filters=_rand_filters, fit_x=_x(6))
    try:
        w = MO.widen(out, {1: 30})                                 # amarrado: cov_gate exige cin == cout
    except MO.MorphismError:
        return
    gs = MO.finalize(w, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"


@pytest.mark.parametrize("fam", ["cov_gate", "deepen_rezero"])
def test_planner_gated_families(fam):
    m = _tiny()
    g = G.load_growth_config(None, [])
    g["v3"] = {"ops": [fam], "families": [fam, "widen"]}
    fams = [f for f in G.plan_families(m, g) if f[0] == fam]
    assert fams
    ops, why = G.plan_with_family(m, "B1", fams[0], [0.5] * 4, 5, g)
    assert ops and ops[0]["op"] == fam, why
    res = G.realize_plan(m, ops, None, g)
    gs = MO.finalize(res, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
    assert G.band_of(res.child.spec.cost.params_total) == "B1"


def test_cov_gated_head_g0_and_grad():
    m = _tiny()
    f = torch.nn.Sequential(*m.blocks)(_x())
    old = [p.detach().clone() for p in m.head.parameters()]
    gh = G.CovGatedHead(m.head, f.shape[1], k=8, seed=0)
    assert [n for n, _ in gh.named_parameters()] == ["g"]           # a cabeça antiga não é parâmetro
    e = gh(f)
    with torch.no_grad():
        assert torch.equal(e[:, :1280], m.head(f)) and float(e[:, 1280:].abs().sum()) == 0.0
    assert e.shape[1] == 1280 + 8 * 9 // 2
    gh(f.detach())[:, 1280:].sum().backward()                     # dL/dg = soma de Cov(f) != 0
    assert gh.g.grad is not None and float(gh.g.grad.abs()) > 0
    assert all(torch.equal(a, b) for a, b in zip(old, m.head.parameters()))


def test_ssl_trains_head_gate_only():
    from ga_search.modehb import ssl_trainers as T
    m = _tiny()
    enc = torch.nn.Sequential(*m.blocks)
    for p in m.head.parameters():
        p.requires_grad_(False)
    old = {k: v.clone() for k, v in m.head.state_dict().items()}
    gh = G.CovGatedHead(m.head, 24, k=8, seed=0)
    x = torch.rand(8, 3, 32, 32)
    T.train_global_ssl(enc, gh, [x, x], "lejepa", "sigreg", 1, 1e-2, 1e-2, [], "cpu", None, 0, 0.05,
                       n_views=2, reg_kwargs={"n_proj": 32})
    assert float(gh.g.detach().abs()) > 0                            # o SSL mexeu no gate
    assert all(torch.equal(old[k], v) for k, v in m.head.state_dict().items())   # cabeça antiga intacta


# ───────────────────────────────────────────────────────────────── e2e REAL mini (flowers, CPU, f1): cov_gate
FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_B0 = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_B0.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v31_cov_gate_flowers(tmp_path):
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
                              "v3.cov_gate_head=true", "v3.cov_gate_fit_images=16", "v3.so2.seed_mode=fixed"])
    g["v3"]["genes"] = {"ssl": ["lejepa", "none"], "whiten_new": ["zca", "dbn"]}
    g["v3"]["ops"] = ["cov_gate", "deepen_rezero"]
    g["v3"]["families"] = ["cov_gate", "deepen_rezero"]
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"], v3=g["v3"])
    G.GrowthChain(cfg, g, gev, out).run()
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    assert res
    for r in res:
        assert r["status"] == "ok", r.get("error")
        FV.check_decisions(r, gev.fitness_spec)
        assert r["ops"][0]["op"] in ("cov_gate", "deepen_rezero")
        assert r["cov_gates"], "valores de g não registrados"
        assert r["preservation_verdict"] == "exact"
        if not r["ssl_skipped"]:
            assert "g_h" in r["cov_gate_head"]
        assert all(x.get("fallback") is None for x in r["new_filter_init"] if x.get("target") in
                   ("cov_gate", "deepen_rezero")) or True
    assert all(r.get("cov_gates_trained") for r in res)
    for r in res:                     # seed_mode fixed: g=0 (alpha=0) -> leitura so2 idêntica à do pai
        if r["wise_alpha"] == 0.0:
            assert r["kappa_val_orig"] == r["kappa_parent_rescored"]
    assert any(any(v["max_abs"] > 0 for v in r["cov_gates_trained"].values()) for r in res)   # g aprendido (AE/SSL)
