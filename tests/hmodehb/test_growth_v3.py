"""growth v3 (cadeia de crescimento v3): cabeça de 2ª ordem fixa, genes (SSL/DA/whitening/SoftHebb), ops novas
(inception_branch_add, dense_concat, replace_bad_filters), TTA8 no teste, saturação/adoção e e2e mini REAL
(flowers, CPU, f1, poucas imagens, B0->B1->B2 com genes variados)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.hmodehb import so2_head as SO2  # noqa: E402

READOUT_X = ROOT / "research/spifil_b0b4/readout_x"


def _tiny(seed=0, res=32):
    L = [LayerSpec(channels=12, kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=24, kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor="spifil_ops", init="inherited"),
         LayerSpec(channels=24, kernel=3, conv="conv", norm="none", activation="relu", pool="none",
                   flavor="spifil_ops", init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


def _x(n=4, res=32, seed=1):
    return torch.rand(n, 3, res, res, generator=torch.Generator().manual_seed(seed))


# ───────────────────────────────────────────────────────────────── (1) cabeça de 2ª ordem == readout_x
@pytest.mark.skipif(not (READOUT_X / "eval_backbone_head.py").exists(), reason="readout_x ausente")
@pytest.mark.parametrize("C,nblocks", [(120, 1), (40, 3)])
def test_so2_head_equals_readout_x(C, nblocks):
    sys.path.insert(0, str(READOUT_X))
    import eval_backbone_head as E
    g = torch.Generator().manual_seed(3)
    train = [[torch.randn(5, C, 7, 7, generator=g) for _ in range(nblocks)] for _ in range(3)]
    val = [torch.randn(4, C, 7, 7, generator=g) for _ in range(nblocks)]
    ro = E.Readouts(n_px=64)
    head = SO2.So2Head(4096, 64, block_index=nblocks - 1)
    off = 0
    for bl in train:
        ids = list(range(off, off + bl[0].shape[0]))
        ro.add_train(bl, ids)
        head.add_train(bl[-1], ids)
        off += bl[0].shape[0]
    ro.fit([C] * nblocks)
    head.finish_fit()
    want = ro(val)["so2_4096_pca"]
    got = head(val[-1])
    assert got.shape == (4, 4096)
    assert torch.equal(got, want)
    assert sum(p.numel() for p in head.parameters()) == 0          # sem parâmetros treináveis


def test_so2_fit_and_read_train_only_and_deterministic():
    m = _tiny()
    enc = torch.nn.Sequential(*m.blocks)
    tr, va = _x(10, seed=2), _x(6, seed=5)
    h1, (a1, b1), gaps = SO2.fit_and_read(enc, tr, [va], "cpu", batch_size=4, block_index=2, gap_head=m.head)
    h2, (a2, b2), _ = SO2.fit_and_read(enc, tr, [va * 0 + 0.5], "cpu", batch_size=4, block_index=2)
    assert torch.equal(h1.V, h2.V) and torch.equal(a1, a2)            # a validação não entra na PCA
    assert a1.shape[1] == 4096 and gaps[0].shape == (10, 1280)
    h3, (a3, _), _ = SO2.fit_and_read(enc, tr, [va], "cpu", batch_size=3, block_index=2)
    assert float((a1 - a3).abs().max()) < 1e-4    # amostragem por imagem; só o arredondamento da soma fp64 muda
    _, (t8, _), _ = SO2.fit_and_read(enc, tr, [va], "cpu", batch_size=4, block_index=2, tta=8)
    assert not torch.equal(t8, a1)
    p = h1.payload()
    assert torch.equal(SO2.So2Head.from_payload(p)(enc(va)), b1)


def test_fitness_sig_changes_with_so2():
    spec = {"readout": "head"}
    s1 = G.FV.signature(spec)
    spec2 = {"readout": SO2.READOUT, "head_v3": SO2.spec_entry()}
    assert G.FV.signature(spec2) != s1
    assert G.FV.ruler({"readout": SO2.READOUT})["readout"] == SO2.READOUT


# ───────────────────────────────────────────────────────────────── genes
def test_genes_cover_space_deterministic():
    g = G.load_growth_config(None, [])
    g["v3"] = {"genes": {"ssl": ["lejepa", "simclr", "vicreg", "none"], "aug": ["weak", "medium", "strong"],
                         "whiten_new": ["none", "zca", "dbn"], "softhebb": [False, True]}}
    sp = G.gene_space(g)
    assert len(sp) == 4 * 3 * 3 * 2
    a = [G.child_genes(g, "B0:0", i) for i in range(len(sp))]
    assert a == [G.child_genes(g, "B0:0", i) for i in range(len(sp))]
    assert sorted(map(json.dumps, a)) == sorted(map(json.dumps, sp))   # rodízio cobre todas
    e = G.apply_genes(g, {"ssl": "vicreg", "aug": "strong", "whiten_new": "zca", "softhebb": True})
    assert e["ssl"] == "vicreg" and e["reg"] == "none" and e["v3"]["aug"] == "strong"
    assert G.apply_genes(g, {"ssl": "none"})["ssl"] == "none" and G.ssl_disabled(G.apply_genes(g, {"ssl": "none"}))
    assert G.apply_genes(g, {"ssl": "lejepa"})["reg"] == "sigreg"
    assert G.child_genes(G.load_growth_config(None, []), "B0:0", 0) == {}   # padrão: sem genes (histórico)


def test_defaults_are_historical():
    g = G.load_growth_config(None, [])
    assert not G.so2_on(g) and G.gene_space(g) == [{}] and G._v3(g)["ops"] == []
    m = _tiny()
    assert G.plan_families(m) == G.plan_families(m, g)


# ───────────────────────────────────────────────────────────────── ops novas no planner (exatas)
@pytest.mark.parametrize("fam", ["inception_branch_add", "dense_concat"])
def test_planner_new_ops_exact(fam):
    m = _tiny()
    g = G.load_growth_config(None, [])
    g["v3"] = {"ops": [fam], "families": [fam]}
    fams = G.plan_families(m, g)
    assert fams and all(f[0] == fam for f in fams)
    ok = 0
    for f in fams:
        ops, why = G.plan_with_family(m, "B1", f, [0.5] * 4, 7, g)
        if ops is None:
            continue
        assert ops[0]["op"] == fam
        res = G.realize_plan(m, ops, None, g)
        gs = MO.finalize(res, _x(), data={"split": "synthetic"})
        assert gs.preservation.verdict == "exact", (f, gs.preservation)
        assert G.band_of(res.child.spec.cost.params_total) == "B1"
        ok += 1
    assert ok >= 1


def test_whiten_gene_in_residual_op():
    m = _tiny()
    g = G.apply_genes(G.load_growth_config(None, []), {"whiten_new": "zca"})
    ops, why = G.plan_with_family(m, "B1", ("deepen_residual_zero", 2), [0.5] * 4, 3, g)
    assert ops and ops[0].get("norm") == "zca", why
    res = G.realize_plan(m, ops, None, g)
    gs = MO.finalize(res, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"


def test_replace_plan_is_last_and_noop_when_planning():
    m = _tiny()
    g = G.load_growth_config(None, [])
    g["v3"] = {"ops": ["replace_bad_filters"], "families": ["replace_bad_filters"]}
    ops, why = G.plan_with_family(m, "B1", ("replace_bad_filters", 1), [0.5] * 4, 3, g)
    assert ops and ops[-1]["op"] == "replace_bad_filters", why
    res = G.realize_plan(m, ops, None, g)
    assert getattr(res, "v3_pre_replace", None) is None


# ───────────────────────────────────────────────────────────────── replace nunca aceita pior
def test_replace_never_accepts_worse():
    base = {"status": "ok", "preservation_verdict": "exact", "kappa_parent_rescored": 0.60}
    # alpha=0 com âncora = pai -> é o pai (adotado), nunca "aceito"
    assert G.decide({**base, "wise_alpha": 0.0, "kappa_val_orig": 0.60}, 0.0, False) == ("adopted_as_parent", False)
    assert G.decide({**base, "wise_alpha": 0.5, "kappa_val_orig": 0.599}, 0.0, False) == ("rejected", False)
    assert G.decide({**base, "wise_alpha": 0.5, "kappa_val_orig": 0.60}, 0.0, False) == ("accepted", True)


def test_wise_anchor_parent_alpha0_is_parent_bitwise():
    """A âncora (= pai em função) entra com alpha=0 literalmente: o WiSE nunca fica abaixo do pai."""
    m = _tiny()
    anchor = {k: v.clone() for k, v in m.state_dict().items()}
    trained = {k: (v + 1.0 if v.is_floating_point() else v) for k, v in anchor.items()}
    G._blend_into(m, anchor, trained, 0.0)
    assert all(torch.equal(m.state_dict()[k], anchor[k]) for k in anchor)


# ───────────────────────────────────────────────────────────────── saturação / curva
class _GevStub:
    fitness_sig = "x" * 12
    readout = "so2_4096_pca"


def test_saturation_adopts_parent_and_curve(tmp_path):
    g = G.load_growth_config(None, [])
    g["v3"] = {"saturation": {"adopt_parent": True}}
    ch = G.GrowthChain({"seed": 42}, g, _GevStub(), tmp_path)
    ch.state = {"members": {b: [] for b in G.BANDS}}
    ch._lin_keys = set()
    tmp_path.mkdir(exist_ok=True)
    ch.state["members"]["B0"] = [{"node_id": "seed00", "band": "B0", "kappa_val_orig": 0.6, "params": 100,
                                  "flops_2mac": 1, "checkpoint": "c", "checkpoint_sha256": "s", "fid": 0,
                                  "parent": None, "theta": None}]
    ch._saturate("B0", "B1")
    assert ch.state["members"]["B1"][0]["adopted_from"] == "seed00"
    ch._saturate("B0", "B1")                                   # idempotente
    assert len(ch.state["members"]["B1"]) == 1
    curve = ch._curve()
    assert [r["kappa_val_orig"] for r in curve[:2]] == [0.6, 0.6] and curve[1]["adopted"]
    assert (tmp_path / "kappa_band_curve.json").exists()


# ───────────────────────────────────────────────────────────────── e2e REAL mini (flowers, CPU, f1, B0->B1->B2)
FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_SMALL = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_SMALL.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v3_flowers_b0_b1_b2(tmp_path):
    from ga_search.hmodehb import fitness_v2 as FV
    out = tmp_path / "fl"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=1"]
    cfg = G.build_cfg("flowers", 42, ov, out)
    g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml",
                             ["seeds.top_k=1", "n_parents=1", "n_children=4", "generations=1", "fid_low=1",
                              "fid_high=1", "end_band=B2", "spifil_images_per_class=1", "debug_train=48",
                              "debug_val=32", "test_channel=true", "seeds.external_small=false",
                              f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}",
                              "v3.softhebb_cfg.n_images=16", "v3.replace.n_images=16"])
    g["v3"]["genes"] = {"ssl": ["vicreg", "none", "lejepa", "simclr"], "aug": ["strong", "weak"],
                        "whiten_new": ["zca", "none"], "softhebb": [True, False]}
    g["v3"]["vicreg_kwargs"] = {"expander_dims": [64, 64, 64]}
    g["v3"]["families"] = None
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"], v3=g["v3"])
    assert gev.fitness_spec["readout"] == "so2_4096_pca"
    s = G.GrowthChain(cfg, g, gev, out).run()
    st = json.loads((out / "growth_state.json").read_text())
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    assert st["members"]["B0"] or st["members"]["B1"], "sem semente"
    kids = [r for r in lin if r.get("op") not in ("seed", None, "adopt_parent") and r.get("decision") != "no_plan"
            and r.get("status") != "infeasible_mem"]
    assert kids
    for r in kids:
        assert r["status"] == "ok", r
        assert r["readout"] == "so2_4096_pca" and r.get("genes")
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    assert res
    seen_ssl, seen_aug = set(), set()
    for r in res:
        FV.check_decisions(r, gev.fitness_spec)
        assert r["so2_head_sha256"] and Path(r["so2_head"]).exists()
        assert r["aux"]["gap"]["used_for_selection"] is False
        seen_ssl.add(r["ssl"])
        if not r["ssl_skipped"]:
            assert r["aug"]["crop_scale"] and "rot90_p" in r["aug"]          # DA registrada
            seen_aug.add(r["aug_gene"])
        if r.get("v3_replace"):
            assert r["anchor"].startswith("parent_function")
    assert len(seen_ssl) >= 2
    for r in kids:                                              # aceito => kappa >= pai re-pontuado (nunca pior)
        if r["decision"] == "accepted":
            assert r["kappa_val_orig"] >= r["kappa_parent_rescored"] - 1e-12
        assert r["decision"] in ("accepted", "rejected", "adopted_as_parent", "failed")
    for b in ("B1", "B2"):                                      # curva nunca piora
        for m in st["members"][b]:
            par = next((x for bb in st["members"].values() for x in bb if x["node_id"] == m["parent"]), None)
            if par is not None:
                assert m["kappa_val_orig"] >= par["kappa_val_orig"] - 1e-12
    curve = json.loads((out / "kappa_band_curve.json").read_text())["curve"]
    ks = [c["kappa_val_orig"] for c in curve if c["kappa_val_orig"] is not None]
    assert ks == sorted(ks) or len(ks) <= 1
    tests = [json.loads(x) for x in (out / "test_orig.jsonl").read_text().splitlines()] \
        if (out / "test_orig.jsonl").exists() else []
    if tests:
        for t in tests:
            assert t["never_feeds_fitness"] and t["readout"] == "so2_4096_pca"
            assert "kappa_test_tta8" in t and "kappa_test_gap" in t
    dump = Path(__import__("os").environ.get("GROWTH_E2E_DUMP", tmp_path))
    (dump / "e2e_v3_lineage.jsonl").write_text("\n".join(json.dumps(r) for r in lin))
    (dump / "e2e_v3_tests.jsonl").write_text("\n".join(json.dumps(r) for r in tests))


# ───────────────────────────────────────────────────────────────── receitas dos artigos / regras
def test_ssl_recipe_paper_records_fiel_adaptacao():
    g = G.load_growth_config(None, [])
    g["v3"] = {"ssl_recipe": "paper"}
    r = G.ssl_recipe(G.apply_genes(g, {"ssl": "lejepa"}))
    assert r["kwargs"]["n_views"] == 8 and r["kwargs"]["projector_dims"] == [2048, 2048, 128]
    assert r["kwargs"]["reg_kwargs"]["n_proj"] == 1024 and r["fiel"] and r["adaptacao"] and r["pendente"]
    s = G.ssl_recipe(G.apply_genes(g, {"ssl": "simclr"}))
    assert s["kwargs"]["tau"] == 0.5
    g["v3"]["recipe_variant"] = "fiel_bs256"
    v = G.ssl_recipe(G.apply_genes(g, {"ssl": "vicreg"}))
    assert v["kwargs"]["vicreg_kwargs"]["expander_dims"] == [8192] * 3
    assert G.ssl_recipe(G.apply_genes(g, {"ssl": "none"})) is None
    assert G.ssl_recipe(G.load_growth_config(None, [])) is None            # padrão: chamada histórica


def test_spifil_genes_and_strict_flag():
    g = G.load_growth_config(None, [])
    e = G.apply_genes(g, {"spifil_images_per_class": "all", "spifil_n_superpixels": 16})
    assert e["spifil_images_per_class"] >= 10 ** 6 and e["v3"]["spifil_n_superpixels"] == 16
    assert issubclass(G.SpifilFallback, G.ChainError)
    assert G._v3(g)["spifil_strict"] is False and G._v3(g)["control_no_grow"] is False


def test_ssl_trainer_projector_dims_paper():
    from ga_search.modehb import ssl_trainers as T
    enc = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU())
    head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 16))
    x = torch.rand(8, 3, 16, 16)
    out = T.train_global_ssl(enc, head, [x], "lejepa", "sigreg", 1, 1e-3, 1e-3, [], "cpu", None, 0, 0.05,
                             n_views=4, projector=True, projector_dims=[32, 32, 8], weight_decay=5e-4,
                             reg_kwargs={"n_proj": 64}, return_state=True)
    assert out["steps"] == 1 and "ssl_proj" in out["state"]["extra"]
    w = out["state"]["extra"]["ssl_proj"]
    assert any(v.shape == (8, 32) for v in w.values())
