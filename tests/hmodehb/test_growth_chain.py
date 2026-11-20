"""Cadeia de crescimento B0->B4 (ga_search/hmodehb/growth_chain.py): preservação, WiSE alpha=0, aceitação,
linhagem, teto, kill -9/resume e mini-cadeia e2e REAL (remote_sensing, poucas imagens, CPU, f1)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones import morphisms as MO  # noqa: E402
from ga_search.backbones.spec import HARD_CAP, HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402

PY = sys.executable
RS_B0 = ROOT / "artifacts/modehb_spifil/remote_sensing/state/seed42/candidates/c00306/f100/encoder.pt"


def _tiny(flavor="spifil_ops", seed=0, widths=(12, 24, 24), res=32):
    L = [LayerSpec(channels=widths[0], kernel=3, conv="conv", norm="batch", activation="relu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=widths[1], kernel=3, conv="dwsep", norm="batch_dbn", activation="silu", pool="max",
                   flavor=flavor, init="inherited"),
         LayerSpec(channels=widths[2], kernel=3, conv="conv", norm="none", activation="gelu", residual="identity",
                   pool="none", flavor=flavor, init="inherited")]
    spec = G.spec_from_layers(L, HeadSpec(embed_dim=1280, pooling="meanstd"), "synthetic", res)
    m = BB.build_backbone(spec, seed=seed, enforce_budget=False)
    with torch.no_grad():                      # estatísticas não triviais (BN/DBN) para o teste valer
        m.train()
        for _ in range(3):
            m(torch.rand(8, 3, res, res))
    return m.eval()


def _x(n=4, res=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, res, res, generator=g)


# ───────────────────────────────────────────────────────────────── preservação logo após o morfismo
@pytest.mark.parametrize("flavor", ["spifil_ops", "convblock"])
@pytest.mark.parametrize("target,seed", [("B1", 1), ("B1", 7), ("B2", 3), ("B2", 11)])
def test_preservation_after_morphism_le_1e5(flavor, target, seed):
    m = _tiny(flavor)
    ops = G.propose_plan(m, target, [0.5, 0.5, 0.5, 0.5], seed, G.GROWTH_DEFAULTS)
    assert ops, "sem plano"
    res = G.realize_plan(m, ops, None)
    gs = MO.finalize(res, _x(), data={"split": "synthetic"})
    assert gs.preservation.verdict == "exact"
    assert gs.preservation.emb_rel_l2 is not None and gs.preservation.emb_rel_l2 <= 1e-5
    assert res.child.spec.head.embed_dim == 1280 and res.child(_x()).shape[1] == 1280
    assert G.band_of(res.child.spec.cost.params_total) == target


@pytest.mark.parametrize("op", ["widen_last", "widen_first", "deepen_residual_zero", "add_skip_zero"])
def test_each_morphism_preserves_spifil_flavor(op):
    m = _tiny()
    r = MO.start(m)
    if op == "widen_last":
        r = MO.widen(r, G._tied_widths(m.spec, {1: 40}))
    elif op == "widen_first":
        r = MO.widen(r, {0: 20})
    elif op == "deepen_residual_zero":
        r = MO.deepen_residual_zero(r, 2)
    else:
        r = MO.add_skip_zero(r, 1)
    gs = MO.finalize(r, _x())
    assert gs.preservation.verdict == "exact" and gs.preservation.emb_rel_l2 <= 1e-5


@pytest.mark.skipif(not RS_B0.exists(), reason="checkpoint de produção RS c00306 ausente")
def test_production_conversion_bitwise_and_growth_preserves():
    x = _x(3, 64)
    m, info = G.backbone_from_production(RS_B0, "remote_sensing", 64, verify_x=x)
    assert info["conversion_bitwise"] and m.spec.cost.params_total == 196990
    for tgt, s in (("B1", 5), ("B2", 9)):
        ops = G.propose_plan(m, tgt, [0.9, 0.1, 0.5, 0.5], s, G.GROWTH_DEFAULTS)
        gs = MO.finalize(G.realize_plan(m, ops, None), x)
        assert gs.preservation.verdict == "exact" and gs.preservation.emb_rel_l2 <= 1e-5


# ───────────────────────────────────────────────────────────────── WiSE alpha = 0
def test_wise_alpha0_is_parent_bitwise():
    m = _tiny()
    x = _x()
    for ops in ([{"op": "deepen_residual_zero", "at": 2, "act": "silu", "kernel": 3}],
                [{"op": "add_skip_zero", "layer": 1}],
                [{"op": "deepen_residual_zero", "at": 1, "act": "gelu", "kernel": 3}, {"op": "add_skip_zero", "layer": 2}]):
        res = G.realize_plan(m, ops, None)
        child = res.child.eval()
        anchor = {k: v.detach().clone() for k, v in child.state_dict().items()}
        trained = {k: (v + 0.1 * torch.randn_like(v)) if v.is_floating_point() else v for k, v in anchor.items()}
        G._blend_into(child, anchor, trained, 1.0)
        assert not torch.equal(child(x), m(x))
        G._blend_into(child, anchor, trained, 0.0)
        child.eval()
        for k, v in child.state_dict().items():           # alpha=0 = âncora (pesos do morfismo), bit a bit
            assert torch.equal(v, anchor[k]), k
        with torch.no_grad():
            assert torch.equal(child(x), m(x))              # e a função do PAI, bit a bit


def test_wise_alpha0_widen_state_bitwise_and_function_within_tol():
    """widen: o estado alpha=0 é a âncora bit a bit; a função do pai é igual até o arredondamento da soma
    (canais extras mudam a ordem de acumulação da conv): rel-L2 <= 1e-5, registrado no GrowthSpec."""
    m = _tiny()
    x = _x()
    res = G.realize_plan(m, [{"op": "widen", "widths": {"0": 20, "1": 40, "2": 40}, "seed": 3}], None)
    child = res.child.eval()
    anchor = {k: v.detach().clone() for k, v in child.state_dict().items()}
    trained = {k: (v + 1.0) if v.is_floating_point() else v for k, v in anchor.items()}
    G._blend_into(child, anchor, trained, 0.5)
    G._blend_into(child, anchor, trained, 0.0)
    for k, v in child.state_dict().items():
        assert torch.equal(v, anchor[k])
    with torch.no_grad():
        a, b = m(x), child.eval()(x)
    assert float((a - b).norm() / a.norm()) <= 1e-5


# ───────────────────────────────────────────────────────────────── aceitação
def test_acceptance_rule():
    assert G.accept(0.50, 0.50, 0.0, 0.0, False)            # alpha=0 (= pai) entra com delta=0
    assert not G.accept(0.50, 0.50, 0.0, 0.0, True)         # require_improvement recusa alpha=0
    assert G.accept(0.52, 0.50, 0.02, 0.5, True)
    assert not G.accept(0.519, 0.50, 0.02, 0.5, False)
    assert not G.accept(0.49, 0.50, 0.0, 0.25, False)
    assert not G.accept(float("nan"), 0.1, 0.0, 0.5, False)
    assert not G.accept(None, 0.1, 0.0, 0.5, False)


def test_pareto_and_de_deterministic():
    import numpy as np
    items = [{"node_id": "a", "kappa_val_orig": 0.5, "flops_2mac": 10, "params": 1},
             {"node_id": "b", "kappa_val_orig": 0.6, "flops_2mac": 20, "params": 1},
             {"node_id": "c", "kappa_val_orig": 0.4, "flops_2mac": 30, "params": 1}]
    assert [r["node_id"] for r in G.pareto_order(items)] == ["b", "a", "c"]
    pool = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5], [0.9, 0.8, 0.7, 0.6]]
    a = G.de_theta(pool, np.random.default_rng(3))
    assert a == G.de_theta(pool, np.random.default_rng(3)) and all(0 <= v <= 1 for v in a)


# ───────────────────────────────────────────────────────────────── teto e cabeça
def test_hard_cap_never_exceeded_and_head_never_shrinks():
    m = _tiny(widths=(64, 256, 256), res=32)
    for s in range(6):
        ops = G.propose_plan(m, "B4", [1.0, 0.0, 1.0, 0.5], s, {**G.GROWTH_DEFAULTS, "max_ops": 40})
        if ops is None:
            continue
        r = G.realize_plan(m, ops, None)
        assert r.child.spec.cost.params_total <= HARD_CAP
        assert r.child.spec.head.embed_dim >= 1280
    with pytest.raises(ValueError):
        G.load_growth_config(None, ["hard_cap=6000000"])
    assert G.propose_plan(m, "B3", [0.5] * 4, 0, G.GROWTH_DEFAULTS, hard_cap=1_000) is None
    with pytest.raises(BB.BudgetError):
        r = MO.widen(MO.start(m), G._tied_widths(m.spec, {1: 2000}))
        MO.finalize(r)


def test_controller_never_reads_test_channel():
    src = (ROOT / "ga_search/hmodehb/growth_chain.py").read_text()
    assert "test_orig" not in src and "--split" not in src
    tsrc = (ROOT / "ga_search/hmodehb/growth_test_channel.py").read_text()
    assert '"never_feeds_fitness": True' in tsrc and '"used_for_selection": False' in tsrc


# ───────────────────────────────────────────────────────────────── controlador (sintético) + kill -9
SYN = ["data.synthetic=true", "data.image_size=32", "data.synthetic_train_samples=16", "data.synthetic_val_samples=12",
       "data.batch_size=4", "data.num_workers=0", "data.ram_cache=true", "train.max_steps_per_epoch=1",
       "search.n_classes=2", "search.spifil_images_choices=[1,2]", "device=cpu"]


def _seed_ckpt(tmp: Path) -> Path:
    m = _tiny(widths=(16, 32, 32))
    return Path(G.BB_save(m, tmp / "seed_b0.pt")["path"])


def _gargs(seed_ck: Path, extra=()):
    return ["--gset", "seeds.registry=null", "--gset", "seeds.external_small=false",
            "--gset", f"seeds.checkpoints=[{seed_ck}]", "--gset", "n_children=2", "--gset", "n_parents=1",
            "--gset", "generations=1", "--gset", "fid_low=1", "--gset", "fid_high=2", "--gset", "eta=1",
            "--gset", "end_band=B2", "--gset", "spifil_images_per_class=1", "--gset", "test_channel=false",
            "--gset", "oom.sleep_s=0", *extra]


def _run(out: Path, seed_ck: Path, env_extra=None, extra=()):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PYTHONPATH": str(ROOT),
           **(env_extra or {})}
    cmd = [PY, "-m", "ga_search.hmodehb.growth_chain", "--dataset", "remote_sensing", "--out", str(out),
           *sum((["--set", s] for s in SYN), []), *_gargs(seed_ck, extra)]
    return subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)


def _summary(out: Path):
    st = json.loads((out / "growth_state.json").read_text())
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    mem = {b: sorted((m["node_id"], round(m["kappa_val_orig"], 9)) for m in v) for b, v in st["members"].items()}
    return st, lin, mem


@pytest.fixture(scope="module")
def synth_ref(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gsyn")
    ck = _seed_ckpt(tmp)
    out = tmp / "ref"
    p = _run(out, ck)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return tmp, ck, out


def test_lineage_records(synth_ref):
    _, _, out = synth_ref
    st, lin, mem = _summary(out)
    keys = [r["cand_key"] for r in lin]
    assert len(keys) == len(set(keys))                        # sem duplicata
    kids = [r for r in lin if r.get("op") not in ("seed", None)]
    assert kids, lin
    for r in kids:
        for k in ("parent", "op", "delta_kappa", "preservation_rel_l2", "params", "flops_2mac", "kappa_val_orig",
                  "parent_kappa_val_orig", "accepted", "fid"):
            assert k in r, k
        if r["status"] == "ok":
            assert r["preservation_rel_l2"] <= 1e-5 and r["preservation_verdict"] == "exact"
            assert abs(r["delta_kappa"] - (r["kappa_val_orig"] - r["parent_kappa_val_orig"])) < 1e-12
            assert r["params"] <= HARD_CAP
            # growth_fix4 (b): alpha=0 com preservação exata = adopted_as_parent (não entra na faixa)
            adopted = r["wise_alpha"] == 0.0 and r["preservation_verdict"] == "exact"
            assert r["decision"] == ("adopted_as_parent" if adopted else ("accepted" if r["accepted"] else "rejected"))
            assert r["accepted"] == (not adopted and G.accept(r["kappa_val_orig"], r["parent_kappa_val_orig"], 0.0,
                                                              r["wise_alpha"], False))
    for b, ms in st["members"].items():                       # todo membro está na própria faixa
        for m in ms:
            assert G.band_of(m["params"]) == b


def test_two_controllers_second_exits(synth_ref, tmp_path):
    from ga_search.hmodehb import integrity as I
    _, ck, _ = synth_ref
    out = tmp_path / "lock"
    with I.ControllerLock(out):
        p = _run(out, ck)
    assert p.returncode == 3, p.stdout[-2000:] + p.stderr[-2000:]


def test_config_drift_refused(synth_ref):
    tmp, ck, out = synth_ref
    p = _run(out, ck, extra=["--gset", "delta=0.5"])
    assert p.returncode == 4


def test_kill9_resume_identical(synth_ref):
    tmp, ck, ref = synth_ref
    st, lin, mem = _summary(ref)
    jobs = [r for r in lin if r.get("status") == "ok" and r.get("op") not in ("seed", None)]
    assert jobs
    victim = jobs[0]
    out = tmp / "killed"
    p = _run(out, ck, env_extra={"GROWTH_CRASH": f"{victim['node_id']}:f{victim['fid']}"})
    assert p.returncode == -signal.SIGKILL, (p.returncode, p.stderr[-2000:])
    assert not (out / "summary.json").exists()
    p = _run(out, ck)
    assert p.returncode == 0, p.stderr[-3000:]
    st2, lin2, mem2 = _summary(out)
    assert mem2 == mem                                          # mesmos aceitos, mesmos kappas
    sig = lambda L: sorted((r["node_id"], r["fid"], r.get("status"), r["accepted"],  # noqa: E731
                            None if r.get("kappa_val_orig") is None else round(r["kappa_val_orig"], 9)) for r in L)
    assert sig(lin2) == sig(lin)                                # mesma sequência de jobs e decisões
    assert len({r["cand_key"] for r in lin2}) == len(lin2)      # nenhuma linha duplicada no resume
    adopted = [r for r in lin2 if r.get("adopted")]
    assert any(r["node_id"] == victim["node_id"] for r in adopted) or \
        all(r["node_id"] != victim["node_id"] for r in lin2 if r.get("cand_key") == victim["cand_key"])
    assert not list(out.rglob("*.tmp*"))
    p = _run(out, ck)                                           # 3ª execução: tudo adotado, nada muda
    assert p.returncode == 0
    assert _summary(out)[2] == mem


# ───────────────────────────────────────────────────────────────── e2e REAL: remote_sensing, CPU, f1, B0->B1->B2
RS_ROOT = Path("/dados/home/moliveira/datasets/organized/remote_sensing")


@pytest.mark.skipif(not (RS_ROOT.exists() and RS_B0.exists()), reason="dados/checkpoint de RS ausentes")
def test_e2e_real_remote_sensing_b0_b1_b2(tmp_path):
    from ga_search.hmodehb import fitness_v2 as FV
    out = tmp_path / "rs"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=2"]
    cfg = G.build_cfg("remote_sensing", 42, ov, out)
    g = G.load_growth_config(None, ["seeds.external_small=false", "seeds.top_k=1", "n_parents=1", "n_children=2",
                                    "generations=1", "fid_low=1", "fid_high=1", "end_band=B2",
                                    "spifil_images_per_class=1", "debug_train=60", "debug_val=40",
                                    "test_channel=false", f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}"])
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"])
    s = G.GrowthChain(cfg, g, gev, out).run()
    st = json.loads((out / "growth_state.json").read_text())
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    assert st["members"]["B0"], "sem semente B0"
    kids = [r for r in lin if r.get("op") not in ("seed", None)]
    assert any(r["band"] == "B1" for r in kids), kids
    # B2 só nasce se B1 ganhou membro. O WiSE pode escolher alpha=0 (filho = pai, adopted_as_parent) e aí B1 fica sem
    # pai: resultado válido. Qual lado sai depende da seed do filho = hash do sha do checkpoint do pai, e o TorchScript
    # grava caminho/linha do código nesse checkpoint -> o teste não pode exigir um dos lados.
    assert any(r["band"] == "B2" for r in kids) == bool(st["members"]["B1"]), (kids, st["members"]["B1"])
    for r in kids:
        assert r["status"] == "ok", r
        assert r["preservation_rel_l2"] <= 1e-5
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    for r in res:
        FV.check_decisions(r, gev.fitness_spec)                  # alpha e kappa pela régua ORIG-val
        assert r["aux"]["wise_kappas"]["0.0"] is not None
        assert any(x["n_fit"] > 0 for x in r["new_filter_init"]) or not any(o["op"] == "widen" for o in r["ops"]) \
            or all(x["fallback"] for x in r["new_filter_init"])
    for b in ("B1", "B2"):                                      # cada aceito >= pai
        for m in st["members"][b]:
            par = next(x for bb in st["members"].values() for x in bb if x["node_id"] == m["parent"])
            assert m["kappa_val_orig"] >= par["kappa_val_orig"]
    (Path(os.environ.get("GROWTH_E2E_DUMP", tmp_path)) / "e2e_lineage.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lin))


# ───────────────────────────────────────────────────────────────── torch 2.2: weights_only sem bytes
def _pickle_globals(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return b"".join(z.read(n) for n in z.namelist() if n.endswith("data.pkl"))


def test_checkpoint_has_no_bytes_for_torch22(tmp_path):
    """torch 2.2 (athena) com weights_only=True recusa ``_codecs.encode`` (bytes): o v2 grava o TorchScript como
    tensor uint8; a simulação procura o global proibido no pickle."""
    from ga_search.backbones import checkpoint as CK
    m = _tiny()
    info = G.BB_save(m, tmp_path / "v2.pt")
    pk = _pickle_globals(info["path"])
    assert b"_codecs" not in pk
    ck = torch.load(info["path"], weights_only=True)
    assert ck["torchscript"].dtype == torch.uint8
    ts = torch.jit.load(__import__("io").BytesIO(CK.ts_bytes(ck["torchscript"])))
    x = _x()
    with torch.no_grad():
        assert torch.allclose(ts(x), m(x), atol=1e-6)
    assert G._is_backbone_ckpt(info["path"])


def test_old_v1_checkpoint_fallback_requires_sha(tmp_path, monkeypatch):
    from ga_search.backbones import checkpoint as CK
    m = _tiny()
    info = CK.save_backbone(m, tmp_path / "v1.pt", torchscript=False)
    ck = torch.load(info["path"], weights_only=True)
    ck["torchscript"] = b"\x00\x01legacy"                       # formato v1 (bytes)
    torch.save(ck, tmp_path / "v1.pt")
    sha = CK.file_sha256(tmp_path / "v1.pt")
    real = torch.load

    def load22(path, *a, **k):                                   # simula o torch 2.2
        if k.get("weights_only") and b"_codecs" in _pickle_globals(path):
            raise RuntimeError("Unsupported class _codecs.encode")
        return real(path, *a, **k)
    monkeypatch.setattr(torch, "load", load22)
    with pytest.raises(ValueError):
        CK.load_backbone(tmp_path / "v1.pt")
    with pytest.raises(ValueError):
        CK.load_backbone(tmp_path / "v1.pt", expected_sha256="0" * 64)
    m2, _ = CK.load_backbone(tmp_path / "v1.pt", expected_sha256=sha)
    with torch.no_grad():
        assert torch.equal(m2(_x()), m(_x()))
