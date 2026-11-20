"""growth v3.1 blocos 3-6, e2e REAL mini (flowers, CPU, f1): add_dense_layer + deepen_rezero + cov_gate(zca) com
ZCA re-ajustada, SoftHebb do paper, poda dos canais novos, replace ReDo, SPiFiL no dwsep, AE em passada completa e
aceitação com bootstrap pareado da val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402

FL_ROOT = Path("/dados/home/moliveira/datasets/organized/flowers")
FL_B0 = ROOT / "artifacts/modehb_spifil/flowers/state/seed42/candidates/c00003/f15/encoder.pt"


@pytest.mark.skipif(not (FL_ROOT.exists() and FL_B0.exists()), reason="dados/semente de flowers ausentes")
def test_e2e_v31_blocks_3_to_6_flowers(tmp_path):
    from ga_search.hmodehb import fitness_v2 as FV
    out = tmp_path / "fl"
    ov = ["device=cpu", "wandb.enabled=false", "data.image_size=64", "data.batch_size=16", "data.num_workers=0",
          "train.max_steps_per_epoch=1"]
    cfg = G.build_cfg("flowers", 42, ov, out)
    g = G.load_growth_config(ROOT / "ga_search/config/hmodehb_b0b4/growth_v3.yaml",
                             ["seeds.top_k=2", "n_parents=2", "n_children=6", "generations=1", "fid_low=1",
                              "fid_high=1", "end_band=B1", "spifil_images_per_class=1", "debug_train=48",
                              "debug_val=32", "test_channel=false", "seeds.external_small=false",
                              f"seeds.registry={ROOT / 'research/spifil_b0b4/agent1/seed_registry.csv'}",
                              "v3.so2.seed_mode=fixed", "v3.zca_refit=true", "v3.zca_refit_images=16",
                              "v3.softhebb_paper=true", "v3.softhebb_cfg.n_images=16", "v3.prune_redundant_new=true",
                              "v3.prune.n_images=16", "v3.replace.criterion=redo", "v3.replace.n_images=16",
                              "v3.spifil_dwsep=true", "v3.full_epoch_ae=true", "v3.accept_bootstrap={n: 200, alpha: 0.05}",
                              "v3.cov_gate_fit_images=16", "v3.so2.eps_rel=0.001"])
    g["v3"]["genes"] = {"ssl": ["none", "lejepa"], "softhebb": [True, False], "whiten_new": ["zca"]}
    g["v3"]["ops"] = ["add_dense_layer", "deepen_rezero", "cov_gate", "replace_bad_filters"]
    g["v3"]["families"] = ["add_dense_layer", "deepen_rezero", "cov_gate", "replace_bad_filters", "widen"]
    gev = G.GrowthEvaluator(cfg, out / "eval", g["debug_train"], g["debug_val"], v3=g["v3"])
    assert gev.fitness_spec["head_v3"]["eps_rel"] == 0.001
    G.GrowthChain(cfg, g, gev, out).run()
    res = [json.loads(p.read_text()) for p in out.glob("nodes/*/f1/result.json")]
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    assert res
    ops_seen = set()
    for r in res:
        assert r["status"] == "ok", r.get("error")
        FV.check_decisions(r, gev.fitness_spec)
        assert "bootstrap" in r and r["bootstrap"]["split"] == "validation"
        assert r["ae_full_epoch"] is True and "_val_pred" not in r
        ops_seen |= {o["op"] for o in r["ops"]}
        for z in r.get("zca_refit") or []:
            assert z["split"] == "train"
        assert r.get("prune_log") is not None
    for x in lin:
        if x.get("decision") == "accepted":
            assert x["kappa_val_orig"] >= x["kappa_parent_rescored"]
    assert len(ops_seen & {"add_dense_layer", "deepen_rezero", "cov_gate", "replace_bad_filters"}) >= 2, ops_seen
