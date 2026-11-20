"""HOTFIX member_fid: o membro da faixa é o melhor de f_low/f_high pela ORIG-val em TODOS os caminhos da promoção
(retomada, retrain, ssl: none); casos reais do athena (f15 > f45)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.hmodehb import growth_chain as G  # noqa: E402


def _r(fid, k, a, ck, **kw):
    return {"status": "ok", "fid": fid, "node_id": "B1g0c03", "cand_key": f"ck{fid}", "kappa_val_orig": k,
            "quality": k, "wise_alpha": a, "checkpoint": ck, "checkpoint_sha256": ck + "sha", "band": "B1",
            "params": 400000, "flops_2mac": 1, "aux": {"wise_kappas": {str(a): k}}, "kappa_parent_rescored": 0.5,
            "preservation_verdict": "exact", "parent": "seed01", "theta": [0.5] * 4, **kw}


@pytest.mark.parametrize("k15,k45", [(0.5714, 0.5293), (0.5913, 0.5219), (0.5493, 0.5370)])
def test_low_wins(k15, k45):
    s = G.select_low_high(_r(15, k15, 0.25, "low.pt"), _r(45, k45, 0.75, "high.pt"), 15, 45)
    assert s["selected_fid"] == 15 and s["kappa_val_orig"] == k15 and s["checkpoint"] == "low.pt"
    assert s["wise_alpha"] == 0.25 and s["aux"]["wise_kappas"] == {"0.25": k15}
    assert s["kappa_val_orig_f_high"] == k45 and s["checkpoint_f_high"] == "high.pt" and s["cand_key"] == "ck45"
    assert G.decide(s, 0.0, False) == ("accepted", True)


def test_tie_and_high_wins_and_high_failed():
    assert G.select_low_high(_r(15, 0.7415, 0.5, "l"), _r(45, 0.7415, 0.5, "h"), 15, 45)["checkpoint"] == "h"
    assert G.select_low_high(_r(15, 0.60, 0.5, "l"), _r(45, 0.65, 0.5, "h"), 15, 45)["selected_fid"] == 45
    f = G.select_low_high(_r(15, 0.60, 0.5, "l"), {"status": "failed", "cand_key": "ck45", "node_id": "x"}, 15, 45)
    assert f["selected_fid"] == 15 and f["status"] == "ok" and f["checkpoint"] == "l"


class _Gev:
    fitness_sig = "s" * 12
    readout = "so2_4096_pca"


@pytest.mark.parametrize("mode,genes", [("retrain", {}), ("resume", {"ssl": "none"}), ("resume", {})])
def test_promote_all_paths_select_best(tmp_path, monkeypatch, mode, genes):
    g = G.load_growth_config(None, [f"promotion.mode={mode}"])
    ch = G.GrowthChain({"seed": 42, "train": {}}, g, _Gev(), tmp_path)
    low = _r(15, 0.5714, 0.25, "low.pt")                       # sem resume_state -> retrain fallback no 3º caso
    monkeypatch.setattr(ch, "_job", lambda *a, **k: _r(45, 0.5293, 0.75, "high.pt"))
    meta = {"parent": {"node_id": "seed01"}, "ops": [], "theta": None, "node": "B1g0c03", "seed": 1, "lkey": "lk",
            "genes": genes}
    r = ch._promote(low, meta, 15, 45)
    assert r["selected_fid"] == 15 and r["checkpoint"] == "low.pt" and r["kappa_val_orig"] == 0.5714
    assert r["promotion"]["mode"] == "retrain"
