"""Unitarios: faixas B0-B4 / teto duro e fitness v2 (regua ORIG, assinatura, aux, NaN)."""
import math

import numpy as np
import pytest
import torch

import _helpers as H
from ga_search.hmodehb import bk as BK
from ga_search.hmodehb import fitness_v2 as FV
from ga_search.hmodehb import run_hsearch as R
from ga_search.modehb import strata as S


# ---------------------------------------------------------------- B0-B4
@pytest.mark.parametrize("p,want", [(0, "B0"), (1, "B0"), (300_000, "B0"), (300_001, "B1"), (800_000, "B1"),
                                    (800_001, "B2"), (1_500_000, "B2"), (1_500_001, "B3"), (3_000_000, "B3"),
                                    (3_000_001, "B4"), (5_500_000, "B4"), (5_500_001, None), (10**8, None)])
def test_bk_boundaries(p, want):
    assert BK.bk_of(p) == want
    assert S.stratum_of(p, S.strat_config(BK.stratified_block())) == want    # mesma regra do strata.py


def test_bands_fixed_and_hard_cap():
    sb = BK.stratified_block({"min_share": 0.2})
    assert sb["strata"] == BK.bands_dict() and sb["param_cap"] == 5_500_000 and sb["enabled"]
    with pytest.raises(BK.BandError):
        BK.stratified_block({"strata": {"tiny": [0, 300000], "huge": [300000, 3500000]}})
    with pytest.raises(BK.BandError):
        BK.stratified_block({"param_cap": 5_775_000})                      # o x1,05 do PNAS-CS
    bad = S.strat_config({"strata": {"B0": [0, 300000], "B1": [300000, 5500000]}, "param_cap": 5_500_000})
    with pytest.raises(BK.BandError):
        BK.check_strat(bad)
    BK.check_strat(S.strat_config(BK.stratified_block()))


def test_config_resolves_bands_and_v2():
    cfg = H.search_cfg("A")
    sc = S.strat_config(cfg["search"]["stratified"])
    BK.check_strat(sc)
    assert cfg["search"]["objective_bounds"] == {"flops_lo": 1e6, "flops_hi": 1e11}
    assert cfg["output_dir"].startswith("artifacts/hmodehb_b0b4/arm_A/remote_sensing/state/seed42")
    sv = FV.validate_eval(cfg["eval"])
    assert (sv["C"], sv["scaler"], sv["readout"], sv["kappa_weight"], sv["tta"], sv["whiten"]) == \
        (100.0, "none", "head", 1.0, 1, "none")
    assert cfg["data"]["val_max_samples"] is None


def test_arm_c_and_missing_arms_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.load_hconfig("D", "remote_sensing", 42)
    with pytest.raises(ValueError):
        R.load_hconfig("Z", "remote_sensing", 42)
    from ga_search.hmodehb.hsearch import HMODEHB
    cfg = H.search_cfg("A")
    with pytest.raises(NotImplementedError):
        HMODEHB({**cfg["search"], "pnas_cs": {"enabled": True}}, H.FakeEval(tmp_path), tmp_path, arm="C",
                fitness_spec=H.SPEC, full_config=cfg)


def test_protected_promotions_cover_bk():
    sc = S.strat_config(BK.stratified_block())
    rng = np.random.default_rng(0)
    cands = [{"F": rng.random(2).tolist(), "stratum": BK.BK_NAMES[i % 5], "slope": None} for i in range(15)]
    from ga_search.modehb.modehb import select_promotions
    idx = S.stratified_promotions(cands, 5, sc, lambda F, k: select_promotions(F, k))
    assert sorted(cands[i]["stratum"] for i in idx) == list(BK.BK_NAMES)       # 1 quota por B_k
    idx = S.stratified_promotions(cands[:3], 2, sc, lambda F, k: select_promotions(F, k), counts={"B0": 5})
    assert {cands[i]["stratum"] for i in idx} == {"B1", "B2"}                    # menos promovidas primeiro


# ---------------------------------------------------------------- fitness v2
BASE_EV = {"fitness_mode": "svm_kappa", "svm_fitness": {"C": 100, "scaler": "none", "readout": "head",
                                                         "kappa_weight": 1.0, "tta": 1, "whiten": "none"}}


@pytest.mark.parametrize("bad", [{"C": 0.1}, {"scaler": "standard"}, {"readout": "cov_head"}, {"whiten": "pw"},
                                 {"kappa_weight": 0.75}, {"tta": 8}, {"solver": "liblinear"},
                                 {"dimensional_collapse_penalty": True}])
def test_validate_rejects_other_rulers(bad):
    ev = {**BASE_EV, "svm_fitness": {**BASE_EV["svm_fitness"], **bad}}
    with pytest.raises(FV.FitnessConfigError):
        FV.validate_eval(ev)


def test_validate_requires_svm_kappa_and_fills_defaults():
    with pytest.raises(FV.FitnessConfigError):
        FV.validate_eval({"svm_fitness": BASE_EV["svm_fitness"]})
    sv = FV.validate_eval({"fitness_mode": "svm_kappa"})                      # tudo implicito = regua v2
    assert sv["kappa_weight"] == 1.0 and sv["scaler"] == "none" and sv["C"] == 100.0


def test_signature_changes_with_any_field():
    sp = FV.spec(FV.validate_eval(BASE_EV), val_sha256="v", n_val=10, dataset="d", split=1)
    s0 = FV.signature(sp)
    assert FV.signature(dict(sp)) == s0
    for k, v in [("C", 0.1), ("scaler", "standard"), ("readout", "cov_head"), ("max_train", 10),
                 ("wise_alphas", [0.0]), ("val_sha256", "w"), ("n_val", 11), ("flops_bounds", [1, 2]),
                 ("dataset", "e"), ("kappa_weight", 0.75), ("whiten", "pw"), ("tta", 8)]:
        assert FV.signature({**sp, k: v}) != s0, k


def test_objectives_fixed_bounds_and_failure():
    assert FV.objectives({"status": "ok", "quality": 0.7, "flops": 1e6}) == pytest.approx([0.3, 0.0])
    assert FV.objectives({"status": "ok", "quality": 0.7, "flops": 1e11}) == pytest.approx([0.3, 1.0])
    assert FV.objectives({"status": "ok", "quality": 0.5, "flops": 1e8 * 1.0}) == pytest.approx([0.5, 0.4])
    assert FV.objectives({"status": "failed"}) == [1.0, 1.0]


def _base_result(**kw):
    r = {"status": "ok", "quality": 0.1, "quality_raw": 0.2, "collapse_multiplier": 0.0,
         "collapse": {"total": True, "dimensional": True}, "fitness_kappa": 0.42, "quality_label_free": 0.9,
         "fitness_macro_f1": 0.5, "fitness_balanced_acc": 0.6, "wise_kappas": {"0.0": 0.3, "1.0": 0.42},
         "metrics": {"rank": 3}, "flops": 1e8, "params": 1000, "ssl_final_loss": 1.5}
    r.update(kw)
    return r


def test_finalize_quality_is_orig_kappa_aux_separated():
    r = FV.finalize(_base_result(), kappa=0.42)
    assert r["quality"] == r["quality_raw"] == r["fitness_kappa"] == r["kappa_val_orig"] == 0.42
    assert r["collapse_multiplier"] == 1.0 and r["collapse_total"] is True     # colapso: so flag
    for k in ("quality_label_free", "fitness_macro_f1", "fitness_balanced_acc", "wise_kappas", "metrics"):
        assert k not in r and k in r["aux"]
    assert r["aux"]["kappa_val_alpha0"] == 0.3 and r["aux"]["quality_legacy"] == 0.1
    assert FV.finalize(_base_result(), kappa=-0.2)["quality"] == 0.0


@pytest.mark.parametrize("kw", [dict(nonfinite=True), dict(kappa=float("nan")), dict(base_loss=float("nan")),
                                dict(base_loss=float("inf"))])
def test_finalize_nan_raises(kw):
    base = _base_result(**({"ssl_final_loss": kw.pop("base_loss")} if "base_loss" in kw else {}))
    with pytest.raises(FV.NonFiniteFitness):
        FV.finalize(base, **{"kappa": 0.4, **kw})


def _hev(monkeypatch, calls):
    from ga_search.hmodehb.hevaluate import HEvaluator
    from ga_search.modehb import supervised_fitness as SF
    ev = HEvaluator.__new__(HEvaluator)
    ev.sv2 = FV.validate_eval({**BASE_EV, "svm_fitness": {**BASE_EV["svm_fitness"], "max_train": 77}})
    ev.svm_train_y, ev.val_y, ev.seed = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1]), 3
    monkeypatch.setattr(SF, "svm_fitness", lambda *a, **k: calls.append((a, k)) or {"kappa": 0.5})
    return ev


def test_hevaluator_svm_call_is_orig(monkeypatch):
    calls = []
    ev = _hev(monkeypatch, calls)
    ev._svm_score(np.zeros((4, 1280)), np.zeros((2, 1280)))
    (a, k), = calls
    assert k == {"C": 100.0, "scaler": "none", "whiten": "none", "kappa_weight": 1.0, "max_train": 77, "seed": 3}
    assert a[0].shape == (4, 1280)


def test_hevaluator_refuses_small_embedding(monkeypatch):
    from ga_search.hmodehb.hevaluate import EmbedDimError
    ev = _hev(monkeypatch, [])
    with pytest.raises(EmbedDimError):
        ev._svm_score(np.zeros((4, 48)), np.zeros((2, 48)))
    with pytest.raises(EmbedDimError):
        ev._evaluate({"embed_dim": 512}, 1.0, {}, None, None)


def test_hevaluator_refuses_subsampled_val_and_cov_ruler(tmp_path):
    from ga_search.hmodehb.hevaluate import HEvaluator
    cfg = H.search_cfg("A")
    with pytest.raises(FV.FitnessConfigError):
        HEvaluator({**cfg, "data": {**cfg["data"], "val_max_samples": 512}}, tmp_path)
    with pytest.raises(FV.FitnessConfigError):
        HEvaluator({**cfg, "eval": {**cfg["eval"], "svm_fitness": {"readout": "cov_head"}}}, tmp_path)


def test_quality_of_nan():
    with pytest.raises(FV.NonFiniteFitness):
        FV.quality_of(math.nan)


# ---------------------------------------------------------------- regua em TODA decisao (A1b)
def _ok(**kw):
    r = {"status": "ok", "quality": 0.4, "kappa_val_orig": 0.4, "decision_rulers": [FV.ruler_sig(H.SPEC)],
         "wise_alpha": 0.5, "aux": {"wise_kappas": {"0.0": 0.3, "0.5": 0.4, "1.0": 0.35}}}
    r.update(kw)
    return r


def test_check_decisions_accepts_orig():
    FV.check_decisions(_ok(), H.SPEC)
    FV.check_decisions({"status": "failed"}, H.SPEC)
    FV.check_decisions(_ok(quality=0.0, kappa_val_orig=-0.1, wise_alpha=0.0,
                           aux={"wise_kappas": {"0.0": -0.1, "1.0": -0.05}}), H.SPEC)   # empate em quality 0


@pytest.mark.parametrize("bad", [
    {"decision_rulers": []},                                                         # nao auditavel
    {"decision_rulers": [FV.ruler_sig({**H.SPEC, "readout": "cov_head"})]},           # WiSE/fitness por COV
    {"decision_rulers": [FV.ruler_sig(H.SPEC), FV.ruler_sig({**H.SPEC, "scaler": "standard"})]},
    {"decision_rulers": [FV.ruler_sig({**H.SPEC, "C": 0.1})]},
    {"decision_rulers": [FV.ruler_sig({**H.SPEC, "tta": 8})]},
    {"quality": 0.9},                                                                # F de outra leitura
    {"wise_alpha": 1.0},                                                             # alpha nao e o argmax ORIG
    {"aux": {"wise_kappas": {"0.0": 0.3, "0.5": 0.4, "1.0": 0.45}}},                  # idem
])
def test_check_decisions_rejects_other_ruler(bad):
    with pytest.raises(FV.RulerViolation):
        FV.check_decisions(_ok(**bad), H.SPEC)


def test_hevaluator_stamps_cov_if_it_ever_feeds_a_decision(monkeypatch):
    calls = []
    ev = _hev(monkeypatch, calls)
    ev._readouts, ev._svm_stamps = ["head", "cov_head"], []
    ev._svm_score(np.zeros((4, 1280)), np.zeros((2, 1280)))
    sigs = {FV.ruler_sig({"readout": r, "tta": 1, "solver": "libsvm", **st})
            for r in set(ev._readouts) for st in ev._svm_stamps}
    assert FV.ruler_sig({**ev.sv2}) in sigs and len(sigs) == 2
