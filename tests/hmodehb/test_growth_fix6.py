"""growth_fix6: a promoção f_low -> f_high RETOMA o filho de f_low (pesos treinados PRÉ-WiSE + AdamW + RNG),
treina as épocas restantes, WiSE com âncora = morfismo e fica com o melhor de f_low/f_high pela ORIG-val."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ga_search.backbones import build as BB  # noqa: E402
from ga_search.backbones.spec import HeadSpec, LayerSpec  # noqa: E402
from ga_search.hmodehb import growth_chain as G  # noqa: E402
from ga_search.hmodehb import integrity as I  # noqa: E402
from ga_search.modehb import ssl_trainers as ST  # noqa: E402

SYN = ["data.synthetic=true", "data.image_size=32", "data.synthetic_train_samples=16", "data.synthetic_val_samples=12",
       "data.batch_size=4", "data.num_workers=0", "data.ram_cache=true", "train.max_steps_per_epoch=2",
       "search.n_classes=2", "search.spifil_images_choices=[1,2]", "device=cpu", "wandb.enabled=false"]
OPS = [{"op": "widen", "widths": {"0": 16}, "seed": 1}]


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
    out = tmp_path_factory.mktemp("gev6")
    cfg = G.build_cfg("remote_sensing", 42, SYN, out)
    return G.GrowthEvaluator(cfg, out / "eval")


def _same(a: dict, b: dict):
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _same_obj(a, b, path="opt"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _same_obj(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same_obj(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


# ───────────────────────────────────────────────────────────────── trainer: run(e1)+resume(e2) == run(e1+e2)
@pytest.mark.parametrize("ssl", ["lejepa", "simclr"])
def test_ssl_trainer_resume_equals_continuous_bitwise(ssl):
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.rand(12, 3, 32, 32, generator=torch.Generator().manual_seed(0))

    def make():
        torch.manual_seed(3)
        enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 16))
        dl = DataLoader(TensorDataset(X), batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(5))
        return enc, head, dl

    reg = "sigreg" if ssl == "lejepa" else "none"
    e, h, dl = make()
    ST.train_global_ssl(e, h, dl, ssl, reg, 3, 1e-3, 1e-3, [], "cpu", None, 11)
    ref = {**e.state_dict(), **{f"h.{k}": v for k, v in h.state_dict().items()}}
    e2, h2, dl2 = make()
    r1 = ST.train_global_ssl(e2, h2, dl2, ssl, reg, 1, 1e-3, 1e-3, [], "cpu", None, 11, return_state=True)
    torch.manual_seed(999)                                  # o que acontece entre f_low e f_high não importa
    torch.rand(7)
    dl2.generator.manual_seed(123)
    st = copy.deepcopy(r1["state"])
    r2 = ST.train_global_ssl(e2, h2, dl2, ssl, reg, 2, 1e-3, 1e-3, [], "cpu", None, 11, resume=st)
    got = {**e2.state_dict(), **{f"h.{k}": v for k, v in h2.state_dict().items()}}
    _same(ref, got)
    assert r2["steps"] == 2 * 3 and r1["state"]["step"] == 3


def test_ssl_trainer_defaults_unchanged():
    """Sem resume/return_state o resultado não ganha chaves e o treino é o mesmo de antes (mesmo seed)."""
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.rand(8, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    outs = []
    for rs in (False, True):
        torch.manual_seed(3)
        enc = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 8))
        dl = DataLoader(TensorDataset(X), batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(5))
        r = ST.train_global_ssl(enc, head, dl, "lejepa", "sigreg", 2, 1e-3, 1e-3, [], "cpu", None, 1, return_state=rs)
        assert ("state" in r) == rs
        outs.append(enc.state_dict())
    _same(outs[0], outs[1])                                 # return_state não muda o treino


# ───────────────────────────────────────────────────────────────── filho: f_low + retomada == contínuo (f_high)
def _child(m):
    return G.realize_plan(copy.deepcopy(m), OPS, None)


def _lg(gev):
    return ST._loader_generator(gev.ev.train_loader)


@pytest.mark.parametrize("save_opt", [True, False])
def test_child_resume_equals_continuous_bitwise(gev, tmp_path, save_opt):
    from ga_search.modehb.run_search import split_epochs
    m = _tiny()
    fl, fh, seed = 2, 4, 7
    g = {**G.GROWTH_DEFAULTS, "wise_alphas": [0.0, 0.5, 1.0]}
    lg = _lg(gev)
    assert lg is not None
    s0 = lg.get_state()
    # (A) f_low com captura -> grava/relê o estado -> retoma (fh - fl) épocas de SSL
    lg.set_state(s0)
    cap = {}
    m_low, o_low = G.train_child(gev, _child(m), fl, g, seed, 0.5, capture=cap)
    info = G.save_resume_state(tmp_path / G.RESUME_FILE, cap, save_optimizer=save_opt, meta={"t": 1})
    assert info["has_optimizer"] is save_opt and info["has_rng"] and info["sha256"] == I.file_sha256(info["path"])
    st = G.load_resume_state(info["path"], info["sha256"])
    _same(st["trained"], cap["trained"])
    torch.manual_seed(12345)                                # outros jobs no meio: não afetam a retomada
    lg.manual_seed(777)
    model = copy.deepcopy(m_low)                            # arquitetura do filho de f_low (blend do WiSE)
    model.load_state_dict(st["trained"])                    # pesos PRÉ-WiSE
    n_extra = G.resume_extra_epochs(fl, fh, g)
    assert n_extra == fh - fl
    capA = {}
    m_high, o_high = G.continue_child(gev, model, st, n_extra, g, seed, 0.5, capture=capA)
    assert o_high["resume"]["optimizer_resumed"] is save_opt and o_high["resume"]["rng_resumed"]
    assert o_high["ae_epochs"] == o_low["ae_epochs"] and o_high["ssl_epochs"] == o_low["ssl_epochs"] + n_extra
    # (B) treino contínuo: o mesmo AE de f_low + SSL (fh - ae(fl)) épocas de uma vez
    ae_l, ssl_l = split_epochs(fl, gev.cfg["train"].get("ae_ssl_split", [1, 2]))
    lg.set_state(s0)
    capB = {}
    m_c, o_c = G.train_child(gev, _child(m), fh, g, seed, 0.5, epochs=(ae_l, ssl_l + n_extra), capture=capB)
    _same(capA["anchor"], capB["anchor"])                   # âncora do WiSE = o morfismo (pai em função)
    if save_opt:
        _same(capA["trained"], capB["trained"])             # bit a bit
        _same_obj(capA["ssl_state"]["opt"], capB["ssl_state"]["opt"])
        assert capA["ssl_state"]["step"] == capB["ssl_state"]["step"]
        assert o_high["aux"]["wise_kappas"] == o_c["aux"]["wise_kappas"]
        assert o_high["wise_alpha"] == o_c["wise_alpha"] and o_high["kappa_val_orig"] == o_c["kappa_val_orig"]
        _same(m_high.state_dict(), m_c.state_dict())
    else:                                                   # AdamW novo: registrado (não é bit a bit)
        assert o_high["resume"]["optimizer"].startswith("restarted")
        assert any(not torch.equal(capA["trained"][k], capB["trained"][k]) for k in capA["trained"]
                   if capA["trained"][k].is_floating_point())


def test_resume_state_rejects_bad_sha(tmp_path):
    p = tmp_path / "x.pt"
    torch.save({"format": G.RESUME_FORMAT}, p)
    with pytest.raises(G.ChainError):
        G.load_resume_state(p, "0" * 64)


# ───────────────────────────────────────────────────────────────── melhor de f_low/f_high pela val
def test_best_of_low_high_rule():
    assert G.best_of_low_high(0.620, 0.593) == 0              # o caso B2g1c02 RS: f15 vence
    assert G.best_of_low_high(0.60, 0.61) == 1
    assert G.best_of_low_high(0.60, 0.60) == 1                # empate = f_high
    assert G.best_of_low_high(0.60, float("nan")) == 0


def _chain(tmp_path, extra=()):
    seed_ck = Path(G.BB_save(_tiny(widths=(16, 32, 32)), tmp_path / "seed_b0.pt")["path"])
    out = tmp_path / "ctl"
    cfg = G.build_cfg("remote_sensing", 42, SYN, out)
    g = G.load_growth_config(None, ["seeds.registry=null", "seeds.external_small=false",
                                    f"seeds.checkpoints=[{seed_ck}]", "n_children=2", "n_parents=1",
                                    "generations=1", "fid_low=1", "fid_high=3", "eta=1", "end_band=B1",
                                    "spifil_images_per_class=1", "test_channel=false", "oom.sleep_s=0",
                                    "wise_alphas=[0.0,1.0]", *extra])
    gev = G.GrowthEvaluator(cfg, out / "eval")
    return cfg, g, gev, out


def _accept_all(monkeypatch):
    monkeypatch.setattr(G, "decide", lambda r, d, ri: ("accepted", True) if r.get("status") == "ok"
                        else ("failed", False))


@pytest.mark.parametrize("force_low", [False, True])
def test_controller_promotion_resumes_and_keeps_best_by_val(tmp_path, monkeypatch, force_low):
    _accept_all(monkeypatch)
    if force_low:                                           # f_high pior na val -> o nó fica com f_low
        real = G.continue_child

        def worse(*a, **k):
            model, out = real(*a, **k)
            out["kappa_val_orig"] = -1.0
            return model, out
        monkeypatch.setattr(G, "continue_child", worse)
    cfg, g, gev, out = _chain(tmp_path)
    G.GrowthChain(cfg, g, gev, out).run()
    st = json.loads((out / "growth_state.json").read_text())
    lin = [json.loads(x) for x in (out / "lineage.jsonl").read_text().splitlines()]
    highs = [r for r in lin if r.get("fid") == 3]
    assert highs, lin
    for r in highs:
        node = r["node_id"]
        low = json.loads((out / "nodes" / node / "f1" / "result.json").read_text())
        high = json.loads((out / "nodes" / node / "f3" / "result.json").read_text())
        assert low["resume_state_has_optimizer"] is True and Path(low["resume_state"]).exists()
        assert high["promotion"]["mode"] == "resume" and high["promotion"]["optimizer_resumed"] is True
        assert high["promotion"]["from_cand_key"] == low["cand_key"] and high["promotion"]["extra_ssl_epochs"] == 2
        assert high["ae_epochs"] == low["ae_epochs"] and high["ssl_epochs"] == low["ssl_epochs"] + 2
        assert high["ae_logs"] == []                                     # AE local não é repetido
        assert high["kappa_val_orig_f_low"] == low["kappa_val_orig"] and high["wise_alpha_f_low"] == low["wise_alpha"]
        assert high["cand_key"] != I.cand_key("x", 3)
        want = 1 if high["kappa_val_orig_f_high"] >= high["kappa_val_orig_f_low"] else 0
        assert want == (0 if force_low else want)
        if force_low:
            assert high["selected_fid"] == 1 and high["checkpoint"] == low["checkpoint"]
            assert high["kappa_val_orig"] == low["kappa_val_orig"] and high["wise_alpha"] == low["wise_alpha"]
        else:
            assert high["selected_fid"] == (3 if want else 1)
        assert Path(high["checkpoint_f_high"]).exists()
        assert "teste nunca" in high["selection"]
        assert r["selected_fid"] == high["selected_fid"] and r["kappa_val_orig_f_low"] == low["kappa_val_orig"]
        mem = {m["node_id"]: m for m in st["members"]["B1"]}
        if node in mem:
            assert mem[node]["checkpoint"] == high["checkpoint"] and mem[node]["selected_fid"] == high["selected_fid"]
    # 2ª execução: tudo adotado (cand_key da retomada estável), nada muda
    st1 = st["members"]
    G.GrowthChain(cfg, g, gev, out).run()
    assert json.loads((out / "growth_state.json").read_text())["members"] == st1


def test_promote_falls_back_to_retrain_and_records(tmp_path, monkeypatch):
    cfg, g, gev, out = _chain(tmp_path)
    ch = G.GrowthChain(cfg, g, gev, out)
    calls = []

    def fake_job(parent, ops, theta, fid, node, seed, ckey, resume_from=None):
        calls.append((fid, ckey, resume_from))
        return {"status": "ok", "fid": fid, "cand_key": ckey}
    monkeypatch.setattr(ch, "_job", fake_job)
    meta = {"parent": {"node_id": "p"}, "ops": OPS, "theta": [0.5] * 4, "seed": 1, "node": "B1g0c00", "lkey": "LK"}
    r = ch._promote({"cand_key": "c1", "fid": 1}, meta, 1, 3)           # f_low antigo (sem resume_state)
    assert calls[-1] == (3, I.cand_key("LK", 3), None)                   # cand_key histórica: adota f_high antigo
    assert r["promotion"]["mode"] == "retrain" and "sem resume_state" in r["promotion"]["reason"]
    r = ch._promote({"cand_key": "c1", "fid": 1, "resume_state": str(tmp_path / "nada.pt"),
                     "resume_state_sha256": "ab"}, meta, 1, 3)
    assert "sha256" in r["promotion"]["reason"]
    ch.g = {**g, "promotion": {"mode": "retrain"}}
    assert ch._promote({"cand_key": "c1", "fid": 1}, meta, 1, 3)["promotion"]["reason"] == "promotion.mode=retrain"
    ch.g = {**g, "ssl": "none"}
    assert "ssl: none" in ch._promote({"cand_key": "c1", "fid": 1}, meta, 1, 3)["promotion"]["reason"]


def test_drift_and_fitness_sig(gev):
    """A chave nova ``growth.promotion`` muda o config_sha (retomar um --out antigo: --allow-drift
    growth.promotion); a fitness_sig não depende do growth.yaml."""
    old = {"run": gev.cfg, "growth": {k: v for k, v in G.GROWTH_DEFAULTS.items() if k != "promotion"}, "kind": G.KIND}
    new = {"run": gev.cfg, "growth": dict(G.GROWTH_DEFAULTS), "kind": G.KIND}
    st = {"config_full": json.loads(json.dumps(old, default=str)), "config_sha": I.config_sha(old)}
    with pytest.raises(I.ConfigDrift):
        I.check_drift(st, new)
    d = I.check_drift(st, new, ["growth.promotion"])
    assert d is not None and all(k.startswith("growth.promotion") for k in d["keys"])
    assert "promotion" not in json.dumps(gev.fitness_spec)


def test_hotfix_resume_flow_wins_keeps_low_wise_kappas():
    """HOTFIX: f_low vence pela val -> aux.wise_kappas = os de f_low (check_decisions passa; f_high em aux_f_high)."""
    import inspect
    src = inspect.getsource(G.GrowthChain._run_resume_job)
    assert 'out["aux_f_high"]' in src and '"wise_kappas": aux_low.get("wise_kappas")' in src
    from ga_search.hmodehb import fitness_v2 as FV
    sp = {"readout": "head", "C": 100.0, "scaler": "none", "whiten": "none", "tta": 1, "kappa_weight": 1.0,
          "solver": "libsvm", "max_train": 4000}
    low = {"0.0": 0.61, "0.25": 0.62, "0.5": 0.60}
    out = {"status": "ok", "kappa_val_orig": 0.62, "quality": 0.62, "wise_alpha": 0.25,
           "aux": {"wise_kappas": low}, "decision_rulers": [FV.ruler_sig(sp)]}
    FV.check_decisions(out, sp)
