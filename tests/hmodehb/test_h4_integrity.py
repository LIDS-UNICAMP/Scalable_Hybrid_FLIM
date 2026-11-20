"""Unitarios de integridade: hash de config + deriva, trava flock, ids, chaves, adocao, monitor_report."""
import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

import _helpers as H
from ga_search.hmodehb import integrity as I
from ga_search.hmodehb import run_hsearch as R


# ---------------------------------------------------------------- config hash / deriva
def test_config_sha_canonical_and_excludes():
    a = {"search": {"x": 1, "y": [1, 2]}, "eval": {"svm_fitness": {"C": 100}}, "device": "cuda",
         "wandb": {"mode": "online", "project": "p"}}
    b = {"wandb": {"project": "p", "mode": "offline"}, "device": "cpu",
         "eval": {"svm_fitness": {"C": 100}}, "search": {"y": [1, 2], "x": 1}}
    assert I.config_sha(a) == I.config_sha(b)                                  # ordem, device, wandb.mode
    assert I.config_sha(a) != I.config_sha({**a, "wandb": {"mode": "online", "project": "q"}})
    assert I.config_sha(a) != I.config_sha({**a, "train": {"lr_new": 1e-3}})


def test_drift_detected_on_eval_C():
    cfg = H.search_cfg("A")
    st = {"config_full": I._plain(cfg), "config_sha": I.config_sha(cfg)}
    assert I.check_drift(st, cfg) is None
    new = json.loads(json.dumps(cfg))
    new["eval"]["svm_fitness"]["C"] = 0.1
    with pytest.raises(I.ConfigDrift, match="eval.svm_fitness.C"):
        I.check_drift(st, new)
    new2 = json.loads(json.dumps(cfg))
    new2["train"]["lr_new"] = 5e-4
    new2["search"]["max_brackets"] = 9
    with pytest.raises(I.ConfigDrift):
        I.check_drift(st, new2, allow=["search.max_brackets"])                 # lr_new nao permitido
    log = I.check_drift(st, new2, allow=["search.max_brackets", "train"])
    assert log["keys"] == ["search.max_brackets", "train.lr_new"] and log["old_sha"] != log["new_sha"]
    with pytest.raises(I.ConfigDrift):
        I.check_drift({"config": {}}, cfg)                                   # estado legado: nao e H-MODEHB


def test_allow_drift_key_itself_not_hashed():
    cfg = H.search_cfg("A")
    c2 = json.loads(json.dumps(cfg))
    c2["hmodehb"]["allow_drift"] = ["train"]
    assert I.config_sha(cfg) == I.config_sha(c2)


# ---------------------------------------------------------------- trava
def test_lock_blocks_second_controller_in_process(tmp_path):
    a = I.ControllerLock(tmp_path).acquire()
    info = json.loads((tmp_path / "controller.lock").read_text())
    assert info["pid"] == os.getpid() and info["host"] and info["start"]
    with pytest.raises(I.ControllerLocked, match=str(os.getpid())):
        I.ControllerLock(tmp_path).acquire()
    a.release()
    with I.ControllerLock(tmp_path):
        pass


def _hold(path, q, ev):
    lk = I.ControllerLock(path).acquire()
    q.put("held")
    ev.wait(30)
    lk.release()


def test_lock_blocks_other_process_and_dies_with_it(tmp_path):
    ctx = mp.get_context("fork")
    q, ev = ctx.Queue(), ctx.Event()
    p = ctx.Process(target=_hold, args=(tmp_path, q, ev))
    p.start()
    assert q.get(timeout=30) == "held"
    with pytest.raises(I.ControllerLocked):
        I.ControllerLock(tmp_path).acquire()
    p.kill()                                                                  # kill -9 do dono
    p.join(10)
    I.ControllerLock(tmp_path).acquire().release()                           # lock do kernel some com ele


# ---------------------------------------------------------------- ids / chaves
def test_ids():
    assert I.cand_id("A", 12) == "h4-A-c00012" and I.cand_id("D0", 1) == "h4-D0-c00001"
    with pytest.raises(ValueError):
        I.cand_id("X", 1)


def test_cand_key_components():
    g = {"depth": 2, "layers": []}
    o1, o2 = {"kind": "scratch", "init": "spifil"}, {"kind": "inherit", "parent_ckpt_sha256": "ab"}
    l1 = I.lineage_key(g, 0, o1, "s1")
    assert l1 == I.lineage_key(dict(g), 0, dict(o1), "s1")
    assert len({l1, I.lineage_key(g, 0, o2, "s1"), I.lineage_key(g, 1, o1, "s1"), I.lineage_key(g, 0, o1, "s2"),
                I.lineage_key({**g, "depth": 3}, 0, o1, "s1"),
                I.lineage_key(g, 0, {**o2, "parent_ckpt_sha256": "cd"}, "s1")}) == 6
    assert I.cand_key(l1, 5) != I.cand_key(l1, 15) and I.cand_key(l1, 5) == I.cand_key(l1, 5.0)


# ---------------------------------------------------------------- adocao
def test_adoptable_rules(tmp_path):
    ck = tmp_path / "encoder.pt"
    ck.write_bytes(b"weights")
    ok = {"status": "ok", "cand_key": "k", "fitness_sig": "s", "checkpoint": str(ck),
          "checkpoint_sha256": I.file_sha256(ck)}
    assert I.adoptable(ok, ckey="k", fitness_sig="s") is None
    assert I.adoptable(ok, ckey="k2", fitness_sig="s")
    assert I.adoptable(ok, ckey="k", fitness_sig="s2")
    assert I.adoptable({**ok, "checkpoint_sha256": None}, ckey="k", fitness_sig="s")
    ck.write_bytes(b"weights-truncad")                                       # checkpoint trocado/parcial
    assert "sha256" in I.adoptable(ok, ckey="k", fitness_sig="s")
    ck.unlink()
    assert I.adoptable(ok, ckey="k", fitness_sig="s")
    assert I.adoptable({"status": "failed", "cand_key": "k", "fitness_sig": "s"}, ckey="k", fitness_sig="s") is None
    (tmp_path / "r.json").write_text('{"status": "ok", "cand_')                # kill durante a escrita
    assert I.read_json(tmp_path / "r.json") is None
    I.atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert I.read_json(tmp_path / "a.json") == {"x": 1}
    assert not list(tmp_path.glob("*.tmp*"))


# ---------------------------------------------------------------- monitor_report: kind proprio
def test_monitor_kind_and_sig_no_collision(tmp_path, monkeypatch):
    from ga_search.modehb import monitor_report as MR
    assert "artifacts/hmodehb_b0b4/arm_*" in MR.ROOT_GLOBS
    # legado inalterado
    assert MR._kind("artifacts/modehb_hybrid") == "hybrid"
    assert MR._kind("artifacts/modehb_lemonade") == "lemonade"
    assert MR._kind("artifacts/modehb_lemonade_seeded") == "lemonade_seeded"
    assert MR._kind("artifacts/pnas_cs") == "pnas_cs"
    assert MR._kind("artifacts/modehb_lemonade", {"config": {}}) == "lemonade"
    assert MR._fit_sig({"config": {}}, "artifacts/modehb_hybrid") == "C100_kw1"
    # H-MODEHB: kind do estado, mesmo com "lemonade" no caminho
    assert MR._kind("artifacts/hmodehb_b0b4/arm_B", {"kind": "h4_B"}) == "h4_B"
    assert MR._kind("artifacts/hmodehb_b0b4/arm_B") == "h4_B"
    assert MR._kind("artifacts/x/lemonade_like", {"kind": "h4_B"}) == "h4_B"
    assert MR._fit_sig({"fitness_sig": "abc123"}, "artifacts/hmodehb_b0b4/arm_A") == "v2_abc123"
    # arvore falsa: o mesmo id numerico em H-LEM e H4-B nao colide
    monkeypatch.chdir(tmp_path)
    job = {"status": "ok", "role": "search", "fidelity": 5.0, "result": {"checkpoint": "x.pt", "fitness_kappa": 0.5}}
    for root, st in (("artifacts/modehb_lemonade", {"config": {}, "jobs": [{**job, "cand_id": "c00001"}]}),
                     ("artifacts/hmodehb_b0b4/arm_B", {"kind": "h4_B", "fitness_sig": "f00d",
                                                      "jobs": [{**job, "cand_id": "h4-B-c00001"}]})):
        p = Path(root) / "remote_sensing" / "state" / "seed42"
        p.mkdir(parents=True)
        (p / "search_state.json").write_text(json.dumps(st))
    monkeypatch.setattr(MR, "ROOT_GLOBS", ["artifacts/modehb_lemonade", "artifacts/hmodehb_b0b4/arm_*"])
    monkeypatch.setattr(MR, "OUT_JSONL", tmp_path / "none.jsonl")
    keys = sorted(t[8] for t in MR.pending())
    assert keys == ["h4_B|remote_sensing|v2_f00d|h4-B-c00001|5", "lemonade|remote_sensing|C100_kw1|c00001|5"]


# ---------------------------------------------------------------- W&B controller id inclui a regua
def test_controller_id_includes_ruler(monkeypatch):
    cfg = H.search_cfg("A")
    a, b = R.controller_id(cfg, "aaaa"), R.controller_id(cfg, "bbbb")
    assert a != b and a.startswith("hctrl-") and a == R.controller_id(cfg, "aaaa")
    assert R.controller_id(H.search_cfg("B"), "aaaa") != a                    # outro braco, outro dir
    seen = {}
    from ga_search.modehb import wandb_logger as WL
    monkeypatch.setattr(WL, "ControllerRun", lambda **k: seen.update(k) or "run")
    assert R.hcontroller(cfg, "aaaa") is None                                 # wandb desligado
    cfg["wandb"]["enabled"] = True
    assert R.hcontroller(cfg, "aaaa") == "run"
    assert seen["id"] == a and "v2_aaaa" in seen["name"] and "fsig_aaaa" in seen["tags"]
    assert seen["group"] == "hmodehb_b0b4/remote_sensing" and seen["resume"] == "allow"
