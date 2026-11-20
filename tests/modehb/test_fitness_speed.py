"""Speed-ups of the SVM fitness phase (cov_head + WiSE + libsvm) are result-neutral.

* thread cap reaches the lazily loaded BLAS/OpenMP runtimes (scipy OpenBLAS = libsvm's ddot);
* cov_head per-image eigh over concurrent CUDA streams (``_eigh_many``) == the single batched call, bitwise;
* opt-in CPU eigh (``_eigh_host``) == the device eigh up to LAPACK rounding;
* WiSE alphas fitted concurrently == sequential (same kappas, same chosen alpha, same sv);
* old path (cov_eigh=device, 1 worker, BLAS untouched) == new defaults (same kappas / alpha);
* opt-in ``gram_reduce`` (exact train-span isometry) == the plain linear SVM;
* core.svm.fit_svm is never changed: linear SVC, C=100, ovo, max_iter=-1.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ga_search.modehb import pooling_head as PH
from ga_search.modehb import run_search as RS
from ga_search.modehb import supervised_fitness as SF

ROOT = Path(__file__).resolve().parents[2]
CFG_V3 = ROOT / "ga_search" / "config" / "modehb" / "search_v3.yaml"


# ---------------------------------------------------------------- thread cap (priority 1)
def test_cap_covers_lazy_scipy_openblas_and_sklearn_openmp():
    """Fresh process: after _cap_cpu_threads every BLAS/OpenMP pool (incl. libscipy_openblas and
    sklearn's libgomp, loaded lazily by the first SVM fit) has n threads; env vars exported."""
    code = (
        "import os, json\n"
        "os.environ.pop('OMP_NUM_THREADS', None); os.environ.pop('OPENBLAS_NUM_THREADS', None)\n"
        "os.environ['MODEHB_THREADS'] = '3'\n"
        "from ga_search.modehb import run_search as RS\n"
        "n = RS._cap_cpu_threads()\n"
        "from core.svm import fit_svm\n"
        "import numpy as np\n"
        "X = np.random.RandomState(0).randn(30, 12); fit_svm(X, np.arange(30) % 2)\n"
        "from threadpoolctl import threadpool_info\n"
        "print(json.dumps({'n': n, 'env': [os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')],\n"
        "  'pools': [(i['internal_api'], os.path.basename(i['filepath']), i['num_threads']) for i in threadpool_info()]}))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["n"] == 3 and r["env"] == ["3", "3", "3"]
    names = [p[1] for p in r["pools"]]
    assert any("scipy_openblas" in n or "openblas" in n for n in names)
    assert all(p[2] == 3 for p in r["pools"]), r["pools"]


def test_blas_threads_context_limits_and_restores():
    from threadpoolctl import threadpool_info
    import scipy.linalg  # noqa: F401
    before = {i["filepath"]: i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"}
    with RS._blas_threads(1):
        assert all(i["num_threads"] == 1 for i in threadpool_info() if i["user_api"] == "blas")
    after = {i["filepath"]: i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"}
    assert after == {k: v for k, v in before.items() if k in after}
    with RS._blas_threads(None):                 # null knob: no-op
        pass


# ---------------------------------------------------------------- cov_head eigh on the host
def _covs(b=40, k=16, n_px=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(b, n_px, k, generator=g, dtype=torch.float64)
    p[:, :, -3:] = 0.0                                       # dead channels -> repeated zero eigenvalues
    p = p - p.mean(1, keepdim=True)
    return (p.transpose(1, 2) @ p) / n_px                    # rank-deficient (n_px < k), PSD


def _sqrtm(w, U):
    return (U * w.clamp(min=0).sqrt()[:, None, :]) @ U.transpose(1, 2)


@pytest.mark.parametrize("k", [8, 16, 64])
def test_eigh_host_matches_eigh_up_to_rounding(k):
    C = _covs(b=37, k=k)
    w0, U0 = PH._eigh(C)
    w1, U1 = PH._eigh_host(C, workers=1)
    assert w1.dtype == C.dtype and w1.device == C.device and U1.shape == C.shape
    assert torch.allclose(w1, w0, atol=1e-12)
    S0, S1 = _sqrtm(w0, U0), _sqrtm(w1, U1)
    assert (S1 - S0).abs().max() <= 1e-6 * S0.abs().max()     # sqrt of +-1e-17 eigenvalues: ~1e-9
    assert torch.allclose(U1 @ torch.diag_embed(w1) @ U1.transpose(1, 2), C, atol=1e-12)


def test_eigh_host_independent_of_workers():
    C = _covs(b=70, k=32)
    w1, U1 = PH._eigh_host(C, workers=1)
    for W in (2, 3, 8):
        wW, UW = PH._eigh_host(C, workers=W)
        assert torch.equal(w1, wW) and torch.equal(U1, UW)


def test_eigh_host_fallback_on_non_convergence(monkeypatch):
    C = _covs(b=20, k=8)
    real, calls = np.linalg.eigh, []

    def flaky(a, *k, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise np.linalg.LinAlgError("Eigenvalues did not converge")
        return real(a, *k, **kw)
    monkeypatch.setattr(np.linalg, "eigh", flaky)
    w, U = PH._eigh_host(C, workers=1)
    assert torch.allclose(U @ torch.diag_embed(w) @ U.transpose(1, 2), C, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("workers", [2, 8])
def test_eigh_many_cuda_bitwise_equal(workers):
    mats = [_covs(b=70, k=64).cuda(), _covs(b=33, k=63, seed=1).cuda(), _covs(b=40, k=16, seed=2).cuda(),
            _covs(b=1, k=64, seed=3).cuda()]
    ref = [PH._eigh(m) for m in mats]
    for _ in range(2):                                         # pool and per-thread streams reused
        got = PH._eigh_many(mats, workers=workers)
        for (w0, v0), (w1, v1) in zip(ref, got):
            assert w1.device == w0.device and torch.equal(w0, w1) and torch.equal(v0, v1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_eigh_many_cuda_failure_redoes_whole_batch(monkeypatch):
    m = _covs(b=70, k=64).cuda()
    ref = PH._eigh(m)
    real, calls = torch.linalg.eigh, []

    def flaky(a, *k, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("linalg.eigh: The algorithm failed to converge")
        return real(a, *k, **kw)
    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    (w, v), = PH._eigh_many([m], workers=4)
    assert torch.equal(w, ref[0]) and torch.equal(v, ref[1])      # _eigh(m) on the whole batch


def test_eigh_many_cpu_is_plain_eigh():
    mats = [_covs(b=20, k=64), _covs(b=5, k=8, seed=1)]
    for (w0, v0), (w1, v1) in zip([PH._eigh(m) for m in mats], PH._eigh_many(mats, workers=8)):
        assert torch.equal(w0, w1) and torch.equal(v0, v1)


def _enc():
    torch.manual_seed(0)
    return nn.Sequential(nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU()),
                         nn.Sequential(nn.Conv2d(12, 20, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
                         nn.Sequential(nn.Conv2d(20, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))).eval()


def test_cov_readout_cpu_eigh_features_match():
    enc = _enc()
    head = PH.EmbeddingHead(24, 1280, "gap").eval()
    x = torch.randn(10, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    old = PH.CovReadout(k=16, n_img=10, n_px=20).fit(enc, x, "cpu", bs=4)
    new = PH.CovReadout(k=16, n_img=10, n_px=20, eigh_device="cpu", eigh_workers=3).fit(enc, x, "cpu", bs=4)
    for (m0, v0), (m1, v1) in zip(old.proj, new.proj):     # the PCA fit is untouched
        assert torch.equal(m0, m1) and torch.equal(v0, v1)
    f0, f1 = old(enc, head, x), new(enc, head, x)
    assert f0.shape == f1.shape and f0.dtype == f1.dtype == torch.float32
    assert (f1 - f0).abs().max() <= 1e-4
    dev = PH.CovReadout(k=16, n_img=10, n_px=20, eigh_workers=8).fit(enc, x, "cpu", bs=4)
    assert torch.equal(dev(enc, head, x), f0)        # CPU tensors: eigh_workers never changes a bit
    ref = torch.cat([old(enc, head, b) for b in x.split(3)])
    assert torch.equal(dev.embed(enc, head, x, 3, "cpu", group_mats=7), ref)
    with pytest.raises(ValueError):
        PH.CovReadout(eigh_device="tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cov_readout_cuda_streams_bitwise_and_cpu_close():
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Sequential(nn.Conv2d(3, 40, 3, padding=1), nn.ReLU()),
                        nn.Sequential(nn.Conv2d(40, 70, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4))).cuda().eval()
    head = PH.EmbeddingHead(70, 1280, "gap").cuda().eval()
    x = torch.randn(24, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    old = PH.CovReadout(k=64, n_img=24, n_px=40).fit(enc, x, "cuda", bs=8)        # historical: 1 call
    new = PH.CovReadout(k=64, n_img=24, n_px=40, eigh_workers=8).fit(enc, x, "cuda", bs=8)
    cpu = PH.CovReadout(k=64, n_img=24, n_px=40, eigh_device="cpu", eigh_workers=2).fit(enc, x, "cuda", bs=8)
    f0, f1, f2 = (r(enc, head, x.cuda()) for r in (old, new, cpu))
    assert torch.equal(f0, f1)                       # 64 > 32: concurrent cusolver syevd, same bits
    assert f2.device == f0.device and (f2 - f0).abs().max().item() <= 1e-4
    ref = torch.cat([old(enc, head, b.cuda()).cpu() for b in x.split(5)])      # historical loop
    for gm in (1, 64, 10 ** 6):                      # grouped eigh over several batches: same bits
        assert torch.equal(new.embed(enc, head, x, 5, "cuda", group_mats=gm), ref)


# ---------------------------------------------------------------- opt-in gram_reduce (exact isometry)
def _blobs(n=90, d=600, c=4, seed=0, sep=0.6):
    r = np.random.RandomState(seed)
    y = np.arange(n) % c
    mu = r.randn(c, d)
    X = (sep * mu[y] + r.randn(n, d)) * r.uniform(0.5, 3.0, size=d)
    return X.astype(np.float32), y


def test_span_coordinates_preserve_gram():
    X, _ = _blobs()
    Xv, _ = _blobs(n=30, seed=1)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X.astype(np.float64))
    Xs, Xvs = sc.transform(X.astype(np.float64)), sc.transform(Xv.astype(np.float64))
    Xr, Xvr = SF.span_coordinates(X.astype(np.float64), Xv.astype(np.float64), device="cpu")
    assert Xr.shape == (90, 90) and Xvr.shape == (30, 90)
    K, Kr = Xs @ Xs.T, Xr @ Xr.T
    assert np.abs(K - Kr).max() <= 1e-12 * np.abs(K).max()
    assert np.abs(Xvs @ Xs.T - Xvr @ Xr.T).max() <= 1e-12 * np.abs(K).max()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gram_reduce_same_svm_result(seed):
    X, y = _blobs(seed=seed)
    Xv, yv = _blobs(n=60, seed=seed + 10)
    kw = dict(C=100.0, scaler="standard", max_train=4000, kappa_weight=1.0, seed=42)
    a = SF.svm_fitness(X, y, Xv, yv, **kw)
    b = SF.svm_fitness(X, y, Xv, yv, gram_reduce=True, **kw)
    for k in ("kappa", "macro_f1", "balanced_acc", "quality", "n_train_used"):
        assert a[k] == b[k], (k, a[k], b[k])
    small_a = SF.svm_fitness(X[:, :50], y, Xv[:, :50], yv, **kw)     # D <= n: plain path, identical
    small_b = SF.svm_fitness(X[:, :50], y, Xv[:, :50], yv, gram_reduce=True, **kw)
    assert {k: v for k, v in small_a.items() if k != "fit_s"} == {k: v for k, v in small_b.items() if k != "fit_s"}


def test_fit_svm_hyperparameters_untouched():
    """R1: the repo SVM is linear SVC, C=100, ovo, max_iter=-1 (core.svm.fit_svm unchanged)."""
    from core.svm import fit_svm, svm_protocol
    X, y = _blobs(n=40, d=20)
    clf = fit_svm(X.astype(np.float64), y, C=100.0, scaler=True)
    assert svm_protocol(clf) == "scaler=standard;kernel=linear;C=100;ovo;max_iter=-1"


# ---------------------------------------------------------------- WiSE: parallel == sequential == old
DS_FREQ_STEP, DS_NOISE = 0.05, 1.5     # hard enough that kappa < 1 and varies with alpha


class _DS(torch.utils.data.Dataset):
    """Class-dependent textures (frequency + orientation), so cov_head features separate classes."""

    def __init__(self, n, n_cls=3, off=0):
        self.samples = [types.SimpleNamespace(name=f"s{off + i}", label=i % n_cls) for i in range(n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        g = torch.Generator().manual_seed(1000 + int(s.name[1:]))
        yy, xx = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
        f = 0.5 + DS_FREQ_STEP * s.label
        base = torch.stack([torch.sin(f * xx), torch.cos(f * yy), torch.sin(f * (xx + yy))])
        return base + DS_NOISE * torch.randn(3, 16, 16, generator=g), s.label, s.name


def _fake_load_data(cfg, out):
    tr, va = _DS(48), _DS(24, off=100)
    sel = [tr.samples[0], tr.samples[1], tr.samples[2]]
    return (DataLoader(tr, batch_size=8), DataLoader(va, batch_size=8), sel, {"sha256": "x"})


def _evaluator(tmp_path, monkeypatch, overrides=()):
    import ga_search.progressive_data as PD
    monkeypatch.setattr(PD, "load_data", _fake_load_data)
    monkeypatch.setattr(RS, "build_spifil_pool", lambda sel, train, k, seed: {s.label: [s] for s in sel})
    cfg = RS.load_config(CFG_V3, "remote_sensing", None,
                         ["device=cpu", "data.ram_cache=true", "data.batch_size=8", "data.val_max_samples=null",
                          "wandb.enabled=false", "eval.svm_fitness.cov_k=8", *overrides])
    ev = RS.Evaluator.__new__(RS.Evaluator)
    ev.cfg, ev.out, ev.device, ev.seed = cfg, tmp_path, torch.device("cpu"), 0
    ev._load_data()
    return ev


def _wise(ev, knobs):
    sf = ev.cfg["eval"]["svm_fitness"]
    for k in ("cov_eigh", "svm_workers", "svm_blas_threads", "cov_eigh_workers", "gram_reduce"):
        sf.pop(k, None)
    sf.update(knobs)
    enc = _enc()
    head = PH.EmbeddingHead(24, 1280, "gap").eval()
    anchor = {"encoder": {k: v.clone() for k, v in enc.state_dict().items()},
              "head": {k: v.clone() for k, v in head.state_dict().items()}}
    torch.manual_seed(5)
    with torch.no_grad():
        for p in list(enc.parameters()) + list(head.parameters()):
            p.add_(0.3 * torch.randn_like(p))
    trained = {"encoder": {k: v.clone() for k, v in enc.state_dict().items()},
               "head": {k: v.clone() for k, v in head.state_dict().items()}}
    logs = []
    run = types.SimpleNamespace(log=lambda d, **k: logs.append(d))
    w = ev._wise_select(enc, head, anchor, trained, [0.0, 0.25, 0.5, 0.75, 1.0], run)
    with torch.no_grad():
        feats = ev._svm_features(enc, head)
    return w, logs, feats, {k: v.clone() for k, v in enc.state_dict().items()}


def _strip(sv):
    return {k: v for k, v in sv.items() if k != "fit_s"}


def test_fitness_speed_defaults_and_knobs(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    sp = ev._fitness_speed(5)
    n = torch.get_num_threads()
    assert sp == {"cov_eigh": None, "cov_eigh_workers": 8, "svm_workers": min(5, n),
                  "svm_blas_threads": 1}
    ev.cfg["eval"]["svm_fitness"].update(cov_eigh="cpu", svm_workers=1, svm_blas_threads=None,
                                         cov_eigh_workers=2)
    assert ev._fitness_speed(5) == {"cov_eigh": "cpu", "cov_eigh_workers": 2, "svm_workers": 1,
                                    "svm_blas_threads": None}
    ev.cfg["eval"]["svm_fitness"].update(svm_workers=9)
    assert ev._fitness_speed(3)["svm_workers"] == 3              # never more workers than alphas
    ev.cfg["eval"]["svm_fitness"]["cov_eigh"] = "tpu"
    with pytest.raises(ValueError):
        ev._fitness_speed(5)


HISTORICAL = {"cov_eigh": "device", "cov_eigh_workers": 1, "svm_workers": 1, "svm_blas_threads": None}


def test_wise_parallel_equals_sequential_and_old_path(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    old, old_logs, old_f, old_sd = _wise(ev, HISTORICAL)
    seq, _, seq_f, _ = _wise(ev, {"svm_workers": 1})                 # new defaults, sequential SVMs
    par, par_logs, par_f, par_sd = _wise(ev, {})                       # new defaults (parallel)
    for other, f in ((seq, seq_f), (old, old_f)):                    # exactly the same everywhere
        assert par["kappas"] == other["kappas"] and par["alpha"] == other["alpha"]
        assert _strip(par["sv"]) == _strip(other["sv"])
        assert all(np.array_equal(a, b) for a, b in zip(par_f, f))
    assert all(torch.equal(par_sd[k], old_sd[k]) for k in old_sd)  # encoder left at the same blend
    assert par_logs == old_logs                                       # same W&B wise/* values
    assert len(set(old["kappas"].values())) > 1                       # non-degenerate test


def test_wise_cpu_eigh_opt_in_close(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    old, _, old_f, _ = _wise(ev, HISTORICAL)
    cpu, _, cpu_f, _ = _wise(ev, {"cov_eigh": "cpu", "cov_eigh_workers": 3})
    for a, b in zip(cpu_f, old_f):
        assert a.shape == b.shape and np.abs(a - b).max() <= 1e-4


def test_svm_workers_ram_cap(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    tr = np.zeros((4000, 18000), np.float32)                    # 3 x 8 B x 72M = 1.73 GB per fit
    assert ev._svm_workers_for(5, tr) == 4                      # 8 GB default
    ev.cfg["eval"]["svm_fitness"]["svm_ram_gb"] = 2
    assert ev._svm_workers_for(5, tr) == 1
    ev.cfg["eval"]["svm_fitness"]["svm_ram_gb"] = 100
    assert ev._svm_workers_for(5, tr) == 5 and ev._svm_workers_for(1, tr) == 1
    ev.cfg["eval"]["svm_fitness"]["svm_ram_gb"] = 0.06              # tiny budget -> 1 worker, never 0
    assert ev._svm_workers_for(5, np.zeros((48, 1500), np.float32)) == 1
    assert ev._svm_workers_for(5, tr) == 1
    a, _, _, _ = _wise(ev, {})
    b, _, _, _ = _wise(ev, HISTORICAL)
    assert a["kappas"] == b["kappas"] and a["alpha"] == b["alpha"]


def test_wise_parallel_propagates_errors(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    real, n = RS.Evaluator._svm_score, []

    def boom(self, tr, va):
        n.append(1)
        if len(n) == 3:
            raise RuntimeError("svm exploded")
        return real(self, tr, va)
    monkeypatch.setattr(RS.Evaluator, "_svm_score", boom)
    with pytest.raises(RuntimeError, match="svm exploded"):
        _wise(ev, {"svm_workers": 3})


def test_wise_gram_reduce_same_selection(tmp_path, monkeypatch):
    ev = _evaluator(tmp_path, monkeypatch)
    a, _, _, _ = _wise(ev, HISTORICAL)
    b, _, _, _ = _wise(ev, {"gram_reduce": True})
    assert a["kappas"] == b["kappas"] and a["alpha"] == b["alpha"]
    assert _strip(a["sv"]) == _strip(b["sv"])
