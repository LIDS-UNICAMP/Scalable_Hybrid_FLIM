"""growth_fix8: fase muda do train_child (AE local + SSL LeJEPA) e o "MKL ERROR ... DLASCL" do SPiFiL.

Tudo bit a bit igual ao histórico em CPU:
  - make_views: sorteios em lote (mesmo gerador, mesmas chamadas) + 1 cópia para o device;
  - ElementProtect: snapshot inteiro + torch.where (sem nonzero/sincronização);
  - train_global_ssl / train_local_ae: checagem/normas de gradiente numa sincronização; ``progress`` só imprime;
  - spifil0: patches NaN/Inf -> InfeasibleCandidate ANTES do eigh (era a origem do DLASCL).
"""
import math
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ga_search.modehb.ssl_trainers import AugConfig, make_views, train_global_ssl
from ga_search.modehb.local_ae import train_local_ae


# ───────────────────────────────────────────── referências = código histórico (pré-fix8), copiado literalmente
def _make_views_ref(x, V, cfg=None, seed=None, generator=None):
    cfg = AugConfig() if cfg is None else (AugConfig(**cfg) if isinstance(cfg, dict) else cfg)
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed)) if seed is not None else generator.seed()
    N, C, H, W = x.shape
    S = cfg.out_size or H
    dev, dt = x.device, x.dtype

    def rnd(*shape):
        return torch.rand(*shape, generator=generator).to(dev, dt)

    views = []
    for _ in range(V):
        area = cfg.crop_scale[0] + rnd(N) * (cfg.crop_scale[1] - cfg.crop_scale[0])
        lr0, lr1 = math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])
        ratio = torch.exp(lr0 + rnd(N) * (lr1 - lr0))
        sw = torch.sqrt(area * ratio).clamp(max=1.0)
        sh = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = (rnd(N) * 2 - 1) * (1 - sw)
        cy = (rnd(N) * 2 - 1) * (1 - sh)
        fx = torch.where(rnd(N) < cfg.hflip_p, -1.0, 1.0).to(dt)
        fy = torch.where(rnd(N) < cfg.vflip_p, -1.0, 1.0).to(dt)
        theta = torch.zeros(N, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, (N, C, S, S), align_corners=False)
        v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        mean = v.mean(dim=(1, 2, 3), keepdim=True)
        con = 1 + (rnd(N, 1, 1, 1) * 2 - 1) * cfg.contrast
        bri = (rnd(N, 1, 1, 1) * 2 - 1) * cfg.brightness
        ch = (rnd(N, C, 1, 1) * 2 - 1) * cfg.channel_jitter
        v = (v - mean) * con + mean + bri + ch
        views.append(v.clamp(0.0, 1.0))
    return views


class _ElementProtectRef:
    def __init__(self, module, masks, scale):
        self.scale = float(scale)
        params = dict(module.named_parameters())
        self.entries = [(params[n], m.to(params[n].device).bool()) for n, m in (masks or {}).items()
                        if n in params and bool(m.any())] if self.scale != 1.0 else []
        self._hooks, self._snap = [], {}

    def __enter__(self):
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
        if self.entries:
            self._hooks = [register_optimizer_step_pre_hook(self._pre), register_optimizer_step_post_hook(self._post)]
        return self

    def _pre(self, opt, args, kwargs):
        ids = {id(q) for g in opt.param_groups for q in g["params"]}
        self._snap[id(opt)] = [(p, m, p.detach()[m].clone()) for p, m in self.entries if id(p) in ids]

    def _post(self, opt, args, kwargs):
        with torch.no_grad():
            for p, m, old in self._snap.pop(id(opt), []):
                p.data[m] = old + self.scale * (p.data[m] - old)

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        return False


# ───────────────────────────────────────────── make_views
@pytest.mark.parametrize("seed,V,C,N", [(0, 4, 3, 8), (7, 2, 3, 5), (123, 4, 5, 3)])
def test_make_views_bitwise_equal_to_historical(seed, V, C, N):
    x = torch.rand(N, C, 24, 24, generator=torch.Generator().manual_seed(seed + 1))
    g1, g2 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    for _ in range(3):                                   # 3 passos seguidos: o gerador avança igual
        a, b = make_views(x, V, generator=g1), _make_views_ref(x, V, generator=g2)
        assert len(a) == len(b) == V
        for u, v in zip(a, b):
            assert torch.equal(u, v)
    assert torch.equal(g1.get_state(), g2.get_state())


def test_make_views_out_size_and_seed_path():
    x = torch.rand(4, 3, 32, 32)
    a = make_views(x, 4, {"out_size": 16}, seed=5)
    b = _make_views_ref(x, 4, {"out_size": 16}, seed=5)
    assert all(torch.equal(u, v) and u.shape == (4, 3, 16, 16) for u, v in zip(a, b))


# ───────────────────────────────────────────── ElementProtect
def test_element_protect_bitwise_equal_to_historical():
    from ga_search.modehb.external_seed import ElementProtect
    finals = []
    for cls in (ElementProtect, _ElementProtectRef):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Conv2d(6, 4, 1))
        masks = {n: (torch.rand(p.shape, generator=torch.Generator().manual_seed(i)) < 0.5)
                 for i, (n, p) in enumerate(m.named_parameters())}
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.01)
        x = torch.rand(4, 3, 9, 9, generator=torch.Generator().manual_seed(3))
        with cls(m, masks, 0.01):
            for _ in range(4):
                opt.zero_grad()
                m(x).pow(2).mean().backward()
                opt.step()
        finals.append({k: v.clone() for k, v in m.state_dict().items()})
    for k in finals[0]:
        assert torch.equal(finals[0][k], finals[1][k]), k


# ───────────────────────────────────────────── trainers: progresso só imprime; augment_device; mesmo resultado
class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return F.relu(self.c(x))


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(8, 16)

    def forward(self, f):
        return self.l(f.mean((2, 3)))


def _loader():
    x = torch.rand(16, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=8, shuffle=True,
                                       generator=torch.Generator().manual_seed(1))


def _ssl(progress):
    torch.manual_seed(0)
    enc, head = _Enc(), _Head()
    r = train_global_ssl(enc, head, _loader(), "lejepa", "sigreg", 2, 1e-3, 1e-3, [], "cpu", None, 3,
                         progress=progress)
    return r, {**{f"e.{k}": v for k, v in enc.state_dict().items()}, **{f"h.{k}": v for k, v in head.state_dict().items()}}


def test_ssl_progress_prints_and_changes_nothing(capsys):
    r0, w0 = _ssl(None)
    assert capsys.readouterr().out == ""
    r1, w1 = _ssl("[ssl]")
    out = capsys.readouterr().out
    assert out.count("[ssl] lejepa época") == 2 and "passos/s=" in out
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert r0["final_loss_total"] == r1["final_loss_total"]
    assert r0["augment_device"] == "cpu"


def test_ssl_nonfinite_grad_still_skipped():
    torch.manual_seed(0)
    enc, head = _Enc(), _Head()
    calls = {"n": 0}

    def hook(g):                       # 1º backward: gradiente NaN num parâmetro -> passo pulado
        calls["n"] += 1
        return torch.full_like(g, float("nan")) if calls["n"] == 1 else g
    head.l.bias.register_hook(hook)
    r = train_global_ssl(enc, head, _loader(), "lejepa", "sigreg", 1, 1e-3, 1e-3, [], "cpu", None, 3)
    assert r["skipped_steps"] == 1 and r["steps"] == 2


def _ae(progress):
    torch.manual_seed(0)
    prefix, blk = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)), nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU())
    prefix.out_channels = 4
    r = train_local_ae(prefix, blk, _loader(), "pixel", 2, 1e-3, "cpu", None, 0, progress=progress)
    return r, {k: v.clone() for k, v in blk.state_dict().items()}


def test_ae_progress_prints_and_changes_nothing(capsys):
    r0, w0 = _ae(None)
    assert capsys.readouterr().out == ""
    r1, w1 = _ae("[ae] bloco 1")
    out = capsys.readouterr().out
    assert out.count("[ae] bloco 1 época") == 2
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert r0["final_recon_loss"] == r1["final_recon_loss"]


def test_ae_grad_norm_logs_same_values():
    """As normas logadas (agora numa sincronização) == float(p.grad.norm()) por parâmetro."""
    logs = []
    lg = types.SimpleNamespace(log=lambda d, step=None: logs.append(dict(d)), alert=lambda *a, **k: None)
    torch.manual_seed(0)
    prefix, blk = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)), nn.Sequential(nn.Conv2d(4, 6, 3, padding=1))
    prefix.out_channels = 4
    train_local_ae(prefix, blk, _loader(), "pixel", 1, 1e-3, "cpu", lg, 0)
    step_logs = [d for d in logs if "ae/recon_loss" in d]
    assert step_logs and all("ae/grad_norm/new_block.0.weight" in d and math.isfinite(d["ae/grad_norm/new_block.0.weight"])
                             for d in step_logs)


# ───────────────────────────────────────────── MKL DLASCL: patches não finitos -> InfeasibleCandidate antes do eigh
def _learn(feats):
    patches = types.SimpleNamespace(feats=feats)
    return types.SimpleNamespace(n_layers=1, state=[types.SimpleNamespace(patches=patches)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_spifil0_nonfinite_patches_infeasible(bad):
    from ga_search.abstract_init.base import InfeasibleCandidate
    from ga_search.abstract_init.spifil0 import check_finite_patches
    f = torch.randn(50, 12)
    f[3, 4] = bad
    rec = {}
    with pytest.raises(InfeasibleCandidate) as ei:
        check_finite_patches(_learn(f), rec)
    assert ei.value.reason == "spifil0_nonfinite_patches"
    assert rec["nonfinite_patches"][0]["nonfinite_values"] == 1 and rec["nonfinite_patches"][0]["D"] == 12


def test_spifil0_finite_patches_untouched():
    from ga_search.abstract_init.spifil0 import check_finite_patches
    f = torch.randn(50, 12)
    rec = {}
    check_finite_patches(_learn(f), rec)
    assert rec == {}


def test_folded_grow_checks_before_super_run(monkeypatch):
    """O FoldedGrowOneLayer recusa ANTES de chamar o GrowOneLayer.run (que faria o eigh no MKL)."""
    from ga_search.abstract_init.base import InfeasibleCandidate
    from ga_search.abstract_init import spifil0
    import flim.spifil as FS
    called = {"n": 0}
    monkeypatch.setattr(FS.GrowOneLayer, "run", lambda self, learn: called.__setitem__("n", called["n"] + 1))
    cls = spifil0._folded_grow_class(True, False, 0.0, 0, {})
    f = torch.randn(50, 12)
    f[0, 0] = float("nan")
    with pytest.raises(InfeasibleCandidate):
        cls().run(_learn(f))
    assert called["n"] == 0
