"""SPiFiL-3 — inicialização contextual/preditiva com transformação latente→kernel EXPLÍCITA.

Desenho: research/spifil_b0b4/agent2/prep_literatura_e_desenho.md §5.3 (MISSION B0–B4, Agente 2).

Pipeline (label-free, só treino, prefixo CONGELADO):

* **Estágio 1 — latente preditivo** na grade de features F = prefix(x) (sem gradiente):
  encoder de contexto f_θ (conv k×k C_in→d + GELU + conv 3×3 d→d), encoder-alvo f_ξ = EMA(f_θ)
  (stop-grad), preditor h_φ (2 convs 3×3 + embedding posicional 2D nos alvos) e uma cabeça de
  reconstrução 1×1 d→C_in. Tarefa: (a) context→target à la I-JEPA, h_φ(f_θ(F_ctx))_p ≈ sg(f_ξ(F))_p
  nos blocos-alvo; (b) masked feature reconstruction, rec(h_φ(...))_p ≈ F_p nos mesmos alvos.
  Anti-colapso: EMA + stop-grad, piso de variância VICReg e covariância sobre f_θ no contexto, e o
  monitor erank(Z)/d ≥ ``erank_min`` (abaixo: ``InfeasibleCandidate("ssl_collapse")``).
  f_θ, h_φ, a cabeça de reconstrução e o token de máscara são TEMPORÁRIOS: são descartados ao fim do
  ``fit`` e nenhum deles entra no backbone (só ``conv.weight``/``conv.bias`` [+ ``skip.*``] saem).
* **Estágio 2 — protótipos latentes**: Z = f_ξ(F) em N posições amostradas, ZCA, k-means esférico
  (K = k_mult·C_out, k-means++ semeado) e seleção de C_out protótipos por massa + FPS.
  Atribuição ao protótipo j: T_pj = cos(z̃_p, μ_j)/τ, centrada.
* **Estágio 3 — latente→kernel**: regressão ridge do patch X_p = unfold(F, k)_p (D = C_in·k², mesmo
  suporte k×k centrado em p, todos os canais), padronizado, para T. λ escolhido por MSE em imagens de
  TREINO separadas (holdout de treino; nunca validação). Kernel j = coluna j de W normalizada no
  espaço z e dobrada: weight_j = (w_j/σ_X).reshape(C_in, k, k), b_j = −Σ μ_X·w_j/σ_X.
  Diagnóstico obrigatório: R² por kernel (holdout e ajuste).
* **Controle S3-rand** (``config["latent"] = "random"``): o mesmo pipeline com f_ξ aleatório (a
  mesma inicialização semeada de f_θ) e SEM o Estágio 1.

Escopo de blocos: ``conv`` (k ímpar) e ``pointwise``; skip ``identity`` (sem pesos) e ``project``
(1×1 por PCA label-free do patch 1×1, linhas excedentes a C_in são aleatórias semeadas, registrado
em ``approximations``). Os demais tipos (depthwise, dwsep, inception) lançam
``InfeasibleCandidate("spifil3: unsupported ...")`` de forma determinística.

Interface: ``ga_search.abstract_init.base.KernelInitializer`` (registro ``spifil3`` e ``spifil3_rand``).
``fit(prefix, train_data, block_spec, context) -> (weights, bias, provenance)``; o runner chama
``init_block`` (herdado). Config = ``DEFAULT_CONFIG`` < kwargs do construtor < ``context.config``
(plano ou aninhado em ``context.config["spifil3"]``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import weakref
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ga_search.abstract_init.base import (InfeasibleCandidate, KernelInitializer, assemble_block,
                                          check_prefix_unchanged, check_train_only, make_provenance,
                                          prefix_fingerprint, register)


NAME = "spifil3"
VERSION = "0.1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    # Estágio 1
    "latent": "trained",            # "trained" (S3) | "random" (S3-rand: sem Estágio 1)
    "encoder": "default",           # "default" (k×k+GELU+3×3, RF k+2) | "local" (k×k+GELU+1×1, RF k) | "linear" (k×k)
    "latent_dim": 64,               # d (≤ 128)
    "steps": 300,                   # orçamento de passos (pequeno e configurável; desenho: até 2000)
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 0.04,
    "ema_start": 0.996,
    "ema_end": 1.0,
    "n_targets": 4,                 # M (2 se a grade < 14×14)
    "target_scale": (0.15, 0.2),
    "target_aspect": (0.75, 1.5),
    "min_side": 2,
    "context_scale": (0.85, 1.0),
    "rec_weight": 1.0,              # masked feature reconstruction
    "var_weight": 0.1,              # piso de variância VICReg (γ = 1)
    "cov_weight": 0.004,            # covariância VICReg (0,04 relativo ao termo de variância)
    "target_layernorm": True,       # alvos normalizados por posição (como no I-JEPA)
    "erank_min": 0.2,               # erank(Z)/d mínimo
    # Estágio 2
    "positions_per_image": 400,
    "k_mult": 3,
    "kmeans_iters": 30,
    "a_rep": 0.5,
    "tau": 0.1,
    "zca_eps_rel": 0.01,
    # Estágio 3
    "ridge_lambdas": (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    "holdout_frac": 0.2,
    "r2_warn": 0.3,
    "chunk": 16,
}

_SUPPORTED_CONV = ("conv", "pointwise")


# ----------------------------------------------------------------------------- utilidades


def _cfg(base: Mapping[str, Any], context) -> dict:
    user = dict(getattr(context, "config", None) or {})
    user = dict(user.get("spifil3", user))
    merged = {**dict(base), **user}
    unknown = sorted(set(merged) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"spifil3: chaves de config desconhecidas {unknown}")
    cfg = {**DEFAULT_CONFIG, **merged}
    if cfg["latent"] not in ("trained", "random"):
        raise ValueError(f"spifil3: latent deve ser 'trained' ou 'random', não {cfg['latent']!r}")
    if cfg["encoder"] not in ("default", "local", "linear"):
        raise ValueError(f"spifil3: encoder inválido {cfg['encoder']!r}")
    return cfg


def _json_default(o):
    if isinstance(o, (tuple, set)):
        return list(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.dtype):
        return str(o)
    return str(o)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def effective_rank(z: Tensor) -> float:
    """exp(entropia dos valores singulares normalizados) de Z centrado (0 se degenerado).

    Z constante (a menos de ruído de arredondamento, relativo à escala de Z) conta como erank 0.
    """
    z = z.double()
    scale = float(z.abs().max()) if z.numel() else 0.0
    z = z - z.mean(0, keepdim=True)
    if z.numel() == 0 or float(z.abs().max()) <= 1e-6 * scale + 1e-12:
        return 0.0
    s = torch.linalg.svdvals(z)
    s = s[s > 1e-7 * s.max()]
    tot = s.sum()
    if not torch.isfinite(tot) or tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _sincos_2d(d: int, h: int, w: int) -> Tensor:
    """Embedding posicional 2D fixo (d, h, w); metade dos canais para y, metade para x."""
    pe = torch.zeros(d, h, w)
    dq = max(1, d // 4)
    omega = 1.0 / (10000 ** (torch.arange(dq, dtype=torch.float32) / dq))
    ys = torch.arange(h, dtype=torch.float32)[:, None] * omega[None]  # (h, dq)
    xs = torch.arange(w, dtype=torch.float32)[:, None] * omega[None]  # (w, dq)
    blocks = [ys.sin().T[:, :, None].expand(dq, h, w), ys.cos().T[:, :, None].expand(dq, h, w),
              xs.sin().T[:, None, :].expand(dq, h, w), xs.cos().T[:, None, :].expand(dq, h, w)]
    full = torch.cat(blocks, 0)[:d]
    pe[: full.shape[0]] = full
    return pe


# ----------------------------------------------------------------------------- módulos temporários


def _conv(cin: int, cout: int, k: int) -> nn.Conv2d:
    # padding "replicate": borda sem artefato (entrada constante -> latente constante -> colapso detectável)
    return nn.Conv2d(cin, cout, k, padding=k // 2, padding_mode="replicate" if k > 1 else "zeros")


def _make_encoder(c_in: int, d: int, k: int, kind: str) -> nn.Sequential:
    if kind == "linear":
        return nn.Sequential(_conv(c_in, d, k))
    second = _conv(d, d, 3) if kind == "default" else _conv(d, d, 1)
    return nn.Sequential(_conv(c_in, d, k), nn.GELU(), second)


class _AuxJEPA(nn.Module):
    """Todos os módulos TEMPORÁRIOS do Estágio 1 (descartados ao fim do fit)."""

    def __init__(self, c_in: int, d: int, k: int, kind: str):
        super().__init__()
        self.context_encoder = _make_encoder(c_in, d, k, kind)          # f_θ
        self.target_encoder = copy.deepcopy(self.context_encoder)       # f_ξ (EMA, stop-grad)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.target_token = nn.Parameter(torch.zeros(1, d, 1, 1))
        self.predictor = nn.Sequential(_conv(d, 64, 3), nn.GELU(), _conv(64, d, 3))   # h_φ
        self.reconstructor = nn.Conv2d(d, c_in, 1)                      # masked feature reconstruction

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith("target_encoder.")]

    @torch.no_grad()
    def ema_update(self, m: float):
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)


# ----------------------------------------------------------------------------- máscaras


def _block(gen: torch.Generator, h: int, w: int, scale, aspect, min_side: int):
    s = float(scale[0] + (scale[1] - scale[0]) * torch.rand((), generator=gen))
    lo, hi = math.log(aspect[0]), math.log(aspect[1])
    ar = math.exp(lo + (hi - lo) * float(torch.rand((), generator=gen)))
    area = s * h * w
    bh = int(round(math.sqrt(area * ar)))
    bw = int(round(math.sqrt(area / ar)))
    bh = max(min(min_side, h), min(bh, h))
    bw = max(min(min_side, w), min(bw, w))
    top = int(torch.randint(0, h - bh + 1, (1,), generator=gen))
    left = int(torch.randint(0, w - bw + 1, (1,), generator=gen))
    return top, left, bh, bw


def block_masks(gen: torch.Generator, b: int, h: int, w: int, cfg: Mapping[str, Any]):
    """Máscaras I-JEPA adaptadas à grade: (contexto, alvo), ambas (b, 1, h, w) float {0,1}.

    M blocos-alvo (M = 2 se a grade < 14×14), escala 0,15–0,2, aspecto 0,75–1,5, lado mínimo 2;
    contexto = bloco de escala 0,85–1,0 menos os alvos (se ficar vazio, o complemento dos alvos).
    """
    m = int(cfg["n_targets"]) if min(h, w) >= 14 else min(2, int(cfg["n_targets"]))
    ctx = torch.zeros(b, 1, h, w)
    tgt = torch.zeros(b, 1, h, w)
    for i in range(b):
        for _ in range(m):
            t, l, bh, bw = _block(gen, h, w, cfg["target_scale"], cfg["target_aspect"], int(cfg["min_side"]))
            tgt[i, 0, t:t + bh, l:l + bw] = 1.0
        if tgt[i].sum() >= h * w:  # grade minúscula: deixa ao menos uma posição de contexto
            tgt[i, 0, int(torch.randint(0, h, (1,), generator=gen)), int(torch.randint(0, w, (1,), generator=gen))] = 0.0
        t, l, bh, bw = _block(gen, h, w, cfg["context_scale"], (1.0, 1.0), 1)
        ctx[i, 0, t:t + bh, l:l + bw] = 1.0
        ctx[i] = ctx[i] * (1.0 - tgt[i])
        if ctx[i].sum() == 0:
            ctx[i] = 1.0 - tgt[i]
    return ctx, tgt


# ----------------------------------------------------------------------------- álgebra


def _vicreg(z: Tensor, mask: Tensor):
    """Piso de variância (γ = 1) e covariância fora da diagonal de z (b, d, h, w) nas posições de mask."""
    d = z.shape[1]
    zz = z.permute(0, 2, 3, 1).reshape(-1, d)[mask.permute(0, 2, 3, 1).reshape(-1) > 0]
    if zz.shape[0] < 2:
        return z.new_zeros(()), z.new_zeros(())
    zz = zz - zz.mean(0, keepdim=True)
    std = torch.sqrt(zz.var(0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (zz.T @ zz) / (zz.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, off.pow(2).sum() / d


def _zca(z: Tensor, eps_rel: float):
    mu = z.mean(0, keepdim=True)
    zc = (z - mu).double()
    cov = zc.T @ zc / max(1, zc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(0)
    eps = eps_rel * float(evals.mean()) + 1e-12
    wz = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
    return mu, wz.float()


def spherical_kmeans(x: Tensor, k: int, iters: int, gen: torch.Generator):
    """k-means esférico em linhas unitárias, k-means++ semeado. Retorna (C (k,d) unitários, assign (N,))."""
    n = x.shape[0]
    k = min(k, n)
    first = int(torch.randint(0, n, (1,), generator=gen))
    idx = [first]
    best = 1.0 - x @ x[first]
    for _ in range(1, k):
        p = best.clamp_min(0).double()
        if float(p.sum()) <= 0:
            nxt = int(torch.randint(0, n, (1,), generator=gen))
        else:
            nxt = int(torch.multinomial(p / p.sum(), 1, generator=gen))
        idx.append(nxt)
        best = torch.minimum(best, 1.0 - x @ x[nxt])
    c = x[idx].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, iters)):
        sim = x @ c.T
        assign = sim.argmax(1)
        newc = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = (counts == 0).nonzero().flatten().tolist()
        if empty:  # re-semeia vazios nos pontos pior representados (determinístico)
            worst = sim.max(1).values.argsort()[: len(empty)]
            for e, wi in zip(empty, worst.tolist()):
                newc[e] = x[wi]
        nrm = newc.norm(dim=1, keepdim=True)
        newc = torch.where(nrm > 1e-12, newc / nrm.clamp_min(1e-12), c)
        if torch.equal(newc, c):
            break
        c = newc
    assign = (x @ c.T).argmax(1)
    return c, assign


def select_mass_fps(c: Tensor, assign: Tensor, n_select: int, a_rep: float):
    """Seleção gulosa: a_rep·log-massa normalizada + (1 − a_rep)·min(1 − cos) aos já escolhidos."""
    k = c.shape[0]
    mass = torch.bincount(assign, minlength=k).double() / max(1, assign.numel())
    lm = torch.log(mass.clamp_min(1e-12))
    alive = mass > 0
    lo, hi = lm[alive].min(), lm[alive].max()
    rep = torch.where(alive, (lm - lo) / (hi - lo) if hi > lo else torch.ones_like(lm), torch.zeros_like(lm))
    sel: list[int] = []
    div = torch.ones(k, dtype=torch.float64)
    for _ in range(min(n_select, k)):
        score = a_rep * rep + (1 - a_rep) * div
        if sel:
            score[sel] = -math.inf
        j = int(score.argmax())
        sel.append(j)
        div = torch.minimum(div, (1.0 - c.double() @ c[j].double()).clamp_min(0))
    return sel, mass


def ridge_fit(x_fit: Tensor, t_fit: Tensor, x_ho: Tensor, t_ho: Tensor, lambdas):
    """Ridge em X padronizado (float64). Retorna (W (D,C), λ, mse_por_λ, R² holdout, R² ajuste)."""
    x = x_fit.double()
    t = t_fit.double()
    n = x.shape[0]
    g = x.T @ x / n
    evals, evecs = torch.linalg.eigh(g)
    evals = evals.clamp_min(0)
    b = evecs.T @ (x.T @ t / n)
    xh, th = x_ho.double(), t_ho.double()
    mses, sols = [], []
    for lam in lambdas:
        w = evecs @ (b / (evals + float(lam))[:, None])
        sols.append(w)
        mses.append(float(((xh @ w - th) ** 2).mean()))
    i = int(np.argmin(mses))
    w = sols[i]

    def r2(xx, tt):
        sse = ((xx @ w - tt) ** 2).sum(0)
        sst = ((tt - tt.mean(0, keepdim=True)) ** 2).sum(0)
        return torch.where(sst > 1e-12, 1.0 - sse / sst.clamp_min(1e-12), torch.zeros_like(sse))

    return w, float(lambdas[i]), mses, r2(xh, th), r2(x, t)


# ----------------------------------------------------------------------------- dados


def _read_train_images(train_data, seed: int, max_images: int):
    check_train_only(train_data)  # I5
    xs, n = [], 0
    digest = hashlib.sha256()
    for batch in train_data.loader_factory(seed):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch  # o rótulo NUNCA é lido
        x = x.detach().float().cpu()
        take = x[: max_images - n]
        xs.append(take)
        digest.update(take.contiguous().numpy().tobytes())
        n += take.shape[0]
        if n >= max_images:
            break
    if not xs:
        raise InfeasibleCandidate("spifil3: no training images")
    return torch.cat(xs, 0), digest.hexdigest()


@torch.no_grad()
def _prefix_features(prefix, x: Tensor, device, chunk: int) -> Tensor:
    if prefix is None:
        return x.clone()
    out = [prefix(x[i:i + chunk].to(device)).float().cpu() for i in range(0, x.shape[0], chunk)]
    return torch.cat(out, 0)


@torch.no_grad()
def _gather(feats: Tensor, fn: Tensor, encoder: nn.Module, pos: Tensor, k: int, device, chunk: int):
    """Z = encoder(fn) e X = unfold(feats, k) nas posições ``pos`` (n_img, P) -> (N, d), (N, C·k²)."""
    zs, xs = [], []
    for i in range(0, feats.shape[0], chunk):
        zf = encoder(fn[i:i + chunk].to(device)).float().cpu()          # (b, d, h, w)
        b, d = zf.shape[:2]
        zf = zf.reshape(b, d, -1)
        xf = F.unfold(feats[i:i + chunk], k, padding=k // 2)            # (b, C·k², h·w), channel-major
        p = pos[i:i + chunk]
        zs.append(torch.gather(zf, 2, p[:, None, :].expand(b, d, p.shape[1])).permute(0, 2, 1).reshape(-1, d))
        dd = xf.shape[1]
        xs.append(torch.gather(xf, 2, p[:, None, :].expand(b, dd, p.shape[1])).permute(0, 2, 1).reshape(-1, dd))
    return torch.cat(zs, 0), torch.cat(xs, 0)


# ----------------------------------------------------------------------------- método


@register("spifil3")
class Spifil3Predictive(KernelInitializer):
    """SPiFiL-3 (S3). Ver docstring do módulo. ``Spifil3Rand`` é o controle S3-rand."""

    name = NAME
    version = VERSION
    label_policy = "label_free"

    def __init__(self, **config):
        super().__init__(**config)
        # weakrefs dos módulos temporários do último fit (para provar o descarte nos testes)
        self.last_aux_refs: list[weakref.ref] = []

    # -- API --------------------------------------------------------------------------
    def fit(self, prefix: Optional[nn.Module], train_data, block_spec: dict, context):
        t0 = time.monotonic()
        cfg = _cfg(self.config, context)
        seed = int(getattr(context, "seed", 0))
        device = torch.device(getattr(context, "device", None) or "cpu")
        budget = getattr(context, "budget", None)
        max_images = int(getattr(budget, "max_images", 256))
        max_patches = int(getattr(budget, "max_patches", 100_000))
        max_seconds = float(getattr(budget, "max_seconds", 600.0))
        spec = dict(block_spec)
        kind = spec.get("conv", "conv")
        if kind not in _SUPPORTED_CONV:
            raise InfeasibleCandidate(f"spifil3: unsupported conv {kind!r}", details={"supported": list(_SUPPORTED_CONV)})
        k = 1 if kind == "pointwise" else int(spec["kernel"])
        if k % 2 == 0:
            raise InfeasibleCandidate(f"spifil3: even kernel {k}")
        c_out = int(spec["channels"])
        c_in = int(spec["in_channels"])
        skip = spec.get("skip")
        label_policy = getattr(train_data, "label_policy", "label_free")

        # ---- prefixo congelado (I1) ---------------------------------------------------
        fp_before = prefix_fingerprint(prefix)
        saved = None
        if prefix is not None:
            saved = (prefix.training, [p.requires_grad for p in prefix.parameters()])
            prefix.eval()
            for p in prefix.parameters():
                p.requires_grad_(False)
        try:
            out = self._fit_frozen(prefix, train_data, spec, cfg, seed, device, k, c_in, c_out, skip,
                                   max_images, max_patches, max_seconds, t0, context)
        finally:
            if prefix is not None:
                prefix.train(saved[0])
                for p, rg in zip(prefix.parameters(), saved[1]):
                    p.requires_grad_(rg)
        check_prefix_unchanged(prefix, fp_before)  # I1 (InfeasibleCandidate se mudou)
        weights, bias, extra = out
        data_extra = extra.pop("data")
        prov = make_provenance(
            self.name, VERSION, context=context, train_data=train_data, requested_spec=dict(block_spec),
            effective_spec=spec, labels_used=False, label_source="none", mask_used=False,
            n_images=data_extra["n_images"], n_patches=data_extra["n_patches"], patch_dim=data_extra["patch_dim"],
            approximations=extra.pop("approximations"), fit_seconds=time.monotonic() - t0,
            extra={**extra, "faithful_to": None,
                   "variant": "S3" if cfg["latent"] == "trained" else "S3-rand",
                   # config efetiva (defaults + construtor + context) e o seu hash
                   "config": cfg, "config_sha256": _sha(cfg),
                   "prefix_state_sha256": fp_before, "prefix_unchanged": True,
                   "label_policy_requested": label_policy})
        prov["data"].update(data_extra)
        prov["env"].update({"python": platform.python_version(), "git_sha": _git_sha(), "device": str(device)})
        prov = json.loads(json.dumps(prov, default=_json_default))  # serializável (tuplas -> listas)
        return weights, bias, prov

    # -- interno ----------------------------------------------------------------------
    def _fit_frozen(self, prefix, train_data, spec, cfg, seed, device, k, c_in, c_out, skip,
                    max_images, max_patches, max_seconds, t0, context):
        gen = torch.Generator().manual_seed(seed)
        x, img_sha = _read_train_images(train_data, seed, max_images)
        feats = _prefix_features(prefix, x, device, int(cfg["chunk"]))
        if feats.dim() != 4 or feats.shape[1] != c_in:
            raise ValueError(f"spifil3: prefixo dá {tuple(feats.shape)}, spec pede in_channels={c_in}")
        n_img, _, h, w = feats.shape

        # split por imagem: ajuste × holdout de TREINO (escolha de λ e R²)
        perm = torch.randperm(n_img, generator=gen)
        if n_img >= 2:
            n_ho = max(1, int(round(cfg["holdout_frac"] * n_img)))
            ho_idx, fit_idx = perm[:n_ho].sort().values, perm[n_ho:].sort().values
            holdout_mode = "images"
        else:
            ho_idx = fit_idx = perm
            holdout_mode = "positions"
        f_fit, f_ho = feats[fit_idx], feats[ho_idx]
        ch_mu = f_fit.mean((0, 2, 3), keepdim=True)
        ch_sd = f_fit.std((0, 2, 3), keepdim=True).clamp_min(1e-6) if f_fit.numel() > c_in else torch.ones_like(ch_mu)
        fn_fit, fn_ho = (f_fit - ch_mu) / ch_sd, (f_ho - ch_mu) / ch_sd

        d = int(cfg["latent_dim"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            aux = _AuxJEPA(c_in, d, k, cfg["encoder"]).to(device)
        self.last_aux_refs = [weakref.ref(m) for m in (aux, aux.context_encoder, aux.target_encoder,
                                                         aux.predictor, aux.reconstructor)]
        curve: list[dict] = []
        if cfg["latent"] == "trained":
            curve = self._train_stage1(aux, fn_fit, cfg, gen, device, t0, max_seconds)
        target_encoder = aux.target_encoder.eval()
        n_aux_params = sum(p.numel() for p in aux.parameters())
        del aux  # f_θ, h_φ, reconstrutor e tokens: descartados (só f_ξ segue até o fim do Estágio 3)

        # ---- Estágio 2: latentes e patches nas mesmas posições ------------------------
        hw = h * w
        per_img = max(1, min(int(cfg["positions_per_image"]), hw, max_patches // max(1, len(fit_idx))))
        pos_fit = torch.stack([torch.randperm(hw, generator=gen)[:per_img] for _ in range(len(fit_idx))])
        if holdout_mode == "images":
            per_ho = max(1, min(int(cfg["positions_per_image"]), hw, max_patches // max(1, len(ho_idx))))
            pos_ho = torch.stack([torch.randperm(hw, generator=gen)[:per_ho] for _ in range(len(ho_idx))])
            z_fit, x_fit = _gather(f_fit, fn_fit, target_encoder, pos_fit, k, device, int(cfg["chunk"]))
            z_ho, x_ho = _gather(f_ho, fn_ho, target_encoder, pos_ho, k, device, int(cfg["chunk"]))
        else:  # uma imagem: holdout por posições disjuntas
            allp = torch.randperm(hw, generator=gen)[: min(hw, max_patches)]
            n_ho = max(1, int(round(cfg["holdout_frac"] * allp.numel())))
            z_all, x_all = _gather(f_fit, fn_fit, target_encoder, allp[None], k, device, int(cfg["chunk"]))
            z_ho, x_ho, z_fit, x_fit = z_all[:n_ho], x_all[:n_ho], z_all[n_ho:], x_all[n_ho:]
        del target_encoder
        n_fit = z_fit.shape[0]
        if n_fit < max(c_out, 2):
            raise InfeasibleCandidate("spifil3: too few positions", details={"n": n_fit, "c_out": c_out})

        z_erank = effective_rank(z_fit)
        z_min_std = float(z_fit.std(0).min()) if n_fit > 1 else 0.0
        if z_erank / d < float(cfg["erank_min"]):
            raise InfeasibleCandidate("ssl_collapse", details={"erank_ratio": z_erank / d, "min_std": z_min_std,
                                                               "latent": cfg["latent"]})

        z_mu, wz = _zca(z_fit, float(cfg["zca_eps_rel"]))
        zt_fit = F.normalize((z_fit - z_mu) @ wz, dim=1)
        zt_ho = F.normalize((z_ho - z_mu) @ wz, dim=1)
        k_clusters = min(int(cfg["k_mult"]) * c_out, n_fit)
        protos, assign = spherical_kmeans(zt_fit, k_clusters, int(cfg["kmeans_iters"]), gen)
        sel, mass = select_mass_fps(protos, assign, c_out, float(cfg["a_rep"]))
        if len(sel) < c_out:
            raise InfeasibleCandidate("spifil3: fewer prototypes than channels",
                                      details={"prototypes": len(sel), "c_out": c_out})
        mu_j = protos[sel]                                         # (C_out, d)
        tau = float(cfg["tau"])
        t_fit = zt_fit @ mu_j.T / tau
        t_mean = t_fit.mean(0, keepdim=True)
        t_fit = t_fit - t_mean
        t_ho = zt_ho @ mu_j.T / tau - t_mean

        # ---- Estágio 3: ridge patch k×k×C_in -> atribuição ----------------------------
        x_mu = x_fit.mean(0, keepdim=True)
        x_var = x_fit.var(0, unbiased=False, keepdim=True)
        x_sd = torch.sqrt(x_var + 1e-6 * float(x_var.mean()) + 1e-12)
        xs_fit, xs_ho = (x_fit - x_mu) / x_sd, (x_ho - x_mu) / x_sd
        wr, lam, mses, r2_ho, r2_fit = ridge_fit(xs_fit, t_fit, xs_ho, t_ho, list(cfg["ridge_lambdas"]))
        norms = wr.norm(dim=0)
        if bool((norms <= 1e-12).any()):
            raise InfeasibleCandidate("spifil3: degenerate kernel (zero ridge solution)",
                                      details={"n_zero": int((norms <= 1e-12).sum())})
        wz_unit = (wr / norms).float().T                           # (C_out, D) unitário no espaço z
        w_in = wz_unit / x_sd                                       # dobra do z-score
        b_in = -(w_in * x_mu).sum(1)
        weights = {"conv.weight": w_in.reshape(c_out, c_in, k, k).contiguous().float()}
        bias = {"conv.bias": b_in.contiguous().float()}
        approximations = ["zscore_fold", "ridge_linear_map", "global_lambda"]
        if skip == "project":
            sw, sb, note = self._skip_pca(f_fit, c_in, c_out, gen)
            weights["skip.weight"], bias["skip.bias"] = sw, sb
            approximations.append(note)

        r2 = r2_ho.float()
        r2_med = float(r2.median())
        diag = {
            "stage1": {"latent": cfg["latent"], "encoder": cfg["encoder"],
                       "receptive_field": k + 2 if cfg["encoder"] == "default" else k,
                       "steps_run": len(curve), "loss_curve": curve[:: max(1, len(curve) // 50)] + curve[-1:],
                       "aux_params_discarded": n_aux_params,
                       "discarded_modules": ["context_encoder", "target_encoder", "predictor",
                                             "reconstructor", "mask_token", "target_token"]},
            "latent_health": {"erank_Z": z_erank, "erank_ratio": z_erank / d, "min_std_Z": z_min_std, "d": d},
            "prototypes": {"K": k_clusters, "selected": sel, "mass_selected": mass[sel].tolist(),
                           "mass_selected_sum": float(mass[sel].sum()), "tau": tau},
            "latent_to_kernel": {"lambda": lam, "lambda_grid": list(cfg["ridge_lambdas"]), "holdout_mse": mses,
                                 "r2_per_kernel": r2.tolist(), "r2_median": r2_med,
                                 "r2_min": float(r2.min()), "r2_max": float(r2.max()),
                                 "r2_fit_per_kernel": r2_fit.float().tolist(),
                                 "r2_low": r2_med < float(cfg["r2_warn"]),
                                 "r2_low_options": ["k_plus_2 (outro braço: muda params/FLOPs)", "encoder=local (RF=k)"],
                                 "holdout_mode": holdout_mode},
            "init": self._init_diag(prefix, weights, bias, k, device, getattr(context, "diag_batch", None)),
        }
        prov = {
            "approximations": approximations,
            "data": {"split_key": getattr(train_data, "split_key", "train"),
                     "split_fingerprint": getattr(train_data, "split_fingerprint", None),
                     "n_images": int(n_img), "n_fit_images": int(len(fit_idx)) if holdout_mode == "images" else 1,
                     "n_holdout_images": int(len(ho_idx)) if holdout_mode == "images" else 0,
                     "images_sha256": img_sha, "n_patches": int(n_fit), "n_holdout_patches": int(z_ho.shape[0]),
                     "patch_dim": int(c_in * k * k), "N_over_D": n_fit / float(c_in * k * k), "grid": [h, w]},
            "diagnostics": diag,
        }
        return weights, bias, prov

    def _train_stage1(self, aux: _AuxJEPA, fn: Tensor, cfg, gen, device, t0, max_seconds):
        n_img, _, h, w = fn.shape
        d = int(cfg["latent_dim"])
        pe = _sincos_2d(d, h, w)[None].to(device)
        opt = torch.optim.AdamW(aux.trainable(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        steps = int(cfg["steps"])
        bs = min(int(cfg["batch_size"]), n_img)
        curve = []
        aux.train()
        for step in range(steps):
            if time.monotonic() - t0 > max_seconds:
                raise InfeasibleCandidate("budget: max_seconds", details={"step": step, "max_seconds": max_seconds})
            idx = torch.randint(0, n_img, (bs,), generator=gen)
            ctx_m, tgt_m = block_masks(gen, bs, h, w, cfg)
            fb = fn[idx].to(device)
            ctx_m, tgt_m = ctx_m.to(device), tgt_m.to(device)
            inp = fb * ctx_m + aux.mask_token * (1.0 - ctx_m)
            z_ctx = aux.context_encoder(inp)
            q = z_ctx + (aux.target_token + pe) * tgt_m
            pred = aux.predictor(q)
            with torch.no_grad():
                z_tgt = aux.target_encoder(fb)
                if cfg["target_layernorm"]:
                    z_tgt = F.layer_norm(z_tgt.permute(0, 2, 3, 1), (d,)).permute(0, 3, 1, 2)
            denom = tgt_m.sum().clamp_min(1.0)
            jepa = (((pred - z_tgt) ** 2).mean(1, keepdim=True) * tgt_m).sum() / denom
            rec = (((aux.reconstructor(pred) - fb) ** 2).mean(1, keepdim=True) * tgt_m).sum() / denom
            var_l, cov_l = _vicreg(z_ctx, ctx_m)
            loss = jepa + float(cfg["rec_weight"]) * rec + float(cfg["var_weight"]) * var_l + float(cfg["cov_weight"]) * cov_l
            if not torch.isfinite(loss):
                raise InfeasibleCandidate("spifil3: non-finite stage-1 loss", details={"step": step})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m = float(cfg["ema_start"]) + (float(cfg["ema_end"]) - float(cfg["ema_start"])) * step / max(1, steps - 1)
            aux.ema_update(m)
            curve.append({"step": step, "loss": float(loss), "jepa": float(jepa), "rec": float(rec),
                          "var": float(var_l), "cov": float(cov_l)})
        aux.eval()
        return curve

    @staticmethod
    def _skip_pca(f_fit: Tensor, c_in: int, c_out: int, gen):
        xs = f_fit.permute(0, 2, 3, 1).reshape(-1, c_in).double()
        mu = xs.mean(0)
        cov = (xs - mu).T @ (xs - mu) / max(1, xs.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        comps = evecs.flip(1).T[: min(c_in, c_out)].float()       # top componentes, linhas unitárias
        note = "skip_project_pca"
        if c_out > c_in:
            extra = F.normalize(torch.randn(c_out - c_in, c_in, generator=gen), dim=1)
            comps = torch.cat([comps, extra], 0)
            note = "skip_project_pca+random_rows"
        b = -(comps @ mu.float())
        return comps.reshape(c_out, c_in, 1, 1).contiguous(), b.contiguous(), note

    @staticmethod
    @torch.no_grad()
    def _init_diag(prefix, weights, bias, k, device, diag_batch):
        w = weights["conv.weight"].flatten(1)
        wn = F.normalize(w, dim=1)
        cos = wn @ wn.T
        off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
        out = {"weight_norm_mean": float(w.norm(dim=1).mean()), "weight_norm_std": float(w.norm(dim=1).std()) if w.shape[0] > 1 else 0.0,
               "bias_abs_mean": float(bias["conv.bias"].abs().mean()),
               "redundancy_frac_abs_cos_gt_0.9": float((off.abs() > 0.9).float().mean()) if off.numel() else 0.0,
               "weight_erank": effective_rank(w)}
        if diag_batch is not None:
            fd = _prefix_features(prefix, diag_batch.float().cpu(), device, 16)
            y = F.relu(F.conv2d(fd, weights["conv.weight"], bias["conv.bias"], padding=k // 2))
            yc = y.transpose(0, 1).reshape(y.shape[1], -1)
            dead = (yc.std(1) < 1e-8) | ((yc > 0).float().mean(1) < 0.01)
            out.update({"act_mean": float(yc.mean()), "act_pos_frac": float((yc > 0).float().mean()),
                        "dead_frac": float(dead.float().mean())})
        return out


@register("spifil3_rand")
class Spifil3Rand(Spifil3Predictive):
    """Controle obrigatório S3-rand: o mesmo pipeline com f_ξ aleatório (não treinado), sem Estágio 1."""

    name = "spifil3_rand"

    def __init__(self, **config):
        super().__init__(**{"latent": "random", **config})


def build_block(spec: dict, weights: Mapping[str, Tensor], bias: Mapping[str, Tensor]):
    """Monta o ``SpifilOpsBlock`` final (só conv/skip; nada dos módulos temporários) via ``assemble_block``."""
    return assemble_block(spec, dict(weights), dict(bias))


__all__ = ["Spifil3Predictive", "Spifil3Rand", "build_block", "block_masks", "ridge_fit", "spherical_kmeans",
           "select_mass_fps", "effective_rank", "DEFAULT_CONFIG", "NAME", "VERSION",
           "InfeasibleCandidate"]
