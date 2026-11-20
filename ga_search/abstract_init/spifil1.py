"""SPiFiL-1 — protótipos hierárquicos (desenho: research/spifil_b0b4/agent2/prep_literatura_e_desenho.md §5.1).

HIPÓTESE, não método comprovado. Troca "escolher K patches individuais por Fisher" (SPiFiL-0) por
"resumir a distribuição densa de patches da camada em protótipos" (Coates 2011, FLIM, PCANet), num
espaço branqueado, escolhendo C_out protótipos por REPRESENTATIVIDADE (massa do cluster) + DIVERSIDADE
(FPS). Refinamento SoftHebb opcional (``ga_search.modehb.softhebb``, usado sem modificação).

Pipeline por "banco" (um banco = um conjunto de filtros que vê o mesmo tipo de patch):

1. Patches DENSOS do prefixo congelado: imagens de TREINO de ``train_data.loader_factory(seed)``
   (até ``max_images``), ``positions_per_image`` posições uniformes (semeadas) por imagem, vetor
   channel-major ``C_in·k·k`` (mesma ordem de ``conv.weight.flatten(1)``, zero padding = o da conv).
   N ≤ ``budget.max_patches`` por construção; nenhuma matriz N×N é formada.
2. μ = média; P = I − 11ᵀ/D (``patch_norm="mean_only"``) ou I; Z = ZCA simétrica de (X−μ)P com
   eps = ``eps_zca``·média(autovalores) (igual ao ``GrowOneLayer``); x̃ = Z·P·(x − μ).
3. k-means esférico (ou euclidiano) com semeadura k-means++ sobre as linhas não nulas de x̃,
   K = ``k_mult``·C_out; massa = fração de patches atribuídos.
4. Seleção: ``mass_fps`` (guloso: a_rep·log-massa normalizada + (1−a_rep)·(1−cos mín)/2),
   ``fps`` (farthest point), ``pca`` (top autovetores de cov((X−μ)P), sem ZCA, à la PCANet) ou
   ``all`` (K = C_out, todos os centróides; Coates).
5. DOBRA EXATA do branqueamento nos pesos: c·x̃ = c·Z·P·(x−μ) ⇒ **w = c·Z·P, b = −w·μ**
   (Z e P simétricas; testado numericamente contra a conv).
6. ``unit_response_std`` (opcional, padrão ligado): w ← w/σ(resposta) medido em patches de treino
   SEPARADOS do ajuste (holdout); b recalculado (= −w·μ). Registrado em ``approximations``.
7. SoftHebb (opcional, ``softhebb.epochs > 0``): ``fit_softhebb`` em cada Conv2d do caminho principal,
   na ordem do bloco, com o prefixo como ``input_transform``. A regra não toca no bias
   (``recenter_bias=True`` o recalcula como −w·μ, registrado).

Bancos por tipo de bloco (chaves de ``SpifilOpsBlock``):

* ``conv``/``pointwise``: um banco (C_out, C_in·k²).
* ``depthwise`` (mult. m): um banco por canal de entrada (D = k²), m filtros, linhas [c·m, (c+1)·m).
* ``dwsep``: depthwise m = 1 (por canal) e depois um banco 1×1 ajustado na SAÍDA LINEAR do depthwise
  já ajustado (sem ativação entre os dois, como o bloco).
* ``inception``: um banco por ramo (b1/b3/b5 no input; ``pool`` no ``max_pool2d(3,1,1)`` do input),
  mesmas posições amostradas; ramos independentes (sem diversidade entre ramos — registrado).
* ``residual``/``project``: o skip 1×1 recebe as top-C_out componentes principais do 1×1 de entrada
  (``skip_init="pca"``; se C_out > C_in, o restante vem de protótipos 1×1, registrado) ou um banco
  de protótipos (``skip_init="prototypes"``). ``identity``: nada a ajustar.

Rótulos: LABEL-FREE por padrão (``label_policy="label_free"``): ``y`` NUNCA é lido, amostragem
uniforme de imagens, máscara nunca lida ⇒ invariante a qualquer permutação de rótulos (testado).
Versão com rótulos (``label_policy="supervised"`` no config E ``train_data.label_policy ==
"supervised"``): amostragem estratificada por classe e k-means POR CLASSE (estilo FLIM), quotas
uniformes (resto às classes com mais patches, desempate pelo id), diversidade global entre classes.
Marcada em ``labels_used``/``label_source``/``variant``. Máscaras de objeto NUNCA são lidas.

Canais mortos / degenerados: dimensões de variância zero recebem ganho finito (eps > 0) e conteúdo
zero; linhas nulas de x̃ ficam fora do k-means; um banco sem nenhuma linha válida (p.ex. canal
depthwise constante) dá filtros ZERO (registrado em ``degenerate_banks``), nunca NaN.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ga_search.abstract_init.base import (
    InfeasibleCandidate, InitContext, KernelInitializer, TrainData, check_prefix_unchanged,
    check_train_only, config_sha256, make_provenance, prefix_fingerprint, register, validate_init,
)

__all__ = ["Spifil1Prototypes", "DEFAULT_CONFIG", "fold_whitening", "zca_matrix", "spherical_kmeans",
           "kmeans_euclidean", "select_prototypes", "dense_patches_at"]

DEFAULT_CONFIG: dict[str, Any] = {
    "label_policy": "label_free",      # "label_free" (padrão) | "supervised" (marcado)
    "max_images": None,                # None → budget.max_images
    "positions_per_image": 400,
    "batch_size": 16,
    "whiten": "zca",                   # "zca" | "none"
    "eps_zca": 0.01,                   # relativo: eps = eps_zca · média(autovalores)
    "patch_norm": "none",              # "none" | "mean_only"
    "k_mult": 3,                       # K = k_mult · C_out
    "kmeans": {"spherical": True, "init": "kmeans++", "iters": 30, "chunk": 16384},
    "select": "mass_fps",              # "mass_fps" | "fps" | "pca" | "all"
    "a_rep": 0.5,
    "unit_response_std": True,
    "holdout_frac": 0.1,
    "skip_init": "pca",                # "pca" | "prototypes"
    "softhebb": {"epochs": 0, "lr": 0.01, "temperature": 1.0, "recenter_bias": False},
    "diag_max_rows": 20000,
}

_TINY = 1e-12


# ─── utilidades numéricas (puras, testáveis) ─────────────────────────────────

def _merge(base: dict, over: Optional[dict]) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def mean_removal_matrix(d: int, enabled: bool, dtype=torch.float64) -> torch.Tensor:
    """P = I − 11ᵀ/D (remove a média do patch; linear e simétrica) ou I."""
    eye = torch.eye(d, dtype=dtype)
    if not enabled or d < 2:
        return eye
    return eye - torch.full((d, d), 1.0 / d, dtype=dtype)


def zca_matrix(cov: torch.Tensor, eps_rel: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """ZCA simétrica Z = V·diag(1/√(λ+eps))·Vᵀ e sua inversa; eps = eps_rel·média(λ) (> 0 sempre)."""
    cov = 0.5 * (cov + cov.T)
    e, v = torch.linalg.eigh(cov)
    e = e.clamp_min(0)
    eps = float(eps_rel) * float(e.mean())
    if not eps > 0:
        eps = 1e-8        # todas as dimensões mortas: ganho finito, conteúdo zero
    g = (e + eps).rsqrt()
    z = (v * g) @ v.T
    zinv = (v * (e + eps).sqrt()) @ v.T
    return z, zinv, eps


def fold_whitening(c: torch.Tensor, z: torch.Tensor, p: torch.Tensor,
                   mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dobra exata: c·[Z·P·(x−μ)] = w·x + b com w = c·Z·P e b = −w·μ (Z, P simétricas)."""
    w = c @ z @ p
    return w, -(w @ mu)


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(_TINY)


def _chunked_argmax_sim(x: torch.Tensor, c: torch.Tensor, chunk: int, euclid: bool):
    """Para cada linha de x: índice do centróide mais próximo e o score (cos ou −dist²)."""
    idx, best = [], []
    cn = (c * c).sum(1) if euclid else None
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        sim = xs @ c.T
        if euclid:
            sim = 2 * sim - cn            # −‖x−c‖² + ‖x‖² (constante por linha)
        b, i = sim.max(1)
        idx.append(i)
        best.append(b)
    return torch.cat(idx), torch.cat(best)


def _kmeanspp(x: torch.Tensor, k: int, gen: torch.Generator, euclid: bool) -> torch.Tensor:
    """Semeadura k-means++ (D² euclidiano, ou 1−cos no esférico). Determinística com ``gen``."""
    n = x.shape[0]
    first = int(torch.randint(n, (1,), generator=gen))
    cents = [x[first]]

    def dist(c):
        if euclid:
            return ((x - c) ** 2).sum(1)
        return (1.0 - x @ c).clamp_min(0)

    d = dist(cents[0])
    for _ in range(1, k):
        tot = float(d.sum())
        if tot > 0:
            j = int(torch.multinomial((d / tot).float(), 1, generator=gen))
        else:                              # todos os pontos já cobertos (duplicatas)
            j = int(torch.randint(n, (1,), generator=gen))
        cents.append(x[j])
        d = torch.minimum(d, dist(x[j]))
    return torch.stack(cents)


def spherical_kmeans(xn: torch.Tensor, k: int, gen: torch.Generator, *, iters: int = 30,
                     init: str = "kmeans++", chunk: int = 16384):
    """k-means esférico (Lloyd, cos) em linhas unitárias. Retorna (C unitários, assign, counts, iters)."""
    n = xn.shape[0]
    k = max(1, min(int(k), n))
    if init == "kmeans++":
        c = _kmeanspp(xn, k, gen, euclid=False)
    else:
        c = xn[torch.randperm(n, generator=gen)[:k]].clone()
    c = _row_normalize(c)
    assign = None
    it = 0
    for it in range(1, int(iters) + 1):
        new, _ = _chunked_argmax_sim(xn, c, chunk, euclid=False)
        if assign is not None and torch.equal(new, assign):
            break
        assign = new
        s = torch.zeros_like(c).index_add_(0, assign, xn)
        cnt = torch.bincount(assign, minlength=k)
        nz = (cnt > 0) & (s.norm(dim=1) > _TINY)
        c = torch.where(nz[:, None], _row_normalize(s), c)     # cluster vazio: mantém o centróide
    if assign is None:
        assign, _ = _chunked_argmax_sim(xn, c, chunk, euclid=False)
    return c, assign, torch.bincount(assign, minlength=k), it


def kmeans_euclidean(x: torch.Tensor, k: int, gen: torch.Generator, *, iters: int = 30,
                     init: str = "kmeans++", chunk: int = 16384):
    """k-means euclidiano em x̃ (variante não esférica). Centróides devolvidos UNITÁRIOS (direção do filtro)."""
    n = x.shape[0]
    k = max(1, min(int(k), n))
    c = _kmeanspp(x, k, gen, euclid=True) if init == "kmeans++" else \
        x[torch.randperm(n, generator=gen)[:k]].clone()
    assign = None
    it = 0
    for it in range(1, int(iters) + 1):
        new, _ = _chunked_argmax_sim(x, c, chunk, euclid=True)
        if assign is not None and torch.equal(new, assign):
            break
        assign = new
        s = torch.zeros_like(c).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k)
        c = torch.where((cnt > 0)[:, None], s / cnt.clamp_min(1)[:, None].to(s.dtype), c)
    if assign is None:
        assign, _ = _chunked_argmax_sim(x, c, chunk, euclid=True)
    return _row_normalize(c), assign, torch.bincount(assign, minlength=k), it


def select_prototypes(c: torch.Tensor, mass: torch.Tensor, n_out: int, method: str = "mass_fps",
                      a_rep: float = 0.5, chosen: Optional[torch.Tensor] = None) -> list[int]:
    """Seleção gulosa determinística de ``n_out`` linhas de ``c`` (unitárias).

    ``mass_fps``: score_j = a_rep·nlog(massa_j) + (1−a_rep)·min_s (1−cos(c_j,c_s))/2;
    ``fps``: argmax da distância mínima (1º = maior massa); ``all``: ordem de massa.
    ``chosen``: filtros já escolhidos (outras classes) que entram no termo de diversidade.
    Se ``n_out`` > K, repete os protótipos (duplicatas registradas pelo chamador).
    """
    k = c.shape[0]
    m = mass.double()
    lm = torch.log(m + 1.0 / max(float(m.sum()) * 1e6, 1.0))
    nlog = (lm - lm.min()) / (lm.max() - lm.min()) if float(lm.max() - lm.min()) > 0 else torch.ones_like(lm)
    order_mass = sorted(range(k), key=lambda j: (-float(m[j]), j))
    if method == "all":
        sel = order_mass[:n_out]
    else:
        cd = c.double()
        if chosen is not None and chosen.shape[0]:
            mind = ((1.0 - cd @ chosen.double().T) / 2).min(1).values
        else:
            mind = torch.full((k,), float("inf"), dtype=torch.float64)
        avail = torch.ones(k, dtype=torch.bool)
        sel = []
        for _ in range(min(n_out, k)):
            if not torch.isfinite(mind).any():          # 1º filtro: maior massa (determinístico)
                j = next(j for j in order_mass if avail[j])
            else:
                div = torch.where(torch.isfinite(mind), mind, torch.ones_like(mind))
                if method == "fps":
                    score = div
                elif method == "mass_fps":
                    score = a_rep * nlog + (1.0 - a_rep) * div
                else:
                    raise ValueError(f"select desconhecido: {method!r}")
                score = torch.where(avail, score, torch.full_like(score, -math.inf))
                j = int(torch.argmax(score))            # empate → menor índice
            sel.append(j)
            avail[j] = False
            mind = torch.minimum(mind, (1.0 - cd @ cd[j]) / 2)
    while len(sel) < n_out:                              # K < n_out: duplicatas
        sel.append(order_mass[(len(sel) - k) % k] if k else 0)
    return sel


def _erank_from_cov(cov: torch.Tensor) -> float:
    """exp(entropia dos valores singulares normalizados) de X centrado, via autovalores de XᵀX."""
    e = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0).sqrt()
    if float(e.sum()) <= 0:
        return 0.0
    p = e / e.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _erank_rows(w: torch.Tensor) -> float:
    w = w.double() - w.double().mean(0, keepdim=True)
    if w.shape[0] < 2:
        return float(w.norm() > 0)
    return _erank_from_cov(w.T @ w)


def _bank_redundancy(w: torch.Tensor) -> dict:
    n = w.shape[0]
    if n < 2:
        return {"redundancy_frac_abscos_gt_0.9": float("nan"), "diversity_1_minus_cos": float("nan")}
    wn = _row_normalize(w.double())
    cos = wn @ wn.T
    iu = torch.triu_indices(n, n, 1)
    pc = cos[iu[0], iu[1]]
    return {"redundancy_frac_abscos_gt_0.9": float((pc.abs() > 0.9).double().mean()),
            "diversity_1_minus_cos": float((1 - pc).mean())}


# ─── extração densa de patches ───────────────────────────────────────────────

def dense_patches_at(f: torch.Tensor, pos_r: torch.Tensor, pos_c: torch.Tensor, k: int) -> torch.Tensor:
    """Patches channel-major (B·P, C·k·k) de ``f`` (B,C,H,W) nas posições (B,P), zero padding k//2.

    A ordem (C, dy, dx) é a de ``conv.weight.flatten(1)``; a resposta w·patch é exatamente a da conv.
    """
    b, ch, _, _ = f.shape
    pad = k // 2
    fp = F.pad(f, (pad, pad, pad, pad))
    d = torch.arange(k, device=f.device)
    rr = (pos_r[:, :, None, None] + d[None, None, :, None]).expand(-1, -1, k, k)   # (B,P,k,k)
    cc = (pos_c[:, :, None, None] + d[None, None, None, :]).expand(-1, -1, k, k)
    bi = torch.arange(b, device=f.device)[:, None, None, None].expand_as(rr)
    out = fp.permute(0, 2, 3, 1)[bi, rr, cc]          # (B,P,k,k,C)
    return out.permute(0, 1, 4, 2, 3).reshape(b * pos_r.shape[1], ch * k * k)


# ─── o inicializador ─────────────────────────────────────────────────────────

@register("spifil1")
class Spifil1Prototypes(KernelInitializer):
    """SPiFiL-1: protótipos hierárquicos em patches densos branqueados (label-free por padrão)."""

    name = "spifil1"
    version = "0.1.0"
    label_policy = "label_free"

    def __init__(self, **config):
        super().__init__(**config)
        self.last_fit_internals: dict = {}     # NÃO serializado: Z, P, μ, c, escala por banco (testes)

    # -- configuração / dados -------------------------------------------------
    def _cfg(self, context: InitContext) -> dict:
        over = dict(self.config)
        ctx_cfg = dict(context.config or {})
        ctx_cfg = dict(ctx_cfg.get("spifil1", ctx_cfg))
        return _merge(_merge(DEFAULT_CONFIG, over), ctx_cfg)

    def _load_images(self, train_data: TrainData, context: InitContext, cfg: dict, supervised: bool):
        """Imagens de TREINO (só ``loader_factory``). Label-free: ``y`` nunca é lido."""
        if train_data.loader_factory is None:
            raise InfeasibleCandidate("spifil1: train_data.loader_factory ausente")
        max_img = int(cfg["max_images"] or context.budget.max_images)
        n_cls = int(train_data.n_classes or 0)
        quota = math.ceil(max_img / n_cls) if (supervised and n_cls > 0) else None
        xs, ys, counts, n = [], [], {}, 0
        for batch in train_data.loader_factory(int(context.seed)):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.detach().float().cpu()
            if not supervised:
                take = x[: max_img - n]
                xs.append(take)
                n += take.shape[0]
            else:
                y = batch[1].detach().long().cpu()
                for i in range(x.shape[0]):
                    yi = int(y[i])
                    if n >= max_img or counts.get(yi, 0) >= quota:
                        continue
                    counts[yi] = counts.get(yi, 0) + 1
                    xs.append(x[i:i + 1])
                    ys.append(yi)
                    n += 1
            if n >= max_img:
                break
        if n == 0:
            raise InfeasibleCandidate("spifil1: nenhuma imagem de treino lida")
        images = torch.cat(xs)
        labels = torch.tensor(ys, dtype=torch.long) if supervised else None
        return images, labels

    # -- features / patches ---------------------------------------------------
    @torch.no_grad()
    def _feature_batches(self, prefix, images: torch.Tensor, bs: int, device, amp_dtype,
                         post: Optional[Callable] = None):
        for s in range(0, images.shape[0], bs):
            xb = images[s:s + bs].to(device)
            if prefix is None:
                f = xb
            elif amp_dtype is not None and torch.device(device).type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    f = prefix(xb)
            else:
                f = prefix(xb)
            f = f.float()
            if post is not None:
                f = post(f)
            yield s, f

    @torch.no_grad()
    def _collect(self, feats_iter, k: int, pos: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        rows = []
        for s, f in feats_iter:
            b = f.shape[0]
            rows.append(dense_patches_at(f, pos[0][s:s + b].to(f.device), pos[1][s:s + b].to(f.device), k)
                        .float().cpu())
        return torch.cat(rows)

    # -- um banco de filtros --------------------------------------------------
    def _fit_bank(self, x: torch.Tensor, n_out: int, cfg: dict, gen: torch.Generator, tag: str,
                  labels: Optional[torch.Tensor] = None, select: Optional[str] = None,
                  hold_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """x: (N, D) patches do TREINO. Retorna w (n_out, D) e b (n_out,) em float64 + estatísticas."""
        select = select or cfg["select"]
        n_all, d = x.shape
        fit_rows = ~hold_mask if hold_mask is not None else torch.ones(n_all, dtype=torch.bool)
        xf = x[fit_rows].double()
        xh = x[hold_mask].double() if (hold_mask is not None and bool(hold_mask.any())) else xf
        n = xf.shape[0]
        stats: dict[str, Any] = {"bank": tag, "n_patches": int(n), "n_holdout": int(xh.shape[0]),
                                 "patch_dim": int(d), "n_over_d": float(n) / max(d, 1), "n_out": int(n_out)}
        mu = xf.mean(0)
        xc = xf - mu
        mean_only = cfg["patch_norm"] == "mean_only" and d >= 2
        if cfg["patch_norm"] == "mean_only" and d < 2:
            stats["patch_norm_disabled"] = "D<2"
        p = mean_removal_matrix(d, mean_only)
        xp = xc @ p
        cov = (xp.T @ xp) / max(n, 1)
        stats["candidate_erank"] = _erank_from_cov(cov * max(n, 1))
        stats["dead_input_dims"] = int((torch.diagonal(cov) <= 1e-12).sum())
        if cfg["whiten"] == "zca":
            z, zinv, eps = zca_matrix(cov, cfg["eps_zca"])
            stats["eps_zca"] = eps
        else:
            z = zinv = torch.eye(d, dtype=torch.float64)
        xw = (xp @ z).float()
        norms = xw.norm(dim=1)
        valid = norms > max(float(norms.max()) * 1e-6, 1e-10) if n else torch.zeros(0, dtype=torch.bool)
        stats["n_valid"] = int(valid.sum())
        degenerate = int(valid.sum()) == 0
        stats["degenerate"] = degenerate
        if degenerate:
            w = torch.zeros(n_out, d, dtype=torch.float64)
            stats.update(n_prototypes=0, duplicates=0, kmeans_iters=0)
            self.last_fit_internals[tag] = {"c": torch.zeros(n_out, d, dtype=torch.float64), "z": z, "p": p,
                                            "mu": mu, "scale": torch.ones(n_out, dtype=torch.float64)}
            return w, -(w @ mu), stats

        km = cfg["kmeans"]
        spherical = bool(km.get("spherical", True))
        xv = xw[valid]
        xvn = _row_normalize(xv)

        def run_kmeans(rows: torch.Tensor, kk: int):
            if spherical:
                return spherical_kmeans(_row_normalize(rows), kk, gen, iters=km["iters"], init=km["init"],
                                        chunk=km["chunk"])
            return kmeans_euclidean(rows, kk, gen, iters=km["iters"], init=km["init"], chunk=km["chunk"])

        pca_rows = None
        if select == "pca":
            e, v = torch.linalg.eigh(0.5 * (cov + cov.T))
            order = torch.argsort(e, descending=True)
            r = min(n_out, int((e > max(float(e.max()), 0) * 1e-10).sum()))
            vt = v[:, order[:r]].T.contiguous()                 # (r, D) direções principais (sem ZCA)
            sgn = torch.sign(vt.gather(1, vt.abs().argmax(1, keepdim=True)))
            vt = vt * torch.where(sgn == 0, torch.ones_like(sgn), sgn)   # sinal canônico (determinismo)
            pca_rows = vt @ p
            stats["pca_components"] = int(r)

        n_proto_needed = n_out - (pca_rows.shape[0] if pca_rows is not None else 0)
        c_sel = torch.zeros(0, d, dtype=torch.float64)
        mass_sel: list[float] = []
        duplicates = 0
        if n_proto_needed > 0:
            sel_method = "mass_fps" if select == "pca" else select
            if select == "pca":
                stats["pca_fill"] = int(n_proto_needed)
            if labels is None:
                kk = n_proto_needed if sel_method == "all" else int(cfg["k_mult"]) * n_proto_needed
                c, assign, counts, iters = run_kmeans(xv, kk)
                mass = counts.double() / max(int(counts.sum()), 1)
                sel = select_prototypes(c, mass, n_proto_needed, sel_method, cfg["a_rep"])
                duplicates = len(sel) - len(set(sel))
                c_sel = c[sel].double()
                mass_sel = [float(mass[j]) for j in sel]
                stats.update(n_prototypes=int(c.shape[0]), kmeans_iters=int(iters))
            else:                                               # versão COM RÓTULOS (marcada)
                lv = labels[valid]
                classes = sorted(torch.unique(lv).tolist())
                sizes = {cl: int((lv == cl).sum()) for cl in classes}
                base_q, rem = divmod(n_proto_needed, len(classes))
                bonus = set(sorted(classes, key=lambda cl: (-sizes[cl], cl))[:rem])
                quotas = {cl: base_q + (1 if cl in bonus else 0) for cl in classes}
                chosen = torch.zeros(0, d, dtype=torch.float64)
                n_proto, per_class = 0, {}
                for cl in classes:
                    q = quotas[cl]
                    if q == 0:
                        continue
                    rows = xv[lv == cl]
                    kk = q if sel_method == "all" else int(cfg["k_mult"]) * q
                    c, assign, counts, _ = run_kmeans(rows, kk)
                    mass = counts.double() / max(int(counts.sum()), 1)
                    sel = select_prototypes(c, mass, q, sel_method, cfg["a_rep"], chosen=chosen)
                    duplicates += len(sel) - len(set(sel))
                    chosen = torch.cat([chosen, c[sel].double()])
                    mass_sel += [float(mass[j]) for j in sel]
                    n_proto += int(c.shape[0])
                    per_class[str(cl)] = int(q)
                c_sel = chosen
                stats.update(n_prototypes=n_proto, filters_per_class=per_class)
        stats["duplicates"] = int(duplicates)
        stats["selected_mass_sum"] = float(sum(mass_sel))

        w_parts, c_white = [], []
        if pca_rows is not None:
            w_parts.append(pca_rows)
            c_white.append(pca_rows @ zinv)                     # mesma resposta expressa em x̃
        if c_sel.shape[0]:
            w_p, _ = fold_whitening(c_sel, z, p, mu)
            w_parts.append(w_p)
            c_white.append(c_sel)
        w = torch.cat(w_parts)
        cw = _row_normalize(torch.cat(c_white))

        # cobertura/representatividade no espaço transformado x̃
        rows_cov = xvn
        lim = int(cfg["diag_max_rows"])
        if rows_cov.shape[0] > lim:
            rows_cov = rows_cov[:: math.ceil(rows_cov.shape[0] / lim)]
        maxcos = (rows_cov.double() @ cw.T).max(1).values
        stats["coverage_frac_cos_gt_0.8"] = float((maxcos > 0.8).double().mean())
        stats["coverage_mean_maxcos"] = float(maxcos.mean())

        scale = torch.ones(w.shape[0], dtype=torch.float64)
        if cfg["unit_response_std"]:
            resp = (xh - mu) @ w.T
            sd = resp.std(0, unbiased=False)
            ok = sd > 1e-8
            scale = torch.where(ok, 1.0 / sd.clamp_min(1e-8), torch.ones_like(sd))
            w = w * scale[:, None]
            stats["response_std_floor_hits"] = int((~ok).sum())
        b = -(w @ mu)
        stats["selected_erank"] = _erank_rows(w)
        stats.update(_bank_redundancy(w))
        stats["weight_norm_mean"] = float(w.norm(dim=1).mean())
        self.last_fit_internals[tag] = {"c": c_sel, "z": z, "p": p, "mu": mu, "scale": scale,
                                        "n_pca_rows": 0 if pca_rows is None else int(pca_rows.shape[0])}
        return w, b, stats

    # -- fit -------------------------------------------------------------------
    def fit(self, prefix, train_data: TrainData, block_spec: dict, context: InitContext):
        t0 = time.perf_counter()
        check_train_only(train_data)
        cfg = self._cfg(context)
        self.last_fit_internals = {}
        want_sup = cfg["label_policy"] == "supervised"
        supervised = want_sup and train_data.label_policy == "supervised"
        approximations: list[str] = []
        if want_sup and not supervised:
            approximations.append("label_policy_forced_label_free_by_train_data")
        before = prefix_fingerprint(prefix)
        modes = None
        if prefix is not None:
            modes = [(m, m.training) for m in prefix.modules()]
            rg = [(p_, p_.requires_grad) for p_ in prefix.parameters()]
            prefix.eval()
            for p_ in prefix.parameters():
                p_.requires_grad_(False)
        try:
            weights, bias, banks, eff, n_images, img_sha, extra_approx, sh_info = self._fit_impl(
                prefix, train_data, dict(block_spec), context, cfg, supervised, t0)
        finally:
            if prefix is not None:
                for m, tr in modes:
                    m.train(tr)
                for p_, r in rg:
                    p_.requires_grad_(r)
        check_prefix_unchanged(prefix, before)
        approximations += extra_approx
        if cfg["whiten"] == "zca":
            approximations.append("zca_folded")
        if cfg["patch_norm"] == "mean_only":
            approximations.append("patch_norm:mean_only_folded")
        if cfg["unit_response_std"]:
            approximations.append("unit_response_std")
        validate_init(eff, weights, bias)
        n_patches = sum(bk["n_patches"] + bk["n_holdout"] for bk in banks if bk.get("primary"))
        patch_dim = max((bk["patch_dim"] for bk in banks if bk.get("primary")), default=None)
        cfg_hash = config_sha256(cfg)
        extra = {
            "config": cfg, "config_sha256": cfg_hash,
            "variant": "spifil1-sup" if supervised else "spifil1-lf",
            "faithful_to": None,
            "prefix_state_sha256": before,
            "candidates": {
                "n_candidates": n_patches, "patch_dim": patch_dim,
                "n_prototypes": sum(int(bk.get("n_prototypes", 0)) for bk in banks),
                "candidate_erank": [bk.get("candidate_erank") for bk in banks if bk.get("primary")],
                "selected_erank": [bk.get("selected_erank") for bk in banks if bk.get("primary")],
                "redundancy_frac_abscos_gt_0.9": [bk.get("redundancy_frac_abscos_gt_0.9") for bk in banks
                                                  if bk.get("primary")],
                "coverage_frac_cos_gt_0.8": [bk.get("coverage_frac_cos_gt_0.8") for bk in banks
                                             if bk.get("primary")],
            },
            "banks": banks if len(banks) <= 64 else _summarize_banks(banks),
            "degenerate_banks": [bk["bank"] for bk in banks if bk.get("degenerate")],
            "softhebb": sh_info,
            "mask_note": "spifil1 nunca lê máscaras de objeto",
        }
        prov = make_provenance(
            self.name, self.version, context=context, train_data=train_data, requested_spec=dict(block_spec),
            effective_spec=eff, labels_used=supervised,
            label_source="image_labels" if supervised else "none", mask_used=False,
            n_images=n_images, n_patches=n_patches, patch_dim=patch_dim, approximations=approximations,
            fit_seconds=time.perf_counter() - t0, extra=extra)
        prov["data"]["images_sha256"] = img_sha
        json.dumps(prov)                                   # proveniência serializável (falha cedo)
        return weights, bias, prov

    def _check_time(self, t0, context, where):
        el = time.perf_counter() - t0
        if el > float(context.budget.max_seconds):
            raise InfeasibleCandidate(f"budget: spifil1 excedeu max_seconds em {where}",
                                      details={"elapsed": el, "max_seconds": context.budget.max_seconds})

    def _fit_impl(self, prefix, train_data, spec, context, cfg, supervised, t0):
        device = torch.device(context.device) if context.device is not None else torch.device("cpu")
        gen = torch.Generator().manual_seed(int(context.seed))
        images, img_labels = self._load_images(train_data, context, cfg, supervised)
        n_img = images.shape[0]
        img_sha = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        bs = int(cfg["batch_size"])
        amp = context.amp_dtype

        def feats(post=None):
            return self._feature_batches(prefix, images, bs, device, amp, post)

        # geometria da grade e posições (as MESMAS para todos os bancos do bloco)
        _, f0 = next(iter(self._feature_batches(prefix, images[:1], 1, device, amp)))
        cin, hh, ww = int(f0.shape[1]), int(f0.shape[2]), int(f0.shape[3])
        if cin != int(spec["in_channels"]):
            raise ValueError(f"spifil1: prefixo dá {cin} canais, spec pede in_channels={spec['in_channels']}")
        hw = hh * ww
        ppi = min(int(cfg["positions_per_image"]), hw,
                  max(1, int(context.budget.max_patches) // n_img))
        if ppi >= hw:
            flat = torch.arange(hw).expand(n_img, hw)
        else:
            flat = torch.stack([torch.randperm(hw, generator=gen)[:ppi] for _ in range(n_img)])
        pos = (flat // ww, flat % ww)
        n_rows = n_img * flat.shape[1]
        patch_labels = img_labels.repeat_interleave(flat.shape[1]) if img_labels is not None else None
        hold = torch.zeros(n_rows, dtype=torch.bool)
        if cfg["unit_response_std"] and float(cfg["holdout_frac"]) > 0:
            n_h = int(round(n_rows * float(cfg["holdout_frac"])))
            if 0 < n_h < n_rows:
                hold[torch.randperm(n_rows, generator=gen)[:n_h]] = True

        def bank(x, n_out, tag, select=None, primary=True, use_labels=True):
            lab = patch_labels[~hold] if (patch_labels is not None and use_labels) else None
            w, b, st = self._fit_bank(x, n_out, cfg, gen, tag, labels=lab, select=select, hold_mask=hold)
            st["primary"] = primary
            banks.append(st)
            self._check_time(t0, context, tag)
            return w, b

        banks: list[dict] = []
        kind, k = spec["conv"], int(spec["kernel"])
        cout = int(spec["channels"])
        eff = dict(spec)
        eff.setdefault("builder", "spifil_ops")
        eff.setdefault("norm_in_block", True)
        W: dict[str, torch.Tensor] = {}
        B: dict[str, torch.Tensor] = {}
        extra_approx: list[str] = []
        mus: dict[str, torch.Tensor] = {}

        def put(key, w, b, shape):
            W[f"{key}.weight"] = w.reshape(shape).float().contiguous()
            B[f"{key}.bias"] = b.reshape(shape[0]).float().contiguous()

        if kind in ("conv", "pointwise"):
            kk = 1 if kind == "pointwise" else k
            x = self._collect(feats(), kk, pos)
            w, b = bank(x, cout, "conv")
            put("conv", w, b, (cout, cin, kk, kk))
            mus["conv"] = self.last_fit_internals["conv"]["mu"]
        elif kind == "depthwise":
            m = cout // cin
            if m * cin != cout:
                raise InfeasibleCandidate(f"spifil1: depthwise exige channels % in_channels == 0 ({cout}/{cin})")
            x = self._collect(feats(), k, pos)
            ws, bs_ = [], []
            for ch in range(cin):
                w, b = bank(x[:, ch * k * k:(ch + 1) * k * k], m, f"dw{ch}", primary=(ch == 0))
                ws.append(w)
                bs_.append(b)
            put("conv", torch.cat(ws), torch.cat(bs_), (cout, 1, k, k))
            extra_approx.append("depthwise_per_channel_banks")
        elif kind == "dwsep":
            x = self._collect(feats(), k, pos)
            ws, bs_ = [], []
            for ch in range(cin):
                w, b = bank(x[:, ch * k * k:(ch + 1) * k * k], 1, f"dw{ch}", primary=False)
                ws.append(w)
                bs_.append(b)
            wd = torch.cat(ws).float().reshape(cin, 1, k, k)
            bd = torch.cat(bs_).float()
            put("conv.depthwise", wd.double(), bd.double(), (cin, 1, k, k))

            def dw_post(f, wd=wd.to(device), bd=bd.to(device)):
                return F.conv2d(f, wd, bd, padding=k // 2, groups=cin)

            x2 = self._collect(feats(dw_post), 1, pos)
            w, b = bank(x2, cout, "pointwise")
            put("conv.pointwise", w, b, (cout, cin, 1, 1))
            extra_approx.append("dwsep:depthwise_per_channel_then_pointwise_on_linear_dw_output")
        elif kind == "inception":
            from ga_search.modehb.ops_norm_act import inception_split
            branches = [list(br) for br in (spec.get("inception_branches") or inception_split(cout))]
            eff["inception_branches"] = branches
            for name, bk_, ch in branches:
                bk_, ch = int(bk_), int(ch)
                if name == "pool":
                    x = self._collect(feats(lambda f: F.max_pool2d(f, 3, 1, 1)), 1, pos)
                    w, b = bank(x, ch, "pool")
                    put("conv.branches.pool.proj", w, b, (ch, cin, 1, 1))
                else:
                    x = self._collect(feats(), bk_, pos)
                    w, b = bank(x, ch, name)
                    put(f"conv.branches.{name}", w, b, (ch, cin, bk_, bk_))
            extra_approx.append("inception_per_branch_no_cross_branch_diversity")
        else:
            raise InfeasibleCandidate(f"spifil1: conv {kind!r} não suportado")

        # residual: skip
        skip = spec.get("skip") if spec.get("residual") else None
        if skip == "project":
            x1 = self._collect(feats(), 1, pos)
            if cfg["skip_init"] == "pca":
                w, b = bank(x1, cout, "skip", select="pca", primary=False)
                extra_approx.append("skip_project:pca" + (":fill_prototypes" if cout > cin else ""))
            else:
                w, b = bank(x1, cout, "skip", primary=False)
                extra_approx.append("skip_project:prototypes")
            put("skip", w, b, (cout, cin, 1, 1))
        elif skip == "identity" and cin != cout:
            raise InfeasibleCandidate("spifil1: skip identity exige in_channels == channels")

        sh_info = None
        if int(cfg["softhebb"].get("epochs", 0)) > 0:
            sh_info = self._softhebb(W, B, eff, prefix, images, cfg, device, context)
            extra_approx.append("softhebb_refined" + (":bias_recentered" if cfg["softhebb"].get("recenter_bias")
                                                      else ":bias_untouched"))
        return W, B, banks, eff, n_img, img_sha, extra_approx, sh_info

    # -- SoftHebb (código existente, sem modificação) -------------------------
    def _softhebb(self, W, B, eff, prefix, images, cfg, device, context):
        from ga_search.abstract_init.base import assemble_block
        from ga_search.modehb.softhebb import fit_softhebb
        sh = cfg["softhebb"]
        blk = assemble_block(eff, W, B, device=device)
        bs = int(cfg["batch_size"])
        batches = [images[s:s + bs] for s in range(0, images.shape[0], bs)]

        def pre(x):
            return x if prefix is None else prefix(x).float()

        kind = eff["conv"]
        plan: list[tuple[str, nn.Conv2d, Callable]] = []
        if kind in ("conv", "pointwise", "depthwise"):
            plan.append(("conv", blk.conv, pre))
        elif kind == "dwsep":
            dw = blk.conv.depthwise
            plan.append(("conv.depthwise", dw, pre))
            plan.append(("conv.pointwise", blk.conv.pointwise, lambda x: dw(pre(x))))
        elif kind == "inception":
            for name, mod in blk.conv.branches.items():
                if name == "pool":
                    plan.append(("conv.branches.pool.proj", mod.proj, lambda x: F.max_pool2d(pre(x), 3, 1, 1)))
                else:
                    plan.append((f"conv.branches.{name}", mod, pre))
        info = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(context.seed))
            for key, conv, tf in plan:
                res = fit_softhebb(conv, batches, int(sh["epochs"]), float(sh["lr"]), float(sh["temperature"]),
                                   device=device, input_transform=tf)
                info[key] = {k: v for k, v in res.items() if k != "history"}
                W[f"{key}.weight"] = conv.weight.detach().float().cpu().clone()
                if sh.get("recenter_bias"):
                    tag = {"conv": "conv", "conv.pointwise": "pointwise", "conv.branches.pool.proj": "pool"}.get(
                        key, key.split(".")[-1])
                    mu = self.last_fit_internals.get(tag, {}).get("mu")
                    if mu is not None:                 # só bancos densos (depthwise: por canal, não recentrado)
                        w = W[f"{key}.weight"].double().flatten(1)
                        B[f"{key}.bias"] = (-(w @ mu)).float()
                        info.setdefault("recentered", []).append(key)
        return info


def _summarize_banks(banks: list[dict]) -> dict:
    """Resumo (média/mín/máx) quando há muitos bancos (depthwise/dwsep com C_in grande)."""
    out: dict[str, Any] = {"n_banks": len(banks), "first": banks[0], "degenerate": sum(bool(b.get("degenerate"))
                                                                                       for b in banks)}
    for key in ("candidate_erank", "selected_erank", "coverage_frac_cos_gt_0.8", "n_valid"):
        vals = [float(b[key]) for b in banks if b.get(key) is not None and not
                (isinstance(b[key], float) and math.isnan(b[key]))]
        if vals:
            out[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    return out
