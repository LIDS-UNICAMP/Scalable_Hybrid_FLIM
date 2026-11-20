"""pooling_head.py -- cabeca de embedding do MO-DEHB (prompt §5.5, §5.6, §6).

    ultimo mapa conv [B, in_ch, H, W] -> Conv 1x1 (treinavel) -> Pooling -> Embedding [B, embed_dim]

Poolings (``POOLINGS``):
    gap        media global                                  (k=1 bloco)
    gem        GeM com p=3 fixo                              (k=1)
    gem_learn  GeM com p aprendivel, p = 1 + 1e-3 + softplus(raw)  (k=1, p > 1 sempre)
    meanstd    media || desvio                               (k=2)
    moments    media || desvio || skew padronizada [|| kurtose excedente]  (k=3 ou 4)
    combo      gap || gem(p=3)                               (k=2)

Largura do Conv 1x1: escolhida para que a dimensao final (apos concatenar os
blocos) seja EXATAMENTE ``embed_dim``.  Com ``k`` blocos, ``c = embed_dim // k +
embed_dim % k``: o bloco da media usa os ``c`` canais e os demais blocos os
primeiros ``embed_dim // k`` -> ``c + (k-1)*(embed_dim//k) = embed_dim``.  Para as
dims da grade (1280/1536/2048/2560) isso so tem resto em ``moments`` sem kurtose.

GeM e valores negativos: depois de GELU/SELU/BN o mapa pode ter negativos, e
``x**p`` com ``p`` fracionario em x<0 vira NaN.  Seguimos a convencao de
``eval/svm.py::_encode_pooled`` (modo ``gem``): ``x.clamp(min=eps)`` -- negativos
contam como ``eps`` (equivale a um ReLU antes do GeM).  Todo pooling roda em fp32
(upcast local) e devolve no dtype de entrada: ``pow(3)``/``pow(4)`` estouram fp16
e ``eps=1e-6`` some em bf16.

TTA: ``tta_transforms`` reproduz as vistas do modo ``gem_tta`` de
``eval/svm.py:_encode_pooled`` (8 = 4 rotacoes x {id, hflip}, na mesma ordem);
nao adiciona parametros, custa ``n`` forwards (``tta_cost``).
"""

from __future__ import annotations

import threading
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EMBED_DIMS = (1280, 1536, 2048, 2560)
MIN_EMBED_DIM = 1280
POOLINGS = ("gap", "gem", "gem_learn", "meanstd", "moments", "combo")
GEM_P = 3.0
GEM_EPS = 1e-6
GEM_P_MARGIN = 1e-3  # p = 1 + margin + softplus: softplus underflowa p/ 0 em fp32
STAT_EPS = 1e-5
TTA_NS = (1, 4, 8)
TTA_GEOMETRIES = ("rot_flip", "flip_only")


# ─── funcoes de pooling (fp32 interno) ────────────────────────────────────────

def gem_pool(x: Tensor, p: Tensor | float = GEM_P, eps: float = GEM_EPS) -> Tensor:
    """GeM ``[B,C,H,W] -> [B,C]``; ``clamp(min=eps)`` torna finito com negativos."""
    xf = x.float().clamp(min=eps)
    if isinstance(p, Tensor):
        p = p.float()
    return xf.pow(p).mean(dim=(2, 3)).pow(1.0 / p)


def _central_stats(x: Tensor, order: int) -> list[Tensor]:
    """[mean, std, skew(, kurt excedente)] por canal, fp32, ``[B,C]`` cada."""
    xf = x.float().flatten(2)
    mean = xf.mean(2)
    d = xf - mean[..., None]
    var = d.pow(2).mean(2)
    std = (var + STAT_EPS).sqrt()
    out = [mean, std]
    if order >= 3:
        out.append(d.pow(3).mean(2) / std.pow(3))
    if order >= 4:
        out.append(d.pow(4).mean(2) / var.add(STAT_EPS).pow(2) - 3.0)
    return out


# ─── cabeca ───────────────────────────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """Conv 1x1 treinavel -> pooling -> embedding de dimensao exata ``embed_dim``.

    Args:
        in_ch:     canais do ultimo mapa conv do encoder.
        embed_dim: dimensao final (>= 1280; a grade usa ``EMBED_DIMS``).
        pooling:   um de ``POOLINGS``.
        kurtosis:  so ``moments``: inclui a kurtose excedente (k=4 blocos).
        act:       ativacao opcional apos o conv ("none", "relu", "gelu").
        bn:        BatchNorm2d apos o conv (como os heads 1x1 de
                   ``src/models/distillation.py:727``).
        gem_p_init: p inicial de ``gem_learn`` (> 1).
    """

    def __init__(self, in_ch: int, embed_dim: int = 1280, pooling: str = "gap", *,
                 kurtosis: bool = True, act: str = "none", bn: bool = False,
                 gem_p_init: float = GEM_P) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling invalido {pooling!r}; use um de {POOLINGS}")
        assert embed_dim >= MIN_EMBED_DIM, f"embed_dim={embed_dim} < {MIN_EMBED_DIM}"
        self.in_ch, self.pooling, self.nominal_dim = int(in_ch), pooling, int(embed_dim)
        self.n_blocks = {"meanstd": 2, "combo": 2, "moments": 4 if kurtosis else 3}.get(pooling, 1)
        self.block = embed_dim // self.n_blocks
        self.width = self.block + embed_dim % self.n_blocks
        layers: list[nn.Module] = [nn.Conv2d(in_ch, self.width, kernel_size=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(self.width))
        if act != "none":
            layers.append({"relu": nn.ReLU, "gelu": nn.GELU}[act]())
        self.proj = nn.Sequential(*layers)
        if pooling == "gem_learn":
            assert gem_p_init > 1.0 + GEM_P_MARGIN, "gem_p_init precisa ser > 1"
            # inverso de softplus: raw = log(exp(p-1-margin) - 1)
            self.p_raw = nn.Parameter(
                torch.tensor(float(gem_p_init) - 1.0 - GEM_P_MARGIN).expm1().log())
        self.out_dim = self.width + (self.n_blocks - 1) * self.block
        assert self.out_dim == self.nominal_dim, (self.out_dim, self.nominal_dim)

    @property
    def p(self) -> Tensor:
        """p efetivo de ``gem_learn`` (sempre > 1)."""
        return 1.0 + GEM_P_MARGIN + F.softplus(self.p_raw)

    @property
    def proj_params(self) -> int:
        return sum(t.numel() for t in self.parameters())

    def proj_flops(self, h: int, w: int) -> int:
        """FLOPs (2*MAC) do conv 1x1 num mapa ``h x w``; pooling e desprezivel."""
        return 2 * self.in_ch * self.width * h * w

    def pool(self, z: Tensor) -> Tensor:
        b = self.block
        if self.pooling == "gap":
            y = z.float().mean(dim=(2, 3))
        elif self.pooling == "gem":
            y = gem_pool(z)
        elif self.pooling == "gem_learn":
            y = gem_pool(z, self.p)
        elif self.pooling == "combo":
            y = torch.cat([z.float().mean(dim=(2, 3)), gem_pool(z[:, :b])], 1)
        else:  # meanstd / moments
            stats = _central_stats(z, self.n_blocks)
            y = torch.cat([stats[0]] + [s[:, :b] for s in stats[1:]], 1)
        return y.to(z.dtype)

    #: projected-map elements above which the head runs in batch chunks (~1 GiB fp32 each).
    CHUNK_ELEMS = 1 << 28

    def _proj_pool(self, x: Tensor) -> Tensor:
        return self.pool(self.proj(x))

    def forward(self, feat_map: Tensor) -> Tensor:
        """Exact chunked evaluation: the 1x1 projection to ``embed_dim`` channels at full
        resolution (e.g. 256 x 1280 x 112^2 = 16 GB) is never materialised for the whole
        batch. Without BatchNorm the head is per-sample, so chunking is bit-for-bit the
        same math; under autograd each chunk is checkpointed (recomputed in backward)."""
        n, _, h, w = feat_map.shape
        per = self.width * h * w
        if n <= 1 or n * per <= self.CHUNK_ELEMS or any(isinstance(m, nn.BatchNorm2d) for m in self.proj):
            return self._proj_pool(feat_map)
        c = max(1, self.CHUNK_ELEMS // per)
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            outs = [checkpoint(self._proj_pool, x, use_reentrant=False) for x in feat_map.split(c, 0)]
        else:
            outs = [self._proj_pool(x) for x in feat_map.split(c, 0)]
        return torch.cat(outs, 0)


# ─── TTA ──────────────────────────────────────────────────────────────────────

def _rot(k: int) -> Callable[[Tensor], Tensor]:
    return lambda x: torch.rot90(x, k, (2, 3))


def _hflip_rot(k: int) -> Callable[[Tensor], Tensor]:
    return lambda x: torch.rot90(x.flip(3), k, (2, 3))


def tta_transforms(n: int = 1, geometry: str = "rot_flip") -> list[Callable[[Tensor], Tensor]]:
    """Vistas de TTA sobre ``[B,C,H,W]``.

    n=1: identidade.  n=4: id, hflip, vflip, rot180 (so flips -> vale p/ H != W).
    n=8 (``rot_flip``): as 8 vistas de ``eval/svm.py`` modo ``gem_tta`` -- para
    x em (id, hflip), rot90^k, k=0..3.  ``flip_only`` so tem 4 vistas distintas.
    """
    if n not in TTA_NS:
        raise ValueError(f"n de TTA invalido {n}; use um de {TTA_NS}")
    if geometry not in TTA_GEOMETRIES:
        raise ValueError(f"geometria invalida {geometry!r}; use um de {TTA_GEOMETRIES}")
    if n == 1:
        return [lambda x: x]
    if n == 4:
        return [lambda x: x, lambda x: x.flip(3), lambda x: x.flip(2), lambda x: x.flip((2, 3))]
    if geometry == "flip_only":
        raise ValueError("flip_only tem so 4 vistas distintas; use n<=4 ou rot_flip")
    return [_rot(k) for k in range(4)] + [_hflip_rot(k) for k in range(4)]


def _eigh(a: Tensor):
    """``torch.linalg.eigh``; on non-convergence (LAPACK / cusolver: ill-conditioned input or many
    repeated eigenvalues, e.g. the zero eigenvalues of dead ReLU channels) retry on the symmetrized
    matrix with a tiny relative diagonal jitter, then on CPU float64. The result is identical whenever
    the plain eigh converges; non-finite input still raises (a real failure)."""
    try:
        return torch.linalg.eigh(a)
    except RuntimeError:                         # torch.linalg.LinAlgError subclasses RuntimeError
        s = (a + a.transpose(-1, -2)) / 2
        eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
        for scale in (1e-10, 1e-7):
            jit = scale * (s.diagonal(dim1=-2, dim2=-1).abs().mean(-1) + 1.0)
            try:
                return torch.linalg.eigh(s + jit[..., None, None] * eye)
            except RuntimeError:
                continue
        c = s.detach().double().cpu()
        jit = 1e-6 * (c.diagonal(dim1=-2, dim2=-1).abs().mean(-1) + 1.0)
        w, v = torch.linalg.eigh(c + jit[..., None, None] * torch.eye(c.shape[-1], dtype=c.dtype))
        return w.to(a.device, a.dtype), v.to(a.device, a.dtype)


HOST_EIGH_CHUNK = 16          # matrices per CPU LAPACK task; fixed, so the result never depends on `workers`
GPU_EIGH_CHUNK = 32           # matrices per concurrent cusolver task (stream)
_POOLS: dict = {}
_TLS = threading.local()


def _pool(workers: int):
    """Persistent thread pool shared by the eigh helpers (one per worker count)."""
    from concurrent.futures import ThreadPoolExecutor
    if workers not in _POOLS:
        _POOLS[workers] = ThreadPoolExecutor(workers, thread_name_prefix="cov_eigh")
    return _POOLS[workers]


def _eigh_host(a: Tensor, workers: int = 1):
    """OPT-IN (``eigh_device="cpu"``): batched fp64 ``eigh`` of the per-image covariances on the CPU
    (numpy / OpenBLAS LAPACK ``dsyevd``), ``HOST_EIGH_CHUNK`` matrices per task over ``workers`` threads.

    ~0.25 ms per 64x64 matrix vs 1.5-3 ms for the per-matrix cusolver loop, but NOT bitwise equal to
    cusolver: the sqrt of the clamped ~1e-17 eigenvalues of rank-deficient covariances (spatial size or
    live channels < k) differs (~1e-9), and the signed-sqrt + StandardScaler of the cov_head readout
    amplify those noise-only coordinates, so the validation kappa of some WiSE alphas can move (measured:
    P4 remote_sensing, 2 of 5 alphas). Per matrix the result is independent of ``workers``.
    A chunk that does not converge falls back to ``_eigh`` (symmetrize + jitter retries)."""
    import numpy as np
    c = a.detach().to("cpu", torch.float64).contiguous().numpy()
    parts = [c[i:i + HOST_EIGH_CHUNK] for i in range(0, len(c), HOST_EIGH_CHUNK)]

    def one(m):
        try:
            return np.linalg.eigh(m)
        except np.linalg.LinAlgError:
            w_, v_ = _eigh(torch.from_numpy(np.array(m)))
            return w_.numpy(), v_.numpy()
    workers = max(1, int(workers))
    res = list(_pool(workers).map(one, parts)) if workers > 1 and len(parts) > 1 else [one(p) for p in parts]
    w = torch.from_numpy(np.concatenate([r[0] for r in res]))
    v = torch.from_numpy(np.concatenate([r[1] for r in res]))
    return w.to(a.device, a.dtype), v.to(a.device, a.dtype)


def _thread_stream(device) -> "torch.cuda.Stream":
    streams = getattr(_TLS, "streams", None)
    if streams is None:
        streams = _TLS.streams = {}
    if device not in streams:
        streams[device] = torch.cuda.Stream(device=device)
    return streams[device]


def _eigh_cuda_chunk(chunk: Tensor, main):
    """``torch.linalg.eigh`` of one chunk on this worker thread's own CUDA stream."""
    with torch.cuda.device(chunk.device):
        s = _thread_stream(chunk.device)
        s.wait_stream(main)                      # the covariances were produced on the main stream
        chunk.record_stream(s)
        with torch.cuda.stream(s):
            w, v = torch.linalg.eigh(chunk)
        w.record_stream(main)                    # consumed on the main stream after we return
        v.record_stream(main)
        s.synchronize()
    return w, v


def _eigh_many(mats: "list[Tensor]", workers: int = 1) -> "list[tuple[Tensor, Tensor]]":
    """``[_eigh(m) for m in mats]``, BITWISE identical, faster on CUDA.

    torch's CUDA ``eigh`` of a [B, n, n] fp64 batch with n > 32 is a loop of one cusolver ``syevd`` per
    matrix (batched Jacobi only for n <= 32), so it is launch/sync-latency bound: 1.5-3.3 ms per 64x64
    matrix with the GPU nearly idle (75% of the cov_head fitness time of a P4). Here the batches are split
    in ``GPU_EIGH_CHUNK`` chunks run concurrently on ``workers`` threads, each on its own stream: every
    matrix still gets the very same ``syevd`` call (the per-matrix result does not depend on the batch),
    measured bitwise equal; 5-7x faster with 8 workers in a micro-benchmark, ~3.5x inside the fitness
    (jaci, shared GPU at 98% util). n <= 32 (batched Jacobi, batch-dependent),
    CPU tensors and ``workers`` <= 1 keep the plain ``_eigh`` call. If any chunk fails to converge, that
    whole matrix batch is redone with ``_eigh`` (same jitter fallback as before, on the whole batch)."""
    workers = max(1, int(workers))
    par = [m.is_cuda and m.shape[-1] > 32 and m.shape[0] > 1 for m in mats]
    if workers <= 1 or not any(par):
        return [_eigh(m) for m in mats]
    main = torch.cuda.current_stream(next(m.device for m, p in zip(mats, par) if p))
    jobs = {i: [_pool(workers).submit(_eigh_cuda_chunk, c, main) for c in m.split(GPU_EIGH_CHUNK)]
            for i, (m, p) in enumerate(zip(mats, par)) if p}
    out = []
    for i, m in enumerate(mats):
        if i not in jobs:
            out.append(_eigh(m))
            continue
        try:
            res = [f.result() for f in jobs[i]]
        except RuntimeError:                     # LinAlgError: redo the whole batch as before
            for f in jobs[i]:
                f.cancel()
            out.append(_eigh(m))
            continue
        out.append((torch.cat([r[0] for r in res]), torch.cat([r[1] for r in res])))
    return out


class CovReadout:
    """Second-order readout ``cov_head`` (measured on remote_sensing, 7 NAS checkpoints: mean
    test kappa 0.51 -> 0.61 vs the head readout). Label-free and fitted per candidate:

    1. per block, PCA of the channel covariance on train pixels (``n_img`` images x ``n_px``
       random pixels), top ``k`` components;
    2. per image and block: project, centre over pixels, covariance k x k, matrix square root
       (fp64 eigh), upper triangle;
    3. append the head embedding; each group gets signed-sqrt + L2, then concatenation.
    Use with ``svm_fitness(scaler="standard", C=100)`` (repo default).

    ``eigh_device`` (step 2): None / ``"device"`` = fp64 eigh where the covariance lives (the GPU),
    over ``eigh_workers`` concurrent streams (``_eigh_many``: bitwise identical to the historical
    single call, 1 worker = that call); ``"cpu"`` = opt-in ``_eigh_host`` (CPU LAPACK, faster, NOT
    bitwise equal: see its docstring)."""

    def __init__(self, k: int = 64, n_img: int = 160, n_px: int = 256, seed: int = 0,
                 eigh_device: str | None = None, eigh_workers: int = 1):
        self.k, self.n_img, self.n_px, self.seed = int(k), int(n_img), int(n_px), int(seed)
        if eigh_device not in (None, "device", "cpu"):
            raise ValueError(f"eigh_device must be None|'device'|'cpu', got {eigh_device!r}")
        self.eigh_device, self.eigh_workers = eigh_device, max(1, int(eigh_workers))
        self.proj = None

    @staticmethod
    @torch.no_grad()
    def _blocks(encoder, x):
        outs, h = [], x
        for blk in encoder:
            h = blk(h)
            outs.append(h.float())
        return outs

    @torch.no_grad()
    def fit(self, encoder: nn.Module, x_train: Tensor, device, bs: int = 32,
            micro: int | None = None) -> "CovReadout":
        """``micro`` < ``bs`` runs the encoder in smaller chunks (less GPU memory) while drawing
        the SAME random pixels as ``micro=None`` (one draw per block per ``bs`` batch)."""
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        pix = None
        for b in x_train[: self.n_img].split(bs):
            idxs = None
            for m in b.split(micro or bs):
                outs = self._blocks(encoder, m.to(device))
                pix = pix or [[] for _ in outs]
                if idxs is None:
                    idxs = [torch.randint(0, o.shape[2] * o.shape[3], (min(self.n_px, o.shape[2] * o.shape[3]),),
                                          generator=g) for o in outs]
                for i, o in enumerate(outs):
                    f = o.permute(0, 2, 3, 1).reshape(o.shape[0], -1, o.shape[1])
                    pix[i].append(f[:, idxs[i].to(f.device)].reshape(-1, o.shape[1]).double().cpu())
        self.proj = []
        for P in pix:
            P = torch.cat(P)
            mu = P.mean(0)
            cov = torch.atleast_2d(torch.cov((P - mu).T))
            _, V = _eigh(cov)
            k = min(self.k, P.shape[1])
            self.proj.append((mu.float().to(device), V[:, -k:].float().to(device)))
        return self

    @staticmethod
    def _pn_l2(z: Tensor) -> Tensor:
        z = z.sign() * z.abs().sqrt()
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def _covs(self, outs):
        covs = []
        for o, (mu, V) in zip(outs, self.proj):
            f = o.permute(0, 2, 3, 1).reshape(o.shape[0], -1, o.shape[1]) - mu
            p = f @ V
            p = p - p.mean(1, keepdim=True)
            covs.append(((p.transpose(1, 2) @ p) / p.shape[1]).double())
        return covs

    def _eig(self, covs):
        if self.eigh_device == "cpu":
            return [_eigh_host(C, self.eigh_workers) for C in covs]
        return _eigh_many(covs, self.eigh_workers)   # all matrices at once: more chunks in flight

    def _finish(self, covs, eig, hz, norm):
        groups = []
        for C, (w, U) in zip(covs, eig):
            S = (U * w.clamp(min=0).sqrt()[:, None, :]) @ U.transpose(1, 2)
            iu = torch.triu_indices(C.shape[1], C.shape[1], device=C.device)
            groups.append(norm(S[:, iu[0], iu[1]].float()))
        groups.append(norm(hz))
        return torch.cat(groups, 1)

    @torch.no_grad()
    def __call__(self, encoder: nn.Module, head: nn.Module, x: Tensor, pn: bool = True) -> Tensor:
        """``pn=False`` skips the per-group signed-sqrt + L2 (raw sqrtm triangles + head)."""
        norm = self._pn_l2 if pn else (lambda z: z)
        outs = self._blocks(encoder, x)
        covs = self._covs(outs)
        return self._finish(covs, self._eig(covs), head(outs[-1]).float(), norm)

    @torch.no_grad()
    def embed(self, encoder: nn.Module, head: nn.Module, X: Tensor, bs: int, device,
              group_mats: int = 2048) -> Tensor:
        """``torch.cat([self(encoder, head, b.to(device)).cpu() for b in X.split(bs)])``, bitwise: the
        encoder still runs on the same ``bs`` batches, but the per-image eigh of several batches (up to
        ``group_mats`` matrices) is solved in one ``_eigh_many`` call, so more chunks run concurrently."""
        out, pend, n = [], [], 0

        def flush():
            if not pend:
                return
            eig = iter(self._eig([C for covs, _ in pend for C in covs]))
            for covs, hz in pend:
                out.append(self._finish(covs, [next(eig) for _ in covs], hz, self._pn_l2).cpu())
            pend.clear()
        for b in X.split(bs):
            outs = self._blocks(encoder, b.to(device))
            covs = self._covs(outs)
            pend.append((covs, head(outs[-1]).float()))
            del outs
            n += sum(len(C) for C in covs)
            if n >= group_mats:
                flush()
                n = 0
        flush()
        return torch.cat(out)


def readout_embed(encoder: nn.Module, head: nn.Module, x: Tensor, readout: str = "head") -> Tensor:
    """Embedding read by the SVM.
    ``head``    : head(encoder(x)) (the searched >=1280-d embedding);
    ``cat``     : GeM of EVERY block output concatenated (hypercolumn-style, eval/svm.py gem_cat);
    ``cat_head``: ``cat`` + the head embedding."""
    if readout == "head":
        return head(encoder(x))
    feats, h = [], x
    for blk in encoder:
        h = blk(h)
        feats.append(gem_pool(h))
    if readout == "cat_head":
        feats.append(head(h).float())
    elif readout != "cat":
        raise ValueError(f"readout must be head|cat|cat_head, got {readout!r}")
    return torch.cat(feats, 1)


def encode_tta(encoder: nn.Module, head: nn.Module, x: Tensor, n: int = 1,
               geometry: str = "rot_flip", readout: str = "head") -> Tensor:
    """Media de ``readout_embed(t(x))`` sobre as vistas; sem parametros novos."""
    views = tta_transforms(n, geometry)
    if n == 1:
        return readout_embed(encoder, head, x, readout)
    return torch.stack([readout_embed(encoder, head, t(x), readout).float() for t in views]).mean(0)


def tta_cost(n: int) -> float:
    """Multiplicador de latencia/FLOPs do TTA (n forwards completos)."""
    if n not in TTA_NS:
        raise ValueError(f"n de TTA invalido {n}; use um de {TTA_NS}")
    return float(n)
