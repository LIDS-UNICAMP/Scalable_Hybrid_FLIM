"""Operators for the MO-DEHB search space: activations, normalizations, convs, ConvBlock.

Prompt §5.1-5.3. Design choices (documented here, referenced from the report):

* ``layer`` norm = channel-wise LayerNorm for conv maps (ConvNeXt style): normalizes the C
  values at every pixel independently (permute -> nn.LayerNorm(C) -> permute). GroupNorm(1, C)
  was NOT chosen because it normalizes jointly over (C, H, W), i.e. it mixes spatial statistics.
* ``zca`` = statistical ZCA (Coates & Ng 2011), same formula as flim/spifil.py:340
  (W = U diag(1/sqrt(S + eps*mean(S))) U^T, relative eps). That code is inline inside a
  callback, so the formula is mirrored in ``zca_matrix`` instead of imported. ``fit`` must be
  called on TRAINING features only; buffers are frozen; before ``fit`` the layer is identity.
* ``dbn`` = Decorrelated Batch Norm (Huang et al., CVPR 2018): differentiable ZCA whitening
  per channel group, gradients flow through ``torch.linalg.eigh``. Cost guard: group_size is
  capped (default 16), so no covariance larger than group_size x group_size is decomposed.
* ``depthwise`` with cout % cin != 0: repaired by using groups = gcd(cin, cout) (keeps the
  requested cout; becomes a grouped conv). ``conv.repaired`` is set to True.
* Residual in ConvBlock: identity only when input and output shapes match (cin == cout and
  stride == 1). On mismatch the default (``residual_mismatch="disable"``) drops the skip;
  ``"project"`` adds a 1x1 (strided) projection instead. The skip is added after act, before pool.
* SELU: self-normalization is NOT assumed; ``selu_compat_check`` only reports the conditions.
* ``batch_zca`` / ``batch_dbn`` = BatchNorm2d followed by statistical ZCA / DBN
  (``nn.Sequential(bn, zca|dbn)``). The ZCA part is fitted like plain ``zca`` (the runner's
  forward pre-hook on every ``ZCAWhitening`` sees the BN output on TRAINING batches).
* ``inception`` (GoogLeNet-style, Szegedy et al. 2015) = ``InceptionConv``: parallel branches
  1x1, 3x3, 5x5 and maxpool3x3(s1)->1x1, concatenated to EXACTLY ``cout`` channels. Branch
  kernels are fixed (the ``kernel`` gene is ignored). Split: ``cout // n`` per branch,
  remainder to the 1x1 branch. Branches are direct k x k convs (no 1x1 reduction), so every
  branch is a dense conv on the block input (SPiFiL-fittable). If ``cout < 4`` fewer branches
  are built, in the order 1x1, 3x3, 5x5, pool (cout=3 -> no pool branch; 2 -> 1x1+3x3;
  1 -> 1x1 only), so every branch has >= 1 channel. norm/act/residual/pool follow the concat.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = ("relu", "leaky_relu", "prelu", "gelu", "silu", "selu", "relu6")
NORMS = ("none", "batch", "group", "layer", "instance", "zca", "dbn", "batch_zca", "batch_dbn")
CONVS = ("conv", "depthwise", "pointwise", "dwsep", "inception")


# ----------------------------------------------------------------------------- activations
def make_act(name: str, leaky_slope: float = 0.01) -> nn.Module:
    table = {
        "relu": lambda: nn.ReLU(inplace=False),
        "leaky_relu": lambda: nn.LeakyReLU(leaky_slope),
        "prelu": lambda: nn.PReLU(),
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(),
        "selu": lambda: nn.SELU(),
        "relu6": lambda: nn.ReLU6(),
    }
    if name not in table:
        raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")
    return table[name]()


# ----------------------------------------------------------------------------- norms
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm on (N, C, H, W): normalizes over C at each pixel."""

    def __init__(self, C: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _flatten_channels(x: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) or (N, C) -> (M, C) samples-by-channels."""
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    if x.dim() == 2:
        return x
    raise ValueError(f"expected (N,C,H,W) or (N,C), got {tuple(x.shape)}")


ZCA_MIN_FLOOR = 1e-8   # absolute floor of the relative ZCA eps (all-zero / dead features)


def safe_eigh(a: torch.Tensor, on_cpu: bool = True):
    """``torch.linalg.eigh`` on the symmetrized matrix; on non-convergence retries with a
    tiny diagonal jitter, then on CPU float64. Identical result whenever eigh converges.

    on_cpu=True (default; one-off fits such as the ZCA whitening matrix): CUDA input is solved on
    CPU float64 -- cusolver syevd once failed with EXECUTION_FAILED + illegal memory access that
    poisoned the CUDA context (athena larvae c00040).
    on_cpu=False (per-training-step DBN whitening): stay on the GPU. A CPU round-trip every step
    starved the GPU (0% util) on a CPU-oversubscribed host; a fatal CUDA error now crashes the
    process unrecorded and the search resumes (modehb._fatal_cuda), so the GPU path is safe."""
    a = (a + a.transpose(-1, -2)) / 2
    if a.is_cuda and on_cpu:
        return _cpu_eigh(a)
    try:
        return torch.linalg.eigh(a)
    except RuntimeError:                 # torch.linalg.LinAlgError subclasses RuntimeError
        if not a.is_cuda:
            return _cpu_eigh(a)
        jit = 1e-10 * (a.diagonal(dim1=-2, dim2=-1).abs().mean() + 1.0)
        try:
            return torch.linalg.eigh(a + jit * torch.eye(a.shape[-1], dtype=a.dtype, device=a.device))
        except RuntimeError:
            return _cpu_eigh(a)


def _cpu_eigh(a: torch.Tensor):
    c = a.detach().double().cpu()
    try:
        s, u = torch.linalg.eigh(c)
    except RuntimeError:
        jit = 1e-10 * (c.diagonal().abs().mean() + 1.0)
        s, u = torch.linalg.eigh(c + jit * torch.eye(c.shape[-1], dtype=c.dtype))
    return s.to(a.device, a.dtype), u.to(a.device, a.dtype)


def zca_matrix(cov: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """U diag(1/sqrt(S + eps*mean(S))) U^T — same as flim/spifil.py:340 (relative eps).

    Robustness (no change when mean(S) > ZCA_MIN_FLOOR/eps): the relative floor
    eps*mean(S) is clamped to ZCA_MIN_FLOOR, so an all-zero covariance (every channel
    dead, e.g. single-filter SPiFiL banks) gives a finite W instead of 1/sqrt(0)=inf -> NaN.
    """
    cov = torch.atleast_2d(cov)          # 1 channel: torch.cov returns a 0-d tensor
    if not bool(torch.isfinite(cov).all()):
        raise FloatingPointError("ZCA covariance contains non-finite values (upstream NaN/Inf)")
    s, u = safe_eigh(cov)
    s = s.clamp_min(0)
    floor = (eps * s.mean()).clamp_min(ZCA_MIN_FLOOR)
    return u @ torch.diag(1 / torch.sqrt(s + floor)) @ u.T


class _InvSqrtPSD(torch.autograd.Function):
    """W = U diag(clamp(S, floor)^-1/2) U^T for a symmetric PSD matrix.

    Forward is exactly DBN's eigh formula. Backward uses the Daleckii-Krein (Loewner)
    divided differences L_ij = (f(s_i)-f(s_j))/(s_i-s_j), with L_ij = f'(s_i) when
    s_i ~= s_j. For distinct eigenvalues this IS the gradient autograd computes through
    eigh; for (near-)degenerate ones (e.g. >= 2 dead channels in a group -> eigenvalues
    all == eps) eigh's backward divides by 0 and returns NaN, this one stays exact/finite.
    """

    @staticmethod
    def forward(ctx, cov, floor: float):
        c = cov.double()
        s, u = safe_eigh(c, on_cpu=False)    # per step: keep it on the GPU (see safe_eigh)
        f = s.clamp_min(floor).rsqrt()
        ctx.save_for_backward(s, u)
        ctx.floor = floor
        return ((u * f) @ u.T).to(cov.dtype)

    @staticmethod
    def backward(ctx, g):
        s, u = ctx.saved_tensors
        sc = s.clamp_min(ctx.floor)
        f = sc.rsqrt()
        fp = torch.where(s > ctx.floor, -0.5 * sc.pow(-1.5), torch.zeros_like(s))
        ds = s[:, None] - s[None, :]
        close = ds.abs() <= 1e-9 * s.abs().max().clamp_min(1e-300)
        L = torch.where(close, 0.5 * (fp[:, None] + fp[None, :]),
                        (f[:, None] - f[None, :]) / torch.where(close, torch.ones_like(ds), ds))
        gd = g.double()
        gd = (gd + gd.T) / 2
        return (u @ ((u.T @ gd @ u) * L) @ u.T).to(g.dtype), None


def _bf16_autocast(x: torch.Tensor) -> bool:
    """True inside a bf16 autocast region (PNAS-CS ``train.amp_dtype: bf16``): whitening then runs in
    fp32 (eigh-based, numerically sensitive). fp16 autocast / no autocast: unchanged behaviour."""
    dev = x.device.type
    try:                                   # torch >= 2.4: per-device API
        return bool(torch.is_autocast_enabled(dev)) and torch.get_autocast_dtype(dev) == torch.bfloat16
    except (TypeError, RuntimeError, AttributeError):
        pass
    # torch 2.2 (athena): is_autocast_enabled() takes no argument and means CUDA; CPU has its own API
    if dev == "cuda":
        return bool(torch.is_autocast_enabled()) and torch.get_autocast_gpu_dtype() == torch.bfloat16
    if dev == "cpu":
        return bool(torch.is_autocast_cpu_enabled()) and torch.get_autocast_cpu_dtype() == torch.bfloat16
    return False


class ZCAWhitening(nn.Module):
    """Statistical ZCA. ``fit`` on training data only; frozen buffers; forward = W (x - mu).

    Unfitted -> identity (so a block can be built/trained before the whitening is fitted).
    """

    def __init__(self, C: int, eps: float = 1e-2):
        super().__init__()
        self.C, self.eps = C, eps
        self.register_buffer("mu", torch.zeros(C))
        self.register_buffer("W", torch.eye(C))
        self.register_buffer("fitted", torch.tensor(False))

    @staticmethod
    @torch.no_grad()
    def moments(features: torch.Tensor, chunk_elems: int = 1 << 25):
        """(n, sum, X^T X) in float64, streamed over chunks of <= ``chunk_elems`` values
        (256 MiB float64): neither the (rows, C) flattened copy nor its float64 cast ever
        exists for the whole batch (both scale with C x H x W x batch)."""
        f = features.detach()
        C = f.shape[1]
        sm = torch.zeros(C, dtype=torch.float64, device=f.device)
        xtx = torch.zeros(C, C, dtype=torch.float64, device=f.device)
        n = 0
        per_sample = max(1, f[:1].numel())
        for fb in f.split(max(1, chunk_elems // per_sample), 0):   # batch chunks
            x = _flatten_channels(fb)
            for r in x.split(max(1, chunk_elems // C), 0):        # row chunks (one huge sample)
                r = r.double()
                n += r.shape[0]
                sm += r.sum(0)
                xtx += r.T @ r
        return n, sm, xtx

    @torch.no_grad()
    def fit_moments(self, n: int, sm: torch.Tensor, xtx: torch.Tensor) -> "ZCAWhitening":
        """Same estimate as ``fit`` (torch.cov, unbiased) from accumulated moments."""
        if n < 2:   # n <= C: rank-deficient cov, still well-posed thanks to the eps floor
            raise ValueError(f"ZCA fit needs >= 2 samples, got {n}")
        mu = sm / n
        cov = (xtx - n * torch.outer(mu, mu)) / (n - 1)
        W = zca_matrix(cov, self.eps)
        self.mu.copy_(mu.to(self.mu.dtype))
        self.W.copy_(W.to(self.W.dtype))
        self.fitted.fill_(True)
        return self

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "ZCAWhitening":
        x = _flatten_channels(features.detach()).double()
        if x.shape[0] < 2:   # n <= C: rank-deficient cov, still well-posed thanks to the eps floor
            raise ValueError(f"ZCA fit needs >= 2 samples, got {x.shape[0]}")
        mu = x.mean(0)
        W = zca_matrix(torch.cov((x - mu).T), self.eps)
        self.mu.copy_(mu.to(self.mu.dtype))
        self.W.copy_(W.to(self.W.dtype))
        self.fitted.fill_(True)
        return self

    def forward(self, x):
        if not bool(self.fitted):
            return x
        if _bf16_autocast(x):
            with torch.autocast(x.device.type, enabled=False):
                return self.forward(x.float())
        W, mu = self.W.to(x.dtype), self.mu.to(x.dtype)
        if x.dim() == 4:
            return torch.einsum("dc,nchw->ndhw", W, x - mu.view(1, -1, 1, 1))
        return (x - mu) @ W.T


class DecorrelatedBatchNorm(nn.Module):
    """Decorrelated BN (Huang et al., CVPR 2018), ZCA whitening per channel group.

    Train: per group g, Sigma_g = cov over (N,H,W) + eps I, W_g = U diag(S^-1/2) U^T computed
    with differentiable eigh (gradients flow). Running mean / running W_g for eval.
    Then an affine per-channel scale/shift. Groups = ceil(C / group_size); the last group may be
    smaller. group_size is capped at ``max_group_size`` so eigh never sees > cap^2 matrices.
    """

    def __init__(self, C: int, group_size: int = 16, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, max_group_size: int = 16):
        super().__init__()
        self.C, self.eps, self.momentum = C, eps, momentum
        self.group_size = max(1, min(group_size, max_group_size, C))
        self.bounds = [(s, min(s + self.group_size, C)) for s in range(0, C, self.group_size)]
        self.register_buffer("running_mean", torch.zeros(C))
        for i, (a, b) in enumerate(self.bounds):
            self.register_buffer(f"running_W{i}", torch.eye(b - a))
        if affine:
            self.weight = nn.Parameter(torch.ones(C))
            self.bias = nn.Parameter(torch.zeros(C))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _whiten_group(self, xg: torch.Tensor, i: int) -> torch.Tensor:
        # xg: (G, M) centered
        G, M = xg.shape
        cov = xg @ xg.T / max(M - 1, 1) + self.eps * torch.eye(G, device=xg.device, dtype=xg.dtype)
        W = _InvSqrtPSD.apply(cov, float(self.eps))   # = U diag(clamp(S,eps)^-1/2) U^T
        with torch.no_grad():
            rw = getattr(self, f"running_W{i}")
            rw.mul_(1 - self.momentum).add_(self.momentum * W.detach().to(rw.dtype))
        return W @ xg

    def forward(self, x):
        if _bf16_autocast(x):
            with torch.autocast(x.device.type, enabled=False):
                return self.forward(x.float())
        N, C, H, W = x.shape
        flat = x.permute(1, 0, 2, 3).reshape(C, -1)  # (C, M)
        if self.training:
            mu = flat.mean(1, keepdim=True)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.squeeze(1).detach())
            xc = flat - mu
            # eigh in float32 at least (float16 eigh is unsupported / unstable)
            work = xc.float() if xc.dtype in (torch.float16, torch.bfloat16) else xc
            out = torch.cat([self._whiten_group(work[a:b], i)
                             for i, (a, b) in enumerate(self.bounds)], 0).to(x.dtype)
        else:
            # memória (growth_mem): centra e branqueia POR GRUPO direto no buffer de saída, sem a cópia
            # centrada inteira (C, M) nem o cat. Mesmas fatias contíguas, mesmo GEMM -> bit a bit igual.
            rm = self.running_mean[:, None].to(x.dtype)
            if torch.is_grad_enabled() and (x.requires_grad or any(
                    getattr(self, f"running_W{i}").requires_grad for i in range(len(self.bounds)))):
                xc = flat - rm
                out = torch.cat([getattr(self, f"running_W{i}").to(x.dtype) @ xc[a:b]
                                 for i, (a, b) in enumerate(self.bounds)], 0)
            else:
                out = torch.empty_like(flat)
                for i, (a, b) in enumerate(self.bounds):
                    torch.matmul(getattr(self, f"running_W{i}").to(x.dtype), flat[a:b] - rm[a:b], out=out[a:b])
            del flat
        out = out.reshape(C, N, H, W).permute(1, 0, 2, 3)
        if self.weight is not None:
            if (torch.is_grad_enabled() and (out.requires_grad or self.weight.requires_grad)) \
                    or out.dtype != self.weight.dtype or self.bias.dtype != self.weight.dtype:
                out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            else:                                   # sem autograd: in-place, mesmas 2 operações elementares
                out.mul_(self.weight.view(1, -1, 1, 1)).add_(self.bias.view(1, -1, 1, 1))
        return out


def make_norm(name: str, C: int, groups: int = 8, dbn_group_size: int = 16,
              zca_eps: float = 1e-2, dbn_eps: float = 1e-5) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm2d(C)
    if name == "group":
        g = math.gcd(C, groups)  # repair: C % groups != 0 -> gcd(C, groups) groups
        return nn.GroupNorm(g, C)
    if name == "layer":
        return LayerNorm2d(C)
    if name == "instance":
        return nn.InstanceNorm2d(C, affine=True)
    if name == "zca":
        return ZCAWhitening(C, eps=zca_eps)
    if name == "dbn":
        return DecorrelatedBatchNorm(C, group_size=dbn_group_size, eps=dbn_eps)
    if name == "batch_zca":
        return nn.Sequential(nn.BatchNorm2d(C), ZCAWhitening(C, eps=zca_eps))
    if name == "batch_dbn":
        return nn.Sequential(nn.BatchNorm2d(C),
                             DecorrelatedBatchNorm(C, group_size=dbn_group_size, eps=dbn_eps))
    raise ValueError(f"unknown norm {name!r}; choose from {NORMS}")


# ----------------------------------------------------------------------------- convs
INCEPTION_ORDER = (("b1", 1), ("b3", 3), ("b5", 5), ("pool", 1))


def inception_split(cout: int) -> list[list]:
    """``[[name, kernel, channels], ...]`` summing to ``cout`` (see module doc)."""
    cout = int(cout)
    if cout < 1:
        raise ValueError("inception needs cout >= 1")
    n = min(len(INCEPTION_ORDER), cout)
    base, rem = divmod(cout, n)
    return [[name, k, base + (rem if i == 0 else 0)]
            for i, (name, k) in enumerate(INCEPTION_ORDER[:n])]


class InceptionConv(nn.Module):
    """GoogLeNet-style parallel branches, concatenated on channels (order of ``branches``).

    ``branches``: ``[[name, kernel, channels], ...]`` (default ``inception_split(cout)``);
    names b1/b3/b5 are Conv2d(cin, ch, k), ``pool`` is MaxPool2d(3, 1, 1) -> Conv2d(cin, ch, 1).
    """

    def __init__(self, cin: int, cout: int | None = None, stride: int = 1, bias: bool = True,
                 branches: list | None = None):
        super().__init__()
        branches = [list(b) for b in (branches or inception_split(cout))]
        if cout is not None and sum(int(b[2]) for b in branches) != int(cout):
            raise ValueError(f"inception branches {branches} do not sum to {cout}")
        self.branch_spec = branches
        self.branches = nn.ModuleDict()
        for name, k, ch in branches:
            k, ch = int(k), int(ch)
            if name == "pool":
                self.branches[name] = nn.Sequential(OrderedDict(
                    pool=nn.MaxPool2d(3, 1, 1), proj=nn.Conv2d(cin, ch, 1, stride, 0, bias=bias)))
            else:
                self.branches[name] = nn.Conv2d(cin, ch, k, stride, k // 2, bias=bias)
        self.in_channels = cin
        self.out_channels = sum(int(b[2]) for b in branches)

    def forward(self, x):
        return torch.cat([m(x) for m in self.branches.values()], 1)


def make_conv(kind: str, cin: int, cout: int, k: int, stride: int = 1, bias: bool = True) -> nn.Module:
    pad = k // 2
    if kind == "inception":
        return InceptionConv(cin, cout, stride, bias)
    if kind == "conv":
        return nn.Conv2d(cin, cout, k, stride, pad, bias=bias)
    if kind == "pointwise":
        return nn.Conv2d(cin, cout, 1, stride, 0, bias=bias)
    if kind == "depthwise":
        repaired = cout % cin != 0
        groups = math.gcd(cin, cout) if repaired else cin
        m = nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=bias)
        m.repaired = repaired
        return m
    if kind == "dwsep":
        return nn.Sequential(nn.Conv2d(cin, cin, k, stride, pad, groups=cin, bias=False),
                             nn.Conv2d(cin, cout, 1, 1, 0, bias=bias))
    raise ValueError(f"unknown conv {kind!r}; choose from {CONVS}")


# ----------------------------------------------------------------------------- block
class ConvBlock(nn.Module):
    """conv -> norm -> act [-> dropout] (+ skip) -> pool, from a spec dict.

    spec keys: in_channels (or pass cin), channels, kernel, conv, residual, activation, norm,
    pool ("none" | "max" | "avg"; pool_size, default 2), stride (1), dropout (0.0),
    residual_mismatch ("disable" | "project"), groups (GroupNorm, 8), dbn_group_size (16).
    """

    def __init__(self, spec: dict, cin: int | None = None):
        super().__init__()
        self.spec = dict(spec)
        cin = cin if cin is not None else spec["in_channels"]
        cout, k = spec["channels"], spec.get("kernel", 3)
        stride = spec.get("stride", 1)
        self.act_name = spec.get("activation", "relu")
        self.norm_name = spec.get("norm", "none")
        self.conv = make_conv(spec.get("conv", "conv"), cin, cout, k, stride,
                              bias=self.norm_name == "none")
        self.norm = make_norm(self.norm_name, cout, groups=spec.get("groups", 8),
                              dbn_group_size=spec.get("dbn_group_size", 16))
        self.act = make_act(self.act_name)
        p = spec.get("dropout", 0.0)
        self.dropout = ((nn.AlphaDropout(p) if self.act_name == "selu" else nn.Dropout(p))
                        if p > 0 else nn.Identity())
        self.skip: nn.Module | None = None
        self.residual_mode = "off"
        if spec.get("residual", False):
            if cin == cout and stride == 1:
                self.skip, self.residual_mode = nn.Identity(), "identity"
            elif spec.get("residual_mismatch", "disable") == "project":
                self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False)
                self.residual_mode = "project"
        pool, ps = spec.get("pool", "none"), spec.get("pool_size", 2)
        self.pool = {"none": nn.Identity(), "max": nn.MaxPool2d(ps, ps),
                     "avg": nn.AvgPool2d(ps, ps)}[pool if pool else "none"]
        self.init_scheme = "pytorch_default"
        if self.act_name == "selu":
            init_lecun_normal(self)

    @property
    def has_residual(self) -> bool:
        return self.skip is not None

    def forward(self, x):
        y = self.dropout(self.act(self.norm(self.conv(x))))
        if self.skip is not None:
            y = y + self.skip(x)
        return self.pool(y)


def init_lecun_normal(block: nn.Module) -> nn.Module:
    """LeCun normal (std = 1/sqrt(fan_in)) for every conv; zero bias. Marks block.init_scheme."""
    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    block.init_scheme = "lecun_normal"
    return block


def selu_compat_check(block: nn.Module) -> dict:
    """Report the SELU self-normalization preconditions; does NOT assume they hold.

    Self-normalization (Klambauer et al. 2017) needs lecun_normal init, no normalization layer,
    and AlphaDropout (not Dropout). Residual skips also break the fixed-point argument.
    """
    acts = [m for m in block.modules() if isinstance(m, nn.SELU)]
    norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm,
                  ZCAWhitening, DecorrelatedBatchNorm)
    has_norm = any(isinstance(m, norm_types) for m in block.modules())
    drops = [m for m in block.modules() if isinstance(m, (nn.Dropout, nn.AlphaDropout))]
    report = {
        "uses_selu": bool(acts),
        "init_lecun_normal": getattr(block, "init_scheme", None) == "lecun_normal",
        "norm_none": not has_norm,
        "dropout_alpha": all(isinstance(m, nn.AlphaDropout) for m in drops),
        "no_residual": getattr(block, "skip", None) is None,
    }
    report["compatible"] = all(report.values())
    return report
