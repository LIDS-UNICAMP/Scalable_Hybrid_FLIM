"""SoftHebb: a *local weight-update rule* for convolutional filters.

Reference
---------
Journé, Rodriguez, Guo, Moraitis. "Hebbian Deep Learning Without Feedback",
ICLR 2023, arXiv:2209.11883. Official code: github.com/NeuromorphicComputing/SoftHebb
(``hebbconv.py``: ``HebbSoftKrotovConv2d.get_wta`` mode 0, ``HebbHardConv2d.delta_weight``,
``update_lr``; ``activation.py``: ``Triangle``). Rule introduced in Moraitis et al.,
"SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks" (2022).

SoftHebb is NOT an activation nor a normalization: it is a plasticity rule that
changes ``conv.weight`` in place, without gradients, from local quantities only.

Equations (neuron = output filter k, input patch x of size C*kh*kw):

  (1) pre-activation      u_k = w_k . x                      (conv without bias)
  (2) soft WTA            s_k = softmax_k(u_k / tau)          (paper: inverse temp. 1/tau)
  (3) soft anti-Hebbian   y_k = +s_k if k = argmax_j u_j  else  -s_k
  (4) plasticity          Dw_k = eta_k * sum_{patches} y_k (x - u_k w_k)
                          (official code normalises Dw by max|Dw| before applying)
  (5) adaptive lr         eta_k = eta * (| ||w_k||_2 - 1 | + 1e-10) ** p    (p = ``lr_power``)
  (6) triangle (output activation for the *next* layer, not part of the rule)
                          tri(u)_k = ReLU(u_k - mean_j u_j) ** power

Fixed point of (4): E[y x] = E[y u] w_k  =>  w_k . E[y x] = E[y u] ||w_k||^2 and since
w_k . E[y x] = E[y u], ||w_k||_2 -> 1 (unit-norm filters; R = 1 in the paper).

Protocol (progressive MO-DEHB, prompt §5.4) -- the three phases are SEQUENTIAL,
NEVER simultaneous:
  1. SPiFiL init        : filters initialised from image markers / patches.
  2. SoftHebb refinement: ``fit_softhebb`` -- weights change ONLY through (1)-(5);
                          ``requires_grad`` is False and no optimizer may hold a
                          reference to these weights during this phase.
  3. Local AE (backprop): ``requires_grad`` restored; an optimizer built *after*
                          ``fit_softhebb`` returns may then train the layer.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["softhebb_update", "fit_softhebb", "triangle", "soft_wta"]

#: (batch x filters x H x W) elements per SoftHebb chunk (~128 MiB per fp32 temporary).
SOFTHEBB_CHUNK_ELEMS = 1 << 25


def triangle(u: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Eq. (6): triangle activation, ReLU(u - mean over channels)**power (activation.py:Triangle)."""
    return F.relu(u - u.mean(dim=1, keepdim=True)) ** power


def soft_wta(u: torch.Tensor, temperature: float, anti_hebbian: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (2)-(3). ``u``: (B, K, H, W). Returns (y, s) where s is the softmax over filters."""
    s = torch.softmax(u / temperature, dim=1)
    if not anti_hebbian:
        return s, s
    winner = F.one_hot(u.argmax(dim=1), num_classes=u.shape[1]).permute(0, 3, 1, 2).to(s.dtype)
    y = s * (2.0 * winner - 1.0)  # +s for winner, -s for the others
    return y, s


def _standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Non-affine BatchNorm (the paper places BN(affine=False) before each Hebbian conv)."""
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def softhebb_update(
    conv: nn.Conv2d,
    x: torch.Tensor,
    lr: float,
    temperature: float,
    *,
    anti_hebbian: bool = True,
    adaptive_lr: bool = True,
    lr_power: float = 0.5,
    normalize_delta: bool = True,
    input_triangle: bool = False,
    standardize_input: bool = False,
) -> dict[str, float]:
    """Apply ONE SoftHebb update (Eqs. 1-5) to ``conv.weight`` in place, under no_grad.

    Uses the conv's own stride/padding/dilation. Bias is ignored by the rule and left
    untouched. Returns scalar stats for logging.

    Grouped / depthwise convs (``groups > 1``): SoftHebb is applied PER GROUP -- group g's
    output neurons compete (soft-WTA) only among themselves on group g's input channels,
    i.e. exactly the groups=1 rule run independently on each (input slice, filter slice)
    pair, which is what a grouped conv computes. With one filter per group (depthwise
    multiplier 1) the soft-WTA is trivially 1 and the rule reduces to Oja's rule
    (y = u, delta = u x - u^2 w). Stats are averaged over groups
    (``softhebb/groups`` reports the count).
    """
    if conv.groups != 1:
        return _grouped_update(conv, x, lr, temperature, anti_hebbian=anti_hebbian,
                               adaptive_lr=adaptive_lr, lr_power=lr_power,
                               normalize_delta=normalize_delta, input_triangle=input_triangle,
                               standardize_input=standardize_input)
    with torch.no_grad():
        w = conv.weight
        x = x.to(device=w.device, dtype=w.dtype)
        if input_triangle:
            x = triangle(x)
        if standardize_input:
            x = _standardize(x)
        kh, kw = w.shape[-2:]
        # Batch-chunked (exact): yx, yu and the entropy are sums over (b, l), so they are
        # accumulated chunk by chunk; the ~12 temporaries of size (b, K, H, W) of
        # (1)-(3) (one_hot is int64) never exist for the whole batch at once.
        per = max(1, int(w.shape[0]) * int(x.shape[-2]) * int(x.shape[-1]))
        bs = max(1, int(SOFTHEBB_CHUNK_ELEMS // per))
        yx = torch.zeros_like(w)
        yu = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
        ent_sum, n_pos = torch.zeros((), dtype=torch.float64, device=w.device), 0
        for xc in x.split(bs, 0):
            # (1) u = w * x  (no bias)
            u = F.conv2d(xc, w, None, conv.stride, conv.padding, conv.dilation)
            # (2)-(3)
            y, s = soft_wta(u, temperature, anti_hebbian)
            # (4) sum_{b,l} y_k x  -- conv trick of the official code (hebbconv.py delta_weight):
            # correlate input channels with y maps; stride/dilation swap roles.
            yx += F.conv2d(
                xc.transpose(0, 1), y.transpose(0, 1), None,
                stride=conv.dilation, padding=conv.padding, dilation=conv.stride,
            ).transpose(0, 1)[:, :, :kh, :kw]
            yu += (y * u).sum(dim=(0, 2, 3))
            ent_sum += (-(s * torch.log(s.clamp_min(1e-30))).sum(dim=1)).sum().double()
            n_pos += s.shape[0] * s.shape[2] * s.shape[3]
            del u, y, s
        delta = yx - yu.view(-1, 1, 1, 1) * w
        if normalize_delta:
            delta = delta / (delta.abs().amax() + 1e-30)
        # (5) per-neuron adaptive learning rate
        if adaptive_lr:
            radius = w.flatten(1).norm(dim=1)
            eta = lr * ((radius - 1.0).abs() + 1e-10) ** lr_power
            step = eta.view(-1, 1, 1, 1) * delta
        else:
            step = lr * delta
        w.add_(step)
        ent = (ent_sum / max(n_pos, 1)).to(w.dtype)
        return {
            "softhebb/weight_norm": float(w.flatten(1).norm(dim=1).mean()),
            "softhebb/delta_norm": float(step.norm()),
            "softhebb/winner_entropy": float(ent),
            "softhebb/winner_entropy_normalized": float(ent / math.log(max(w.shape[0], 2))),
        }


def _grouped_update(conv: nn.Conv2d, x: torch.Tensor, lr: float, temperature: float,
                    **kw) -> dict[str, float]:
    """Per-group SoftHebb (see ``softhebb_update``); updates ``conv.weight`` in place."""
    g = conv.groups
    cin_g, cout_g = conv.in_channels // g, conv.out_channels // g
    stats = []
    with torch.no_grad():
        for i in range(g):
            sub = nn.Conv2d(cin_g, cout_g, conv.kernel_size, conv.stride, conv.padding,
                            conv.dilation, groups=1, bias=False).to(conv.weight.device,
                                                                    conv.weight.dtype)
            sub.weight.data = conv.weight.data[i * cout_g:(i + 1) * cout_g]   # view: in place
            stats.append(softhebb_update(sub, x[:, i * cin_g:(i + 1) * cin_g], lr, temperature, **kw))
    out = {k: float(sum(s[k] for s in stats) / len(stats)) for k in stats[0]}
    out["softhebb/delta_norm"] = float(math.sqrt(sum(s["softhebb/delta_norm"] ** 2 for s in stats)))
    out["softhebb/groups"] = float(g)
    return out


def _emit(logger: Any, metrics: dict, step: int) -> None:
    if logger is None:
        return
    if hasattr(logger, "log"):
        logger.log(metrics, step=step)
    elif hasattr(logger, "log_metrics"):
        logger.log_metrics(metrics, step=step)
    elif callable(logger):
        logger(metrics, step)


def fit_softhebb(
    conv: nn.Conv2d,
    loader_or_tensor_iter: Iterable[Any] | Callable[[], Iterable[Any]],
    epochs: int,
    lr: float,
    temperature: float,
    device: torch.device | str = "cpu",
    logger: Any = None,
    *,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **update_kwargs: Any,
) -> dict[str, Any]:
    """SoftHebb refinement phase (phase 2 of: SPiFiL init -> SoftHebb -> local AE).

    Protocol:
      * ``conv.weight.requires_grad`` (and bias) are set False for the phase and asserted
        False before every update; their original values are restored on exit (even on error).
      * Weights change ONLY through ``softhebb_update``; no optimizer is created or stepped.
        Callers must build any backprop optimizer AFTER this function returns.
      * ``loader_or_tensor_iter``: a re-iterable (DataLoader / list) or a zero-arg callable
        returning a fresh iterator per epoch. Items are tensors (B,C,H,W) or tuples whose
        first element is the image batch. ``input_transform`` (e.g. frozen previous layers)
        is applied under no_grad.

    Logs ``softhebb/weight_norm``, ``softhebb/delta_norm``, ``softhebb/winner_entropy``
    per step through ``logger`` (``.log(dict, step=)``, ``.log_metrics`` or callable).
    """
    conv.to(device)
    params = [p for p in (conv.weight, conv.bias) if p is not None]
    saved = [p.requires_grad for p in params]
    history: list[dict[str, float]] = []
    step = 0
    try:
        for p in params:
            p.requires_grad_(False)
        for epoch in range(epochs):
            it = loader_or_tensor_iter() if callable(loader_or_tensor_iter) else loader_or_tensor_iter
            for batch in it:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                if isinstance(xb, dict):
                    xb = xb.get("image", next(iter(xb.values())))
                xb = xb.to(device)
                assert conv.weight.requires_grad is False, "SoftHebb phase: weights must not track grads"
                with torch.no_grad():
                    if input_transform is not None:
                        xb = input_transform(xb)
                    m = softhebb_update(conv, xb, lr, temperature, **update_kwargs)
                m["softhebb/epoch"] = epoch
                history.append(m)
                _emit(logger, m, step)
                step += 1
    finally:
        for p, rg in zip(params, saved):
            p.requires_grad_(rg)
    radius = conv.weight.detach().flatten(1).norm(dim=1)
    return {
        "steps": step,
        "history": history,
        "final_weight_norm_mean": float(radius.mean()),
        "final_weight_norm_dev": float((radius - 1.0).abs().mean()),
        "final_winner_entropy": history[-1]["softhebb/winner_entropy"] if history else float("nan"),
    }
