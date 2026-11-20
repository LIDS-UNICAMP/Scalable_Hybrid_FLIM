"""Interface base dos inicializadores de kernels (SPiFiL-0..4) — Agente 2, missão B0–B4.

Contrato publicado para os outros agentes (SPiFiL-1/2/3 implementam contra ELE):

    class MeuInit(KernelInitializer):
        name, version = "spifil1", "0.1.0"
        def fit(self, prefix, train_data, block_spec, context):
            ...                                   # só TREINO (train_data), prefixo congelado
            return weights, bias, provenance      # WeightDict, WeightDict, dict

* ``prefix``: ``nn.Sequential`` dos blocos já treinados (eval, sem gradiente) ou ``None`` (camada 0).
  ``prefix.out_channels`` = canais de entrada do bloco novo.
* ``train_data``: :class:`TrainData` (doadores SPiFiL + ``loader_factory`` de imagens/rótulos de TREINO).
* ``block_spec``: **ops spec** de ``spifil_ops.to_ops_spec`` (chaves ``conv, kernel, in_channels, channels,
  requested_channels, dw_multiplier, activation, norm, pool, pool_stride, residual, skip, dropout`` …).
* ``context``: :class:`InitContext` (seed da camada, device, config do método, orçamento, lote de diagnóstico).
* retorno: ``weights``/``bias`` com as chaves ``conv.*``/``skip.*`` do ``state_dict`` de
  ``SpifilOpsBlock(effective_spec)`` (ver :func:`conv_param_keys`); ``provenance`` com pelo menos
  ``effective_spec`` (ops spec com os canais REAIS; ``builder == "spifil_ops"``) e os campos de
  :func:`make_provenance`.

Invariantes verificadas por :func:`validate_init` / testes (desenho §3.4): I1 prefixo intocado,
I2 chaves e shapes exatos, float32 finito, I3 1 ≤ C_out efetivo ≤ pedido (diferença registrada em
``channel_repair``), I4 determinismo em CPU, I5 só treino, I6 label-free invariante a permutação de
rótulos, I7 falha determinística → ``InfeasibleCandidate``.

O runner (``run_ablation.py``) chama :meth:`KernelInitializer.init_block`, que tem a MESMA assinatura de
retorno de ``ga_search.modehb.init.init_block`` (``(block, provenance)``) — por isso os blocos novos
passam pelo AE local, LeJEPA+SIGReg, WiSE e SVM do ``Evaluator`` sem nenhuma mudança na produção.
Nada em ``ga_search/modehb``, ``flim`` ou na biblioteca SPiFiL é modificado por este módulo.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

import torch
from torch import nn

__all__ = [
    "LabelPolicy", "WeightDict", "TrainData", "InitBudget", "ParentInfo", "InitContext",
    "KernelInitializer", "InfeasibleCandidate", "conv_param_keys", "validate_init", "assemble_block",
    "split_state", "make_provenance", "config_sha256", "check_train_only", "check_prefix_unchanged",
    "prefix_fingerprint", "register", "get_initializer", "available", "INTERFACE_VERSION",
]

INTERFACE_VERSION = "1.0.0"

LabelPolicy = Literal["supervised", "label_free"]
WeightDict = dict  # dict[str, Tensor]; chaves = nomes do state_dict de SpifilOpsBlock(effective_spec)


from flim.progressive import InfeasibleCandidate  # noqa: E402  (o MODEHB trata como candidato inviável)


# ─── tipos ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainData:
    """SOMENTE TREINO. Nunca contém validação nem teste (I5).

    * ``donors``: doadores SPiFiL (``spifil.data.Sample`` com rótulo + máscara) = ``run_search.spifil_subset``.
    * ``loader_factory(seed)``: iterável de ``(x [B,3,S,S] float32 LAB em [0,1], y [B] int64)`` do split de
      TREINO, em ordem semeada (o runner usa o subconjunto estratificado de treino do SVM-fitness).
      Com ``label_policy == "label_free"`` o método NÃO pode ler ``y``.
    * ``split_fingerprint``: sha256 do split (``progressive_data`` identity) + subconjunto usado.
    """
    donors: Sequence[Any] = ()
    loader_factory: Optional[Callable[[int], Iterable[tuple]]] = None
    split_key: str = "train"
    split_fingerprint: str = ""
    n_classes: int = 0
    image_size: int = 224
    n_images: int = 0                      # nº de imagens que loader_factory percorre (uma passada)
    label_policy: LabelPolicy = "supervised"
    mask_policy: Literal["mask", "no_mask"] = "mask"   # máscara = anotação (conta como uso de rótulo)


@dataclass(frozen=True)
class InitBudget:
    max_images: int = 256          # imagens de treino lidas pelo método
    max_patches: int = 100_000     # N máximo de patches
    max_seconds: float = 600.0
    max_gpu_gb: float = 8.0        # acima disso: dividir em blocos (bit a bit igual) ou InfeasibleCandidate


@dataclass(frozen=True)
class ParentInfo:                  # só SPiFiL-4 / crescimento (Agente 3)
    parent_block: nn.Module
    keep: Sequence[int]
    in_map: Sequence[int]
    n_new: int
    inherited_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class InitContext:
    seed: int                                  # seed da camada (= genome.seed + layer_idx, como hoje)
    device: torch.device = torch.device("cpu")
    layer_idx: int = 0
    prefix_id: str = "input"
    budget: InitBudget = field(default_factory=InitBudget)
    config: Mapping[str, Any] = field(default_factory=dict)   # hiperparâmetros do MÉTODO (hash na proveniência)
    spifil_config: Mapping[str, Any] = field(default_factory=dict)  # cfg["spifil"] do run (image_size, n_superpixels…)
    parent: Optional[ParentInfo] = None
    genome: Optional[dict] = None              # só o adaptador SPiFiL-0 usa (equivalência com init.init_block)
    diag_batch: Optional[torch.Tensor] = None  # lote FIXO de TREINO para diagnóstico (nunca validação)
    logger: Any = None
    amp_dtype: Optional[torch.dtype] = None    # só no forward do prefixo; estatísticas sempre em fp32


# ─── utilidades de shape / montagem ───────────────────────────────────────────

def _ops_block(spec: dict):
    from ga_search.modehb.spifil_ops import SpifilOpsBlock
    return SpifilOpsBlock(spec, int(spec["in_channels"]))


def conv_param_keys(spec: dict) -> dict[str, tuple]:
    """``{nome: shape}`` dos parâmetros que o inicializador deve devolver (conv.* e skip.*)."""
    ref = _ops_block(spec).state_dict()
    return {k: tuple(v.shape) for k, v in ref.items() if k.startswith(("conv.", "skip."))}


def validate_init(spec: dict, weights: WeightDict, bias: WeightDict) -> None:
    """I2: chaves exatamente iguais às de ``SpifilOpsBlock(spec)`` (conv/skip), shapes exatos, fp32 finito."""
    ref = conv_param_keys(spec)
    got = set(weights) | set(bias)
    if set(weights) & set(bias):
        raise ValueError(f"chave em weights E bias: {sorted(set(weights) & set(bias))}")
    if got != set(ref):
        raise ValueError(f"chaves diferentes de SpifilOpsBlock: faltam {sorted(set(ref) - got)}, "
                         f"sobram {sorted(got - set(ref))}")
    for k, v in {**weights, **bias}.items():
        if tuple(v.shape) != ref[k]:
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {ref[k]}")
        if v.dtype != torch.float32:
            raise ValueError(f"{k}: dtype {v.dtype} (esperado float32)")
        if not bool(torch.isfinite(v).all()):
            raise ValueError(f"{k}: valores não finitos")
    c_out = int(spec["channels"])
    req = int(spec.get("requested_channels", c_out))
    if spec.get("conv") != "depthwise" and not 1 <= c_out <= max(req, c_out):   # I3
        raise ValueError(f"C_out efetivo {c_out} fora de [1, {req}]")


def assemble_block(spec: dict, weights: WeightDict, bias: WeightDict, device=None) -> nn.Module:
    """``SpifilOpsBlock(spec)`` com os pesos dados (norm/act/pool do spec; norm pós-hoc como hoje)."""
    validate_init(spec, weights, bias)
    blk = _ops_block(spec)
    with torch.no_grad():
        sd = blk.state_dict()
        for k, v in {**weights, **bias}.items():
            sd[k].copy_(v)
    for p in blk.parameters():
        p.requires_grad_(True)
    blk.out_channels = int(spec["channels"])
    return blk if device is None else blk.to(device)


def split_state(block: nn.Module) -> tuple[WeightDict, WeightDict]:
    """(weights, bias) conv/skip de um bloco já montado (cópias fp32 em CPU)."""
    sd = block.state_dict()
    keep = [k for k in sd if k.startswith(("conv.", "skip."))]
    w = {k: sd[k].detach().float().cpu().clone() for k in keep if not k.endswith("bias")}
    b = {k: sd[k].detach().float().cpu().clone() for k in keep if k.endswith("bias")}
    return w, b


# ─── proveniência / contratos ────────────────────────────────────────────────

def config_sha256(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def prefix_fingerprint(prefix: Optional[nn.Module]) -> str:
    from flim.progressive import _fingerprint
    return _fingerprint(nn.Identity() if prefix is None else prefix)


def check_prefix_unchanged(prefix, before: str) -> None:
    """I1."""
    if prefix_fingerprint(prefix) != before:
        raise InfeasibleCandidate("o inicializador modificou o prefixo treinado")


def check_train_only(train_data: TrainData) -> None:
    """I5: recusa qualquer split que não seja treino."""
    if train_data.split_key != "train":
        raise ValueError(f"KernelInitializer só aceita split de treino, recebeu {train_data.split_key!r}")


def _env() -> dict:
    import numpy as np
    out = {"torch": torch.__version__, "numpy": np.__version__}
    try:
        from importlib.metadata import version
        out["spifil"] = version("spifil")
    except Exception:
        out["spifil"] = "editable/unpackaged"
    return out


def make_provenance(method: str, version: str, *, context: InitContext, train_data: TrainData,
                    requested_spec: dict, effective_spec: dict, labels_used: bool, label_source: str,
                    mask_used: bool, n_images: int, n_patches: int | None = None,
                    patch_dim: int | None = None, approximations: Sequence[str] = (),
                    fit_seconds: float | None = None, extra: Mapping[str, Any] | None = None) -> dict:
    """Schema mínimo de proveniência (desenho §3.5), serializável em JSON."""
    req, got = int(requested_spec.get("requested_channels", requested_spec["channels"])), int(effective_spec["channels"])
    prov = {
        "origin": "abstract_init", "method": method, "version": version,
        "interface_version": INTERFACE_VERSION,
        "config": dict(context.config), "config_sha256": config_sha256(dict(context.config)),
        "approximations": list(approximations),
        "seed": int(context.seed), "layer_idx": int(context.layer_idx), "prefix_id": context.prefix_id,
        "requested_spec": dict(requested_spec), "effective_spec": dict(effective_spec),
        "requested_channels": req, "actual_channels": got,
        "channel_repair": None if req == got else "method_actual_width",
        "labels_used": bool(labels_used), "label_source": label_source, "mask_used": bool(mask_used),
        "data": {"split_key": train_data.split_key, "split_fingerprint": train_data.split_fingerprint,
                 "n_images": int(n_images), "n_patches": n_patches, "patch_dim": patch_dim,
                 "label_policy": train_data.label_policy, "mask_policy": train_data.mask_policy},
        "timing": {"fit_seconds": fit_seconds},
        "env": _env(),
    }
    if extra:
        prov.update(dict(extra))
    return prov


# ─── a interface ─────────────────────────────────────────────────────────────

class KernelInitializer:
    """Base dos inicializadores. Subclasses implementam :meth:`fit`; :meth:`init_block` é o que o runner chama."""

    name: str = "base"
    version: str = "0.0.0"
    label_policy: LabelPolicy = "supervised"

    def __init__(self, **config):
        self.config = dict(config)

    # -- a implementar
    def fit(self, prefix: Optional[nn.Module], train_data: TrainData, block_spec: dict,
            context: InitContext) -> tuple[WeightDict, WeightDict, dict]:
        raise NotImplementedError

    # -- usado pelo runner (mesmo retorno de ga_search.modehb.init.init_block)
    def layer_spec(self, genome: dict, layer_idx: int, in_channels: int) -> dict:
        """Camada do genoma → ops spec pedido (reparo do spifil_ops: mesmo vocabulário do gêmeo aleatório)."""
        from ga_search.modehb.spifil_ops import ops_repair, to_ops_spec
        layer = ops_repair({k: v for k, v in genome["layers"][layer_idx].items() if k != "active"})
        repairs = layer.pop("spifil_repairs", [])
        spec = to_ops_spec(layer, int(in_channels))
        spec["spifil_repairs"] = repairs
        return spec

    def init_block(self, genome: dict, layer_idx: int, prefix: Optional[nn.Module],
                   train_data: TrainData, context: InitContext) -> tuple[nn.Module, dict]:
        check_train_only(train_data)
        cin = 3 if prefix is None or isinstance(prefix, nn.Identity) or not len(list(prefix.children())) \
            else int(getattr(prefix, "out_channels", 0) or _conv_out(prefix))
        spec = self.layer_spec(genome, layer_idx, cin)
        repairs = spec.pop("spifil_repairs", [])
        before = prefix_fingerprint(prefix)
        if prefix is not None:
            was = [p.requires_grad for p in prefix.parameters()]
            prefix.eval()
            for p in prefix.parameters():
                p.requires_grad_(False)
        t0 = time.perf_counter()
        try:
            weights, bias, prov = self.fit(prefix, train_data, spec, context)
        finally:
            if prefix is not None:
                for p, r in zip(prefix.parameters(), was):
                    p.requires_grad_(r)
        check_prefix_unchanged(prefix, before)
        eff = dict(prov["effective_spec"])
        eff.setdefault("builder", "spifil_ops")
        eff.setdefault("norm_in_block", True)
        block = assemble_block(eff, weights, bias, device=context.device)
        prov = dict(prov)
        prov.setdefault("timing", {})
        if prov["timing"].get("fit_seconds") is None:
            prov["timing"]["fit_seconds"] = time.perf_counter() - t0
        prov.update(effective_spec=eff, layer_idx=int(layer_idx), init=self.name,
                    spifil_repairs=repairs, spifil_repaired=bool(repairs), builder="spifil_ops",
                    prefix_state_sha256=before, in_channels=cin)
        block.initialization_provenance = prov
        return block, prov


@torch.no_grad()
def _conv_out(prefix: nn.Module) -> int:
    convs = [m for m in prefix.modules() if isinstance(m, nn.Conv2d)]
    return int(convs[-1].out_channels) if convs else 3


# ─── registro ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., KernelInitializer]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def get_initializer(name: str, **config) -> KernelInitializer:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"inicializador {name!r} desconhecido; registrados: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**config)


def _load_builtin():
    import importlib
    for mod in ("spifil0", "spifil1", "spifil2", "spifil3"):
        try:
            importlib.import_module(f"ga_search.abstract_init.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name != f"ga_search.abstract_init.{mod}":
                raise
