"""BackboneSpec v1 e GrowthSpec v1 (Agente 3, desenho em research/spifil_b0b4/agent3/prep_literatura_e_specs.md §3).

* ``BackboneSpec``: modelo Bk autocontido = stem + estágios (local / intermediate / abstract) + cabeça
  com embedding >= 1280. Serializa em JSON canônico; ``arch_hash`` = sha256 do JSON canônico de
  ``input`` + ``stem`` + ``stages`` + ``head`` (NÃO entram custo, proveniência, dataset, nem o campo
  ``init`` das camadas, que é proveniência dos pesos e não arquitetura).
* ``GrowthSpec``: uma aresta do DAG de linhagens (pai -> filho), com as operações atômicas, a herança
  por camada (pesos herdados x novos, mapas de canais, shapes) e o erro de preservação medido em
  ``eval()``, fp32, antes de qualquer treino.

Nada aqui importa torch: só dataclasses + json + hashlib.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_BACKBONE = "backbone_spec/v1"
SCHEMA_GROWTH = "growth_spec/v1"

#: teto superior de cada classe de porte (encoder + cabeça; decoders/projetores à parte). Sem tolerância.
BUDGETS: Dict[str, int] = {"B0": 300_000, "B1": 800_000, "B2": 1_500_000, "B3": 3_000_000, "B4": 5_500_000}
HARD_CAP = 5_500_000            # proibido passar (MISSION.md, Agente 3)
MIN_EMBED_DIM = 1280            # R11 / decisão do usuário 17/11: nunca abaixo
STAGE_NAMES = ("local", "intermediate", "abstract")
FLOPS_DEFINITION = ("flops = 2 * macs; macs = termos de kernel de Conv2d/Linear (sem bias, pooling, "
                    "normalizações e operações elemento a elemento), metodologia de run_search.inference_macs")

CONVS = ("conv", "dwsep", "depthwise", "pointwise", "inception")
NORMS_EXACT = ("none", "batch", "instance", "zca", "dbn", "batch_zca", "batch_dbn")
NORMS = NORMS_EXACT + ("group", "layer")
ACTS = ("relu", "relu6", "leaky_relu", "prelu", "gelu", "silu", "selu")
RESIDUALS = ("none", "identity", "project", "gated")
POOLS = ("none", "max", "avg")
HEAD_POOLINGS = ("gap", "gem", "gem_learn", "meanstd", "moments", "combo")
HEAD_PROJ = ("full", "lowrank")
FLAVORS = ("convblock", "spifil_ops")


class SpecError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def band_of(params_total: int) -> str:
    """Menor classe Bk cujo teto comporta ``params_total``. > 5,5M -> erro (teto duro)."""
    for k, cap in BUDGETS.items():
        if params_total <= cap:
            return k
    raise SpecError(f"params_total={params_total} > teto duro {HARD_CAP}")


# ─────────────────────────────────────────────────────────────────── camadas / estágios / cabeça
@dataclass
class LayerSpec:
    """Um ``ConvBlock`` do repo: conv -> norm -> act (+ skip) -> pool. stride sempre 1.

    residual: none | identity (cin == cout) | project (1x1 sem bias) | gated (y + lambda*x, cin == cout).
    branches: só inception, ``[[nome, kernel, canais], ...]`` (preenchido no ``normalize``).
    init: proveniência dos pesos (spifil0..4 | random | identity | residual_zero | inherited); fora do hash.
    """
    channels: int
    kernel: int = 3
    conv: str = "conv"
    norm: str = "none"
    activation: str = "relu"
    residual: str = "none"
    pool: str = "none"
    pool_size: int = 2
    branches: Optional[List[List[Any]]] = None
    dbn_group_size: int = 16
    init: str = "spifil0"
    # crescimento a partir de checkpoints de produção (growth_chain): ``spifil_ops`` = bloco montado como
    # ``spifil_ops.SpifilOpsBlock`` (bias em toda conv, pool 3x3 s2 pad 1 da biblioteca, skip project com
    # bias), carregável bit a bit dos encoders da busca. ``pool_padding`` (só spifil_ops): pool da semente
    # externa FLIM (MaxPool2d(pool_size, 2, pad)). Valores padrão ficam FORA do hash (hashes antigos iguais).
    flavor: str = "convblock"
    pool_padding: Optional[int] = None
    # growth v3 (dense_concat): a saída do bloco = cat(saída, adaptive_avg_pool(entrada do bloco)) -> o bloco
    # seguinte vê channels + cin canais (DenseNet). False (padrão) fica FORA do hash (hashes antigos iguais).
    dense_out: bool = False
    # growth v3.1 (cov_gate): ramo de covariância GATED num residual: y = x + g * act(norm(conv(x))), g treinável
    # iniciado em 0 (ReZero; "scalar" | "channel"). "none" (padrão) fica FORA do hash.
    cov_gate: str = "none"

    def arch_dict(self) -> dict:
        d = asdict(self)
        d.pop("init")
        if d.get("flavor") == "convblock":
            d.pop("flavor")
        if d.get("pool_padding") is None:
            d.pop("pool_padding")
        if not d.get("dense_out"):
            d.pop("dense_out", None)
        if d.get("cov_gate", "none") == "none":
            d.pop("cov_gate", None)
        return d

    def validate(self, cin: int) -> None:
        if self.conv not in CONVS:
            raise SpecError(f"conv {self.conv!r} fora de {CONVS}")
        if self.norm not in NORMS:
            raise SpecError(f"norm {self.norm!r} fora de {NORMS}")
        if self.activation not in ACTS:
            raise SpecError(f"activation {self.activation!r} fora de {ACTS}")
        if self.residual not in RESIDUALS:
            raise SpecError(f"residual {self.residual!r} fora de {RESIDUALS}")
        if self.pool not in POOLS:
            raise SpecError(f"pool {self.pool!r} fora de {POOLS}")
        if self.flavor not in FLAVORS:
            raise SpecError(f"flavor {self.flavor!r} fora de {FLAVORS}")
        if self.pool_padding is not None and self.flavor != "spifil_ops":
            raise SpecError("pool_padding só vale com flavor=spifil_ops")
        if int(self.channels) < 1 or int(self.kernel) < 1 or int(self.kernel) % 2 == 0:
            raise SpecError(f"channels/kernel inválidos: {self.channels}/{self.kernel}")
        if getattr(self, "cov_gate", "none") != "none":
            if self.cov_gate not in ("scalar", "channel"):
                raise SpecError(f"cov_gate {self.cov_gate!r} (none|scalar|channel)")
            if cin != self.channels or self.residual != "none" or self.pool != "none" or self.dense_out:
                raise SpecError("cov_gate exige cin == cout, residual none, pool none, sem dense_out")
        if self.residual in ("identity", "gated") and cin != self.channels:
            raise SpecError(f"residual {self.residual} exige cin == cout ({cin} != {self.channels})")
        if self.conv == "depthwise" and self.channels % cin:
            raise SpecError("depthwise exige cout múltiplo de cin (sem reparo silencioso por gcd)")
        if self.conv == "inception":
            if not self.branches or sum(int(b[2]) for b in self.branches) != int(self.channels):
                raise SpecError(f"inception: branches {self.branches} não somam {self.channels}")


@dataclass
class StageSpec:
    name: str
    layers: List[LayerSpec]


@dataclass
class HeadSpec:
    """Cabeça: [1x1 C->rank (lowrank)] -> EmbeddingHead (1x1 -> pooling) -> embedding ``embed_dim`` >= 1280."""
    embed_dim: int = 1280
    pooling: str = "gem"
    proj: str = "full"
    rank: Optional[int] = None
    kurtosis: bool = True
    act: str = "none"
    bn: bool = False
    taps: List[str] = field(default_factory=lambda: ["last"])

    def validate(self) -> None:
        if int(self.embed_dim) < MIN_EMBED_DIM:
            raise SpecError(f"embed_dim={self.embed_dim} < {MIN_EMBED_DIM} (proibido)")
        if self.pooling not in HEAD_POOLINGS:
            raise SpecError(f"pooling {self.pooling!r} fora de {HEAD_POOLINGS}")
        if self.proj not in HEAD_PROJ:
            raise SpecError(f"proj {self.proj!r} fora de {HEAD_PROJ}")
        if self.proj == "lowrank" and not (self.rank and int(self.rank) >= 1):
            raise SpecError("proj=lowrank exige rank >= 1")
        if self.proj == "full" and self.rank is not None:
            raise SpecError("rank só vale com proj=lowrank")
        if list(self.taps) != ["last"]:
            raise SpecError("v1 só implementa taps=['last'] (multiescala: pendente, registrado)")

    @property
    def label(self) -> str:
        s = f"{self.pooling}{self.embed_dim}"
        if self.pooling == "moments" and not self.kurtosis:
            s += "_nokurt"
        return s + (f"_lr{self.rank}" if self.proj == "lowrank" else "")


@dataclass
class InputSpec:
    resolution: int = 224
    in_channels: int = 3
    color: str = "lab01"
    norm: str = "none"


@dataclass
class CostSpec:
    params_encoder: int = 0
    params_head: int = 0
    params_total: int = 0                     # = encoder + head (verificado <= orçamento)
    params_extras: Dict[str, int] = field(default_factory=dict)   # decoders/projetores, NUNCA somados
    macs_encoder: int = 0
    macs_head: int = 0
    macs: int = 0
    flops: int = 0                            # = 2 * macs (explícito)
    flops_definition: str = FLOPS_DEFINITION
    resolution: int = 224
    measured: bool = False                    # True = contado no módulo construído


@dataclass
class Provenance:
    lineage_id: Optional[str] = None
    parent: Optional[str] = None
    growth_spec: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    config_sha256: Optional[str] = None
    seeds: Dict[str, int] = field(default_factory=dict)
    fidelity_epochs: int = 0
    wise_alpha: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class BackboneSpec:
    stem: LayerSpec
    stages: List[StageSpec]
    head: HeadSpec = field(default_factory=HeadSpec)
    input: InputSpec = field(default_factory=InputSpec)
    dataset: str = "unspecified"
    split: Optional[int] = None
    budget_class: Optional[str] = None
    extras_not_counted: Dict[str, int] = field(default_factory=dict)
    cost: CostSpec = field(default_factory=CostSpec)
    provenance: Provenance = field(default_factory=Provenance)
    schema: str = SCHEMA_BACKBONE

    # ── estrutura
    def layers(self) -> List[LayerSpec]:
        return [self.stem] + [L for st in self.stages for L in st.layers]

    def block_in_channels(self, j: int) -> int:
        """Canais que ENTRAM no bloco plano ``j`` (growth v3: conta a passagem dense_out dos anteriores)."""
        cin = int(self.input.in_channels)
        for L in self.layers()[:int(j)]:
            cin = int(L.channels) + (cin if getattr(L, "dense_out", False) else 0)
        return cin

    def block_out_channels(self, j: int) -> int:
        """Canais que SAEM do bloco ``j`` (= ``channels`` + cin_j se dense_out)."""
        L = self.layers()[int(j)]
        return int(L.channels) + (self.block_in_channels(j) if getattr(L, "dense_out", False) else 0)

    def layer_names(self) -> List[str]:
        return ["stem"] + [f"{st.name}.{i}" for st in self.stages for i in range(len(st.layers))]

    def locate(self, flat_index: int):
        """(stage_index, position) de uma camada plana (stem -> (-1, 0))."""
        if flat_index == 0:
            return -1, 0
        i = flat_index - 1
        for s, st in enumerate(self.stages):
            if i < len(st.layers):
                return s, i
            i -= len(st.layers)
        raise IndexError(flat_index)

    def normalize(self) -> "BackboneSpec":
        """Preenche ``branches`` do inception (partição explícita, entra no hash)."""
        from ga_search.modehb.ops_norm_act import inception_split as default_branches
        for L in self.layers():
            if L.conv == "inception" and not L.branches:
                L.branches = default_branches(int(L.channels))
            if L.branches is not None:
                L.branches = [[str(n), int(k), int(c)] for n, k, c in L.branches]
        return self

    def validate(self) -> None:
        if self.schema != SCHEMA_BACKBONE:
            raise SpecError(f"schema {self.schema!r} != {SCHEMA_BACKBONE}")
        names = [st.name for st in self.stages]
        if any(n not in STAGE_NAMES for n in names) or names != [n for n in STAGE_NAMES if n in names]:
            raise SpecError(f"estágios {names}: use um subconjunto ordenado de {STAGE_NAMES}")
        if any(len(st.layers) == 0 for st in self.stages):
            raise SpecError("estágio vazio: remova-o em vez de deixá-lo sem camadas")
        cin = int(self.input.in_channels)
        for L in self.layers():
            L.validate(cin)
            cin = int(L.channels) + (cin if getattr(L, "dense_out", False) else 0)
        if self.layers() and getattr(self.layers()[-1], "dense_out", False):
            raise SpecError("dense_out no último bloco não é suportado (a cabeça leria a passagem)")
        self.head.validate()
        if self.budget_class is not None and self.budget_class not in BUDGETS:
            raise SpecError(f"budget_class {self.budget_class!r} fora de {list(BUDGETS)}")

    # ── hash / id
    def arch_dict(self) -> dict:
        return {"schema": self.schema, "input": asdict(self.input), "stem": self.stem.arch_dict(),
                "stages": [{"name": st.name, "layers": [L.arch_dict() for L in st.layers]} for st in self.stages],
                "head": asdict(self.head)}

    def arch_hash(self) -> str:
        return sha256_hex(self.arch_dict())

    @property
    def hash12(self) -> str:
        return self.arch_hash()[:12]

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.budget_class or 'B?'}/{self.hash12}"

    # ── (de)serialização
    def to_dict(self) -> dict:
        d = asdict(self)
        d["arch_sha256"] = self.arch_hash()
        d["id"] = self.id
        return d

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "BackboneSpec":
        d = copy.deepcopy(d)
        d.pop("arch_sha256", None)
        d.pop("id", None)
        spec = cls(stem=LayerSpec(**d.pop("stem")),
                   stages=[StageSpec(name=s["name"], layers=[LayerSpec(**L) for L in s["layers"]])
                           for s in d.pop("stages")],
                   head=HeadSpec(**d.pop("head")), input=InputSpec(**d.pop("input")),
                   cost=CostSpec(**d.pop("cost")), provenance=Provenance(**d.pop("provenance")), **d)
        return spec

    @classmethod
    def from_json(cls, s: str) -> "BackboneSpec":
        d = json.loads(s)
        stored = d.get("arch_sha256")
        spec = cls.from_dict(d)
        if stored is not None and stored != spec.arch_hash():
            raise SpecError(f"arch_sha256 gravado {stored[:12]} != recalculado {spec.hash12}")
        return spec

    def copy(self) -> "BackboneSpec":
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────── GrowthSpec
@dataclass
class OpRecord:
    op: str                                   # widen | deepen_identity | deepen_residual_zero | add_skip_zero | head_widen | ...
    where: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    exact_claimed: bool = True
    condition: str = ""                       # condição da exatidão (ex.: "eval(); ZCA congelado")


@dataclass
class LayerInheritance:
    child_layer: str
    child_index: int
    parent_layer: Optional[str]
    parent_index: Optional[int]
    origin: str                               # inherited | fp_widen | fp_identity | fp_residual_zero | new
    shape_parent: Dict[str, List[int]]
    shape_child: Dict[str, List[int]]
    out_map: Optional[List[int]]              # canal do filho -> canal do pai (-1 = novo)
    n_inherited_elements: int
    n_new_elements: int
    new_init: str = ""
    skip_added: bool = False
    in_map: Optional[List[int]] = None        # canal de entrada do filho -> do pai (-1 = novo; None = vem de camada nova)


@dataclass
class PreservationRecord:
    data: Dict[str, Any] = field(default_factory=dict)       # nunca teste
    exact_claimed: bool = True
    tolerance_rel: float = 1e-5
    map_max_abs: Optional[float] = None       # mapa final do encoder, canais herdados
    map_rel_l2: Optional[float] = None
    map_cos_min: Optional[float] = None
    per_layer: List[Dict[str, Any]] = field(default_factory=list)
    emb_max_abs: Optional[float] = None       # embedding >= 1280 (só com cabeça herdada)
    emb_rel_l2: Optional[float] = None
    emb_cos_min: Optional[float] = None
    verdict: str = "unmeasured"               # exact | approx | not_preserving | violated | unmeasured


@dataclass
class GrowthSpec:
    parent: str
    child: str
    parent_hash: str
    child_hash: str
    operations: List[OpRecord]
    inheritance: Dict[str, Any]               # {"head": inherit_zero_cols|reinit, "per_layer": [...], "head_detail": {...}}
    params: Dict[str, Any]
    macs: Dict[str, int]
    flops: Dict[str, int]
    preservation: PreservationRecord
    protect: Dict[str, Any] = field(default_factory=lambda: {"mode": "full", "scope": "inherited_elements"})
    twin: Dict[str, Any] = field(default_factory=lambda: {"reinit_twin_id": None})
    schema: str = SCHEMA_GROWTH

    @property
    def op_label(self) -> str:
        return "+".join(dict.fromkeys(o.op for o in self.operations))

    @property
    def id(self) -> str:
        return f"{self.parent_hash[:12]}->{self.child_hash[:12]}:{self.op_label}"

    def content_hash(self) -> str:
        """Hash das operações + pai + filho (não inclui medições)."""
        return sha256_hex({"schema": self.schema, "parent": self.parent_hash, "child": self.child_hash,
                           "operations": [asdict(o) for o in self.operations]})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        d["content_sha256"] = self.content_hash()
        return d

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "GrowthSpec":
        d = copy.deepcopy(d)
        d.pop("id", None)
        stored = d.pop("content_sha256", None)
        g = cls(operations=[OpRecord(**o) for o in d.pop("operations")],
                preservation=PreservationRecord(**d.pop("preservation")), **d)
        if stored is not None and stored != g.content_hash():
            raise SpecError("content_sha256 do GrowthSpec não confere")
        return g

    @classmethod
    def from_json(cls, s: str) -> "GrowthSpec":
        return cls.from_dict(json.loads(s))


def dataclass_fields(cls) -> List[str]:
    return [f.name for f in dataclasses.fields(cls)]
