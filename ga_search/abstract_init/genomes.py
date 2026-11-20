"""Genomas FIXOS das ablações do Agente 2 (fonte: search_state.json, seleção só por VALIDAÇÃO).

* ``rs_b1``      remote_sensing, melhor ≤800k por κ val (A1a_best_by_budget.csv: B1 = B0 = hybrid c00009 f5,
                 133k, 2 blocos 70-280, κ val 0,591 na fitness ATUAL cov_head+scaler — não na régua ORIG).
* ``rs_b1_band`` remote_sensing, melhor DENTRO da faixa 300k–800k na Jaci (LEMONADE c00054, 736k, κ val 0,569).
                 Opcional (o c00042 do PNAS-CS, 0,577, só existe no athena).
* ``flowers_b1`` flowers, B1 = hybrid c00002 (684k, 4 blocos, κ val 0,748 em f200).
* ``seed_small`` a "semente SPiFiL pequeno" (topologia FLIM L0 de ``artifacts/external_ssl_pequeno``; RS =
                 RegNet 40-80-152-152-152-312 k3). Aqui é só a TOPOLOGIA: os pesos vêm da init do braço
                 (SPiFiL-0/aleatório/…); nada do checkpoint externo é carregado. O pool do FLIM (3×3 s2 sem
                 padding) vira o pool da biblioteca (3×3 s2 pad 1) — registrado em ``topology_note``.

Os genomas são devolvidos com ``seed`` = 42 (o braço troca o seed e ``ae_schedule``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_INACTIVE = {"activation": "relu", "active": False, "channels": 16, "conv": "conv", "kernel": 3,
             "norm": "none", "pool": "none", "residual": False, "softhebb": False}

GENOMES = {
    "rs_b1": {
        "dataset": "remote_sensing",
        "source": "artifacts/modehb_hybrid/remote_sensing/state/seed42 c00009 (f5, kval_cov 0.5914)",
        "genome": {"ae_schedule": "joint", "ae_target": "pixel", "depth": 2, "embed_dim": 1280, "init": "spifil",
                   "layers": [{"activation": "relu", "active": True, "channels": 73, "conv": "conv", "kernel": 3,
                               "norm": "batch_dbn", "pool": "max", "residual": False, "softhebb": False},
                              {"activation": "gelu", "active": True, "channels": 287, "conv": "dwsep", "kernel": 3,
                               "norm": "none", "pool": "max", "residual": True, "softhebb": False}],
                   "pooling": "moments", "projector": False, "protect": "freeze", "reg": "sigreg", "seed": 42,
                   "spifil_images_per_class": 4, "ssl": "lejepa"}},
    "rs_b1_band": {
        "dataset": "remote_sensing",
        "source": "artifacts/modehb_lemonade/remote_sensing/state/seed42 c00054 (f5, kval_cov 0.5693)",
        "genome": {"ae_schedule": "joint", "ae_target": "masked", "depth": 6, "embed_dim": 2560, "init": "spifil",
                   "layers": [{"activation": "gelu", "active": True, "channels": 42, "conv": "conv", "kernel": 5, "norm": "batch", "pool": "max", "residual": False, "softhebb": True},
                              {"activation": "relu", "active": True, "channels": 53, "conv": "inception", "kernel": 7, "norm": "zca", "pool": "avg", "residual": True, "softhebb": False},
                              {"activation": "selu", "active": True, "channels": 53, "conv": "conv", "kernel": 7, "norm": "batch", "pool": "max", "residual": True, "softhebb": True},
                              {"activation": "relu", "active": True, "channels": 139, "conv": "inception", "kernel": 7, "norm": "batch_zca", "pool": "max", "residual": True, "softhebb": False},
                              {"activation": "gelu", "active": True, "channels": 188, "conv": "inception", "kernel": 3, "norm": "dbn", "pool": "max", "residual": True, "softhebb": False},
                              {"activation": "relu", "active": True, "channels": 188, "conv": "inception", "kernel": 7, "norm": "batch_zca", "pool": "none", "residual": False, "softhebb": True}],
                   "pooling": "moments", "projector": False, "protect": "freeze", "reg": "sigreg", "seed": 42,
                   "spifil_images_per_class": 16, "ssl": "lejepa"}},
    "flowers_b1": {
        "dataset": "flowers",
        "source": "artifacts/modehb_hybrid/flowers/state/seed42 c00002 (f200, kval_cov 0.7476)",
        "genome": {"ae_schedule": "joint", "ae_target": "pixel", "depth": 4, "embed_dim": 1280, "init": "spifil",
                   "layers": [{"activation": "relu", "active": True, "channels": 15, "conv": "conv", "kernel": 5, "norm": "batch_zca", "pool": "max", "residual": False, "softhebb": True},
                              {"activation": "gelu", "active": True, "channels": 54, "conv": "conv", "kernel": 3, "norm": "zca", "pool": "avg", "residual": False, "softhebb": True},
                              {"activation": "selu", "active": True, "channels": 136, "conv": "conv", "kernel": 3, "norm": "dbn", "pool": "max", "residual": True, "softhebb": False},
                              {"activation": "relu", "active": True, "channels": 1024, "conv": "dwsep", "kernel": 5, "norm": "zca", "pool": "none", "residual": True, "softhebb": True}],
                   "pooling": "moments", "projector": True, "protect": "freeze", "reg": "sigreg", "seed": 42,
                   "spifil_images_per_class": 32, "ssl": "lejepa"}},
}

# topologia L0 da semente externa (fallback quando o architecture.json não existe na máquina)
SEED_SMALL_FALLBACK = {
    "remote_sensing": [(3, 40, "max"), (3, 80, "max"), (3, 152, "max"), (3, 152, "max"), (3, 152, "none"), (3, 312, "none")],
    "flowers": [(3, 104, "max"), (3, 104, "max"), (3, 104, "max"), (5, 104, "max"), (5, 104, "none"), (5, 104, "none")],
}
SEED_SMALL_ARCH = "artifacts/external_ssl_pequeno/{dataset}_split1_pct100/L0/architecture.json"


def _seed_small_layers(dataset: str) -> tuple[list, str]:
    p = Path(SEED_SMALL_ARCH.format(dataset=dataset))
    if p.exists():
        from ga_search.modehb.external_seed import _flim_layers
        geo = [(g["kernel"], None, g["pool"]) for g in _flim_layers(json.loads(p.read_text()))]
        arch = json.loads(p.read_text())
        widths = [int(arch[f"layer{i}"]["conv"]["noutput_channels"]) for i in range(1, int(arch["nlayers"]) + 1)]
        return [(k, w, pool) for (k, _, pool), w in zip(geo, widths)], str(p)
    if dataset not in SEED_SMALL_FALLBACK:
        raise FileNotFoundError(p)
    return SEED_SMALL_FALLBACK[dataset], "fallback:SEED_SMALL_FALLBACK"


def seed_small_genome(dataset: str, n_classes: int, choices=(1, 2, 4, 8, 16, 32), pool_sizes=None) -> dict:
    from ga_search.modehb.external_seed import DEFAULT_GENES
    from ga_search.modehb.strata import spifil_feasible
    layers, src = _seed_small_layers(dataset)
    act = [{"activation": "relu", "active": True, "channels": int(w), "conv": "conv", "kernel": int(k),
            "norm": "none", "pool": pool, "residual": False, "softhebb": False} for k, w, pool in layers]
    g = {"init": "spifil", "seed": 42, "depth": len(act), "layers": act,
         **{k: DEFAULT_GENES[k] for k in ("ssl", "reg", "embed_dim", "pooling", "ae_target", "projector",
                                          "ae_schedule")},
         "protect": DEFAULT_GENES["protect"], "spifil_images_per_class": int(choices[-1])}
    for k in sorted(int(c) for c in choices):          # menor ipc que fecha o orçamento N > 1,05·D
        if spifil_feasible(g, k, n_classes, 90.0, pool_sizes):
            g["spifil_images_per_class"] = k
            break
    return {"dataset": dataset, "source": src,
            "topology_note": "FLIM MaxPool2d(3,2,pad0) -> pool da biblioteca (3x3,s2,pad1); pesos NÃO importados",
            "genome": g}


def get_genome(name: str, dataset: str, n_classes: int = 0, choices=(1, 2, 4, 8, 16, 32), pool_sizes=None) -> dict:
    """``{"dataset", "source", "genome"}`` (cópia). ``name`` pode ser um caminho .json com o mesmo formato."""
    if name == "seed_small":
        return seed_small_genome(dataset, n_classes, choices, pool_sizes)
    if name.endswith(".json"):
        info = json.loads(Path(name).read_text())
    elif name in GENOMES:
        info = copy.deepcopy(GENOMES[name])
    else:
        raise KeyError(f"genoma {name!r} desconhecido: {sorted(GENOMES) + ['seed_small']}")
    if info.get("dataset") and info["dataset"] != dataset:
        raise ValueError(f"genoma {name} é de {info['dataset']}, não de {dataset}")
    full = json.loads((Path(__file__).with_name("genomes_b1.json")).read_text()).get(name)
    if full is not None:               # genoma EXATO do search_state (8 camadas, inativas incluídas)
        assert [L for L in full["layers"][:full["depth"]]] == info["genome"]["layers"][:full["depth"]], name
        info["genome"] = full
    g = info["genome"]
    g["layers"] = [dict(L) for L in g["layers"]] + [dict(_INACTIVE) for _ in range(max(0, 8 - len(g["layers"])))]
    return info
