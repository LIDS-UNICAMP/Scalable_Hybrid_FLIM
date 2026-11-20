# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""encdeep.py — L4/L5 do `spf_grade` sem rotulo e sem backprop (prompt_encdeep §5-§8, §11, §12).

Tudo opt-in em `amostra.encdeep` (chaves em `experiments/ray/schema.py`, ENCDEEP_*): sem a chave,
`spf_grade` so chama `ends`, que devolve [""]. Braco = `<metodo>~<geometria>~<mecanismo>[~nw]`:
g0 = pools da receita, g1 = sem pool apos L4, g2 = g1 + dilatacao 2 na L5; `nw` tira o whitening das
camadas encdeep (H0/H2). Os controles levam so a geometria.

O mecanismo recebe a direcao fps `u` (linhas de norma 1) no espaco v = N0(ZCA(p)) da camada, com A, m
congelados, e devolve (W, b_v) nesse espaco: a dobra de `chain` faz K = (W/σ)·G e b = b_v − K·m_total.
Os patches v saem do mapa cru com o mesmo unfold da conv (kernel, dilatacao, padding zero, stride 1):
na borda, p = 0 da A(0 − m), o mesmo que a conv dobrada. So `images["train"][0]` entra: nem rotulo nem
mascara. Vistas (clapp, sfa, fu): jitter fotometrico leve de `TrainAug`, geometria intacta, semente fixa;
negativo do CLAPP = a vista b da imagem vizinha do lote (`roll`), sem olhar classe.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from core.data.parasite_dataset import ParasiteDataset
from core.data.train_aug import TrainAug
from eval.svm import DEVICE
from experiments.ray.schema import ENCDEEP_MECHS
from flim.local_plasticity import learn_layer_hebb
from flim.local_predictive import learn_layer_clapp
from flim.slow_features import fit_fu, fit_sfa

VIEWS = {"l_jitter": 0.05, "ab_jitter": 0.016}  # o `strength` leve de core/data/train_aug.py
FIELDS = ["geometry", "kernels", "dilations", "pools", "maps", "strides", "rfs", "grid", "trunk_hash", "mech",
          "mech_steps", "drift", "drift_cos", "rel_update", "dead", "entropy", "hinge_active", "score_pos",
          "score_neg", "delta_min", "delta_med", "delta_max", "sfa_rank", "deploy_params", "aux_params", "mech_s",
          "class_recall"]
ERRORS = ["dataset", "split", "percentage", "arm", "plan", "kernel", "seed", "layer", "readout", "id", "y", "pred"]


def ends(spec: dict, control: bool) -> list:
    """Sufixos encdeep de um braco: geometria x mecanismo; controle so geometria."""
    deep = spec.get("encdeep")
    if not deep:
        return [""]
    geos = deep.get("geometrias", ["g0"])
    return [f"~{g}" for g in geos] if control else [f"~{g}~{m}" for g in geos for m in deep["mecanismos"]]


def geometry(spec: dict, tags: list, pool: list) -> tuple:
    """Kernel, dilatacao e pool por camada; g1/g2 mexem so em L4 (pool) e L5 (dilatacao)."""
    deep = spec["encdeep"]
    ks, ds, ps = list(deep["kernels"]), list(deep.get("dilacao", [1] * len(deep["kernels"]))), list(pool)
    if {"g1", "g2"} & set(tags):
        ps[3] = 1
    if "g2" in tags:
        ds[4] = 2
    return ks, ds, ps


def geo_log(tags: list, size: int, ks: list, ds: list, ps: list) -> dict:
    """H=W, stride acumulado e campo receptivo apos cada camada (conv `same`, MaxPool 3 stride p sem padding)."""
    side, rf, jump, maps, strides, rfs = size, 1, 1, [], [], []
    for k, d, p in zip(ks, ds, ps):
        rf += (k - 1) * d * jump
        if p > 1:
            side, rf, jump = (side - 3) // p + 1, rf + 2 * jump, jump * p
        maps, strides, rfs = maps + [side], strides + [jump], rfs + [rf]
    join = lambda v: "-".join(map(str, v))
    return {"geometry": next((t for t in tags if t in ("g0", "g1", "g2")), "g0"), "kernels": join(ks),
            "dilations": join(ds), "pools": join(ps), "maps": join(maps), "strides": join(strides), "rfs": join(rfs)}


@contextmanager
def no_autograd():
    """no_grad e backward/grad proibidos enquanto o mecanismo roda."""
    saved = torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad

    def refuse(*_, **__):
        raise RuntimeError("encdeep: backward/grad proibido no encoder")
    torch.Tensor.backward = torch.autograd.backward = torch.autograd.grad = refuse
    try:
        with torch.no_grad():
            yield
    finally:
        torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad = saved


@torch.no_grad()
def layer_rows(encoder, x: torch.Tensor, fold, mean: torch.Tensor, stdev: torch.Tensor, k: int, dil: int) -> tuple:
    """(B, H*W, d) em v-space, uma linha por posicao da conv, e a grade (H, W)."""
    from eval.spf_grade import whitened  # spf_grade importa este modulo
    raw = encoder(x)
    span = 2 * (k // 2) + 1
    rows = F.unfold(raw, span, dilation=dil, padding=dil * (k // 2)).transpose(1, 2).flatten(0, 1)
    if fold:
        rows = whitened(rows, *fold, span**2).to(DEVICE)
    v = (rows - mean.to(rows.device)) / stdev.to(rows.device)
    return v.view(len(x), -1, v.shape[1]), tuple(raw.shape[-2:])


def deep_mechanism(spec: dict, n: int, tags: list):
    """None fora de `encdeep`, fora de `camadas` ou sem tag de mecanismo: o caminho de sempre."""
    deep = spec.get("encdeep")
    mech = next((t for t in tags if t in ENCDEEP_MECHS), None)
    if not deep or mech is None or n + 1 not in deep.get("camadas", [4, 5]):
        return None
    return lambda **kw: learn(mech, deep, **kw)


def batches(train: torch.Tensor, to_float, cfg: dict, seed: int):
    """`steps` lotes de `batch` imagens, sorteados entre `images` do treino (o sorteio de `learn_softhebb`)."""
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(train), generator=gen)[:cfg["images"]]
    for _ in range(cfg["steps"]):
        yield to_float(train[pick[torch.randint(len(pick), (cfg["batch"],), generator=gen)]].to(DEVICE))


def learn(mech: str, deep: dict, *, u, encoder, fold, mean, stdev, k, dil, train, to_float, seed) -> tuple:
    """(W, b_v, colunas) em v-space, CPU float32."""
    start, u0 = time.perf_counter(), u.to(DEVICE)
    zero, cfg = torch.zeros(len(u0), device=DEVICE), deep.get(mech.rstrip("02"), {})
    rows = lambda x: layer_rows(encoder, x, fold, mean, stdev, k, dil)[0]
    grid = layer_rows(encoder, to_float(train[:1].to(DEVICE)), fold, mean, stdev, k, dil)[1]
    aug = TrainAug(p=1, **deep.get("vistas", VIEWS))
    info, aux = {}, 0
    with torch.random.fork_rng(devices=[DEVICE] if DEVICE.type == "cuda" else []):
        torch.manual_seed(seed)
        if mech == "z0":
            W, b = u0, zero
        elif mech == "hebb":
            Q, info = learn_layer_hebb((rows(x).flatten(0, 1) for x in batches(train, to_float, cfg, seed)), u0, zero, cfg)
            W, b = F.normalize(Q, dim=1), zero  # direcoes, como todo braco da grade; o raio so serve ao lr
        elif mech.startswith("clapp"):
            def pairs():
                for x in batches(train, to_float, cfg, seed):
                    a, v = rows(aug.corrupt(x)), rows(aug.corrupt(x))
                    q = torch.cat([torch.ones(len(a)), -torch.ones(len(a))])
                    yield torch.cat([a, a]), torch.cat([v, v.roll(1, 0)]), q
            W, b, info = learn_layer_clapp(pairs(), u0, zero, grid, cfg | {"saida": (2, 2) if mech == "clapp2" else (1, 1)})
            aux = info["predictor_params"]
        elif mech.startswith("sfa"):
            def pre():
                for x in batches(train, to_float, cfg, seed):
                    yield tuple((rows(aug.corrupt(x)).flatten(0, 1) @ u0.T).cpu() for _ in range(2))
            ta, tb = itertools.tee(pre())
            sub = cfg if mech == "sfa" else cfg | {"lentos": cfg["lentos"] + cfg.get("residuo", 0), "residuo": 0}
            W, b, info = fit_sfa((p[0] for p in ta), (p[1] for p in tb), u0.cpu(), zero.cpu(), sub)
            aux = info["R_all"].numel() + (info["P_rec"].numel() if "P_rec" in info else 0)
            slow = info["delta"][:info["n_slow"]]
            info |= {"delta_min": float(slow.min()), "delta_med": float(slow.median()), "delta_max": float(slow.max()),
                     "sfa_rank": info["rank"]}
        else:  # fu
            def views():
                for x in batches(train, to_float, cfg, seed):
                    yield tuple(rows(aug.corrupt(x)).flatten(0, 1).cpu() for _ in range(2))
            ta, tb = itertools.tee(views())
            W, info = fit_fu((p[0] for p in ta), (p[1] for p in tb), u0.cpu(), cfg)
            b = zero
    W, b = W.float().cpu(), b.float().cpu()
    same = W.shape == u0.shape
    cols = {"mech": mech, "mech_steps": info.get("steps", cfg.get("steps", 0) * (mech != "z0")),
            "drift": float((W - u0.cpu()).norm() / u0.norm()) if same else "",
            "drift_cos": float(F.cosine_similarity(W, u0.cpu(), dim=1).mean()) if same else "",
            "aux_params": aux, "mech_s": round(time.perf_counter() - start, 2), "grid": "x".join(map(str, grid))}
    cols |= {key: info[key] for key in ("rel_update", "dead", "entropy", "hinge_active", "score_pos", "score_neg",
                                        "delta_min", "delta_med", "delta_max", "sfa_rank") if info.get(key) is not None}
    return W, b, cols


def trunk_hash(folder: Path, layers: int) -> str:
    """sha256 dos pesos de L1..L`layers` como gravados (conv*-kernels.npy, conv*-bias.txt)."""
    digest = hashlib.sha256()
    for n in range(1, layers + 1):
        for name in (f"conv{n}-kernels.npy", f"conv{n}-bias.txt"):
            digest.update((Path(folder) / name).read_bytes())
    return digest.hexdigest()[:16]


def persist(job: dict, arm: str, kernel: int, seed: int, n: int, out: Path, norm: dict) -> None:
    """`salvar`: L{n} identificavel em work_dir/encdeep/, com a dobra (m, G, μ, σ) ao lado."""
    dest = Path(job["work_dir"]) / "encdeep" / f"{arm}_k{kernel}_s{seed}" / f"L{n + 1}"
    shutil.copytree(out, dest, dirs_exist_ok=True)
    torch.save(norm, dest / "norm.pt")


def val_ids(job: dict) -> list:
    """Nomes da validacao na ordem de `load_images` (mesmo ParasiteDataset, sem shuffle)."""
    data = ParasiteDataset(set_name="validation", split=job["split"], percentage=job["percentage"],
                           path_dataset=job["dataset_name"])
    return [Path(s).name for s in data.samples]


def save_errors(path: Path, row: dict, ids: list, y, pred, guard) -> None:
    """`erros`: id, rotulo e predicao de cada imagem de validacao, ao lado do CSV."""
    with guard, open(path, "a", newline="") as handle:
        sheet = csv.DictWriter(handle, ERRORS, extrasaction="ignore")
        if handle.tell() == 0:
            sheet.writeheader()
        sheet.writerows(row | {"id": i, "y": int(t), "pred": int(p)} for i, t, p in zip(ids, y, pred))
