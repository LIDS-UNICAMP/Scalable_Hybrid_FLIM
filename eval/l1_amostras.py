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
"""l1_amostras.py — o SPiFiL melhora com mais imagens por classe e mais kernels? Sem backprop.

Um job = (dataset, split), escrito por `experiments/ray/runners/eval.py` (`source: l1_amostras`).
Celula = (k imagens por classe, largura da L1, semente): `grow` oficial camada a camada e a
leitura de `eval/l1_fast.py` em cada profundidade. So treino e validacao: o teste nem sobe.

Bracos:
    spifil    o `grow` de hoje, `images_per_class = k`; a semente sorteia as imagens
    random    `random_layer` oficial: mesma largura real, norma copiada, bias 0
    random_z  so na L1: a dobra do SPiFiL (media e desvio do proprio banco) com direcao
              sorteada. E o controle que so muda a direcao; `random` muda tambem o bias.

Cobertura (so na L1, janelas 5x5 de validacao, que nunca doam filtro): `cover_*` e o cosseno
da janela com o melhor kernel no espaco do z-score, o mesmo que a conv mede; `fire_*` e a
fracao de pares (kernel, imagem) em que o pico passa de FIRE vezes o pico nas doadoras.
`donor` le as imagens doadoras, `same` a mesma classe fora delas, `other` as outras classes.
"""
from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from core.constants import DATASET_NUM_CLASSES
from eval import l1_fast
from eval.l1_fast import COPY_COS, fit_pending, load_images
from eval.svm import DEVICE, _encode_pooled, cli_kwargs
from experiments.ray.runners.growth import _paths, _short
from flim import spifil
from flim.build import build
from flim.kernel_redundancy import effective_rank

l1_fast.SETS = {"val": "validation"}  # decisao so por validacao: o teste nem sobe para a RAM
l1_fast.BATCH_PIXELS = 2**20          # 255 canais a 200 px: o lote de l1_fast passa do teto de 5 GB

MODES = {"gem_pw": "gem", "gem_tta_pw": "gem_tta"}
KERNEL = 5
COVER_STEP = 8       # janelas por imagem: (200 / 8)^2 = 625
COVER_IMAGES = 400   # imagens de validacao na cobertura
FLAT = 1e-3          # janela chapada (letterbox, fundo preto) nao conta como coberta
TIGHT = 0.7          # `cover_c70`: janela com um kernel a menos de ~45 graus
FIRE = 0.5
FIELDS = ["dataset", "split", "percentage", "image_size", "n_superpixels", "arm", "k", "width_req",
          "seed", "layer", "readout", "width", "spifil_filters", "images", "candidates", "distinct",
          "donors", "copies", "copies_frac", "cos_max", "cos_mean", "erank", "cover_all", "cover_c70",
          "cover_donor", "cover_same", "cover_other", "fire_same", "fire_other", "dim", "n_train",
          "n_val", "val_kappa", "val_acc", "val_f1", "fit_status", "grow_s", "extract_s", "fit_s",
          "lock_wait_s", "status", "error"]


class Spy(spifil.GrowOneLayer):
    """`grow` nao devolve o `Learner`: e daqui que saem candidatos, doadoras e banco."""
    learn = None

    def run(self, learn) -> None:
        super().run(learn)
        Spy.learn = learn


spifil.GrowOneLayer = Spy
_prepare_sample, _prepared = spifil.preparation.prepare_sample, {}


def prepare_sample(dataset, index, **kwargs):
    """O superpixel so depende da imagem: sem isso cada celula refaz o DISF de todas elas."""
    key = dataset[index].image_path, kwargs["n_superpixels"]
    if key not in _prepared:
        _prepared[key] = _prepare_sample(dataset, index, **kwargs)
    return _prepared[key]


spifil.preparation.prepare_sample = prepare_sample


def bank_stats(weights: Path, layer: int) -> dict:
    return kernel_stats(np.load(weights / f"conv{layer}-kernels.npy").T.astype(np.float64))


def kernel_stats(kernels: np.ndarray) -> dict:
    """Largura real, copias (a regra de l1_fast), |cosseno| entre kernels e posto efetivo."""
    unit = kernels / np.linalg.norm(kernels, axis=1, keepdims=True).clip(1e-12)
    cos = unit @ unit.T
    pairs = np.abs(cos[np.tril_indices(len(cos), -1)])
    found = int((np.tril(cos, -1) > COPY_COS).any(1).sum())
    return {"width": len(unit), "copies": found, "copies_frac": round(found / len(unit), 4),
            "cos_max": float(pairs.max()), "cos_mean": float(pairs.mean()),
            "erank": effective_rank(torch.from_numpy(unit))}


def pool_stats(learn) -> dict:
    """Candidatos = superpixels x imagens x classes; distintos = sem janelas iguais."""
    feats = learn.state[learn.n_layers - 1].patches.feats.double().cpu().numpy()
    hashed = feats @ np.random.default_rng(0).standard_normal(feats.shape[1])
    chosen = learn.state[learn.n_layers].selected
    return {"images": len(learn.data), "candidates": len(feats),
            "distinct": len(np.unique(hashed.round(6))), "spifil_filters": len(chosen),
            "donors": int(chosen.image_ids.unique().numel())}


def windows(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Janelas nao chapadas, canal primeiro como em `extract_patches`, e a imagem de cada uma."""
    rows = F.unfold(images, KERNEL, stride=COVER_STEP).transpose(1, 2)
    owner = torch.arange(len(images), device=images.device)[:, None].expand(rows.shape[:2]).flatten()
    rows = rows.flatten(0, 1)
    keep = rows.std(1) > FLAT
    return rows[keep], owner[keep]


def reach(w, b, mean, stdev, rows, owner, n_images) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosseno janela x kernel e o pico de cada kernel em cada imagem (K.x + b = u.z(x))."""
    act = rows @ w.T + b
    cos = act / ((rows - mean) / stdev).norm(dim=1, keepdim=True).clamp_min(1e-6)
    peak = act.new_zeros(n_images, act.shape[1]).scatter_reduce(
        0, owner[:, None].expand_as(act), act, "amax", include_self=False)
    return cos, peak


def coverage(w, b, mean, stdev, sample: dict, labels=None, donors=None, donor_labels=None) -> dict:
    """Ver o docstring do modulo. Sem `labels` (banco sem classe) sai so a cobertura geral."""
    rows, owner, classes = sample["rows"], sample["owner"], sample["classes"]
    cos, peak = reach(w, b, mean, stdev, rows, owner, len(classes))
    best = cos.amax(1)
    out = {"cover_all": float(best.mean()), "cover_c70": float((best >= TIGHT).float().mean())}
    if labels is None:
        return out
    drows, downer = windows(donors)
    dcos, dpeak = reach(w, b, mean, stdev, drows, downer, len(donors))
    parts = {name: [] for name in ("cover_donor", "cover_same", "cover_other", "fire_same", "fire_other")}
    for label in labels.unique():
        mine, here, there = labels == label, classes == label, classes != label
        reached, own = cos[:, mine].amax(1), donor_labels == label
        parts["cover_same"].append(reached[here[owner]].mean())
        parts["cover_other"].append(reached[there[owner]].mean())
        parts["cover_donor"].append(dcos[own[downer]][:, mine].amax(1).mean())
        fired = peak[:, mine] >= FIRE * dpeak[own][:, mine].amax(0).clamp_min(1e-6)
        parts["fire_same"].append(fired[here].float().mean())
        parts["fire_other"].append(fired[there].float().mean())
    return out | {name: float(torch.stack(v).mean()) for name, v in parts.items()}


def cover_sample(images: dict, to_float) -> dict:
    """As janelas de validacao, uma vez por job, com rotulo no mesmo dominio do `Sample.label`."""
    x, y = images["validation"]
    step = max(1, len(x) // COVER_IMAGES)
    rows, owner = windows(to_float(x[::step].to(DEVICE)))
    return {"rows": rows, "owner": owner, "classes": torch.as_tensor(y[::step], device=DEVICE) + 1}


def grow_layer(job: dict, n: int, k: int, width: int, seed: int, encoder, ckpt: str, arch: str,
               weights: str, out: Path, **extra) -> None:
    """Uma rodada do `grow` oficial; o que nao cabe no teto da GPU refaz em CPU."""
    spec = job["amostra"]
    point = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    kwargs = dict(encoder=encoder, ckpt=ckpt, arch_json=arch, flim_weights_path=weights,
                  dataset=_short(point), split=job["split"], percentage=job["percentage"], out=out,
                  n_superpixels=job["superpixels"][0], image_size=job["image_size"],
                  no_crop=job["no_crop"], images_per_class=k, seed=seed,
                  out_channels=spec["channels"][n] or width, kernel_size=spec["kernels"][n],
                  pool_stride=spec["pools"][n],
                  **job["grow"], **extra)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            spifil.grow(device=str(DEVICE), **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            spifil.grow(device="cpu", **kwargs)


@torch.no_grad()
def features(encoder, x: torch.Tensor, to_float, size: int, readouts: list) -> dict:
    """So as leituras pedidas: a TTA custa 8 forwards, e em 255 canais isso domina a celula."""
    out = {r: [] for r in readouts}
    step = max(1, l1_fast.BATCH_PIXELS // size**2)
    for i in range(0, len(x), step):
        batch = to_float(x[i:i + step].to(DEVICE))
        for r in readouts:
            out[r].append(_encode_pooled(encoder, batch, MODES[r]))
    return {r: torch.cat(v).numpy() for r, v in out.items()}


def read(job: dict, images: dict, to_float, encoder, row: dict, tag: str) -> list:
    """Features de treino e validacao; devolve os SVMs a fazer, um por leitura."""
    start, size = time.perf_counter(), job["image_size"]
    train = features(encoder, images["train"][0], to_float, size, job["readouts"])
    val = features(encoder, images["validation"][0], to_float, size, job["readouts"])
    row = row | {"extract_s": round(time.perf_counter() - start, 2)}
    return [(row | {"readout": readout},
             (train[readout], images["train"][1], {"val": val[readout]},
              {"val": images["validation"][1]}, DATASET_NUM_CLASSES[job["dataset_name"]], tag))
            for readout in job["readouts"]]


@torch.no_grad()
def twin(job: dict, images: dict, to_float, encoder, bank, seed: int, row: dict, tag: str) -> tuple:
    """`random_z`: troca a direcao dos filtros da L1 e mantem a dobra do proprio banco."""
    mean, stdev = bank.mean.to(DEVICE), bank.stdev.to(DEVICE)
    noise = torch.randn(bank.weight.flatten(1).shape, generator=torch.Generator().manual_seed(seed))
    w = F.normalize(noise.to(DEVICE), dim=1) / stdev
    b = -(w * mean).sum(1)
    conv = encoder.conv1[0]
    conv.weight.copy_(w.view_as(conv.weight))
    conv.bias.copy_(b)
    return read(job, images, to_float, encoder, row | {"arm": "random_z"}, tag), (w, b, mean, stdev)


def cell(job: dict, images: dict, to_float, sample: dict, arm: str, k: int, width: int, seed: int,
         write) -> list:
    """Banco e features de uma celula; devolve os SVMs a fazer. Falha vira linha `error`."""
    spec = job["amostra"]
    point = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    base = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"],
            "image_size": job["image_size"], "n_superpixels": job["superpixels"][0], "arm": arm,
            "k": k, "width_req": width, "seed": seed, "n_train": len(images["train"][1]),
            "n_val": len(images["validation"][1]), "status": "ok"}
    tag = f"amostra {job['stem']} {arm} k{k} w{width} s{seed}"
    extra = {"random_layer": True} if arm == "random" else {}
    arch, weights = _paths(point)
    encoder, ckpt, pending = None, "", []
    with tempfile.TemporaryDirectory(dir=job["work_dir"]) as tmp:
        try:
            for n in range(spec["depth"]):
                out, start = Path(tmp) / f"L{n + 1}", time.perf_counter()
                grow_layer(job, n, k, width, seed, encoder, ckpt, arch, weights, out, **extra)
                grow_s = round(time.perf_counter() - start, 2)
                learn = Spy.learn
                row = base | pool_stats(learn) | bank_stats(out, n + 1) | {"layer": n + 1, "grow_s": grow_s}
                arch, weights, ckpt = str(out / "architecture.json"), str(out), str(out)
                encoder = build(arch, "flim", weights).to(DEVICE).eval()
                pair = []
                if arm == "spifil" and spec["depth"] == 1:
                    bank = learn.state[1].bank
                    w, b = bank.weight.flatten(1).to(DEVICE), bank.bias.to(DEVICE)
                    mean, stdev = bank.mean.to(DEVICE), bank.stdev.to(DEVICE)
                    labels = bank.labels.to(DEVICE)
                    donors = torch.stack(learn.state[0].features).to(DEVICE)
                    donor_labels = torch.tensor([learn.data[i].label for i in range(len(learn.data))],
                                                device=DEVICE)
                    row |= coverage(w, b, mean, stdev, sample, labels, donors, donor_labels)
                    twin_enc = build(arch, "flim", weights).to(DEVICE).eval()
                    pair, folded = twin(job, images, to_float, twin_enc, bank, seed, row, tag)
                    cover = coverage(*folded, sample, labels, donors, donor_labels)
                    cover |= kernel_stats(folded[0].double().cpu().numpy())
                    pair = [(r | cover, args) for r, args in pair]
                pending += read(job, images, to_float, encoder, row, tag) + pair
                Spy.learn = None
        except Exception as exc:  # noqa: BLE001
            write(base | {"layer": 0, "readout": "", "status": "error", "error": repr(exc)[:500]})
    return pending


def cells(job: dict) -> list:
    """Aleatorio primeiro, nas larguras reais que o SPiFiL vai dar; depois k crescente."""
    spec, classes = job["amostra"], DATASET_NUM_CLASSES[job["dataset_name"]]
    todo = []
    for width in spec["widths"]:
        # abaixo de `full` imagens por classe faltam candidatos e a largura real encolhe
        full = -(-(width // classes) // job["superpixels"][0])
        ks = [k for k in spec["ipc"] if k < full] + [min(k for k in spec["ipc"] if k >= full)]
        todo += [("random", k, width, seed) for k in ks for seed in spec["rand_seeds"]]
    return todo + [("spifil", k, width, seed) for k in spec["ipc"] for width in spec["widths"]
                   for seed in spec["draws"]]


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    with open(config, encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    path = Path(job["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path, newline="") as handle:
            done = {(r["arm"], int(r["k"]), int(r["width_req"]), int(r["seed"]), int(r["layer"]), r["readout"])
                    for r in csv.DictReader(handle) if r["status"] == "ok"}
    depth = job["amostra"]["depth"]

    def finished(arm, k, width, seed) -> bool:
        arms = ("spifil", "random_z") if arm == "spifil" and depth == 1 else (arm,)
        return all((a, k, width, seed, n + 1, r) in done for a in arms for n in range(depth)
                   for r in job["readouts"])

    todo = [c for c in cells(job) if not finished(*c)]
    if not todo:
        return
    images, to_float = load_images(job)
    sample = cover_sample(images, to_float)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if handle.tell() == 0:
            writer.writeheader()

        def write(row: dict) -> None:
            writer.writerow(row)
            handle.flush()

        pending, group = [], None
        for item in todo:
            if item[:2] != group:
                fit_pending(pending, write)
                pending, group = [], item[:2]
            pending += cell(job, images, to_float, sample, *item, write)
        fit_pending(pending, write)


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
