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

"""l1_fast.py — so a 1a camada SPiFiL, sem backprop nem decoder: tamanho da imagem x superpixels.

Um job = (dataset, split, percentual, tamanho), escrito por `experiments/ray/runners/eval.py`
(`source: l1_fast`). As imagens sobem para a RAM uma vez e cada celula so refaz o `grow`, a
conv e o SVM. Grava CSV proprio do job, entao jobs concorrentes nunca disputam arquivo, e
relancar retoma da primeira celula sem linha `ok`.

Le teste e validacao de proposito: o usuario pediu a lei medida direto no teste. Quem escolhe
celula pelo teste contamina o numero, por isso as duas colunas vivem lado a lado.
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from core.constants import DATASET_NUM_CLASSES
from core.data.parasite_dataset import ParasiteDataset, image_cache_path
from core.data.transforms import build_test
from core.metrics import compute_metrics
from core.svm import fit_svm
from eval.svm import DEVICE, _encode_pooled, cli_kwargs, pw_whiten
from experiments.ray.runners.growth import _paths, _short
from flim.build import build
from flim.spifil import grow

SVM_LOCK = "/exps/moliveira/queues/svm.lock"
COPY_COS = 0.9999    # so copia exata: L_kernels mostrou que cosseno nao mede mais que isso
COPY_LIMIT = 0.10    # acima disso o banco e refeito sem copias, com reposicao
BATCH_PIXELS = 2**23
LOCK_HOLD_S = 120    # a fila do lock custa ~1 min por entrada: entra uma vez e faz varios ajustes
SETS = {"val": "validation", "test": "test"}
METRICS = ("kappa", "acc", "f1")
FIELDS = ["dataset", "split", "percentage", "image_size", "n_superpixels", "arm", "readout",
          "seeds", "seeds_min", "seeds_max", "n_kernels", "copies", "copies_frac", "dedup",
          "removed", "replaced", "short", "grow_device", "dim", "n_train", "n_val", "n_test",
          *[f"{s}_{m}" for s in SETS for m in METRICS], "fit_status", "grow_s", "extract_s",
          "fit_s", "lock_wait_s", "weights", "status", "error"]


def load_images(job: dict) -> tuple[dict, v2.Transform]:
    """uint8 em RAM de cada conjunto, e o ultimo passo de `build_test`, que vira float na GPU."""
    size, no_crop = job["image_size"], job["no_crop"]
    test = build_test(size, imagenet_norm=False, no_crop=no_crop)
    tf = v2.Compose(test.transforms[:-1])
    cache = image_cache_path(job["image_cache_dir"], job["dataset_name"], size, no_crop)
    images = {}
    for name in ("train", *SETS.values()):
        data = ParasiteDataset(set_name=name, split=job["split"], percentage=job["percentage"],
                               transform=tf, loader="ift_lab", path_dataset=job["dataset_name"],
                               cache_dir=cache)
        pairs = list(DataLoader(data, batch_size=32, num_workers=8))
        images[name] = (torch.cat([x.as_subclass(torch.Tensor) for x, _ in pairs]),
                        torch.cat([y for _, y in pairs]).numpy())
    return images, test.transforms[-1]


def copies(weights: Path) -> tuple[int, int]:
    """Filtros que repetem um anterior, e o total do banco."""
    kernels = np.load(weights / "conv1-kernels.npy").T.astype(np.float64)
    unit = kernels / np.linalg.norm(kernels, axis=1, keepdims=True).clip(1e-12)
    return int((np.tril(unit @ unit.T, -1) > COPY_COS).any(1).sum()), len(kernels)


def run_grow(job: dict, n: int, out: Path, **extra) -> tuple[dict, float]:
    """O `grow` oficial; o que nao cabe no teto da GPU (matriz seeds x seeds) refaz em CPU."""
    point = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    arch, weights = _paths(point)
    kwargs = dict(encoder=None, ckpt="", arch_json=arch, flim_weights_path=weights,
                  dataset=_short(point), split=job["split"], percentage=job["percentage"],
                  out=out, n_superpixels=n, image_size=job["image_size"],
                  no_crop=job["no_crop"], **job["grow"], **extra)
    start, log, device = time.perf_counter(), io.StringIO(), str(DEVICE)
    with contextlib.redirect_stdout(log):
        try:
            grow(device=device, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            device = "cpu"
            grow(device=device, **kwargs)
    seeds = re.search(r"\((\d+) sementes em \d+ imagens, (\d+)\.\.(\d+)", log.getvalue())
    return ({"seeds": seeds[1], "seeds_min": seeds[2], "seeds_max": seeds[3], "grow_device": device},
            time.perf_counter() - start)


def spifil_bank(job: dict, n: int, cell: Path) -> tuple[Path, dict, float]:
    """Banco da celula. Passou de COPY_LIMIT em copias: refaz tirando as copias e repondo."""
    out = cell / "L1_grow"
    info, grow_s = run_grow(job, n, out)
    found, total = copies(out)
    info |= {"copies": found, "copies_frac": round(found / total, 4), "dedup": 0}
    if found / total > COPY_LIMIT:
        os.replace(out, cell / "L1_grow_raw")
        again, spent = run_grow(job, n, out, redundancy_cos=COPY_COS, redundancy_replace=True)
        report = json.loads((cell / "kernel_redundancy.json").read_text())["1"]
        info |= again | {"dedup": 1, **{k: report[k] for k in ("removed", "replaced", "short")}}
        grow_s += spent
    return out, info, grow_s


@torch.no_grad()
def features(encoder, x: torch.Tensor, to_float, size: int) -> dict:
    gem, tta = [], []
    step = max(1, BATCH_PIXELS // size**2)
    for i in range(0, len(x), step):
        batch = to_float(x[i:i + step].to(DEVICE))
        gem.append(_encode_pooled(encoder, batch, "gem"))
        tta.append(_encode_pooled(encoder, batch, "gem_tta"))
    return {"gem_pw": torch.cat(gem).numpy(), "gem_tta_pw": torch.cat(tta).numpy()}


def score(train, train_y, evals: dict, labels: dict, num_classes: int, tag: str) -> dict:
    """Whitening fitado no treino, um `fit_svm`, e `compute_metrics` nos dois conjuntos."""
    z, *others = pw_whiten(train, *evals.values())
    clf = fit_svm(z, train_y, tag=tag)
    row = {"fit_status": clf.fit_diagnostics_["svm_fit_status"], "fit_s": clf.fit_diagnostics_["svm_fit_seconds"],
           "dim": z.shape[1]}
    for (name, _), other in zip(evals.items(), others):
        metrics = compute_metrics(y_true=labels[name], y_pred=clf.predict(other), num_classes=num_classes)
        row |= {f"{name}_{m}": metrics[m] for m in METRICS}
    return row


def fit_pending(pending: list, write) -> None:
    """Todos os SVMs do job sob a trava, soltando a cada LOCK_HOLD_S para nao segurar a fila."""
    held = None
    with open(SVM_LOCK, "w") as lock:
        for row, args in pending:
            if held is None:
                start = time.perf_counter()
                fcntl.flock(lock, fcntl.LOCK_EX)
                held = time.perf_counter()
                wait = held - start
            try:
                write(row | score(*args) | {"lock_wait_s": round(wait, 1)})
            except Exception as exc:  # noqa: BLE001
                write(row | {"status": "error", "error": repr(exc)[:500]})
            wait = 0.0
            if time.perf_counter() - held > LOCK_HOLD_S:
                fcntl.flock(lock, fcntl.LOCK_UN)
                held = None


def run_cell(job: dict, images: dict, to_float, arm: str, n: int, write) -> list:
    """Banco e features de uma celula; devolve os SVMs a fazer. Falha vira linha `error`."""
    size = job["image_size"]
    base = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"],
            "image_size": size, "n_superpixels": n, "arm": arm, "n_train": len(images["train"][1]),
            "n_val": len(images["validation"][1]), "n_test": len(images["test"][1]), "status": "ok"}
    cell = Path(job["work_dir"]) / f"s{size:04d}_n{n:04d}" / arm
    try:
        if arm == "random":
            out, (info, grow_s) = cell / "L1_grow", run_grow(job, n, cell / "L1_grow", random_layer=True)
        else:
            out, info, grow_s = spifil_bank(job, n, cell)
        base |= info | {"weights": str(out), "n_kernels": copies(out)[1], "grow_s": round(grow_s, 2)}
        encoder = build(str(out / "architecture.json"), "flim", str(out)).to(DEVICE).eval()
        start = time.perf_counter()
        train = features(encoder, images["train"][0], to_float, size)
        evals = {k: features(encoder, images[v][0], to_float, size) for k, v in SETS.items()}
        base["extract_s"] = round(time.perf_counter() - start, 2)
        labels = {k: images[v][1] for k, v in SETS.items()}
        return [(base | {"readout": readout},
                 (train[readout], images["train"][1], {k: v[readout] for k, v in evals.items()},
                  labels, DATASET_NUM_CLASSES[job["dataset_name"]], f"l1fast {job['stem']} {arm} n{n}"))
                for readout in job["readouts"]]
    except Exception as exc:  # noqa: BLE001
        write(base | {"readout": "", "status": "error", "error": repr(exc)[:500]})
        return []


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    with open(config, encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    path = Path(job["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path, newline="") as handle:
            done = {(r["arm"], int(r["n_superpixels"]), r["readout"]) for r in csv.DictReader(handle)
                    if r["status"] == "ok"}
    cells = [("spifil", n) for n in job["superpixels"]] + [("random", n) for n in job["random"]]
    cells = [c for c in cells if any((*c, r) not in done for r in job["readouts"])]
    if not cells:
        return
    images, to_float = load_images(job)

    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if handle.tell() == 0:
            writer.writeheader()

        def write(row: dict) -> None:
            if (row["arm"], row["n_superpixels"], row["readout"]) not in done:
                writer.writerow(row)
                handle.flush()

        pending = []
        for arm, n in cells:
            shutil.rmtree(Path(job["work_dir"]) / f"s{job['image_size']:04d}_n{n:04d}" / arm,
                          ignore_errors=True)
            pending += run_cell(job, images, to_float, arm, n, write)
        fit_pending(pending, write)


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
