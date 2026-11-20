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
"""Benchmark de modelos congelados: pesos oficiais do HF, RGB, SVM oficial, nos splits do encoder.

Cada imagem passa 1 vez pelo modelo; split e percentual so escolhem linhas do .npz.
`python -m benchmark.<modelo>` extrai (GPU); `python -m benchmark.common` roda os SVMs,
um por vez sob a trava, 50% de todos os modelos primeiro.
"""
import csv
import fcntl
import json
import os
import signal
import subprocess
import threading
import time

import numpy as np
import timm
import torch
from PIL import ImageFile
from sklearn.metrics import recall_score
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader

from core.constants import NUM_CLASSES, ORGANIZED, PERCENTAGES, PROJECT_ROOT, SPLITS
from core.data.parasite_dataset import ParasiteDataset
from core.metrics import compute_metrics
from core.svm import fit_svm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # aid/orig/000022_00000228.jpg esta cortado e fica no split

EMB_DIR = "/exps/moliveira/benchmark/emb"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/benchmark")
STATUS = "/exps/moliveira/queues/logs/bench.status"
SVM_LOCK = "/exps/moliveira/queues/svm.lock"
GPU_CAP_MIB = 36000  # soma do moliveira na GPU 0
SETS = {"train": "train", "val": "validation", "test": "test"}


def _status(msg):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%F_%T')} {msg}\n")


def _emb_path(model, dataset):
    return os.path.join(EMB_DIR, model.split("/")[-1], f"{dataset}.npz")


def _gpu_guard(peak, stop):
    """Pico real por PID na GPU 0. Acima do teto, este run morre: quem chama nao segue adiante."""
    query = "nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits".split()
    while not stop.wait(5):
        mine = {}
        for line in subprocess.run(query, capture_output=True, text=True).stdout.splitlines():
            pid, mib = line.split(", ")
            if mib.isdigit() and os.path.exists(f"/proc/{pid}") and os.stat(f"/proc/{pid}").st_uid == os.getuid():
                mine[int(pid)] = int(mib)
        peak["own"] = max(peak["own"], mine.get(os.getpid(), 0))
        peak["total"] = max(peak["total"], sum(mine.values()))
        if sum(mine.values()) > GPU_CAP_MIB:
            _status(f"KILL teto de GPU: moliveira {sum(mine.values())} MiB {mine}")
            os.kill(os.getpid(), signal.SIGKILL)


def extract(model, revision, preprocess, load, batch=32, datasets=ORGANIZED, device="cuda"):
    """Grava `emb/<modelo>/<dataset>.npz` com as imagens dos 3 splits; pula o que ja existe.

    `load()` devolve (rede, funcao imagem -> embedding). `preprocess` e o nome timm de onde
    sai so o pre-processamento: nenhum peso vem do timm.
    """
    _status(f"START extract {model}")
    torch.backends.cuda.matmul.allow_tf32 = True
    net, embed = load()
    net.eval().to(device)
    cfg = resolve_data_config(pretrained_cfg=timm.get_pretrained_cfg(preprocess).to_dict())
    params = sum(p.numel() for p in net.parameters())
    os.makedirs(os.path.dirname(_emb_path(model, "x")), exist_ok=True)
    peak, stop = {"own": 0, "total": 0}, threading.Event()
    if device == "cuda":
        threading.Thread(target=_gpu_guard, args=(peak, stop), daemon=True).start()
    for dataset in datasets:
        out = _emb_path(model, dataset)
        if os.path.exists(out):
            continue
        ds = ParasiteDataset("train", 1, None, create_transform(**cfg, is_training=False), "pil", dataset)
        ds.samples = sorted({p for s in SPLITS for n in SETS.values()
                             for p in ParasiteDataset(n, s, None, path_dataset=dataset).samples})
        t0 = time.perf_counter()
        feats, labels = [], []
        with torch.inference_mode():
            for x, y in DataLoader(ds, batch_size=batch, num_workers=8, pin_memory=True):
                feats.append(embed(x.to(device)).float().cpu().numpy())
                labels.append(y.numpy())
        seconds = time.perf_counter() - t0
        np.savez(f"{out}.tmp.npz", X=np.concatenate(feats), y=np.concatenate(labels),
                 names=[os.path.basename(p) for p in ds.samples], seconds=seconds, params=params,
                 image_size=cfg["input_size"][-1], crop_pct=cfg["crop_pct"],
                 source=f"{model}@{revision}", preprocess=preprocess)
        os.replace(f"{out}.tmp.npz", out)
        print(f"[bench] {model} {dataset}: {len(ds)} img, {seconds:.0f}s, pico MiB {peak}", flush=True)
    stop.set()
    _status(f"END extract {model} pico MiB {peak}")


def _metrics(clf, X, y, num_classes):
    pred = clf.predict(X) - 1
    recall = recall_score(y, pred, labels=range(num_classes), average=None, zero_division=0)
    return {**compute_metrics(y, pred, num_classes), "recall": json.dumps(np.round(recall, 4).tolist())}


def _cell(model, dataset, split, pct, emb):
    """Uma linha do CSV. Sem 2 classes no treino nao ha SVM: a linha sai com o status e sem metrica."""
    index = {n: i for i, n in enumerate(emb["names"])}
    rows = {k: [index[os.path.basename(p)] for p in ParasiteDataset(n, split, pct, path_dataset=dataset).samples]
            for k, n in SETS.items()}
    num_classes = NUM_CLASSES[dataset]
    y_train = emb["y"][rows["train"]]
    missing = sorted(set(range(num_classes)) - set(y_train.tolist()))
    row = {"model": model, "source": str(emb["source"]), "preprocess": str(emb["preprocess"]),
           "dataset": dataset, "split": split, "pct": pct,
           "params": int(emb["params"]), "dim": emb["X"].shape[1], "image_size": int(emb["image_size"]),
           "crop_pct": float(emb["crop_pct"]), "n_train": len(y_train), "n_val": len(rows["val"]),
           "n_test": len(rows["test"]), "classes_missing_train": json.dumps(missing),
           "extract_ms_per_img": round(1000 * float(emb["seconds"]) / len(emb["y"]), 2)}
    if len(missing) > num_classes - 2:
        print(f"[bench] {model} {dataset} s{split} p{pct}: so {num_classes - len(missing)} classe no treino", flush=True)
        return {**row, "status": "one_class"}
    if missing:
        print(f"[bench] {model} {dataset} s{split} p{pct}: classes sem treino {missing}", flush=True)
    t0 = time.perf_counter()
    with open(SVM_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait = time.perf_counter() - t0
        clf = fit_svm(emb["X"][rows["train"]], y_train + 1, tag=f"{model} {dataset} s{split} p{pct}")
        t1 = time.perf_counter()
        for k, r in rows.items():
            for name, value in _metrics(clf, emb["X"][r], emb["y"][r], num_classes).items():
                row[f"{k}_{name}"] = value
    return {**row, **clf.fit_diagnostics_, "status": "ok", "lock_wait_s": round(wait, 1),
            "predict_s": round(time.perf_counter() - t1, 1)}


FIELDS = ["model", "source", "preprocess", "dataset", "split", "pct", "status", "params", "dim", "image_size", "crop_pct", "n_train",
          "n_val", "n_test", "classes_missing_train", "extract_ms_per_img",
          *[f"{k}_{m}" for k in SETS for m in ("kappa", "acc", "f1", "recall")],
          "svm_protocol", "svm_fit_status", "svm_n_iter_max", "svm_n_iter_sum", "svm_n_sv", "svm_fit_seconds",
          "predict_s", "lock_wait_s"]


def svm(model, dataset, pct):
    """Roda os splits que faltam no CSV do modelo. Relancar retoma de onde parou."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model.split('/')[-1]}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["dataset"], r["split"], r["pct"]) for r in csv.DictReader(f)}
    todo = [s for s in SPLITS if (dataset, str(s), str(pct)) not in done]
    if not todo:
        return
    with np.load(_emb_path(model, dataset)) as z:
        emb = {k: z[k] for k in z.files}
    for split in todo:
        row = _cell(model, dataset, split, pct, emb)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"[bench] {model} {dataset} s{split} p{pct}: val kappa {row.get('val_kappa')}", flush=True)


def svm_all(models, datasets=ORGANIZED):
    """Espera cada embedding aparecer; `emb/.extract_done` diz que nao vem mais nenhum."""
    for pct in sorted(PERCENTAGES, key=lambda p: p != 50):
        for model in models:
            _status(f"START svm {model} pct{pct}")
            for dataset in datasets:
                while not os.path.exists(_emb_path(model, dataset)):
                    if os.path.exists(os.path.join(EMB_DIR, ".extract_done")):
                        break
                    time.sleep(60)
                else:
                    svm(model, dataset, pct)
            _status(f"END svm {model} pct{pct}")


if __name__ == "__main__":
    from benchmark import aimv2, convnextv2, dinov2, ijepa

    svm_all([m.MODEL for m in (convnextv2, ijepa, dinov2, aimv2)])
