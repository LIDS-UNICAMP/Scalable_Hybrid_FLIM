# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
"""svm_spifil_dense.py — SVM linear em cada estagio da rede densa crescida pelo SPiFiL.

EXPERIMENTO VERSIONADO (nao e script descartavel). Re-executavel e incremental.

Pergunta: a cada bloco que o crescimento encaixa (S, DB1, ..., H), o embedding
fica mais separavel? Uma linha por (dataset, split, estagio).

  slot       | aqui
  -----------|------------------------------------------------------------
  encoder    | estagio salvo, src.models.dense_growth.load_stage
  features   | GAP da saida do estagio -> [B, C] (C = 24 ... 1280); o mesmo
             | reduce do "avgpool2d" de _encode_pooled (evaluate.py:254)
  entrada    | _build_test(200, imagenet_norm=False) — o que o treino viu
  solver     | fit_svm (evaluate.py:371), max_iter do YAML (-1 = convergido)
  metricas   | compute_metrics (src/metrics/classification.py)
  rotulos    | 1-indexed no fit, predict(feats)-1  (src/evaluate/svm.py:135)
  dados      | fit no treino de grid.percentage, score no teste

_encode_pooled nao serve aqui: ele chama conv1/conv2/conv3, o encoder FLIM.

**Uma passada para os 10 estagios.** O crescimento congela tudo antes de cada
bloco e checa que nada mudou, entao o estagio k e prefixo exato do H. Isto e
re-conferido aqui (state_dict de cada estagio == prefixo do H, bit a bit) e,
passando, uma unica forward do H entrega o GAP de todos os estagios.

Tudo vem de configs/model/spifil_dense_growth.yaml (blocos grid, output_root,
evaluate):

    python -m src.evaluate.svm_spifil_dense --config configs/model/spifil_dense_growth.yaml
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from torch.utils.data import DataLoader                                 # noqa: E402

from src.data_modules.datasets.dataset import DatasetParasite           # noqa: E402
from src.data_modules.datasets.lejepa_dataset import _build_test        # noqa: E402
from src.evaluate.constants import IMAGE_SIZE                           # noqa: E402
from src.metrics.classification import compute_metrics                  # noqa: E402
from src.models.dense_growth import load_stage                          # noqa: E402
from src.utils.evaluate import DEVICE, fit_svm                          # noqa: E402

_PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae",
             "protozoan": "protozoan-cysts"}
_NUM_CLASSES = {"eggs": 9, "larvae": 2, "protozoan": 7}

_METHOD = "SVM_spifil_dense_gap"
_FIELDS = ["method", "run_name", "dataset", "split", "percentage", "stage_index",
           "stage", "dim", "resolution", "n_train", "n_test", "imagenet_norm",
           "max_iter", "fit_status", "n_iter_max", "n_sv", "kappa", "acc", "f1",
           "acc_raw", "fit_s"]


def _loader(dataset: str, split: int, percentage: int, set_name: str, ev: dict):
    base = DatasetParasite(
        set_name=set_name, split=split, percentage=percentage,
        transform=_build_test(IMAGE_SIZE, imagenet_norm=False),
        loader="ift_lab", path_dataset=_PARASITE[dataset],
    )
    return DataLoader(base, batch_size=ev["batch_size"], shuffle=False,
                      num_workers=ev["num_workers"], pin_memory=True)


def _stage_dirs(run_dir: str) -> list:
    return sorted(os.path.join(run_dir, d) for d in os.listdir(run_dir)
                  if d.startswith("stage") and os.path.isdir(os.path.join(run_dir, d)))


def _check_prefix(stage_dirs: list, final_state: dict) -> None:
    """Cada estagio tem de ser prefixo bit a bit do ultimo, ou a passada unica mente."""
    for d in stage_dirs:
        state = torch.load(os.path.join(d, "encoder.pt"), map_location="cpu")
        for key, value in state.items():
            if not torch.equal(value, final_state[key]):
                raise AssertionError(f"{d}: {key} difere do ultimo estagio — "
                                     "o crescimento nao congelou este prefixo")


@torch.no_grad()
def _stage_features(net, loader) -> tuple:
    """GAP da saida de cada bloco: lista [n_stages] de (N, C_k), e os rotulos 0-indexed."""
    net.eval().to(DEVICE)
    feats = [[] for _ in net.blocks]
    labels = []
    for inputs, y in loader:
        x = inputs.to(DEVICE)
        for k, block in enumerate(net.blocks):
            x = block(x)
            feats[k].append(x.mean(dim=(2, 3)).cpu())
        labels.append(torch.as_tensor(y))
    return [torch.cat(f).numpy() for f in feats], torch.cat(labels).numpy()


def _evaluate_cell(cfg: dict, dataset: str, split: int, done: set) -> list:
    ev, pct = cfg["evaluate"], cfg["grid"]["percentage"]
    run_name = f"spifil_dense_{dataset}_split{split}_pct{pct}"
    run_dir = os.path.join(_ROOT, cfg["output_root"], run_name)
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if not (os.path.isfile(meta_path) and json.load(open(meta_path)).get("status") == "ok"):
        print(f"[SKIP] {run_name}: crescimento nao terminou")
        return []
    stage_dirs = _stage_dirs(run_dir)
    names = [os.path.basename(d).split("_", 1)[1] for d in stage_dirs]
    if all((dataset, split, n) in done for n in names):
        print(f"[SKIP] {run_name}: ja no CSV")
        return []

    net = load_stage(stage_dirs[-1])
    _check_prefix(stage_dirs[:-1], net.state_dict())
    arch = json.load(open(os.path.join(stage_dirs[-1], "architecture.json")))["layers"]

    t0 = time.perf_counter()
    train_x, train_y = _stage_features(net, _loader(dataset, split, pct, "train", ev))
    test_x, test_y = _stage_features(net, _loader(dataset, split, pct, "test", ev))
    print(f"[INFO] {run_name}: train n={len(train_y)} test n={len(test_y)} "
          f"features em {time.perf_counter() - t0:.0f}s")

    rows = []
    for k, name in enumerate(names):
        if (dataset, split, name) in done:
            continue
        start = time.perf_counter()
        clf = fit_svm(train_x[k], train_y + 1, max_iter=ev["max_iter"],
                      tag=f"SVM_{name}")
        fit_s = time.perf_counter() - start
        y_pred = clf.predict(test_x[k]) - 1                          # svm.py:135
        metrics = compute_metrics(y_true=test_y, y_pred=y_pred,
                                  num_classes=_NUM_CLASSES[dataset])
        diag = clf.fit_diagnostics_
        rows.append({
            "method": _METHOD, "run_name": run_name, "dataset": dataset,
            "split": split, "percentage": pct, "stage_index": k, "stage": name,
            "dim": train_x[k].shape[1], "resolution": arch[k]["out_size"],
            "n_train": len(train_y), "n_test": len(test_y), "imagenet_norm": False,
            "max_iter": ev["max_iter"], "fit_status": diag["svm_fit_status"],
            "n_iter_max": diag["svm_n_iter_max"], "n_sv": diag["svm_n_sv"],
            "kappa": float(metrics["kappa"]), "acc": float(metrics["acc"]),
            "f1": float(metrics["f1"]),
            "acc_raw": float((y_pred == test_y).mean()), "fit_s": round(fit_s, 1),
        })
        print(f"  [RESULT] {dataset} split{split} {name:<4} dim={train_x[k].shape[1]:<5d} "
              f"kappa={metrics['kappa']:+.4f} acc={metrics['acc']:.4f} "
              f"f1={metrics['f1']:.4f} ({fit_s:.1f}s)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(
        _ROOT, "configs/model/spifil_dense_growth.yaml"))
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ev = cfg["evaluate"]
    out = os.path.join(_ROOT, ev["out_csv"])

    # Incremental: o que ja esta no CSV nao e refeito, e o CSV antigo nunca e truncado.
    old = list(csv.DictReader(open(out))) if os.path.isfile(out) else []
    done = {(r["dataset"], int(r["split"]), r["stage"]) for r in old}
    print(f"[CONTRATO] features=GAP por estagio | entrada=_build_test({IMAGE_SIZE}, False) "
          f"| solver=fit_svm(max_iter={ev['max_iter']}) | device={DEVICE}")

    rows = []
    for dataset in ev["datasets"]:
        for split in cfg["grid"]["splits"]:
            rows += _evaluate_cell(cfg, dataset, split, done)
            # Escreve a cada celula: uma queda no meio nao perde o que ja foi ajustado.
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDS)
                writer.writeheader()
                writer.writerows(old + rows)
    print(f"\n[OK] {out}  ({len(rows)} linhas novas, {len(old) + len(rows)} no total)")

    allrows = old + rows
    for dataset in ev["datasets"]:
        sel = [r for r in allrows if r["dataset"] == dataset]
        if not sel:
            continue
        print(f"\n=== {dataset}: kappa medio por estagio ({len({r['split'] for r in sel})} splits) ===")
        print(f"{'stage':>6} {'dim':>5} {'kappa':>9} {'std':>7} {'acc':>7} {'f1':>7}")
        for name in dict.fromkeys(r["stage"] for r in sorted(sel, key=lambda r: int(r["stage_index"]))):
            s = [r for r in sel if r["stage"] == name]
            kap = [float(r["kappa"]) for r in s]
            print(f"{name:>6} {s[0]['dim']:>5} {np.mean(kap):>+9.4f} {np.std(kap):>7.4f} "
                  f"{np.mean([float(r['acc']) for r in s]):>7.4f} "
                  f"{np.mean([float(r['f1']) for r in s]):>7.4f}")


if __name__ == "__main__":
    main()
