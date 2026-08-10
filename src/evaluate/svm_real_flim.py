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
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""svm_real_flim.py — SVM do pipeline de destilacao aplicado ao encoder FLIM RAW.

Pergunta: quanto o FLIM real (pesos dos marcadores, sem destilacao, sem checkpoint
nenhum) entrega sob EXATAMENTE o mesmo protocolo de avaliacao que produziu o
"Distill 4" (``results/svm_distill_proj1280_results.csv``)?

O que e reaproveitado sem alteracao de ``src/evaluate/svm_distill_with_projection.py``:
  * o classificador  — Pipeline(StandardScaler, SVC(C=1e2, gamma='auto',
    kernel='linear', decision_function_shape='ovo', max_iter=-1))
    NOTA: max_iter passou de 20000 para -1 (solver sem limite) nas duas arms;
    numeros novos nao sao comparaveis com os CSVs gerados sob o cap antigo.
    O protocolo efetivo vai gravado na coluna ``svm_protocol`` de cada linha.
  * o transform de eval — ``_build_test(IMAGE_SIZE=200, imagenet_norm=...)``
  * os dataloaders      — ``DatasetParasite`` + ``_OneHotDataset``, batch 32, sem shuffle
  * as metricas         — ``src.metrics.classification.compute_metrics``
  * a grade             — dataset x split x percentage identica ao Distill 4

O que muda e so o extrator de features:

    Distill 4:  input -> student FLIM (ckpt destilado) -> proj_kd -> [B, 1280]
    aqui:       input -> encoder FLIM raw              -> AvgPool -> [B, 48]

O pooling final e ``src.utils.evaluate._encode_pooled`` (``nn.AdaptiveAvgPool2d(1)``
sobre o mapa da conv3), ou seja: average pooling, um valor por canal de saida.

O encoder e construido do zero a partir do ``architecture.json`` + pesos FLIM pelo
mesmo caminho de ``AutoEncoderFlimModule.__init__`` (``get_actual_channels_from_weights``
-> ``override_arch_channels`` -> ``load_FLIM_encoder*``), o que cobre o caso protozoan
(conv2 com 30 canais, nao 32). Nenhum checkpoint e lido.

Saidas:
    artifacts/real_FLIM/real_FLIM_results.csv   (+ copia em results/real_FLIM_results.csv)
    artifacts/real_FLIM/partial_real_FLIM/<celula>.csv

Uso::

    CUDA_VISIBLE_DEVICES=3 conda run -n scalable_FLIM --no-capture-output \
        python -m src.evaluate.svm_real_flim

    # so uma celula, para conferir
    CUDA_VISIBLE_DEVICES=3 conda run -n scalable_FLIM --no-capture-output \
        python -m src.evaluate.svm_real_flim \
        --datasets protozoan --splits 1 --percentages 100 --imagenet-norm true
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Infra reaproveitada do avaliador de destilacao ────────────────────────────
from src.evaluate.svm_distillation import (            # noqa: E402
    _OneHotDataset,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
    DEVICE,
    _RESULTS_DIR,
)
from src.evaluate.constants import IMAGE_SIZE                      # noqa: E402
from src.data_modules.datasets.dataset import DatasetParasite      # noqa: E402
from src.data_modules.datasets.lejepa_dataset import _build_test   # noqa: E402
from src.metrics.classification import compute_metrics             # noqa: E402
from src.modules.autoencoder_flim_module import (                  # noqa: E402
    NUM_CLASSES, AutoEncoderFlimModule,
)
from src.utils.evaluate import (                                   # noqa: E402
    SVM_DIAG_MISSING, _encode_pooled, fit_svm,
)
from autoencoder_flim_ray import _arch_json, _flim_weights_path    # noqa: E402

_METHOD   = "SVM_Real_FLIM"
_OUT_DIR  = os.path.join(_ROOT, "artifacts", "real_FLIM")
_CSV_STEM = "real_FLIM_results"

# Grade do Distill 4 (results/svm_distill_proj1280_results.csv): 3 x 3 x 6 = 54 celulas.
_DATASETS    = ["eggs", "larvae", "protozoan"]
_SPLITS      = [1, 2, 3]
_PERCENTAGES = [1, 5, 25, 50, 75, 100]


# ── Encoder FLIM raw — mesmo caminho de AutoEncoderFlimModule.__init__ ────────

def _build_encoder(dataset: str, split: int):
    """Encoder FLIM construido do architecture.json + pesos dos marcadores.

    Sem checkpoint: os pesos vem direto de ``conv{n}-kernels.npy`` /
    ``conv{n}-bias.txt``. ``AutoEncoderFlimModule`` ja resolve o numero real de
    canais por dataset (protozoan: conv2 com 30, nao 32).
    """
    mod = AutoEncoderFlimModule(
        arch_json=_arch_json(dataset, split),
        dataset=dataset,
        flim_weights_path=_flim_weights_path(dataset, split),
        num_classes=NUM_CLASSES[dataset],
        image_size=IMAGE_SIZE,
        imagenet_norm=True,          # so afeta o alvo da reconstrucao; nao usamos o decoder
        freeze_encoder_flag=True,
    )
    enc = mod.model.encoder.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, mod.channels


def _loader(dataset: str, split: int, pct: int, set_name: str, imagenet_norm: bool,
            num_workers: int, one_hot: int = 0) -> DataLoader:
    """Identico ao dataloader do avaliador de destilacao (batch 32, sem shuffle)."""
    base = DatasetParasite(
        set_name=set_name, split=split, percentage=pct,
        transform=_build_test(IMAGE_SIZE, imagenet_norm=imagenet_norm),
        loader="ift_lab", path_dataset=_DATASET_PARASITE_NAME[dataset],
    )
    ds = _OneHotDataset(base, one_hot) if one_hot else base
    return DataLoader(ds, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ── Features: encoder FLIM -> average pooling -> [B, 48] ─────────────────────

@torch.no_grad()
def _extract_avgpooled(enc, loader: DataLoader, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """conv1->conv2->conv3 -> AdaptiveAvgPool2d(1) -> flatten. Labels 0-indexed."""
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        feats.append(_encode_pooled(enc, x.to(DEVICE)).numpy())
        if y.ndim == 2:                       # one-hot -> inteiro
            labels.extend(np.argmax(y.numpy(), axis=1).tolist())
        else:
            labels.extend(y.tolist())
    return np.concatenate(feats), np.array(labels, dtype=np.int64)


# ── SVM: literalmente o de svm_distill_with_projection._train_svm_proj ───────

def _fit_svm(X: np.ndarray, y: np.ndarray, C: float = 1e2,
             max_iter: int = -1) -> Pipeline:
    """``max_iter=-1`` (padrao) = solver sem limite; passe o cap antigo
    explicitamente so para reproduzir um CSV historico. Diagnosticos do solver
    ficam em ``clf.fit_diagnostics_`` (lidos do step ``svm`` do Pipeline).
    """
    # O StandardScaler vem do Distill 4 e e mantido; a assimetria de scaler
    # entre as arms NAO e unificada aqui, so registrada em svm_protocol.
    return fit_svm(X, y, max_iter=max_iter, C=C, scaler=True, tag=_METHOD)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--datasets",    nargs="+", default=_DATASETS, choices=_DATASETS)
    ap.add_argument("--splits",      nargs="+", type=int, default=_SPLITS)
    ap.add_argument("--percentages", nargs="+", type=int, default=_PERCENTAGES)
    ap.add_argument("--imagenet-norm", choices=["true", "false", "both"], default="both",
                    help="'true' = mesmo transform do Distill 4; 'false' = LAB[0,1] raw.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir",     default=_OUT_DIR)
    ap.add_argument("--csv-stem",    default=_CSV_STEM)
    args = ap.parse_args()

    norms = {"true": [True], "false": [False], "both": [True, False]}[args.imagenet_norm]

    partial_dir = os.path.join(args.out_dir, f"partial_{args.csv_stem}")
    os.makedirs(partial_dir, exist_ok=True)
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    total = len(args.datasets) * len(args.splits) * len(args.percentages) * len(norms)
    print(f"\n{'=' * 78}")
    print(f"{_METHOD}  |  encoder FLIM raw -> AvgPool -> [B, C]  |  SVM do Distill 4")
    print(f"device={DEVICE}  image_size={IMAGE_SIZE}  celulas={total}")
    print(f"imagenet_norm={norms}   saida={args.out_dir}")
    print(f"{'=' * 78}")

    rows: list[dict] = []
    done = 0

    for dataset in args.datasets:
        nc = _DATASET_NUM_CLASSES[dataset]
        for split in args.splits:
            enc, channels = _build_encoder(dataset, split)
            embed_dim = channels[-1]
            print(f"\n── {dataset} split{split}  channels={channels}  embed_dim={embed_dim}")

            for norm in norms:
                for pct in args.percentages:
                    done += 1
                    run_name = f"real_flim_{dataset}_split{split}_pct{pct}" + \
                               ("" if norm else "_no_imagenet_norm")
                    base = {
                        "run_name":     run_name,
                        "method":       _METHOD,
                        "dataset":      dataset,
                        "split":        split,
                        "percentage":   pct,
                        "encoder_init": "flim",
                        "embed_dim":    embed_dim,
                        "pooling":      "avg",
                        "imagenet_norm": norm,
                        "num_classes":  nc,
                    }
                    print(f"   [{done}/{total}] pct={pct:<3} imagenet_norm={norm}")
                    try:
                        X_tr, y_tr = _extract_avgpooled(
                            enc, _loader(dataset, split, pct, "train", norm,
                                         args.num_workers, one_hot=nc), "  train feats")
                        X_te, y_te = _extract_avgpooled(
                            enc, _loader(dataset, split, pct, "test", norm,
                                         args.num_workers), "  test feats")
                        clf = _fit_svm(X_tr, y_tr)
                        m = compute_metrics(y_true=y_te, y_pred=clf.predict(X_te),
                                            num_classes=nc)
                        # Diagnosticos do solver viajam no Pipeline.
                        diag = getattr(clf, "fit_diagnostics_", SVM_DIAG_MISSING)
                        row = {**base, **m, **diag,
                               "n_train": len(y_tr), "n_test": len(y_te),
                               "status": "ok", "error": ""}
                        print(f"        kappa={m['kappa']:.4f}  acc={m['acc']:.4f}  "
                              f"f1={m['f1']:.4f}  (n_train={len(y_tr)})  "
                              f"fit_status={diag['svm_fit_status']}  n_sv={diag['svm_n_sv']}")
                    except Exception as exc:                     # noqa: BLE001
                        print(f"        [ERROR] {exc}")
                        row = {**base, "kappa": float("nan"), "acc": float("nan"),
                               "f1": float("nan"), **SVM_DIAG_MISSING,
                               "n_train": 0, "n_test": 0,
                               "status": "error", "error": str(exc)}
                    rows.append(row)
                    pd.DataFrame([row]).to_csv(
                        os.path.join(partial_dir, f"{run_name}.csv"), index=False)

            del enc
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, f"{args.csv_stem}.csv")
    df.to_csv(csv_path, index=False)
    df.to_csv(os.path.join(_RESULTS_DIR, f"{args.csv_stem}.csv"), index=False)

    ok = df[df["status"] == "ok"]
    print(f"\n{'=' * 78}")
    print(f"[DONE] {csv_path}")
    print(f"       celulas: {len(df)}  ok: {len(ok)}  erros: {len(df) - len(ok)}")
    for norm in norms:
        sub = ok[ok["imagenet_norm"] == norm]
        if len(sub):
            print(f"       imagenet_norm={norm}: kappa={sub['kappa'].mean():.4f} "
                  f"acc={sub['acc'].mean():.4f} f1={sub['f1'].mean():.4f}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
