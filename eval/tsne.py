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

"""tsne_analysis.py — t-SNE visualization of FLIM distillation and LeJEPA trunc_normal embeddings.

Gera plots 2D t-SNE do test set para dois tipos de modelo:

  FLIM   — encoder do student (3x3_BN2d_1280_one_layer), 48-dim via GAP
  LeJEPA — encoder SSL com trunc_normal init, 48-dim via GAP

Estrutura de saída em artifacts/TSNE_analysis/:
    {dataset}/
        FLIM/
            split{s}_pct{p}.png
        lejepa/
            pct{p}/
                split{s}.png

Usage::
    python -m src.evaluate.tsne_analysis
    python -m src.evaluate.tsne_analysis --dataset eggs
    python -m src.evaluate.tsne_analysis --dataset larvae --split 1 --pct 100
    python -m src.evaluate.tsne_analysis --model flim
    python -m src.evaluate.tsne_analysis --max-samples 800
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# Dois niveis: eval/<este arquivo> -> eval -> raiz (eram tres em src/evaluate/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Os dois `_DATASET_*` de src/evaluate/svm_distillation.py:78 e :84 sao copia
# byte-identica de core/constants.py:67 e :192. Vem de la com apelido para o
# corpo deste arquivo continuar como estava.
from core.constants import (
    IMAGE_SIZE,
    NUM_CLASSES as _DATASET_NUM_CLASSES,
    PARASITE_NAME as _DATASET_PARASITE_NAME,
)
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from eval.constants import CLASS_NAMES
# `_ARTIFACTS_DIR` e `find_distillation_runs` sairam do import: eram mortos ja na
# origem (src/evaluate/tsne_analysis.py:62-63 importava, o corpo nunca usava).
# Traze-los para eval/svm.py so para satisfazer um import morto seria codigo novo
# sem consumidor.
from eval.svm import (
    cli_kwargs,
    resolve_available_experiments, find_best_checkpoint, parse_experiment_name,
)
from flim.arch import (
    parse_architecture, get_channels_from_arch,
    get_actual_channels_from_weights, override_arch_channels,
)
from flim.encoder import Encoder
from flim.weights import load_FLIM_encoder
from methods.distillation.one_layer_1x1_conv_distillation_projection_head import (
    OneLayer1x1ConvDistillationProjectionHead,
)
from methods.distillation.teacher_constants import TEACHER_DIM
from methods.lejepa.lejepa_flim_model import LeJEPAFLIMModel
from methods.lejepa.lejepa_line_module import LejepaLineModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TSNE_DIR  = os.path.join(_ROOT, "artifacts", "TSNE_analysis")
_FLIM_BASE = os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5")

_PERCENTAGES = [1, 5, 25, 50, 75, 100]
_SPLITS      = [1, 2, 3]
_DATASETS    = ["eggs", "larvae", "protozoan"]

_DATASET_LONG = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

# Colormaps por número de classes
_CMAP = {9: "tab10", 2: "bwr", 7: "tab10"}


# ── Carregamento de modelos ───────────────────────────────────────────────────

def _load_flim_encoder(ckpt_path: str) -> torch.nn.Module:
    """Carrega apenas o student encoder de um checkpoint de distilação."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hparams = ckpt.get("hyper_parameters", {})
    arch = parse_architecture(hparams.get("arch_json", ""))
    get_channels_from_arch(arch, hparams.get("in_channels", 3))
    student = LeJEPAFLIMModel(
        arch=arch,
        in_channels=hparams.get("in_channels", 3),
        proj_dim=hparams.get("proj_dim", 256),
        proj_hidden=hparams.get("proj_hidden", 2048),
    )
    full_sd = ckpt["state_dict"]
    student_sd = {k[len("student."):]: v for k, v in full_sd.items() if k.startswith("student.")}
    student.load_state_dict(student_sd, strict=True)
    return student.encoder.to(DEVICE).eval()


def _load_lejepa_encoder(run_id: str) -> torch.nn.Module:
    """Carrega encoder do LeJEPA trunc_normal via LejepaLineModule."""
    ckpt = find_best_checkpoint(run_id)
    module = LejepaLineModule.load_from_checkpoint(ckpt, map_location=DEVICE)
    return module.model.encoder.to(DEVICE).eval()


# ── Extração de embeddings ────────────────────────────────────────────────────

@torch.no_grad()
def _extract_embeddings(
    encoder: torch.nn.Module,
    loader: DataLoader,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai embeddings [N, 48] via GAP + labels."""
    feats, labels = [], []
    n = 0
    for x, y in loader:
        x = x.to(DEVICE)
        out = encoder.conv1(x)
        out = encoder.conv2(out)
        out = encoder.conv3(out)          # [B, 48, H, W]
        emb = out.mean(dim=[2, 3])        # GAP → [B, 48]
        feats.append(emb.cpu().numpy())
        if isinstance(y, torch.Tensor):
            labels.extend(y.tolist())
        else:
            labels.extend(y)
        n += len(x)
        if max_samples and n >= max_samples:
            break
    return np.concatenate(feats), np.array(labels, dtype=np.int64)


# ── t-SNE + plot ─────────────────────────────────────────────────────────────

def _tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    dataset: str,
    title: str,
    save_path: str,
    perplexity: int = 30,
) -> None:
    """Ajusta t-SNE e salva o plot 2D com nomes de espécies na legenda."""
    perp = min(perplexity, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(embeddings)

    class_names = CLASS_NAMES.get(dataset, [f"class {i}" for i in range(num_classes)])
    cmap = plt.get_cmap(_CMAP.get(num_classes, "tab10"))
    colors = [cmap(i / max(num_classes - 1, 1)) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        label = class_names[c] if c < len(class_names) else f"class {c}"
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=[colors[c]], label=label,
                   s=12, alpha=0.7, linewidths=0)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", markerscale=2, fontsize=8,
              framealpha=0.85, edgecolor="gray",
              handlelength=1.2, handletextpad=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Loops principais ──────────────────────────────────────────────────────────

def _load_pure_flim_encoder(dataset: str, split: int) -> torch.nn.Module:
    """Carrega encoder FLIM puro de data/to_mateus/model/ (sem distilação)."""
    train_id   = f"train{split}"
    model_dir  = os.path.join(_FLIM_BASE, dataset, train_id)
    arch_json  = os.path.join(model_dir, "architecture.json")
    weights_dir = os.path.join(model_dir, "models")

    arch     = parse_architecture(arch_json)
    channels = get_actual_channels_from_weights(weights_dir, arch, in_channels=3)
    arch     = override_arch_channels(arch, channels)
    encoder  = Encoder(arch, in_channels=3)
    load_FLIM_encoder(encoder, arch_json, weights_dir, channels)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder.to(DEVICE).eval()


def run_flim(
    datasets: list[str],
    splits: list[int],
    percentages: list[int],
    transform,
    max_samples: int | None,
    skip_existing: bool,
) -> None:
    """Gera plots t-SNE para os modelos FLIM puros (pesos originais, sem distilação)."""
    combos = [
        (ds, sp, pct)
        for ds in datasets
        for sp in splits
        for pct in percentages
    ]
    total = len(combos)
    print(f"\n[FLIM] {total} combinações a processar")

    with tqdm(total=total, desc="FLIM t-SNE", unit="run") as pbar:
        for (ds, sp, pct) in combos:
            save_path = os.path.join(_TSNE_DIR, ds, "FLIM", f"split{sp}_pct{pct}.png")
            pbar.set_postfix_str(f"{ds} s{sp} p{pct}")

            if skip_existing and os.path.exists(save_path):
                pbar.update(1)
                continue

            try:
                encoder  = _load_pure_flim_encoder(ds, sp)
                parasite = _DATASET_PARASITE_NAME.get(ds, ds)
                num_cls  = _DATASET_NUM_CLASSES.get(ds, 9)

                test_ds = ParasiteDataset(
                    set_name="test", split=sp, percentage=pct,
                    transform=transform, loader="ift_lab", path_dataset=parasite,
                )
                loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                    num_workers=4, pin_memory=True)

                embs, lbls = _extract_embeddings(encoder, loader, max_samples)
                title = (f"FLIM | {ds} | split={sp} | pct={pct}%\n"
                         f"(N={len(lbls)}, dim=48 GAP)")
                _tsne_plot(embs, lbls, num_cls, ds, title, save_path)
            except Exception as e:
                tqdm.write(f"  [ERR] FLIM {ds} s{sp} p{pct}: {e}")
            finally:
                pbar.update(1)


def run_lejepa(
    datasets: list[str],
    splits: list[int],
    percentages: list[int],
    transform,
    max_samples: int | None,
    skip_existing: bool,
) -> None:
    """Gera plots t-SNE para os modelos LeJEPA trunc_normal."""
    all_exps = resolve_available_experiments(update_wandb=False)
    trunc_exps: dict[tuple, tuple[str, str]] = {}

    ds_alias = {
        "eggs":      "helminth-eggs",
        "larvae":    "helminth-larvae",
        "protozoan": "protozoan-cysts",
    }
    ds_alias_inv = {v: k for k, v in ds_alias.items()}

    for run_id, name in all_exps.items():
        if "trunc_normal" not in name:
            continue
        try:
            info = parse_experiment_name(name)
        except Exception:
            continue
        ds_short = ds_alias_inv.get(info.dataset_name, "")
        if not ds_short:
            continue
        sp  = int(info.split_id)
        pct = int(info.percentage)
        if ds_short in datasets and sp in splits and pct in percentages:
            trunc_exps[(ds_short, sp, pct)] = (run_id, name)

    total = len(trunc_exps)
    print(f"\n[LeJEPA] {total} modelos a processar")

    with tqdm(total=total, desc="LeJEPA t-SNE", unit="run") as pbar:
        for (ds, sp, pct), (run_id, name) in sorted(trunc_exps.items()):
            save_path = os.path.join(_TSNE_DIR, ds, "lejepa", f"pct{pct}", f"split{sp}.png")
            pbar.set_postfix_str(f"{ds} s{sp} p{pct}")

            if skip_existing and os.path.exists(save_path):
                pbar.update(1)
                continue

            try:
                encoder  = _load_lejepa_encoder(run_id)
                parasite = _DATASET_PARASITE_NAME.get(ds, ds)
                num_cls  = _DATASET_NUM_CLASSES.get(ds, 9)

                test_ds = ParasiteDataset(
                    set_name="test", split=sp, percentage=pct,
                    transform=transform, loader="ift_lab", path_dataset=parasite,
                )
                loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                    num_workers=4, pin_memory=True)

                embs, lbls = _extract_embeddings(encoder, loader, max_samples)
                title = (f"LeJEPA (trunc_normal) | {ds} | split={sp} | pct={pct}%\n"
                         f"(N={len(lbls)}, dim=48 GAP)")
                _tsne_plot(embs, lbls, num_cls, ds, title, save_path)
            except Exception as e:
                tqdm.write(f"  [ERR] LeJEPA {ds} s{sp} p{pct}: {e}")
            finally:
                pbar.update(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(model: str = "both",
         dataset: str = "all",
         split: int = 0,
         pct: int = 0,
         max_samples: int | None = None,
         force: bool = False,
         perplexity: int = 30) -> None:
    """t-SNE 2D de embeddings FLIM e LeJEPA trunc_normal.

    Um parametro por flag do argparse antigo, com o mesmo default. `split=0` e
    `pct=0` continuam querendo dizer "todos". `perplexity` ja era inerte no
    argparse (src/evaluate/tsne_analysis.py:351): nunca chegou em `_tsne_plot`,
    que usa o proprio default de 30. Fica aqui so para a linha de comando nao
    perder a flag. Os `choices=` do argparse sairam junto com ele.
    """
    datasets    = _DATASETS if dataset == "all" else [dataset]
    splits      = _SPLITS if split == 0 else [split]
    percentages = _PERCENTAGES if pct == 0 else [pct]
    skip = not force
    transform = build_test(IMAGE_SIZE)

    print(f"{'='*65}")
    print(f"t-SNE Analysis")
    print(f"  datasets:    {datasets}")
    print(f"  splits:      {splits}")
    print(f"  percentages: {percentages}")
    print(f"  model:       {model}")
    print(f"  max_samples: {max_samples}")
    print(f"  output:      {_TSNE_DIR}")
    print(f"{'='*65}")

    if model in ("flim", "both"):
        run_flim(datasets, splits, percentages, transform, max_samples, skip)

    if model in ("lejepa", "both"):
        run_lejepa(datasets, splits, percentages, transform, max_samples, skip)

    print(f"\n[DONE] Plots salvos em: {_TSNE_DIR}")


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
