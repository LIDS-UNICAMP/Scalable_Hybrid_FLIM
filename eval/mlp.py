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

"""mlp.py — MLP fine-tuning evaluation of pretrained LeJEPA Line encoders.

Origem: `src/evaluate/mlp.py` (656 linhas), movido fiel. As unicas edicoes sao o
reponto dos imports para a arvore nova e a troca do argparse (`main()` da origem,
linha 485) por uma funcao com defaults nomeados.

Discovers YAML configs from configs/generated/mlp/{freeze,unfreeze}/ and for
each experiment:
  1. Loads the pretrained encoder from the best checkpoint.
  2. Attaches an MLP head (ClassificationModel from core.blocks).
  3. Fine-tunes with encoder frozen or unfrozen.
  4. Evaluates on the test split using core.metrics.compute_metrics.
  5. Appends all results to results/mlp_results.csv.

Usage:
    python -m eval.mlp                              # all yamls
    python -m eval.mlp --mode freeze                # only freeze mode
    python -m eval.mlp --mode unfreeze              # only unfreeze mode
    python -m eval.mlp --config <path/to/cfg.yaml>  # single yaml

Este e o modulo que `experiments/ray/runners/eval.py:build_cmd` invoca por
subprocesso, com `--config <path> --ckpt-selection=<best|last> [--wandb]`.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.blocks.classification_model import ClassificationModel
from core.blocks.init import freeze_encoder, unfreeze_encoder
from core.constants import DATASET_NUM_CLASSES, IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from core.wandb import ENTITY, PROJECT, resolve_run
from flim.arch import (
    get_channels_from_arch,
    override_arch_channels,
    parse_architecture,
)
from methods.lejepa.lejepa_line_module import LejepaLineModule
from eval.svm import (
    cli_kwargs,
    find_best_checkpoint,
    parse_experiment_name,
    resolve_available_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ROOT = PROJECT_ROOT

# Os 112 YAMLs do MLP vivem em configs/generated/mlp/{freeze,unfreeze}/<ds>/<run_id>.yaml.
# Mesmo diretorio que experiments/constants.py:MLP_CONFIGS_DIR aponta para o runner.
_CONFIGS_DIR = os.path.join(_ROOT, "configs", "generated", "mlp")
_RESULTS_DIR = os.path.join(_ROOT, "results")


# ─── Model loading ────────────────────────────────────────────────────────────


def load_classification_model(
    run_id: str,
    num_classes: int,
    in_channels: int = 3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    conv2_channels: Optional[int] = None,
) -> tuple[ClassificationModel, str]:
    """Load encoder weights from a LejepaLineModule checkpoint into a ClassificationModel.

    Returns the model and the checkpoint path used.

    Args:
        conv2_channels: Override the number of output channels for conv2.
                        Use 30 for protozoan-cysts with FLIM initialisation,
                        whose kernels produced 30 filters instead of the default 32.
    """
    ckpt_path = find_best_checkpoint(run_id)
    module = LejepaLineModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    arch = parse_architecture(module.hparams.arch_json)

    # Auto-detect actual conv2 channel count from the checkpoint weights.
    # This handles FLIM protozoan-cysts checkpoints that have 30 conv2 channels
    # instead of the 32 declared in the architecture JSON, without requiring
    # the caller to know or pass conv2_channels explicitly.
    encoder_sd = module.model.encoder.state_dict()
    ckpt_conv2_ch = encoder_sd.get("blocks.conv2.0.weight", encoder_sd.get("conv2.0.weight"))
    if ckpt_conv2_ch is not None:
        conv2_channels = int(ckpt_conv2_ch.shape[0])

    if conv2_channels is not None:
        channels = get_channels_from_arch(arch, in_channels)
        channels[2] = conv2_channels
        arch = override_arch_channels(arch, channels)

    model = ClassificationModel(arch, num_classes, in_channels, hidden_dim, dropout)
    model.encoder.load_state_dict(module.model.encoder.state_dict(), strict=True)
    del module
    return model, ckpt_path


# ─── Training loop ────────────────────────────────────────────────────────────


def train_and_evaluate(
    model: ClassificationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    frozen: bool,
    num_classes: int,
    log_to_wandb: bool = True,
    weights_path: Optional[str] = None,
    patience: int = 50,
) -> dict[str, float]:
    """Fine-tune model and evaluate on test set. Returns compute_metrics dict.

    Args:
        weights_path: If provided, saves the best-val-acc model state dict to
                      this path after training (via ``torch.save``).
    """
    model.to(DEVICE)

    if frozen:
        freeze_encoder(model)
        optimizer = AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        unfreeze_encoder(model)
        optimizer = AdamW(
            [
                {"params": model.encoder.parameters(), "lr": lr * 0.1},
                {"params": model.head.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state: Optional[dict] = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        if frozen:
            model.encoder.eval()

        epoch_loss = 0.0
        n_batches = 0
        for imgs, labels in tqdm(
            train_loader,
            desc=f"  Epoch {epoch + 1:>3}/{max_epochs}",
            leave=False,
        ):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_preds: list[int] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                lbl = labels.numpy() if isinstance(labels, torch.Tensor) else labels
                val_labels.extend(lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))

        val_metrics = compute_metrics(
            y_true=np.array(val_labels),
            y_pred=np.array(val_preds),
            num_classes=num_classes,
        )
        val_acc = val_metrics["acc"]

        if log_to_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "val/acc": val_acc,
                "val/kappa": val_metrics["kappa"],
                "val/f1": val_metrics["f1"],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step()

        if patience > 0 and epochs_no_improve >= patience:
            if log_to_wandb:
                wandb.log({"early_stop_epoch": epoch + 1})
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if weights_path is not None:
        _weights_dir = os.path.dirname(weights_path)
        if _weights_dir:
            os.makedirs(_weights_dir, exist_ok=True)
        torch.save(best_state if best_state is not None else model.state_dict(), weights_path)

    # Test evaluation
    model.eval()
    model.to(DEVICE)
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Test eval", leave=False):
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            lbl = labels.numpy() if isinstance(labels, torch.Tensor) else labels
            all_labels.extend(lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))

    metrics = compute_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        num_classes=num_classes,
    )

    if log_to_wandb:
        wandb.log({f"test/{k}": v for k, v in metrics.items()})

    return metrics


# ─── Config discovery ─────────────────────────────────────────────────────────


def get_experiment_configs(
    mode: str = "all",
    update_wandb: bool = False,
    available_experiments: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Return a list of experiment config dicts for the given mode.

    Each dict contains all fields from the corresponding YAML file plus a
    ``config_path`` key with the absolute path to the YAML file.

    Args:
        mode:                  ``"freeze"``, ``"unfreeze"``, or ``"all"`` (default).
        update_wandb:          When ``True``, refresh W&B cache before resolving.
                               Ignored if *available_experiments* is provided.
        available_experiments: Pre-built ``{run_id: run_name}`` dict.  When
                               supplied, skips the W&B / cache lookup entirely.

    Returns:
        List of config dicts whose ``run_id`` is present in the set of
        available experiments (W&B record + local checkpoint).  Configs whose
        ``experiment_name`` starts with ``"finetune_"`` are excluded.
    """
    yaml_files: list[str] = []
    if mode in ("freeze", "all"):
        yaml_files += sorted(
            glob.glob(os.path.join(_CONFIGS_DIR, "freeze", "**", "*.yaml"), recursive=True)
        )
    if mode in ("unfreeze", "all"):
        yaml_files += sorted(
            glob.glob(os.path.join(_CONFIGS_DIR, "unfreeze", "**", "*.yaml"), recursive=True)
        )

    if available_experiments is None:
        available_experiments = resolve_available_experiments(update_wandb=update_wandb)

    # Keep only configs whose run_id has weights + W&B entry
    yaml_files = [
        f for f in yaml_files
        if os.path.splitext(os.path.basename(f))[0] in available_experiments
    ]

    configs: list[dict] = []
    for cfg_path in yaml_files:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        # Resolve run_name from available_experiments so it is always current
        run_id = cfg.get("run_id", "")
        run_name = available_experiments.get(run_id, cfg.get("experiment_name", ""))
        cfg["experiment_name"] = run_name
        cfg["config_path"] = cfg_path
        # Exclude legacy finetune_ runs from Ray queue
        if run_name.startswith("finetune_"):
            continue
        configs.append(cfg)

    return configs


# ─── Per-experiment runner ────────────────────────────────────────────────────


def run_from_yaml(
    cfg_path: str,
    rows: list[dict],
    finetune_names: dict[str, str],
    report: dict[str, list],
    available_experiments: dict[str, str],
    log_to_wandb: bool = True,
    conv2_channels: Optional[int] = None,
) -> None:
    """Run one MLP experiment from a YAML config and append result to rows."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_id: str = cfg["run_id"]
    frozen: bool = cfg.get("freeze_encoder", True)
    max_epochs: int = cfg.get("max_epochs", 50)
    lr: float = cfg.get("lr", 1e-3)
    weight_decay: float = cfg.get("weight_decay", 1e-4)
    batch_size: int = cfg.get("batch_size", 32)
    hidden_dim: int = cfg.get("hidden_dim", 256)
    dropout: float = cfg.get("dropout", 0.3)
    patience: int = cfg.get("patience", 50)

    run_name = available_experiments.get(run_id)
    if run_name is None:
        print(f"  [WARN] run_id={run_id!r} not found in W&B — skipping")
        return

    wandb_name = "X_" + finetune_names.get(run_id, f"finetune_{run_name}")
    is_collision_resolved = wandb_name.endswith(f"_{run_id}")

    base_row: dict = {
        "wandb_run_id": run_id,
        "experiment_name": run_name,
        "wandb_run_name": wandb_name,
        "freeze_encoder": frozen,
        "max_epochs": max_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "config_path": os.path.relpath(cfg_path, _ROOT),
    }

    # ── Pre-check: verify weights exist before doing any work ─────────────
    try:
        find_best_checkpoint(run_id)
    except FileNotFoundError as exc:
        print(f"  [SKIP] Missing weights: {exc}")
        rows.append({
            **base_row,
            "kappa": float("nan"), "acc": float("nan"), "f1": float("nan"),
            "status": "missing_weights", "error": str(exc),
        })
        report["skipped_missing_weights"].append(run_name)
        return

    try:
        info = parse_experiment_name(run_name)
        base_row.update({
            "dataset_name": info.dataset_name,
            "split_id": info.split_id,
            "percentage": info.percentage,
            "initialization_type": info.initialization_type,
        })

        num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)
        transform = build_test(IMAGE_SIZE)

        print(f"  [INFO] Loading encoder from checkpoint...")
        model, ckpt_path = load_classification_model(
            run_id, num_classes, hidden_dim=hidden_dim, dropout=dropout,
            conv2_channels=conv2_channels,
        )
        print(f"  [INFO] Checkpoint : {os.path.relpath(ckpt_path, _ROOT)}")

        ds_kwargs = dict(
            split=info.split_id,
            percentage=info.percentage,
            transform=transform,
            loader="ift_lab",
            path_dataset=info.path_dataset,
        )
        loader_kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

        train_loader = DataLoader(
            ParasiteDataset(set_name="train", **ds_kwargs), shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            ParasiteDataset(set_name="validation", **ds_kwargs), shuffle=False, **loader_kwargs
        )
        test_loader = DataLoader(
            ParasiteDataset(set_name="test", **ds_kwargs), shuffle=False, **loader_kwargs
        )

        mode_str = "frozen encoder" if frozen else "unfrozen encoder"
        print(f"  [INFO] Training MLP ({mode_str}, {max_epochs} epochs)...")
        print(f"  [INFO] W&B run name: {wandb_name}")

        mode_tag = "freeze" if frozen else "unfreeze"
        weights_path = os.path.join(
            _RESULTS_DIR, "mlp_weights", mode_tag, run_id, "model_best.pth"
        )

        if log_to_wandb:
            wandb_run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=wandb_name,
                config={**base_row, "num_classes": num_classes},
                reinit=True,
            )
        try:
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
                frozen=frozen, num_classes=num_classes,
                log_to_wandb=log_to_wandb, patience=patience,
                weights_path=weights_path,
            )
            if log_to_wandb and os.path.exists(weights_path):
                artifact = wandb.Artifact(
                    name=f"model_{run_id}_{mode_tag}",
                    type="model",
                    metadata={**base_row, "num_classes": num_classes, **metrics},
                )
                artifact.add_file(weights_path, name="model_best.pth")
                wandb_run.log_artifact(artifact)
        finally:
            if log_to_wandb:
                wandb_run.finish()

        rows.append({**base_row, **metrics, "status": "ok", "error": ""})
        report["executed"].append(run_name)
        if is_collision_resolved:
            report["collision_resolved"].append(run_name)
        print(
            f"  [RESULT] kappa={metrics['kappa']:.4f}  "
            f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
        )

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        rows.append({
            **base_row,
            "kappa": float("nan"),
            "acc": float("nan"),
            "f1": float("nan"),
            "status": "error",
            "error": str(exc),
        })
        report["skipped_errors"].append(f"{run_name}: {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(
    mode: str = "all",
    config: Optional[str] = None,
    dataset: str = "all",
    wandb_update: bool = False,
    ckpt_selection: str = "best",
    wandb: bool = True,
) -> str:
    """Roda a avaliacao MLP sobre os YAMLs de `configs/generated/mlp/`.

    Substitui o `argparse.ArgumentParser` de `src/evaluate/mlp.py:486`. Cada
    parametro abaixo era uma flag daquele parser, com o MESMO default:

        --mode          -> mode          (default "all";  freeze|unfreeze|all)
        --config        -> config        (default None;   sobrepoe --mode)
        --dataset       -> dataset       (default "all";  eggs|protozoan|larvae|all)
        --wandb-update  -> wandb_update  (default False)

    Os dois ultimos parametros NAO existiam no argparse de origem; entraram para
    honrar o comando que `experiments/ray/runners/eval.py:build_cmd` ja monta:

        ckpt_selection  Contrato do runner (`--ckpt-selection=best|last`).
                        `best` e o unico implementado — e exatamente o que
                        `eval/svm.py:find_best_checkpoint` faz (best.ckpt, com
                        fallback para last.ckpt). `last` levanta erro em vez de
                        avaliar silenciosamente o checkpoint errado.
        wandb           Contrato do runner (`--wandb`). Default True para nao
                        mudar o comportamento da origem, que sempre subia o
                        resumo. Com False, o upload de resumo e pulado; o
                        `wandb.init` por experimento dentro de `run_from_yaml`
                        NAO e afetado, porque a origem tambem nao o gateava.

    Returns:
        Caminho do CSV escrito em `results/mlp_results.csv`.
    """
    # Alias local: o parametro `wandb` (bool, exigido pelo nome da flag do
    # runner) sombreia o modulo dentro desta funcao.
    import wandb as wandb_sdk  # noqa: PLC0415

    if ckpt_selection != "best":
        raise NotImplementedError(
            f"ckpt_selection={ckpt_selection!r}: so 'best' existe hoje. "
            "A politica de selecao mora em eval/svm.py:find_best_checkpoint "
            "(best.ckpt, com fallback para last.ckpt) e nao tem seletor 'last'."
        )

    _DATASET_SHORT_TO_FULL = {
        "eggs": "helminth-eggs",
        "protozoan": "protozoan-cysts",
        "larvae": "helminth-larvae",
    }

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []

    # Resolve W&B metadata once (uses cache unless wandb_update was passed)
    available_experiments = resolve_available_experiments(update_wandb=wandb_update)

    # Build collision-safe W&B run names from the already-resolved experiments.
    # `core.wandb.resolve_run` e o antigo `build_finetune_name_dict`.
    finetune_names = resolve_run(available_experiments)

    # Execution report
    report: dict[str, list] = {
        "executed": [],
        "skipped_missing_weights": [],
        "skipped_errors": [],
        "collision_resolved": [],
    }

    if config:
        yaml_files = [config]
    else:
        yaml_files: list[str] = []
        if mode in ("freeze", "all"):
            yaml_files += sorted(
                glob.glob(os.path.join(_CONFIGS_DIR, "freeze", "**", "*.yaml"), recursive=True)
            )
        if mode in ("unfreeze", "all"):
            yaml_files += sorted(
                glob.glob(os.path.join(_CONFIGS_DIR, "unfreeze", "**", "*.yaml"), recursive=True)
            )

        # Dataset filter
        if dataset != "all":
            ds_full = _DATASET_SHORT_TO_FULL[dataset]
            yaml_files = [f for f in yaml_files if os.sep + ds_full + os.sep in f]

    print(f"\n{'=' * 70}")
    print(f"[FILTER] {len(available_experiments)} run(s) available.")
    print(f"{'=' * 70}\n")

    # Keep only configs whose run_id (= yaml filename stem) has weights + W&B
    yaml_files = [
        f for f in yaml_files
        if os.path.splitext(os.path.basename(f))[0] in available_experiments
    ]
    print(f"[INFO] Found {len(yaml_files)} experiment config(s) after filtering")

    for cfg_path in yaml_files:
        print(f"\n{'=' * 70}")
        print(f"  Config : {os.path.relpath(cfg_path, _ROOT)}")
        print(f"{'=' * 70}")
        run_from_yaml(cfg_path, rows, finetune_names, report, available_experiments)

    csv_path = os.path.join(_RESULTS_DIR, "mlp_results.csv")
    _meta = [
        "wandb_run_id", "experiment_name", "wandb_run_name", "dataset_name", "split_id",
        "percentage", "initialization_type", "freeze_encoder",
        "max_epochs", "lr", "weight_decay", "batch_size", "hidden_dim", "dropout",
        "config_path",
    ]
    _metrics = ["kappa", "acc", "f1"]
    _extra = ["status", "error"]
    _col_order = _meta + _metrics + _extra

    df = pd.DataFrame(rows)
    if not df.empty:
        remaining = [c for c in df.columns if c not in _col_order]
        df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    # ── Upload CSV + summary metrics to W&B ──────────────────────────────────
    df_ok = df[df["status"] == "ok"] if not df.empty else df
    summary_metrics: dict = {}
    for m in _metrics:
        if m in df_ok.columns and not df_ok[m].isna().all():
            summary_metrics[f"summary/mean_{m}"] = float(df_ok[m].mean())
            summary_metrics[f"summary/std_{m}"]  = float(df_ok[m].std())

    mode_tag = mode if not config else "single"
    if wandb:
        print(f"\n[W&B] Uploading CSV and summary to project '{PROJECT}'...")
        for k, v in summary_metrics.items():
            print(f"  {k}: {v:.4f}")

        summary_run = wandb_sdk.init(
            project=PROJECT,
            entity=ENTITY,
            name=f"mlp_summary_{mode_tag}",
            job_type="evaluation_summary",
            config={"mode": mode_tag, "n_experiments": len(df_ok)},
            reinit=True,
        )
        wandb_sdk.log(summary_metrics)
        # Log full results as a W&B Table
        table_cols = [c for c in _meta + _metrics if c in df_ok.columns]
        wandb_sdk.log({"mlp/results_table": wandb_sdk.Table(dataframe=df_ok[table_cols].reset_index(drop=True))})
        # Upload CSV as an artifact
        artifact = wandb_sdk.Artifact("mlp_results", type="evaluation_results")
        artifact.add_file(csv_path, name="mlp_results.csv")
        summary_run.log_artifact(artifact)
        summary_run.finish()
        print("[W&B] Upload complete.")

    # ── Final execution report ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("[DONE] Results saved to:", csv_path)
    print(f"{'=' * 70}")
    print("\n=== EXECUTION REPORT ===")
    print(f"  Executed             : {len(report['executed'])}")
    print(f"  Missing weights      : {len(report['skipped_missing_weights'])}")
    print(f"  Other errors         : {len(report['skipped_errors'])}")
    print(f"  Collision-resolved   : {len(report['collision_resolved'])}")

    if report["skipped_missing_weights"]:
        print("\n  [MISSING WEIGHTS]")
        for name in report["skipped_missing_weights"]:
            print(f"    - {name}")

    if report["skipped_errors"]:
        print("\n  [ERRORS]")
        for entry in report["skipped_errors"]:
            print(f"    - {entry}")

    if report["collision_resolved"]:
        print("\n  [COLLISION-RESOLVED — run_id appended to W&B name]")
        for name in report["collision_resolved"]:
            print(f"    - {name}")

    print(f"{'=' * 70}")

    return csv_path


if __name__ == "__main__":
    import sys

    main(**cli_kwargs(sys.argv[1:]))
