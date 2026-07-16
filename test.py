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

"""
Test runner for the LeGEPA Line pipeline.

Trains the FLIM encoder under the LeGEPA regime for every initialiser config,
evaluates with Accuracy / F1-score / Kappa (mean +/- std across splits),
and writes a CSV results table.

Supports:
  - Dataset-percentage training (saves weights with percentage in filename).
  - Mask flag (toggles mask application from dataset folder).
  - Sweep across all initialiser configs.

Usage:
    python test.py [OPTIONS]

    --arch_json        Path to FLIM architecture JSON.          [default: see below]
    --splits           Comma-separated split indices.           [default: 1,2,3]
    --percentages      Comma-separated data percentages.        [default: None]
    --use_mask         Apply mask from dataset folder.          [default: False]
    --max_epochs       Training epochs per run.                 [default: 100]
    --batch_size       Batch size.                              [default: 32]
    --num_workers      DataLoader workers.                      [default: 4]
    --output_csv       Path to output CSV file.                 [default: results.csv]
    --weights_dir      Directory to save weights.               [default: weights]
    --num_classes      Number of target classes.                [default: 15]
    --loader           Image loader: pil or ift_lab.            [default: ift_lab]
    --n_global         Global crop count.                       [default: 2]
    --n_local          Local crop count.                        [default: 6]
    --global_size      Global crop size.                        [default: 200]
    --gpu              GPU device index.                        [default: 0]
    --configs          Comma-separated initialiser names to run.[default: random,he,xavier,flim]
    --flim_weights_path Path to FLIM weights dir (for flim init).[default: see below]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from torchvision.transforms import v2

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modules.lejepa_line_module import LejepaLineModule, ENCODER_INITS
from src.data_modules.lejepa_line import (
    LejepaLineDataModule,
    DatasetLejepaLine,
)
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    MLPHead,
)


def _parse_args():
    p = argparse.ArgumentParser(description="LeJEPA Line test runner")
    p.add_argument("--arch_json", type=str,
                    default="data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json")
    p.add_argument("--splits", type=str, default="1,2,3")
    p.add_argument("--percentages", type=str, default="None",
                    help="Comma-separated percentages (e.g. 1,5,25,50,75,100) or None for full split.")
    p.add_argument("--use_mask", action="store_true", default=False)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_csv", type=str, default="results.csv")
    p.add_argument("--weights_dir", type=str, default="weights")
    p.add_argument("--num_classes", type=int, default=15)
    p.add_argument("--loader", type=str, default="ift_lab")
    p.add_argument("--n_global", type=int, default=2)
    p.add_argument("--n_local", type=int, default=6)
    p.add_argument("--global_size", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--configs", type=str, default="random,he,xavier,flim",
                    help="Comma-separated initialiser names to run.")
    p.add_argument("--flim_weights_path", type=str,
                    default="data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--sigreg_type", type=str, default="simple")
    p.add_argument("--eval_epochs", type=int, default=50,
                    help="Epochs for linear evaluation head training.")
    return p.parse_args()


def _build_weight_filename(encoder_init: str, split: int, percentage, use_mask: bool) -> str:
    """Build a weight filename that includes encoder_init, split, percentage, and mask info."""
    parts = [f"lejepa_line_{encoder_init}", f"split{split}"]
    if percentage is not None:
        parts.append(f"perc{int(percentage)}")
    else:
        parts.append("percFull")
    if use_mask:
        parts.append("masked")
    return "_".join(parts) + ".ckpt"


def _train_ssl(args, encoder_init: str, split: int, percentage, weights_path: str) -> str:
    """Train the LeJEPA SSL model and save the checkpoint. Returns checkpoint path."""
    flim_wp = args.flim_weights_path if encoder_init == "flim" else None

    module = LejepaLineModule(
        arch_json=args.arch_json,
        encoder_init=encoder_init,
        flim_weights_path=flim_wp,
        in_channels=3,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        lam=args.lam,
        sigreg_type=args.sigreg_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    datamodule = LejepaLineDataModule(
        split=split,
        percentage=percentage,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loader=args.loader,
        use_mask=args.use_mask,
        ssl_mode=True,
        n_global=args.n_global,
        n_local=args.n_local,
        global_size=args.global_size,
        image_size=args.global_size,
        pin_memory=True,
    )

    ckpt_path = os.path.join(args.weights_dir, weights_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.dirname(ckpt_path),
        filename=os.path.splitext(os.path.basename(ckpt_path))[0],
        monitor="val/loss",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        logger=False,
    )

    trainer.fit(module, datamodule=datamodule)

    # Return best checkpoint path
    best = checkpoint_cb.best_model_path
    if best and os.path.isfile(best):
        return best
    return ckpt_path


def _evaluate(args, ckpt_path: str, split: int, percentage) -> dict:
    """
    Evaluate a pretrained SSL encoder using a linear probe.

    Returns dict with accuracy, f1, kappa.
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    # Load SSL checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # Parse architecture to build encoder
    arch = parse_architecture(args.arch_json)
    hparams = ckpt.get("hyper_parameters", {})
    encoder_init = hparams.get("encoder_init", "random")

    if encoder_init == "flim" and args.flim_weights_path:
        channels = get_actual_channels_from_weights(args.flim_weights_path, arch, 3)
        arch = override_arch_channels(arch, channels)
    else:
        channels = get_channels_from_arch(arch, 3)

    from src.models.lejepa_flim import LeJEPAFLIMModel
    model = LeJEPAFLIMModel(arch=arch, in_channels=3)

    # Load encoder weights from checkpoint
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("model.encoder."):
            encoder_state[k[len("model.encoder."):]] = v
    if encoder_state:
        model.encoder.load_state_dict(encoder_state, strict=False)

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    embed_dim = channels[-1]

    # Build linear probe head
    head = MLPHead(embed_dim, args.num_classes, hidden_dim=256, dropout=0.3)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    head = head.to(device)

    # Prepare data (non-ssl mode)
    transform = v2.Compose([
        v2.Resize((args.global_size, args.global_size), antialias=True),
        v2.ToDtype(torch.float32, scale=False),
    ])

    train_ds = DatasetLejepaLine(
        set_name="train", split=split, percentage=percentage,
        transform=transform, loader=args.loader, use_mask=args.use_mask,
    )
    test_ds = DatasetLejepaLine(
        set_name="test", split=split, percentage=percentage,
        transform=transform, loader=args.loader, use_mask=args.use_mask,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Train linear probe
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.eval_epochs):
        head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                features = model.encoder(imgs)  # [B, C, H', W']
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            features = model.encoder(imgs)
            logits = head(features)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    return {"accuracy": acc, "f1": f1, "kappa": kappa}


def main():
    args = _parse_args()

    # Parse splits and percentages
    splits = [int(s.strip()) for s in args.splits.split(",")]
    if args.percentages.strip().lower() == "none":
        percentages = [None]
    else:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]

    # Parse initialiser configs to run
    init_configs = [c.strip() for c in args.configs.split(",")]
    for c in init_configs:
        if c not in ENCODER_INITS:
            print(f"[ERROR] Unknown encoder_init '{c}'. Must be one of {ENCODER_INITS}")
            sys.exit(1)

    os.makedirs(args.weights_dir, exist_ok=True)

    # Collect all results
    results = []

    for encoder_init in init_configs:
        for percentage in percentages:
            pct_label = str(int(percentage)) if percentage is not None else "Full"
            split_metrics = {"accuracy": [], "f1": [], "kappa": []}
            weight_files = []

            for split in splits:
                print(f"\n{'='*70}")
                print(f"  encoder_init={encoder_init}  split={split}  "
                      f"percentage={pct_label}  use_mask={args.use_mask}")
                print(f"{'='*70}")

                # Build weight filename with percentage
                wf = _build_weight_filename(encoder_init, split, percentage, args.use_mask)
                weight_files.append(wf)

                # Train SSL
                print(f"\n[TRAIN] SSL pretraining ...")
                ckpt_path = _train_ssl(args, encoder_init, split, percentage, wf)
                print(f"[TRAIN] Checkpoint saved: {ckpt_path}")

                # Evaluate
                print(f"[EVAL] Linear probe evaluation ...")
                metrics = _evaluate(args, ckpt_path, split, percentage)
                print(f"[EVAL] ACC={metrics['accuracy']:.4f}  "
                      f"F1={metrics['f1']:.4f}  Kappa={metrics['kappa']:.4f}")

                for k in split_metrics:
                    split_metrics[k].append(metrics[k])

            # Compute mean and std across splits
            row = {
                "encoder_init": encoder_init,
                "percentage": pct_label,
                "use_mask": args.use_mask,
                "acc_mean": np.mean(split_metrics["accuracy"]),
                "acc_std": np.std(split_metrics["accuracy"]),
                "f1_mean": np.mean(split_metrics["f1"]),
                "f1_std": np.std(split_metrics["f1"]),
                "kappa_mean": np.mean(split_metrics["kappa"]),
                "kappa_std": np.std(split_metrics["kappa"]),
                "weight_files": ";".join(weight_files),
            }
            results.append(row)

            print(f"\n[RESULT] {encoder_init} | perc={pct_label} | "
                  f"ACC={row['acc_mean']:.4f}+/-{row['acc_std']:.4f} | "
                  f"F1={row['f1_mean']:.4f}+/-{row['f1_std']:.4f} | "
                  f"Kappa={row['kappa_mean']:.4f}+/-{row['kappa_std']:.4f}")

    # Write CSV
    fieldnames = [
        "encoder_init", "percentage", "use_mask",
        "acc_mean", "acc_std", "f1_mean", "f1_std",
        "kappa_mean", "kappa_std", "weight_files",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Results written to: {args.output_csv}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'Initialiser':<12} {'Pct':<6} {'Mask':<6} "
          f"{'ACC':>12} {'F1':>12} {'Kappa':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['encoder_init']:<12} {r['percentage']:<6} {str(r['use_mask']):<6} "
              f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f} "
              f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f} "
              f"{r['kappa_mean']:.4f}+/-{r['kappa_std']:.4f}")


if __name__ == "__main__":
    main()
