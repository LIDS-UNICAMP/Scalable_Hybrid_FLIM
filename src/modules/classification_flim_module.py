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

"""classification_flim_module.py — Direct supervised classification: FLIM-init encoder +
2-layer Sigmoid head.

    encoder(x) -> [B, 48, H, W]
        -> AdaptiveAvgPool2d(1) -> flatten -> [B, 48]
        -> Linear(48, 24) -> Sigmoid
        -> Linear(24, n_classes) -> Softmax

No SSL objective, no teacher, no distillation — pure supervised training against
ground-truth labels. The encoder is always initialised with FLIM-estimated
kernels/bias before training (Experiment 3: "FLIM init + perceptron de
classificacao"). Because the head's forward() returns post-Softmax class
probabilities rather than logits, training minimises NLLLoss on log(probs)
instead of CrossEntropyLoss (which expects raw logits and applies its own
softmax internally).

Run name convention: classhead_<dataset>_split<N>_pct<P>_sigmoid2l

Usage (subprocess call from scripts/classification_flim_ray.py)::

    python -m src.modules.classification_flim_module \\
        --dataset protozoan \\
        --split 3 \\
        --percentage 75 \\
        --arch-json data/to_mateus/model/ch24_30_48_a0.5_f5/protozoan/train3/architecture.json \\
        --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/protozoan/train3/models \\
        --num-classes 7 \\
        --run-name classhead_protozoan_split3_pct75_sigmoid2l \\
        --max-epochs 100 \\
        --batch-size 32 \\
        --freeze-encoder   # optional: train only the head, encoder stays FLIM-frozen
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    load_FLIM_encoder,
    load_FLIM_encoder_from_arch_dict,
    PROTOZOAN_FLIM_ARCH,
    SigmoidClassificationModel,
)
from src.metrics.classification import compute_metrics

_log = logging.getLogger(__name__)

ENCODER_INITS = ("flim",)  # Exp.3 is FLIM-init only by design; see distillation_*_module.py for random/he/xavier


class ClassificationFlimModule(pl.LightningModule):
    """FLIM-init encoder + TwoLayerSigmoidHead, trained end-to-end with NLLLoss.

    Args:
        arch_json:          Path to the FLIM architecture JSON file.
        dataset:             ``"eggs"`` | ``"larvae"`` | ``"protozoan"`` — protozoan loads
                              ``PROTOZOAN_FLIM_ARCH`` (ch24_32_48 weights) instead of arch_json,
                              matching the FLIM-init convention used by the distillation modules.
        flim_weights_path:   Directory with FLIM weight files (conv{n}-kernels.npy / conv{n}-bias.txt).
                              Always required — this module has no non-FLIM init path.
        num_classes:         9 (eggs) / 2 (larvae) / 7 (protozoan).
        hidden_dim:          Head hidden size. Defaults to encoder_out_channels // 2 (48 -> 24).
        freeze_encoder:      If True, the FLIM encoder is frozen after init (requires_grad=False)
                              and only the TwoLayerSigmoidHead is trained. Mirrors the freeze/unfreeze
                              distinction already used by the MLP probe configs (configs/evaluate/mlp/).
    """

    def __init__(
        self,
        arch_json: str,
        dataset: str = "",
        flim_weights_path: Optional[str] = None,
        num_classes: int = 9,
        hidden_dim: Optional[int] = None,
        in_channels: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        seed: int = 42,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if flim_weights_path is None:
            raise ValueError(
                "flim_weights_path is required — ClassificationFlimModule always "
                "initialises the encoder with FLIM weights (Experiment 3)."
            )

        if dataset == "protozoan":
            arch = PROTOZOAN_FLIM_ARCH
        else:
            arch = parse_architecture(arch_json)
        channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
        arch = override_arch_channels(arch, channels)

        self.model = SigmoidClassificationModel(
            arch=arch, num_classes=num_classes, in_channels=in_channels, hidden_dim=hidden_dim,
        )
        self.encoder_out_channels: int = channels[-1]

        if dataset == "protozoan":
            load_FLIM_encoder_from_arch_dict(self.model, arch, flim_weights_path, channels)
        else:
            load_FLIM_encoder(self.model, arch_json, flim_weights_path, channels)

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        _log.info(
            "[ClassificationFlimModule] FLIM weights loaded from %s | encoder_out=%d | "
            "hidden=%d | num_classes=%d | freeze_encoder=%s",
            flim_weights_path, self.encoder_out_channels,
            hidden_dim if hidden_dim is not None else self.encoder_out_channels // 2,
            num_classes, freeze_encoder,
        )

        self._val_preds: List[Tensor] = []
        self._val_labels: List[Tensor] = []

    # ── Helpers ────────────────────────────────────────────────────────────

    def _first_view(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
        return views

    @staticmethod
    def _nll(probs: Tensor, labels: Tensor) -> Tensor:
        # forward() returns post-Softmax probabilities, not logits — NLLLoss on
        # log(probs) is the correct counterpart to CrossEntropyLoss here.
        return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)

    # ── Forward / training ─────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        probs = self(x)
        loss = self._nll(probs, y)

        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean()

        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_labels = []

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        probs = self(x)
        loss = self._nll(probs, y)

        self._val_preds.append(probs.argmax(dim=1).detach().cpu())
        self._val_labels.append(y.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        y_pred = torch.cat(self._val_preds)
        y_true = torch.cat(self._val_labels)
        metrics = compute_metrics(y_true, y_pred, num_classes=self.hparams.num_classes)
        self.log("val/kappa", metrics["kappa"], prog_bar=True, on_epoch=True)
        self.log("val/acc", metrics["acc"], prog_bar=False, on_epoch=True)
        self.log("val/f1", metrics["f1"], prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total = self.hparams.max_epochs or 100
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Standalone entry-point ─────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Train one FLIM-init + 2-layer Sigmoid classification experiment (Exp.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split", required=True, type=int)
    p.add_argument("--percentage", required=True, type=int)
    p.add_argument("--arch-json", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--flim-weights-path", required=True,
                   help="Directory with FLIM weight files (conv{n}-kernels.npy / conv{n}-bias.txt).")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Head hidden size. Default: encoder_out_channels // 2 (48 -> 24).")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-imagenet-norm", action="store_true", default=False,
                   help="Disable ImageNet RGB Normalize on ift_lab inputs (correct for FLIM init).")
    p.add_argument("--freeze-encoder", action="store_true", default=False,
                   help="Freeze the FLIM encoder (requires_grad=False); train only the "
                        "TwoLayerSigmoidHead. Default trains encoder+head end-to-end.")
    p.add_argument("--image-size", type=int, default=200)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="flim-ssl")
    p.add_argument("--wandb-entity", default="ophira-ai")
    p.add_argument("--seed", type=int, default=42)
    return p


def _dataset_short_to_parasite_name(dataset: str) -> str:
    return {
        "eggs": "helminth-eggs_split_2",
        "larvae": "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }[dataset]


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)

    args = _build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir or os.path.join(_ROOT, "artifacts", "classification_flim", args.run_name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited

    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(args.dataset),
        split=args.split, percentage=args.percentage,
        image_size=args.image_size, V_train=1, V_eval=1,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        loader="ift_lab",
        imagenet_norm=not args.no_imagenet_norm,
    )

    module = ClassificationFlimModule(
        arch_json=args.arch_json,
        dataset=args.dataset,
        flim_weights_path=args.flim_weights_path,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, warmup_epochs=args.warmup_epochs,
        seed=args.seed, freeze_encoder=args.freeze_encoder,
    )

    checkpoint_kappa = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best_kappa",
        monitor="val/kappa", mode="max",
        save_last=True, save_top_k=1,
    )

    logger_list: list = []
    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.run_name, save_dir=output_dir,
            )
            freeze_tag = "frozen" if args.freeze_encoder else "unfrozen"
            wandb_logger.experiment.tags = tuple(dict.fromkeys(
                list(wandb_logger.experiment.tags or ()) + ["classification_flim", "sigmoid2l", freeze_tag]
            ))
            wandb_logger.log_hyperparams({
                "dataset": args.dataset, "split": args.split,
                "percentage": args.percentage, "encoder_init": "flim",
                "num_classes": args.num_classes,
                "hidden_dim": args.hidden_dim or module.encoder_out_channels // 2,
                "arch_json": args.arch_json,
                "flim_weights_path": args.flim_weights_path,
                "head": "Linear(48,24)->Sigmoid->Linear(24,n_classes)->Softmax",
                "loss": "NLLLoss(log(softmax_probs))",
                "freeze_encoder": args.freeze_encoder,
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s", _e)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, callbacks=[checkpoint_kappa],
        logger=logger_list or False,
        log_every_n_steps=10, enable_progress_bar=True, deterministic=False,
    )

    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0:
        _log.info("[ClassificationFlimModule] Resuming from checkpoint: %s", resume_ckpt)
    else:
        resume_ckpt = None

    t0 = time.time()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except Exception as exc:
        _log.error("Training failed: %s", exc, exc_info=True)
        _save_metadata(args, module, output_dir, ckpt_dir, status="error", error=str(exc),
                       freeze_encoder=args.freeze_encoder)
        return 1

    elapsed = time.time() - t0
    _save_metadata(args, module, output_dir, ckpt_dir, status="ok",
                   best_ckpt=checkpoint_kappa.best_model_path or "",
                   best_kappa=_score(checkpoint_kappa),
                   elapsed=elapsed, freeze_encoder=args.freeze_encoder)

    if args.wandb:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass
    return 0


def _score(checkpoint_cb) -> float:
    s = getattr(checkpoint_cb, "best_model_score", None)
    try:
        return float(s.item() if hasattr(s, "item") else s)
    except (TypeError, ValueError):
        return float("nan")


def _save_metadata(args, module, output_dir, ckpt_dir,
                   status="ok", error="", best_ckpt="",
                   best_kappa=float("nan"), elapsed=0.0, freeze_encoder=False):
    import subprocess as _sp
    try:
        git_sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "run_name": args.run_name, "dataset": args.dataset,
        "split": args.split, "percentage": args.percentage,
        "encoder_init": "flim",
        "freeze_encoder": freeze_encoder,
        "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim or module.encoder_out_channels // 2,
        "head": "Linear(48,24)->Sigmoid->Linear(24,n_classes)->Softmax",
        "arch_json": args.arch_json,
        "flim_weights_path": args.flim_weights_path,
        "image_size": args.image_size, "max_epochs": args.max_epochs,
        "warmup_epochs": args.warmup_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "status": status, "error": error,
        "best_checkpoint": best_ckpt, "best_val_kappa": best_kappa,
        "elapsed_s": elapsed, "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
