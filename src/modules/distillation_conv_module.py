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

"""distillation_conv_module.py — Lightning module for I-JEPA → FLIM CNN distillation
using a purely convolutional projection head (next_layers variant).

Projection head: 48 → 128 → 256 → 512 → 1280 via 1×1 convolutions + AdaptiveAvgPool.
No MLP or linear layer — spatial features are projected channel-wise and then
global-average-pooled.

Two distillation strategies:

    direct
        loss = MSE( conv_proj(student_enc(x)), teacher_enc(x) )

    hybrid
        loss = α · L_SSL + (1-α) · KL( softmax(y_T/T) ∥ log_softmax(y_S/T) ) × T²

Run name convention: distillation_<dataset>_split<N>_pct<P>_next_layers_<type>

Usage (subprocess call from distillation_conv_ray.py)::

    python -m src.modules.distillation_conv_module \\
        --dataset protozoan \\
        --split 3 \\
        --percentage 100 \\
        --distillation-type direct \\
        --encoder-init trunc_normal \\
        --arch-json data/to_mateus/model/ch24_30_48_a0.5_f5/protozoan/train3/architecture.json \\
        --run-name distillation_protozoan_split3_pct100_next_layers_direct \\
        --max-epochs 100 \\
        --batch-size 32
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, List, Optional, Sequence, Union

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

from src.models.distillation import (
    KLDistillationLoss,
    MSEDistillationLoss,
    CosineDistillationLoss,
    ConvDistillationProjectionHead,
    StudentClassificationHead,
    FrozenTeacher,
    TEACHER_DIM,
    prepare_teacher_input,
)
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    load_FLIM_encoder,
    load_FLIM_encoder_from_arch_dict,
    PROTOZOAN_FLIM_ARCH,
    init_weights_trunc_normal,
    init_weights_he,
    init_weights_xavier,
)

from src.losses.lejepa_loss import SimpleSIGReg, RealSIGReg, invariance_loss

_log = logging.getLogger(__name__)

_SIGREG_TYPES      = {"simple": SimpleSIGReg, "real": RealSIGReg}
ENCODER_INITS      = ("random", "he", "xavier", "trunc_normal", "flim")
DISTILLATION_TYPES = ("direct", "direct_cosine", "hybrid")


class DistillationConvModule(pl.LightningModule):
    """Lightning module for knowledge distillation from frozen I-JEPA to FLIM CNN.

    Uses a purely convolutional projection head (next_layers variant):
        48 → 128 → 256 → 512 → 1280  (1×1 convs + BN + ReLU) → AdaptiveAvgPool2d(1)

    Two distillation strategies:

        direct
            L = MSE( conv_proj(student_enc(x)),  teacher_enc(x) )

        hybrid
            L = α · L_SSL + (1-α) · KL( softmax(y_T/T) ∥ log_softmax(y_S/T) ) × T²

    Args:
        arch_json:         Path to the FLIM architecture JSON file.
        distillation_type: ``"direct"`` or ``"hybrid"``.
        encoder_init:      ``"random"`` | ``"he"`` | ``"xavier"`` | ``"trunc_normal"``.
        alpha:             In hybrid mode: weight on L_SSL.
        temperature:       Softmax temperature T for KL divergence.
        lam_ssl:           SIGReg weight inside L_SSL for hybrid mode.
        sigreg_type:       ``"simple"`` or ``"real"`` SIGReg variant.
        in_channels:       Input image channels.
        proj_dim:          Student SSL projection head output dim (hybrid only).
        proj_hidden:       Student SSL projection head hidden dim (hybrid only).
        teacher_model_id:  HuggingFace ID for the I-JEPA teacher.
        lr:                Peak learning rate.
        weight_decay:      AdamW weight decay.
        max_epochs:        Total training epochs.
        warmup_epochs:     Linear warmup epochs.
    """

    def __init__(
        self,
        arch_json: str,
        dataset: str = "",
        distillation_type: str = "direct",
        encoder_init: str = "trunc_normal",
        flim_weights_path: Optional[str] = None,
        alpha: float = 0.7,
        temperature: float = 4.0,
        lam_ssl: float = 0.05,
        sigreg_type: str = "simple",
        in_channels: int = 3,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        num_classes: int = 9,
        teacher_model_id: str = "facebook/ijepa_vith14_1k",
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()

        if distillation_type not in DISTILLATION_TYPES:
            raise ValueError(
                f"distillation_type must be one of {DISTILLATION_TYPES}, "
                f"got '{distillation_type}'"
            )
        if encoder_init not in ENCODER_INITS:
            raise ValueError(
                f"encoder_init must be one of {ENCODER_INITS}, got '{encoder_init}'"
            )
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )

        self.save_hyperparameters()

        # ── Build student (LeJEPAFLIM backbone) ───────────────────────────
        # Protozoan special case: its FLIM weights live in ch24_32_48 but the
        # actual layer2 has 30 filters (not 32), so use PROTOZOAN_FLIM_ARCH and
        # let get_actual_channels_from_weights correct the channels from the bias files.
        if encoder_init == "flim" and dataset == "protozoan":
            arch = PROTOZOAN_FLIM_ARCH
        else:
            arch = parse_architecture(arch_json)
        if encoder_init == "flim":
            channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
            arch = override_arch_channels(arch, channels)
        else:
            channels = get_channels_from_arch(arch, in_channels)

        self.student = LeJEPAFLIMModel(
            arch=arch,
            in_channels=in_channels,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
        )
        self.student_embed_dim: int = self.student.embed_dim

        _log.info(
            "[DistillationConvModule] student_embed_dim=%d  encoder_init=%s  type=%s  num_classes=%d",
            self.student_embed_dim, encoder_init, distillation_type, num_classes,
        )

        if encoder_init == "he":
            init_weights_he(self.student.encoder)
        elif encoder_init == "xavier":
            init_weights_xavier(self.student.encoder)
        elif encoder_init == "trunc_normal":
            init_weights_trunc_normal(self.student.encoder)
        elif encoder_init == "flim":
            if dataset == "protozoan":
                load_FLIM_encoder_from_arch_dict(self.student, arch, flim_weights_path, channels)
            else:
                load_FLIM_encoder(self.student, arch_json, flim_weights_path, channels)
            _log.info("[DistillationConvModule] FLIM weights loaded from %s.", flim_weights_path)

        # ── Frozen teacher ─────────────────────────────────────────────────
        self.teacher = FrozenTeacher(model_id=teacher_model_id)

        # ── Conv projection head: 48 → 128 → 256 → 512 → 1280 + GAP ──────
        # No MLP — purely 1×1 convolutions followed by AdaptiveAvgPool2d(1)
        self.proj_kd = ConvDistillationProjectionHead(
            student_channels=self.student_embed_dim,
            teacher_dim=TEACHER_DIM,
        )

        # ── direct: MSE between conv-projected student and teacher embedding
        if distillation_type == "direct":
            self.mse_loss = MSEDistillationLoss()

        # ── direct_cosine: cosine distance between conv-projected student and teacher
        elif distillation_type == "direct_cosine":
            self.mse_loss = CosineDistillationLoss()

        # ── hybrid: KL distillation + SSL ─────────────────────────────────
        else:
            self.kd_loss = KLDistillationLoss(temperature=temperature)
            if distillation_type == "hybrid":
                self.sigreg = _SIGREG_TYPES[sigreg_type]()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _first_view(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
        return views

    @torch.no_grad()
    def _teacher_emb(self, student_view: Tensor) -> Tensor:
        teacher_input = prepare_teacher_input(student_view.float())
        return self.teacher(teacher_input)  # (B, 1280) on CPU

    # ── Forward / training ─────────────────────────────────────────────────

    def _log_embedding_stats(self, student_emb: Tensor, student_proj: Tensor, teacher_emb: Tensor, prefix: str) -> None:
        with torch.no_grad():
            t_dev = teacher_emb.to(student_proj.device)
            cos_sim = F.cosine_similarity(student_proj, t_dev, dim=-1).mean()
            self.log(f"{prefix}/cosine_sim",        cos_sim,                           on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_emb_norm",  student_emb.norm(dim=-1).mean(),   on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_proj_norm", student_proj.norm(dim=-1).mean(),  on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_emb_norm",  t_dev.norm(dim=-1).mean(),         on_step=False, on_epoch=True)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        first_view = self._first_view(views)  # (B, 3, H, W)

        feat_map     = self.student.encoder(first_view)        # (B, 48, H', W')
        student_emb  = self.student.pool(feat_map).flatten(1)  # (B, 48)   for logging
        student_proj = self.proj_kd(feat_map)                  # (B, 1280) via 1×1 convs + GAP
        teacher_emb  = self._teacher_emb(first_view)           # (B, 1280) CPU, no grad

        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)

        if self.hparams.distillation_type in ("direct", "direct_cosine"):
            loss = self.mse_loss(student_proj, teacher_emb)
            self.log("train/loss_mse", loss, prog_bar=True,  on_step=True, on_epoch=True)
            self.log("train/loss",     loss, prog_bar=False, on_step=True, on_epoch=True)

        else:
            loss_kd = self.kd_loss(student_proj, teacher_emb)
            emb, _  = self.student(views)
            sig_loss = self.sigreg(emb)
            inv_loss = invariance_loss(emb)
            loss_ssl = (
                self.hparams.lam_ssl * sig_loss
                + (1.0 - self.hparams.lam_ssl) * inv_loss
            )
            alpha = self.hparams.alpha
            loss  = alpha * loss_ssl + (1.0 - alpha) * loss_kd

            self.log("train/loss_ssl",   loss_ssl, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/sigreg",     sig_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/invariance", inv_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss_kd",    loss_kd,  prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss",       loss,     prog_bar=True,  on_step=True, on_epoch=True)

        self._log_embedding_stats(student_emb, student_proj, teacher_emb, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        first_view = self._first_view(views)

        feat_map     = self.student.encoder(first_view)
        student_emb  = self.student.pool(feat_map).flatten(1)
        student_proj = self.proj_kd(feat_map)
        teacher_emb  = self._teacher_emb(first_view)

        if self.hparams.distillation_type in ("direct", "direct_cosine"):
            loss = self.mse_loss(student_proj, teacher_emb)
            self.log("val/loss_mse", loss, prog_bar=False, on_epoch=True)
            self.log("val/loss",     loss, prog_bar=True,  on_epoch=True)
        else:
            loss_kd = self.kd_loss(student_proj, teacher_emb)
            emb, _  = self.student(views)
            sig_loss = self.sigreg(emb)
            inv_loss = invariance_loss(emb)
            loss_ssl = (
                self.hparams.lam_ssl * sig_loss
                + (1.0 - self.hparams.lam_ssl) * inv_loss
            )
            alpha = self.hparams.alpha
            loss  = alpha * loss_ssl + (1.0 - alpha) * loss_kd
            self.log("val/loss_ssl", loss_ssl, prog_bar=False, on_epoch=True)
            self.log("val/loss_kd",  loss_kd,  prog_bar=False, on_epoch=True)
            self.log("val/loss",     loss,     prog_bar=True,  on_epoch=True)

        self._log_embedding_stats(student_emb, student_proj, teacher_emb, "val")
        return loss

    # ── Optimiser ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total  = self.hparams.max_epochs    or 100
        params = list(self.student.parameters()) + list(self.proj_kd.parameters())
        optimizer = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[sched_warmup, sched_cosine],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── Standalone training entry-point ───────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Train one distillation experiment (conv projection head — next_layers variant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split",      required=True, type=int)
    p.add_argument("--percentage", required=True, type=int)
    p.add_argument("--arch-json",  required=True, help="Path to FLIM architecture.json")
    p.add_argument("--run-name",          required=True)
    p.add_argument("--distillation-type", required=True, choices=list(DISTILLATION_TYPES))
    p.add_argument("--encoder-init",      default="trunc_normal", choices=list(ENCODER_INITS))
    p.add_argument("--flim-weights-path", default=None,
                   help="Directory with FLIM weight files (conv{n}-kernels.npy, conv{n}-bias.txt). "
                        "Required when --encoder-init flim.")
    p.add_argument("--alpha",             type=float, default=0.5)
    p.add_argument("--temperature",       type=float, default=4.0)
    p.add_argument("--num-classes",       type=int,   default=9)
    p.add_argument("--lam-ssl",           type=float, default=0.05)
    p.add_argument("--sigreg-type",       default="simple", choices=["simple", "real"])
    p.add_argument("--max-epochs",        type=int,   default=100)
    p.add_argument("--warmup-epochs",     type=int,   default=10)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--weight-decay",      type=float, default=5e-2)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--image-size",        type=int,   default=200)
    p.add_argument("--num-views",         type=int,   default=4)
    p.add_argument("--output-dir",        default=None)
    p.add_argument("--wandb",             action="store_true", default=False)
    p.add_argument("--wandb-project",     default="flim-ssl")
    p.add_argument("--wandb-entity",      default="ophira-ai")
    p.add_argument("--seed",              type=int, default=42)
    return p


def _dataset_short_to_parasite_name(dataset: str) -> str:
    _MAP = {
        "eggs":      "helminth-eggs_split_2",
        "larvae":    "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }
    return _MAP[dataset]


def main() -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = _build_parser()
    args   = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir or os.path.join(
        _ROOT, "artifacts", "distillation", args.run_name
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import (
        ParasiteLejepaDataModuleSplited,
    )

    parasite_name = _dataset_short_to_parasite_name(args.dataset)
    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=parasite_name,
        split=args.split,
        percentage=args.percentage,
        image_size=args.image_size,
        V_train=args.num_views,
        V_eval=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        loader="ift_lab",
    )

    module = DistillationConvModule(
        arch_json=args.arch_json,
        dataset=args.dataset,
        distillation_type=args.distillation_type,
        encoder_init=args.encoder_init,
        flim_weights_path=args.flim_weights_path,
        alpha=args.alpha,
        temperature=args.temperature,
        lam_ssl=args.lam_ssl,
        sigreg_type=args.sigreg_type,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"

    logger_list: list = []
    if args.wandb:
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                save_dir=output_dir,
            )
            wandb_logger.log_hyperparams({
                "dataset":           args.dataset,
                "split":             args.split,
                "percentage":        args.percentage,
                "distillation_type": args.distillation_type,
                "encoder_init":      args.encoder_init,
                "lam_ssl":           args.lam_ssl,
                "alpha":             args.alpha,
                "temperature":       args.temperature,
                "arch_json":         args.arch_json,
                "teacher":           "facebook/ijepa_vith14_1k",
                "student_embed_dim": module.student_embed_dim,
                "proj_head":         "conv_next_layers",
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s — continuing without W&B.", _e)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb],
        logger=logger_list or False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )

    t0 = time.time()
    try:
        trainer.fit(module, datamodule=datamodule)
    except Exception as exc:
        _log.error("Training failed: %s", exc, exc_info=True)
        _save_metadata(args, module, output_dir, ckpt_dir, status="error", error=str(exc))
        if args.wandb:
            try:
                import wandb as _wandb
                _wandb.finish(exit_code=1)
            except Exception:
                pass
        return 1

    elapsed = time.time() - t0
    best_ckpt = checkpoint_cb.best_model_path or ""
    _log.info("Training complete in %.1fs. Best ckpt: %s", elapsed, best_ckpt)

    _save_metadata(args, module, output_dir, ckpt_dir, status="ok",
                   best_ckpt=best_ckpt, elapsed=elapsed)

    if args.wandb:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass
    return 0


def _save_metadata(
    args: Any,
    module: "DistillationConvModule",
    output_dir: str,
    ckpt_dir: str,
    status: str = "ok",
    error: str = "",
    best_ckpt: str = "",
    elapsed: float = 0.0,
) -> None:
    import subprocess as _sp
    try:
        git_sha = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "run_name":            args.run_name,
        "dataset":             args.dataset,
        "split":               args.split,
        "percentage":          args.percentage,
        "distillation_type":   args.distillation_type,
        "encoder_init":        args.encoder_init,
        "alpha":               args.alpha,
        "temperature":         args.temperature,
        "lam_ssl":             args.lam_ssl,
        "sigreg_type":         args.sigreg_type,
        "teacher_model":       "facebook/ijepa_vith14_1k",
        "teacher_embed_dim":   TEACHER_DIM,
        "student_embed_dim":   module.student_embed_dim,
        "arch_json":           args.arch_json,
        "image_size":          args.image_size,
        "max_epochs":          args.max_epochs,
        "warmup_epochs":       args.warmup_epochs,
        "batch_size":          args.batch_size,
        "lr":                  args.lr,
        "weight_decay":        args.weight_decay,
        "seed":                args.seed,
        "proj_head":           "conv_next_layers",
        "proj_head_channels":  "48->128->256->512->1280",
        "best_checkpoint":     best_ckpt,
        "checkpoint_dir":      ckpt_dir,
        "output_dir":          output_dir,
        "status":              status,
        "error":               error,
        "elapsed_s":           round(elapsed, 1),
        "git_sha":             git_sha,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    meta_path = os.path.join(output_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    _log.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    sys.exit(main())
