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

"""distillation_module.py — Lightning module for I-JEPA → FLIM CNN knowledge distillation.

Two distillation strategies are supported:

    direct
        The FLIM CNN student learns solely from the frozen I-JEPA teacher:
            loss = MSE( proj(student_emb), teacher_emb )

    hybrid
        The student receives gradients from both its own SSL loss and the
        distillation loss against the frozen teacher:
            loss = ( loss_student * lambda_student_loss + loss_teacher ) / 2
        where loss_student is the standard LeJEPA SIGReg + invariance objective.

Usage (subprocess call from distillation_ray.py)::

    python -m src.modules.distillation_module \\
        --dataset eggs \\
        --split 1 \\
        --percentage 1 \\
        --distillation-type direct \\
        --encoder-init trunc_normal \\
        --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json \\
        --run-name distillation_eggs_split1_pct1_modeldirect \\
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
    StudentClassificationHead,
    DistillationProjectionHead,
    FrozenTeacher,
    TEACHER_DIM,
    prepare_teacher_input,
    kd_loss,
    add_distill_flags,
    resolve_distill_flags,
    derive_flim_paths,
    distill_run_tags,
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
DISTILLATION_TYPES = ("direct", "direct_cosine", "hybrid", "kd_hybrid")


class DistillationModule(pl.LightningModule):
    """Lightning module for knowledge distillation from frozen I-JEPA to FLIM CNN.

    Two distillation strategies:

        direct
            L = L_KD = KL( softmax(y_T/T) ∥ log_softmax(y_S/T) ) × T²
            Only the KL divergence between frozen teacher and student.

        hybrid
            L = α · L_SSL + (1-α) · L_KD
            Student's own SSL loss (SIGReg + invariance) combined with KD.

    In both modes gradients flow only through the student; the teacher is
    permanently frozen.

    Args:
        arch_json:         Path to the FLIM architecture JSON file.
        dataset:           Short dataset name; ``"protozoan"`` selects the built-in
                           FLIM architecture when ``encoder_init="flim"``.
        distillation_type: ``"direct"`` | ``"direct_cosine"`` | ``"hybrid"`` | ``"kd_hybrid"``.
        encoder_init:      ``"random"`` | ``"he"`` | ``"xavier"`` | ``"trunc_normal"`` | ``"flim"``.
        flim_weights_path: Directory with FLIM weights; required for ``encoder_init="flim"``.
        alpha:             In hybrid mode: weight on L_SSL.
                           ``L = alpha * L_SSL + (1-alpha) * L_KD``.
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
        fine_tune:         Build the classification heads and train on labels
                           (``kd_hybrid`` only).
        teacher_frozen:    Keep the I-JEPA teacher frozen; when False it trains
                           in its own param group at ``lr * 0.1``.
        kd_temperature:    Temperature T of the ``kd_loss`` KL term.
        kd_alpha:          Weight alpha of the ``kd_loss`` KL term.
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
        fine_tune: bool = False,
        teacher_frozen: bool = True,
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.7,
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
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")
        if fine_tune != (distillation_type == "kd_hybrid"):
            raise ValueError(
                f"fine_tune=True and distillation_type='kd_hybrid' must be set "
                f"together; got fine_tune={fine_tune}, "
                f"distillation_type='{distillation_type}'"
            )

        self.save_hyperparameters()

        # ── Build student (LeJEPAFLIM backbone) ───────────────────────────
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
            "[DistillationModule] student_embed_dim=%d  encoder_init=%s  type=%s  num_classes=%d",
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
            _log.info("[DistillationModule] FLIM weights loaded from %s.", flim_weights_path)

        # ── Teacher (frozen unless --teacher_unfrozen) ─────────────────────
        self.teacher = FrozenTeacher(model_id=teacher_model_id, frozen=teacher_frozen)

        # ── Projection head: spatial feature maps → teacher_dim (for MSE / L_KD) ───
        # Encoder outputs [B, 48, 24, 24]; partial pool to 6×6 → 1728 flat → 1280 (going down)
        self.proj_kd = DistillationProjectionHead(
            student_channels=self.student_embed_dim,
            pool_size=6,
            teacher_dim=TEACHER_DIM,
        )

        # ── kd_hybrid: class logits on both sides (I-JEPA has no classifier,
        # so teacher_cls_head is the linear probe trained by the CE below) ──
        if fine_tune:
            self.cls_head = StudentClassificationHead(TEACHER_DIM, num_classes)
            self.teacher_cls_head = StudentClassificationHead(TEACHER_DIM, num_classes)

        # ── direct: MSE between student projection and teacher embedding ────
        if distillation_type == "direct":
            self.mse_loss = MSEDistillationLoss()

        # ── direct_cosine: same, but 1 - cosine instead of MSE ─────────────
        elif distillation_type == "direct_cosine":
            self.mse_loss = CosineDistillationLoss()

        # ── hybrid: KL distillation + SSL ─────────────────────────────────
        elif distillation_type == "hybrid":
            self.kd_loss = KLDistillationLoss(temperature=temperature)
            self.sigreg = _SIGREG_TYPES[sigreg_type]()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _first_view(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        """Extract the first augmented view from a multi-view batch."""
        if isinstance(views, (list, tuple)):
            return views[0]
        # views: [B, V, C, H, W] or [V, B, C, H, W]
        if views.ndim == 5:
            return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
        return views

    def _teacher_emb(self, student_view: Tensor) -> Tensor:
        """Compute teacher embeddings from a student view.

        Resizes from student image size to 224×224; the views are already
        ImageNet-normalised (output of _build_aug/_build_test). Grad is kept
        only when the teacher is unfrozen, otherwise it would be built and
        thrown away every step.
        """
        with torch.set_grad_enabled(not self.hparams.teacher_frozen):
            teacher_input = prepare_teacher_input(student_view.float())
            return self.teacher(teacher_input)  # (B, 1280), CPU while frozen

    # ── Forward / training ─────────────────────────────────────────────────

    def _log_embedding_stats(self, student_emb: Tensor, student_proj: Tensor, teacher_emb: Tensor, prefix: str) -> None:
        """Log norms and cosine similarity between student projection and teacher."""
        with torch.no_grad():
            t_dev = teacher_emb.to(student_proj.device)
            cos_sim = F.cosine_similarity(student_proj, t_dev, dim=-1).mean()
            self.log(f"{prefix}/cosine_sim",       cos_sim,                           on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_emb_norm", student_emb.norm(dim=-1).mean(),   on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_proj_norm",student_proj.norm(dim=-1).mean(),  on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_emb_norm", t_dev.norm(dim=-1).mean(),         on_step=False, on_epoch=True)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        first_view = self._first_view(views)  # (B, 3, H, W)

        feat_map     = self.student.encoder(first_view)        # (B, 48, H', W')
        student_emb  = self.student.pool(feat_map).flatten(1)  # (B, 48)   for logging
        student_proj = self.proj_kd(feat_map)                  # (B, 1280) 1728→1280
        teacher_emb  = self._teacher_emb(first_view)           # (B, 1280) CPU, no grad

        # ── LR logged every step so warmup curve is visible ───────────────
        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)

        if self.hparams.distillation_type == "kd_hybrid":
            # ── CE + KL on class logits; teacher probe learns the labels too
            logits_s   = self.cls_head(student_proj)
            logits_t   = self.teacher_cls_head(teacher_emb.to(student_proj.device))
            teacher_ce = F.cross_entropy(logits_t, y)
            loss = kd_loss(logits_s, logits_t.detach(), y,
                           T=self.hparams.kd_temperature,
                           alpha=self.hparams.kd_alpha) + teacher_ce  # + teacher_ce treina teacher_cls_head
            self.log("train/teacher_ce", teacher_ce, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/kd_acc", (logits_s.argmax(1) == y).float().mean(),
                     prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        elif self.hparams.distillation_type in ("direct", "direct_cosine"):
            # ── MSE / 1-cosine between projected student embedding and teacher
            loss = self.mse_loss(student_proj, teacher_emb)
            self.log("train/loss_mse", loss, prog_bar=True,  on_step=True, on_epoch=True)
            self.log("train/loss",     loss, prog_bar=False, on_step=True, on_epoch=True)

        else:
            # ── hybrid: L = α·L_SSL + (1-α)·KL(teacher‖student) ──────────
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

        if self.hparams.distillation_type == "kd_hybrid":
            logits_s   = self.cls_head(student_proj)
            logits_t   = self.teacher_cls_head(teacher_emb.to(student_proj.device))
            teacher_ce = F.cross_entropy(logits_t, y)
            loss = kd_loss(logits_s, logits_t.detach(), y,
                           T=self.hparams.kd_temperature,
                           alpha=self.hparams.kd_alpha)
            self.log("val/teacher_ce", teacher_ce, prog_bar=False, on_epoch=True)
            self.log("val/kd_acc", (logits_s.argmax(1) == y).float().mean(),
                     prog_bar=False, on_epoch=True)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        elif self.hparams.distillation_type in ("direct", "direct_cosine"):
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
        if self.hparams.fine_tune:
            params += list(self.cls_head.parameters()) + list(self.teacher_cls_head.parameters())
        groups = [{"params": params}]
        if not self.hparams.teacher_frozen:
            # Fine-tuning a ViT-H at the student LR destroys the teacher.
            groups.append({"params": list(self.teacher.parameters()),
                           "lr": self.hparams.lr * 0.1})
        optimizer = AdamW(groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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


# ── Standalone training entry-point (called from distillation_ray.py) ─────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Train one distillation experiment (student from frozen I-JEPA teacher).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset",   required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split",     required=True, type=int)
    p.add_argument("--percentage",required=True, type=int)
    # Architecture
    p.add_argument("--arch-json", default=None, help="Path to FLIM architecture.json")
    # Experiment
    p.add_argument("--run-name",          required=True)
    p.add_argument("--distillation-type", default="direct", choices=list(DISTILLATION_TYPES))
    p.add_argument("--encoder-init",      default="trunc_normal", choices=list(ENCODER_INITS))
    p.add_argument("--flim-weights-path", default=None,
                   help="Directory with FLIM weight files. Required when --encoder-init flim.")
    p.add_argument("--alpha",             type=float, default=0.5,
                   help="In hybrid mode: weight on L_SSL. L = alpha*L_SSL + (1-alpha)*L_KD")
    p.add_argument("--temperature",       type=float, default=4.0,
                   help="Softmax temperature T for KL divergence.")
    p.add_argument("--num-classes",        type=int,   default=9,
                   help="Number of target classes for CE loss in direct (CDD) mode.")
    p.add_argument("--lam-ssl",           type=float, default=0.05)
    p.add_argument("--sigreg-type",       default="simple", choices=["simple", "real"])
    # Training hyper-params
    p.add_argument("--max-epochs",  type=int,   default=100)
    p.add_argument("--warmup-epochs",type=int,  default=10)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--weight-decay",type=float, default=5e-2)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--image-size",  type=int,   default=200)
    p.add_argument("--num-views",   type=int,   default=4,
                   help="Number of augmented views per sample (multi-view dataloader).")
    # Infra
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: artifacts/distillation/<run-name>)")
    p.add_argument("--wandb",      action="store_true", default=False)
    p.add_argument("--wandb-project", default="flim-ssl")
    p.add_argument("--wandb-entity",  default="ophira-ai")
    p.add_argument("--seed",          type=int, default=42)
    add_distill_flags(p)
    return p


def _dataset_short_to_parasite_name(dataset: str) -> str:
    _MAP = {
        "eggs":      "helminth-eggs_split_2",
        "larvae":    "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }
    return _MAP[dataset]


def main() -> int:
    """Entry point for subprocess-based distillation training.

    Returns exit code 0 on success, 1 on any error.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = _build_parser()
    args   = parser.parse_args()

    # The new flags win over --encoder-init/--distillation-type when used.
    derive_flim_paths(args)
    _overrides = resolve_distill_flags(args)

    pl.seed_everything(args.seed, workers=True)

    # ── Resolve output directory ───────────────────────────────────────────
    output_dir = args.output_dir or os.path.join(
        _ROOT, "artifacts", "distillation", args.run_name
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Dataset / Datamodule ───────────────────────────────────────────────
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

    # ── Module ────────────────────────────────────────────────────────────
    _module_kwargs = dict(
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
    _module_kwargs.update(_overrides)
    module = DistillationModule(**_module_kwargs)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    # ── Logger ────────────────────────────────────────────────────────────
    # WANDB_CONSOLE=off prevents W&B from trying to capture stdout/stderr,
    # which conflicts with Ray's capture_output=True subprocess mode.
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
            wandb_logger.experiment.tags = tuple(dict.fromkeys(
                list(wandb_logger.experiment.tags or ()) + distill_run_tags(module.hparams)
            ))
            wandb_logger.log_hyperparams({
                "dataset":           args.dataset,
                "split":             args.split,
                "percentage":        args.percentage,
                "distillation_type": module.hparams.distillation_type,
                "encoder_init":      module.hparams.encoder_init,
                "fine_tune":         module.hparams.fine_tune,
                "teacher_frozen":    module.hparams.teacher_frozen,
                "kd_temperature":    module.hparams.kd_temperature,
                "kd_alpha":          module.hparams.kd_alpha,
                "lam_ssl":           args.lam_ssl,
                "alpha":             args.alpha,
                "temperature":       args.temperature,
                "arch_json":         args.arch_json,
                "teacher":           "facebook/ijepa_vith14_1k",
                "student_embed_dim": module.student_embed_dim,
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s — continuing without W&B.", _e)

    # ── Trainer ───────────────────────────────────────────────────────────
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
                import wandb as _wandb  # noqa: PLC0415
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
            import wandb as _wandb  # noqa: PLC0415
            _wandb.finish()
        except Exception:
            pass
    return 0


def _save_metadata(
    args: Any,
    module: "DistillationModule",
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
        "distillation_type":   module.hparams.distillation_type,
        "encoder_init":        module.hparams.encoder_init,
        "flim_weights_path":   args.flim_weights_path,
        "fine_tune":           module.hparams.fine_tune,
        "teacher_frozen":      module.hparams.teacher_frozen,
        "kd_temperature":      module.hparams.kd_temperature,
        "kd_alpha":            module.hparams.kd_alpha,
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
        "projection_used_for_kd": True,
        "svm_embedding":       "raw_student_encode",
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
