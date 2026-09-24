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
"""distillation_twolayer_module.py — Lightning module for I-JEPA → FLIM CNN distillation
using a two-layer 1×1 conv projection head (~402k total params).

Projection head: 48 → 256 → 1280 via two Conv2d(1×1) + BN2d + GELU + AdaptiveAvgPool.
Intermediate bottleneck at 256 channels before jumping to teacher dimensionality.

Run name convention: distillation_<dataset>_split<N>_pct<P>_2l_1x1_BN2d_256_1280

Usage (subprocess call from distillation_conv_ray.py with --proj-type 2l_1x1_bn2d_256_1280)::

    python -m src.modules.distillation_twolayer_module \\
        --dataset eggs \\
        --split 1 \\
        --percentage 100 \\
        --distillation-type direct \\
        --encoder-init trunc_normal \\
        --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json \\
        --run-name distillation_eggs_split1_pct100_2l_1x1_BN2d_256_1280 \\
        --max-epochs 100 \\
        --batch-size 32
"""
from __future__ import annotations

import copy
import gc
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

from src.models.distillation import (
    KLDistillationLoss,
    StudentClassificationHead,
    add_distill_flags,
    add_student_flags,
    add_queue_flags,
    add_runtime_flags,
    apply_cpu_budget,
    run_grid,
    resolve_student,
    resolve_distill_flags,
    derive_flim_paths,
    distill_run_tags,
    kd_loss,
    MSEDistillationLoss,
    CosineDistillationLoss,
    TwoLayer1x1ConvBN2dDistillationProjectionHead,
    FrozenTeacher,
    FrozenTeacherCheckpointMixin,
    KnnKappaProbeMixin,
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
DISTILLATION_TYPES = ("direct", "direct_cosine", "hybrid", "kd_hybrid")

_PROJ_LABEL = "Conv1x1(48→256→1280)+BN2d+GELU×2"


class DistillationTwoLayerModule(FrozenTeacherCheckpointMixin, KnnKappaProbeMixin, pl.LightningModule):
    """Lightning module for knowledge distillation using a two-layer 1×1 conv projection.

    Projection head: Conv2d(48→256, k=1×1) + BN2d + GELU →
                     Conv2d(256→1280, k=1×1) + BN2d + GELU → GAP → [B, 1280]

    ~402k total parameters (encoder 59k + head 343k).

    Two distillation strategies:
        direct:  loss = MSE( proj(encoder(x)), teacher(x) )
        hybrid:  loss = α·L_SSL + (1-α)·KL

    Run name convention: distillation_<dataset>_split<N>_pct<P>_2l_1x1_BN2d_256_1280
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
        freeze_encoder: bool = False,
        fine_tune: bool = False,
        teacher_frozen: bool = True,
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.7,
        teacher_imagenet_norm: bool = False,
        knn_probe: str = "encoder",
        knn_train_subsample: int = 3000,
        knn_every_n_epochs: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if distillation_type not in DISTILLATION_TYPES:
            raise ValueError(f"distillation_type must be one of {DISTILLATION_TYPES}")
        if encoder_init not in ENCODER_INITS:
            raise ValueError(f"encoder_init must be one of {ENCODER_INITS}")
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")
        if knn_probe not in ("encoder", "projection"):
            raise ValueError("knn_probe must be 'encoder' or 'projection'")
        if fine_tune != (distillation_type == "kd_hybrid"):
            raise ValueError(
                f"fine_tune=True and distillation_type='kd_hybrid' must be set "
                f"together; got fine_tune={fine_tune}, "
                f"distillation_type='{distillation_type}'"
            )

        self.save_hyperparameters()

        # ── Student ────────────────────────────────────────────────────────
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
            arch=arch, in_channels=in_channels,
            proj_dim=proj_dim, proj_hidden=proj_hidden,
        )
        self.student_embed_dim: int = self.student.embed_dim

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
            _log.info("[DistillationTwoLayerModule] FLIM weights loaded from %s.", flim_weights_path)

        # ── Frozen teacher ─────────────────────────────────────────────────
        self.teacher = FrozenTeacher(model_id=teacher_model_id, frozen=teacher_frozen)

        # ── Two-layer 1×1 projection head: 48 → 256 → 1280 ───────────────
        self.proj_kd = TwoLayer1x1ConvBN2dDistillationProjectionHead(
            student_channels=self.student_embed_dim,
            teacher_dim=TEACHER_DIM,
        )

        # I-JEPA não tem classificador: teacher_cls_head é o probe linear que
        # produz logits_t, treinado pelo CE do próprio passo.
        if fine_tune:
            self.cls_head         = StudentClassificationHead(TEACHER_DIM, num_classes)
            self.teacher_cls_head = StudentClassificationHead(TEACHER_DIM, num_classes)

        _log.info(
            "[DistillationTwoLayerModule] embed_dim=%d  init=%s  type=%s  proj=%s",
            self.student_embed_dim, encoder_init, distillation_type, _PROJ_LABEL,
        )

        if distillation_type == "direct":
            self.mse_loss = MSEDistillationLoss()
        elif distillation_type == "direct_cosine":
            self.mse_loss = CosineDistillationLoss()
        elif distillation_type == "hybrid":
            self.kd_loss = KLDistillationLoss(temperature=temperature)
            self.sigreg  = _SIGREG_TYPES[sigreg_type]()

        # ── Frozen-encoder mode: only the projection head trains ────────────
        # The FLIM encoder is kept in eval() with requires_grad=False, so its
        # 48d output is constant across epochs — that is why the kNN probe runs
        # on the trainable 1280d projection for this variant (knn_probe).
        if freeze_encoder:
            for p in self.student.parameters():
                p.requires_grad = False
            self.student.eval()
            _log.info("[DistillationTwoLayerModule] FLIM encoder FROZEN — only proj_kd trains.")

        # ImageNet stats for normalising the teacher input when the student is
        # fed raw LAB[0,1] (--teacher-imagenet-norm). Non-persistent: stays out
        # of the checkpoint state_dict.
        self.register_buffer(
            "_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        """Keep the FLIM encoder in eval mode when it is frozen."""
        super().train(mode)
        if getattr(self.hparams, "freeze_encoder", False):
            self.student.eval()
        return self

    def _first_view(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
        return views

    def _teacher_emb(self, student_view: Tensor) -> Tensor:
        # Um @torch.no_grad() fixo cortaria o gradiente do teacher descongelado.
        with torch.set_grad_enabled(not self.hparams.teacher_frozen):
            x = prepare_teacher_input(student_view.float())
            # Decoupled normalisation: the FLIM student may receive raw LAB[0,1],
            # but I-JEPA expects ImageNet stats. Apply them here only for the teacher
            # (matches the svm_ijepa baseline, which feeds I-JEPA LAB + ImageNet norm).
            if getattr(self.hparams, "teacher_imagenet_norm", False):
                x = (x - self._imagenet_mean) / self._imagenet_std
            return self.teacher(x)

    def _log_embedding_stats(self, student_emb, student_proj, teacher_emb, prefix):
        with torch.no_grad():
            t_dev = teacher_emb.to(student_proj.device)
            cos_sim = F.cosine_similarity(student_proj, t_dev, dim=-1).mean()
            self.log(f"{prefix}/cosine_sim",        cos_sim,                          on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_emb_norm",  student_emb.norm(dim=-1).mean(),  on_step=False, on_epoch=True)
            self.log(f"{prefix}/student_proj_norm", student_proj.norm(dim=-1).mean(), on_step=False, on_epoch=True)
            self.log(f"{prefix}/teacher_emb_norm",  t_dev.norm(dim=-1).mean(),        on_step=False, on_epoch=True)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y     = batch
        first_view   = self._first_view(views)
        feat_map     = self.student.encoder(first_view)
        student_emb  = self.student.pool(feat_map).flatten(1)
        student_proj = self.proj_kd(feat_map)
        teacher_emb  = self._teacher_emb(first_view)

        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)

        if self.hparams.distillation_type == "kd_hybrid":
            logits_s   = self.cls_head(student_proj)
            logits_t   = self.teacher_cls_head(teacher_emb.to(student_proj.device))
            teacher_ce = F.cross_entropy(logits_t, y)
            loss       = kd_loss(logits_s, logits_t.detach(), y,
                                 T=self.hparams.kd_temperature,
                                 alpha=self.hparams.kd_alpha) + teacher_ce  # + teacher_ce treina teacher_cls_head
            self.log("train/teacher_ce", teacher_ce, on_step=True, on_epoch=True)
            self.log("train/kd_acc", (logits_s.argmax(1) == y).float().mean(),
                     on_step=True, on_epoch=True)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        elif self.hparams.distillation_type in ("direct", "direct_cosine"):
            loss = self.mse_loss(student_proj, teacher_emb)
            self.log("train/loss_mse", loss, prog_bar=True,  on_step=True, on_epoch=True)
            self.log("train/loss",     loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            loss_kd  = self.kd_loss(student_proj, teacher_emb)
            emb, _   = self.student(views)
            sig_loss = self.sigreg(emb)
            inv_loss = invariance_loss(emb)
            loss_ssl = self.hparams.lam_ssl * sig_loss + (1.0 - self.hparams.lam_ssl) * inv_loss
            alpha    = self.hparams.alpha
            loss     = alpha * loss_ssl + (1.0 - alpha) * loss_kd
            self.log("train/loss_ssl",   loss_ssl, on_step=True, on_epoch=True)
            self.log("train/loss_kd",    loss_kd,  on_step=True, on_epoch=True)
            self.log("train/loss",       loss,     prog_bar=True, on_step=True, on_epoch=True)

        self._log_embedding_stats(student_emb, student_proj, teacher_emb, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y     = batch
        first_view   = self._first_view(views)
        feat_map     = self.student.encoder(first_view)
        student_emb  = self.student.pool(feat_map).flatten(1)
        student_proj = self.proj_kd(feat_map)
        teacher_emb  = self._teacher_emb(first_view)

        if self.hparams.distillation_type == "kd_hybrid":
            logits_s   = self.cls_head(student_proj)
            logits_t   = self.teacher_cls_head(teacher_emb.to(student_proj.device))
            teacher_ce = F.cross_entropy(logits_t, y)
            loss       = kd_loss(logits_s, logits_t.detach(), y,
                                 T=self.hparams.kd_temperature,
                                 alpha=self.hparams.kd_alpha)
            self.log("val/teacher_ce", teacher_ce, on_epoch=True)
            self.log("val/kd_acc", (logits_s.argmax(1) == y).float().mean(), on_epoch=True)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        elif self.hparams.distillation_type in ("direct", "direct_cosine"):
            loss = self.mse_loss(student_proj, teacher_emb)
            self.log("val/loss_mse", loss, on_epoch=True)
            self.log("val/loss",     loss, prog_bar=True, on_epoch=True)
        else:
            loss_kd  = self.kd_loss(student_proj, teacher_emb)
            emb, _   = self.student(views)
            sig_loss = self.sigreg(emb)
            inv_loss = invariance_loss(emb)
            loss_ssl = self.hparams.lam_ssl * sig_loss + (1.0 - self.hparams.lam_ssl) * inv_loss
            alpha    = self.hparams.alpha
            loss     = alpha * loss_ssl + (1.0 - alpha) * loss_kd
            self.log("val/loss_ssl", loss_ssl, on_epoch=True)
            self.log("val/loss_kd",  loss_kd,  on_epoch=True)
            self.log("val/loss",     loss,     prog_bar=True, on_epoch=True)

        self._log_embedding_stats(student_emb, student_proj, teacher_emb, "val")
        return loss

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total  = self.hparams.max_epochs    or 100
        # Frozen encoder params have requires_grad=False → excluded; in frozen
        # mode this leaves only proj_kd ("a única que treina é a projetora").
        params = [p for p in self.student.parameters() if p.requires_grad] \
                 + list(self.proj_kd.parameters())
        if self.hparams.fine_tune:
            params += list(self.cls_head.parameters()) + list(self.teacher_cls_head.parameters())
        groups = [{"params": params}]
        if not self.hparams.teacher_frozen:
            # LR do student destruiria um ViT-H em fine-tune.
            groups.append({"params": list(self.teacher.parameters()),
                           "lr": self.hparams.lr * 0.1})
        optimizer    = AdamW(groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler    = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Standalone entry-point ─────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Train one two-layer 1×1 conv distillation experiment (48→256→1280).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",            required=True, choices=["eggs", "larvae", "protozoan", "all"],
                   help="'all' roda os tres datasets em serie, no mesmo processo.")
    p.add_argument("--split",              required=True, type=str,
                   help="Split, ou lista CSV de splits (ex.: 1,2,3).")
    p.add_argument("--percentage",         required=True, type=str,
                   help="Percentual, ou lista CSV de percentuais (ex.: 1,5,25,50,75,100).")
    p.add_argument("--arch-json",          default=None,
                   help="Derivado de --dataset/--split quando ausente.")
    p.add_argument("--run-name",           required=True)
    p.add_argument("--distillation-type",  default="direct", choices=list(DISTILLATION_TYPES))
    p.add_argument("--encoder-init",       default="trunc_normal", choices=list(ENCODER_INITS))
    p.add_argument("--flim-weights-path",  default=None,
                   help="Directory with FLIM weight files. Required when --encoder-init flim.")
    p.add_argument("--alpha",              type=float, default=0.5)
    p.add_argument("--temperature",        type=float, default=4.0)
    p.add_argument("--num-classes",        type=int,   default=9,
                   help="IGNORADA: num_classes e resolvido por dataset (scripts/constants.py NUM_CLASSES). "
                        "Mantida so para nao quebrar launchers que ainda a passam.")
    p.add_argument("--lam-ssl",            type=float, default=0.05)
    p.add_argument("--sigreg-type",        default="simple", choices=["simple", "real"])
    p.add_argument("--max-epochs",         type=int,   default=100)
    p.add_argument("--warmup-epochs",      type=int,   default=10)
    p.add_argument("--batch-size",         type=int,   default=32)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=5e-2)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--no-imagenet-norm",   action="store_true", default=False,
                   help="Disable ImageNet RGB Normalize on ift_lab inputs (use for FLIM init: input stays LAB[0,1]).")
    p.add_argument("--freeze-encoder",     action="store_true", default=False,
                   help="Freeze the FLIM encoder (eval + requires_grad=False); only proj_kd trains.")
    p.add_argument("--teacher-imagenet-norm", action="store_true", default=False,
                   help="Apply ImageNet Normalize to the teacher input only (student stays LAB[0,1]). "
                        "Use together with --no-imagenet-norm.")
    p.add_argument("--knn-probe",          default="encoder", choices=["encoder", "projection"],
                   help="Embedding for the val/knn_kappa probe & checkpoint selection: "
                        "'encoder'=48d student.encode (default), "
                        "'projection'=1280d proj_kd (use when --freeze-encoder).")
    p.add_argument("--knn-train-subsample", type=int, default=3000,
                   help="Memory-bank subsample size for the kNN probe (seeded).")
    p.add_argument("--knn-every-n-epochs", type=int, default=1,
                   help="Run the kNN probe every N validation epochs.")
    p.add_argument("--image-size",         type=int,   default=200)
    p.add_argument("--num-views",          type=int,   default=4)
    p.add_argument("--output-dir",         default=None)
    p.add_argument("--proj-type",          default="",
                   help="proj_type label (ex.: 2l_1x1_init_flim_256_1280) usado como tag/sinal no W&B "
                        "e gravado no run_metadata.json.")
    p.add_argument("--wandb",              action="store_true", default=False)
    p.add_argument("--wandb-project",      default="flim-ssl")
    p.add_argument("--wandb-entity",       default="ophira-ai")
    p.add_argument("--seed",               type=int, default=42)
    add_distill_flags(p)
    add_student_flags(p)
    add_queue_flags(p)
    add_runtime_flags(p)
    return p


def _csv_ints(value: str) -> list:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _dataset_short_to_parasite_name(dataset: str) -> str:
    return {
        "eggs":      "helminth-eggs_split_2",
        "larvae":    "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }[dataset]


def _run_one(args, dataset: str, split: int, pct: int, num_classes: int,
             idx: int, total: int) -> int:
    """Treina UMA combinacao (dataset, split, percentage). Retorna 0 ok / 1 erro."""
    _log.info("[%d/%d] %s split=%d pct=%d num_classes=%d",
              idx, total, dataset, split, pct, num_classes)

    # Namespace proprio da iteracao: derive_flim_paths so preenche campo vazio,
    # entao reusar o mesmo args congelaria os caminhos do primeiro dataset.
    a = copy.copy(args)
    a.dataset, a.split, a.percentage = dataset, split, pct
    a.arch_json, a.flim_weights_path = args.arch_json, args.flim_weights_path
    a.run_name = f"{args.run_name}_{dataset}_s{split}_p{pct}"

    # As flags novas vencem as antigas quando alguma delas é usada.
    derive_flim_paths(a)
    _overrides = resolve_distill_flags(a)
    # Reseeda a cada combinacao: sem isso a iteracao k herda o RNG da k-1.
    pl.seed_everything(a.seed, workers=True)

    # Com varias combinacoes, --output-dir precisa aninhar o run_name derivado,
    # senao todas compartilham checkpoints/ e a seguinte retoma a anterior.
    output_dir = (os.path.join(a.output_dir, a.run_name) if a.output_dir
                  else os.path.join(_ROOT, "artifacts", "distillation", a.run_name))
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited

    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(a.dataset),
        split=a.split, percentage=a.percentage,
        image_size=a.image_size, V_train=a.num_views, V_eval=1,
        batch_size=a.batch_size, num_workers=a.num_workers,
        pin_memory=True, persistent_workers=(a.num_workers > 0),
        loader="ift_lab",
        imagenet_norm=not a.no_imagenet_norm,
    )

    _module_kwargs = dict(
        arch_json=a.arch_json,
        dataset=a.dataset,
        distillation_type=a.distillation_type,
        encoder_init=a.encoder_init,
        flim_weights_path=a.flim_weights_path,
        alpha=a.alpha, temperature=a.temperature,
        lam_ssl=a.lam_ssl, sigreg_type=a.sigreg_type,
        num_classes=num_classes, lr=a.lr,
        weight_decay=a.weight_decay, max_epochs=a.max_epochs,
        warmup_epochs=a.warmup_epochs,
        freeze_encoder=a.freeze_encoder,
        teacher_imagenet_norm=a.teacher_imagenet_norm,
        knn_probe=a.knn_probe,
        knn_train_subsample=a.knn_train_subsample,
        knn_every_n_epochs=a.knn_every_n_epochs,
        seed=a.seed,
    )
    _module_kwargs.update(_overrides)
    module = DistillationTwoLayerModule(**_module_kwargs)

    # Two checkpoints: PRIMARY by val/knn_kappa (the MSE val/loss is decoupled
    # from / inverted w.r.t. downstream κ — minimising it collapses the encoder),
    # plus a best-by-loss fallback. Both are evaluated downstream by the SVM.
    checkpoint_knn = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_knn_kappa",
        monitor="val/knn_kappa", mode="max",
        save_last=False, save_top_k=1, save_weights_only=True,
    )
    checkpoint_loss = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_loss",
        monitor="val/loss", mode="min",
        save_last=False, save_top_k=1, save_weights_only=True,
    )

    logger_list: list = []
    if a.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        try:
            wandb_logger = WandbLogger(
                project=a.wandb_project, entity=a.wandb_entity,
                name=a.run_name, save_dir=output_dir,
            )
            # W&B tags sinalizam o tipo do experimento de relance: o proj_type
            # (ex.: "2l_1x1_init_flim_256_1280"), além de "frozen"/"no_imagenet_norm".
            _tags = list(wandb_logger.experiment.tags or ())
            if a.proj_type:
                _tags.append(a.proj_type)
            if a.freeze_encoder:
                _tags.append("frozen")
            if a.no_imagenet_norm:
                _tags.append("no_imagenet_norm")
            _tags += distill_run_tags(module.hparams)
            if _tags:
                wandb_logger.experiment.tags = tuple(dict.fromkeys(_tags))
            wandb_logger.log_hyperparams({
                "dataset": a.dataset, "split": a.split,
                "percentage": a.percentage,
                "distillation_type": module.hparams.distillation_type,
                "encoder_init": module.hparams.encoder_init,
                "fine_tune": module.hparams.fine_tune,
                "teacher_frozen": module.hparams.teacher_frozen,
                "kd_temperature": module.hparams.kd_temperature,
                "kd_alpha": module.hparams.kd_alpha,
                "no_imagenet_norm": a.no_imagenet_norm,
                "freeze_encoder": a.freeze_encoder,
                "teacher_imagenet_norm": a.teacher_imagenet_norm,
                "knn_probe": a.knn_probe,
                "proj_type": a.proj_type,
                "arch_json": a.arch_json,
                "teacher": "facebook/ijepa_vith14_1k",
                "proj_head": _PROJ_LABEL,
                "student_embed_dim": module.student_embed_dim,
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s", _e)

    trainer = pl.Trainer(
        max_epochs=a.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # Sem --gpu, devices=1 = a primeira GPU visivel; com --gpu, essa GPU e so ela.
        devices=1 if a.gpu is None else [a.gpu],
        callbacks=[checkpoint_knn, checkpoint_loss],
        logger=logger_list or False,
        log_every_n_steps=10, enable_progress_bar=True, deterministic=False,
    )

    # ── Resume support ──────────────────────────────────────────────────────
    # So acha last.ckpt de treinos antigos: hoje os callbacks gravam
    # save_weights_only e nao emitem save_last. Sem o arquivo, treina do zero.
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0:
        _log.info("[DistillationTwoLayerModule] Resuming from checkpoint: %s", resume_ckpt)
    else:
        resume_ckpt = None

    t0 = time.time()
    rc = 0
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
        _save_metadata(a, module, output_dir, ckpt_dir, status="ok",
                       best_ckpt=checkpoint_knn.best_model_path or "",
                       best_loss_ckpt=checkpoint_loss.best_model_path or "",
                       best_knn_kappa=_score(checkpoint_knn),
                       best_val_loss=_score(checkpoint_loss),
                       elapsed=time.time() - t0)
    except Exception as exc:
        # A combinacao morre sozinha; o laco continua nas outras.
        _log.error("Training failed (%s split=%d pct=%d): %s", dataset, split, pct, exc,
                   exc_info=True)
        _save_metadata(a, module, output_dir, ckpt_dir, status="error", error=str(exc))
        rc = 1
    finally:
        if a.wandb:
            # Tambem no caminho de erro: run pendurada faria a proxima iteracao
            # se anexar a ela.
            try:
                import wandb as _wandb
                _wandb.finish()
            except Exception:
                pass
        # Cada iteracao carrega um teacher de ~2.4 GB — sem soltar aqui a 3a/4a
        # combinacao da OOM.
        del trainer, module, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rc


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)

    args = _build_parser().parse_args()
    resolve_student(args, "distillation_twolayer_module")
    apply_cpu_budget(args)
    datasets = ["eggs", "larvae", "protozoan"] if args.dataset == "all" else [args.dataset]
    splits   = _csv_ints(args.split)
    pcts     = _csv_ints(args.percentage)
    total    = len(datasets) * len(splits) * len(pcts)
    if not total:
        raise SystemExit(
            f"[grid] nada a rodar: --dataset {args.dataset} --split {args.split!r} "
            f"--percentage {args.percentage!r} expandiu para zero combinacoes.")

    cells = []
    for dataset in datasets:
        # Resolvido por dataset, dentro do laco. Valores conferem com
        # scripts/constants.py:101 e src/evaluate/constants.py:54.
        if dataset == "eggs":
            num_classes = 9
        elif dataset == "larvae":
            num_classes = 2
        elif dataset == "protozoan":
            num_classes = 7
        else:
            raise ValueError(f"dataset desconhecido: {dataset}")
        for split in splits:
            for pct in pcts:
                cells.append((dataset, split, pct, num_classes))
    failures = run_grid(cells, _run_one, args, "distillation_twolayer_module")
    if failures:
        _log.error("%d/%d combinacoes falharam.", failures, total)
    return 1 if failures else 0


def _score(checkpoint_cb) -> float:
    """Best monitored score of a ModelCheckpoint as a plain float (NaN if unset)."""
    s = getattr(checkpoint_cb, "best_model_score", None)
    try:
        return float(s.item() if hasattr(s, "item") else s)
    except (TypeError, ValueError):
        return float("nan")


def _save_metadata(args, module, output_dir, ckpt_dir,
                   status="ok", error="", best_ckpt="", best_loss_ckpt="",
                   best_knn_kappa=float("nan"), best_val_loss=float("nan"), elapsed=0.0):
    import subprocess as _sp
    try:
        git_sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "run_name": args.run_name, "dataset": args.dataset,
        "split": args.split, "percentage": args.percentage,
        "distillation_type": module.hparams.distillation_type,
        "encoder_init": module.hparams.encoder_init,
        "fine_tune": module.hparams.fine_tune,
        "teacher_frozen": module.hparams.teacher_frozen,
        "kd_temperature": module.hparams.kd_temperature,
        "kd_alpha": module.hparams.kd_alpha,
        "no_imagenet_norm": args.no_imagenet_norm,
        "freeze_encoder": args.freeze_encoder,
        "teacher_imagenet_norm": args.teacher_imagenet_norm,
        "knn_probe": args.knn_probe,
        "proj_type": getattr(args, "proj_type", ""),
        "monitor": "val/knn_kappa",
        "teacher_model": "facebook/ijepa_vith14_1k",
        "teacher_embed_dim": TEACHER_DIM,
        "student_embed_dim": module.student_embed_dim,
        "proj_head": _PROJ_LABEL,
        "arch_json": args.arch_json,
        "image_size": args.image_size, "max_epochs": args.max_epochs,
        "warmup_epochs": args.warmup_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "best_checkpoint": best_ckpt, "best_loss_checkpoint": best_loss_ckpt,
        "best_knn_kappa": best_knn_kappa, "best_val_loss": best_val_loss,
        "checkpoint_dir": ckpt_dir,
        "output_dir": output_dir, "status": status, "error": error,
        "elapsed_s": round(elapsed, 1), "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
