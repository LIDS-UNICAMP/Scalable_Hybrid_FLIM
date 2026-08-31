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

"""Lightning module for I-JEPA -> FLIM CNN knowledge distillation (MLP head variant).

Movido de src/modules/distillation_module.py:107 (DistillationModule). As duas
steps, o acoplamento com o professor e a ordem das operacoes sao os da origem;
mudou so de quem os blocos sao importados.

Two distillation strategies are supported:

    direct
        The FLIM CNN student learns solely from the frozen I-JEPA teacher:
            loss = MSE( proj(student_emb), teacher_emb )

    hybrid
        The student receives gradients from both its own SSL loss and the
        distillation loss against the frozen teacher:
            loss = alpha * L_SSL + (1 - alpha) * L_KD
        where L_SSL is the standard LeJEPA SIGReg + invariance objective.

O que NAO veio junto (era CLI, agora e train.py + experiment YAML):
``_build_parser`` (src/modules/distillation_module.py:428),
``_dataset_short_to_parasite_name`` (:475), ``main`` (:484) e ``_save_metadata``
(:645). Os grupos de flags de src/models/distillation.py:279-497 vao para
``methods/distillation/cli.py``.

O parse do ``architecture.json``, a correcao de canais pelos kernels reais e a
carga dos pesos FLIM sairam do ``__init__``: agora vivem atras da porta unica
``flim.build``, consumida pelo ``LeJEPAFLIMModel``. Com isso o caso especial do
protozoan (``PROTOZOAN_FLIM_ARCH``) deixa de existir aqui — o
``architecture.json`` de ch24_30_48 ja traz layer2 com 30 canais e
``get_actual_channels_from_weights`` corrige os canais do mesmo jeito de antes.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from core.blocks.init import INIT_FNS
from core.losses import SimpleSIGReg, RealSIGReg, invariance_loss
from core.mixins import FrozenTeacherCheckpointMixin
from methods.lejepa import LeJEPAFLIMModel

from .cosine_distillation_loss import CosineDistillationLoss
from .distillation_projection_head import DistillationProjectionHead
from .frozen_teacher import FrozenTeacher
from .teacher_constants import TEACHER_DIM
from .kd_loss import kd_loss
from .kl_distillation_loss import KLDistillationLoss
from .mse_distillation_loss import MSEDistillationLoss
from .student_classification_head import StudentClassificationHead
from .teacher_input import prepare_teacher_input

_log = logging.getLogger(__name__)

_SIGREG_TYPES      = {"simple": SimpleSIGReg, "real": RealSIGReg}
ENCODER_INITS      = ("random", "he", "xavier", "trunc_normal", "flim")
DISTILLATION_TYPES = ("direct", "direct_cosine", "hybrid", "kd_hybrid")


class DistillationModule(FrozenTeacherCheckpointMixin, pl.LightningModule):
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
        dataset:           Short dataset name; kept as a hparam for run naming.
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
        # arch_json, canais reais e pesos FLIM sao resolvidos dentro do model
        # pela porta unica `flim.build`.
        self.student = LeJEPAFLIMModel(
            arch_json=arch_json,
            init=encoder_init,
            weights_path=flim_weights_path,
            in_channels=in_channels,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
        )
        self.student_embed_dim: int = self.student.embed_dim

        _log.info(
            "[DistillationModule] student_embed_dim=%d  encoder_init=%s  type=%s  num_classes=%d",
            self.student_embed_dim, encoder_init, distillation_type, num_classes,
        )

        # ── Apply encoder initialisation ──────────────────────────────────
        # `flim` e `random` mapeiam para None no INIT_FNS: o primeiro ja foi
        # carregado dentro do model, o segundo e o default do torch. O if/elif
        # da origem cobria he/xavier/trunc_normal, exatamente as tres funcoes
        # que o dicionario tem — a resolucao aqui e 1:1 com o comportamento
        # antigo.
        _init_fn = INIT_FNS.get(encoder_init)
        if _init_fn is not None:
            _init_fn(self.student.encoder)
        elif encoder_init == "flim":
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
