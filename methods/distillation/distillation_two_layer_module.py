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

"""distillation_two_layer_module.py — Lightning module for I-JEPA -> FLIM CNN
distillation using a two-layer 1x1 conv projection head (~402k total params).

Projection head: 48 -> 256 -> 1280 via two Conv2d(1x1) + BN2d + GELU + AdaptiveAvgPool.
Intermediate bottleneck at 256 channels before jumping to teacher dimensionality.

Movido de src/modules/distillation_twolayer_module.py:110 (DistillationTwoLayerModule).
Ficaram na origem, intocados: o ``if __name__ == "__main__"`` (L726), ``main()`` (L635),
``_build_parser()`` (L388), ``_csv_ints`` (L452), ``_dataset_short_to_parasite_name``
(L456), ``_run_one`` (L464), ``_score`` (L673) e ``_save_metadata`` (L682). Aqui so vive
a LightningModule, instanciada por YAML. As steps, os defaults, as chaves logadas e o
state_dict nao mudaram.

Notas do move (nada aqui muda numero):

* **Encoder.** O parse do ``architecture.json``, a resolucao de canais e a carga dos
  pesos FLIM sairam do ``__init__`` (origem L176-201) e passaram para a porta unica
  ``flim.build``, ja consumida pelo ``LeJEPAFLIMModel`` (methods/lejepa/lejepa_flim_model.py:48).
  A sequencia e a mesma: monta o encoder, depois aplica o init. Os ``init_weights_*``
  viraram o dicionario ``core.blocks.init.INIT_FNS``, que tem exatamente as tres
  funcoes do if/elif da origem (he, xavier, trunc_normal) e mapeia ``flim``/``random``
  para ``None``.
* **protozoan.** O ramo ``dataset == "protozoan"`` -> ``PROTOZOAN_FLIM_ARCH`` (origem
  L176-177 e L198-199) nao atravessa a fronteira do pacote ``flim``: ``build`` recebe
  caminho de json, nao dict. Nao muda nada — ``get_actual_channels_from_weights`` le os
  canais dos proprios arquivos de bias e ``override_arch_channels`` sobrescreve
  ``noutput_channels``, entao os unicos campos em que os dois candidatos divergem
  (layer2, 30 vs 32 canais) nunca chegam ao encoder. Mesma analise, mesmo desfecho, de
  methods/classification/classification_flim_module.py:88.
* **``dataset``** continua na assinatura (e no ``save_hyperparameters``) porque e chave
  do hparams de todo checkpoint ja gravado; hoje so nomeia o run.
* **``num_classes`` — preservar o fix de e1ddf39.** A origem passa o ``num_classes``
  resolvido POR DATASET dentro do laco (L656-663, entregue em L510: eggs 9 / larvae 2 /
  protozoan 7), e nao o ``--num-classes`` do argparse, que e fixo em 9 — sem isso,
  ``--dataset all`` treinava larvae e protozoan com cabeca de 9 saidas. Com o argparse
  fora, a mesma regra tem que valer no YAML: **um ``model.init_args.num_classes`` por
  dataset**. O default 9 desta assinatura e o mesmo de antes e vale so para quem nao
  passar nada.
* **``--proj-type``** NAO virou parametro: naquele modulo ela nunca chegava ao
  ``__init__``. Era rotulo puro de telemetria, lido so por ``_run_one`` (tag do W&B,
  origem L550-551; ``log_hyperparams``, L572) e por ``_save_metadata`` (L705) — as duas
  funcoes que ficaram para tras. Aqui a head e unica e fixa, entao nao ha o que a flag
  pudesse escolher.
* **``acc``.** A convencao da origem esta preservada: ``train/kd_acc`` e ``val/kd_acc``
  sao acuracia MICRO calculada inline, ``(logits_s.argmax(1) == y).float().mean()``
  (origem L306 e L344). Nao passa por ``core.metrics.compute_metrics`` (cujo ``acc`` e
  macro) — trocar mudaria o numero logado.
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
from core.mixins import FrozenTeacherCheckpointMixin, KnnKappaProbeMixin
from methods.lejepa import LeJEPAFLIMModel

from .cosine_distillation_loss import CosineDistillationLoss
from .frozen_teacher import TEACHER_DIM, FrozenTeacher
from .kd_loss import kd_loss
from .kl_distillation_loss import KLDistillationLoss
from .mse_distillation_loss import MSEDistillationLoss
from .student_classification_head import StudentClassificationHead
from .teacher_input import prepare_teacher_input
from .two_layer_1x1_conv_bn2d_distillation_projection_head import (
    TwoLayer1x1ConvBN2dDistillationProjectionHead,
)

_log = logging.getLogger(__name__)

_SIGREG_TYPES      = {"simple": SimpleSIGReg, "real": RealSIGReg}
ENCODER_INITS      = ("random", "he", "xavier", "trunc_normal", "flim")
DISTILLATION_TYPES = ("direct", "direct_cosine", "hybrid", "kd_hybrid")

_PROJ_LABEL = "Conv1x1(48→256→1280)+BN2d+GELU×2"


class DistillationTwoLayerModule(FrozenTeacherCheckpointMixin, KnnKappaProbeMixin, pl.LightningModule):
    """Lightning module for knowledge distillation using a two-layer 1x1 conv projection.

    Projection head: Conv2d(48->256, k=1x1) + BN2d + GELU ->
                     Conv2d(256->1280, k=1x1) + BN2d + GELU -> GAP -> [B, 1280]

    ~402k total parameters (encoder 59k + head 343k).

    Two distillation strategies:
        direct:  loss = MSE( proj(encoder(x)), teacher(x) )
        hybrid:  loss = a*L_SSL + (1-a)*KL

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
        # arch_json, canais reais e pesos FLIM sao resolvidos dentro do model,
        # pela porta unica `flim.build`.
        self.student = LeJEPAFLIMModel(
            arch_json=arch_json,
            init=encoder_init,
            weights_path=flim_weights_path,
            in_channels=in_channels,
            proj_dim=proj_dim, proj_hidden=proj_hidden,
        )
        self.student_embed_dim: int = self.student.embed_dim

        # `flim` e `random` mapeiam para None no INIT_FNS: o primeiro ja foi
        # carregado dentro do model, o segundo e o default do torch.
        _init_fn = INIT_FNS.get(encoder_init)
        if _init_fn is not None:
            _init_fn(self.student.encoder)
        elif encoder_init == "flim":
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
