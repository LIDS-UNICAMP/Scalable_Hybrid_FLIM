"""distillation_onelayer_module.py — Lightning module for I-JEPA → FLIM CNN distillation
using a single 3×3 conv projection head (one_layer variant).

Projection head: 48 → 1280 via a single Conv2d(3×3) + BN2d + GELU + AdaptiveAvgPool.
Direct jump from encoder channels to teacher dimensionality in one operation.

Run name convention: distillation_<dataset>_split<N>_pct<P>_3x3_BN2d_1280_one_layer

Usage (subprocess call from distillation_conv_ray.py with --proj-type 3x3_bn2d_1280)::

    python -m src.modules.distillation_onelayer_module \\
        --dataset eggs \\
        --split 1 \\
        --percentage 100 \\
        --distillation-type direct \\
        --encoder-init trunc_normal \\
        --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json \\
        --run-name distillation_eggs_split1_pct100_3x3_BN2d_1280_one_layer \\
        --max-epochs 100 \\
        --batch-size 32
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

from src.models.distillation import (
    KLDistillationLoss,
    MSEDistillationLoss,
    OneLayerConvDistillationProjectionHead,
    OneLayer1x1ConvDistillationProjectionHead,
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
DISTILLATION_TYPES = ("direct", "hybrid")


class DistillationOneLayerModule(FrozenTeacherCheckpointMixin, KnnKappaProbeMixin, pl.LightningModule):
    """Lightning module for knowledge distillation using a single 3×3 conv projection.

    Projection head: Conv2d(48→1280, k=3×3) + BN2d + GELU → GAP → [B, 1280]

    Two distillation strategies:
        direct:  loss = MSE( proj(encoder(x)), teacher(x) )
        hybrid:  loss = α·L_SSL + (1-α)·KL

    Run name convention: distillation_<dataset>_split<N>_pct<P>_3x3_BN2d_1280_one_layer
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
        proj_kernel: int = 3,
        freeze_encoder: bool = False,
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
        if proj_kernel not in (1, 3):
            raise ValueError("proj_kernel must be 1 or 3")
        if knn_probe not in ("encoder", "projection"):
            raise ValueError("knn_probe must be 'encoder' or 'projection'")

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
            _log.info("[DistillationOneLayerModule] FLIM weights loaded from %s.", flim_weights_path)

        # ── Frozen teacher ─────────────────────────────────────────────────
        self.teacher = FrozenTeacher(model_id=teacher_model_id)

        # ── One-layer projection head: 48 → 1280 (3×3 ou 1×1) ─────────────
        if proj_kernel == 1:
            self.proj_kd = OneLayer1x1ConvDistillationProjectionHead(
                student_channels=self.student_embed_dim,
                teacher_dim=TEACHER_DIM,
            )
            _proj_label = "Conv1x1(48→1280)+BN2d+GELU"
        else:
            self.proj_kd = OneLayerConvDistillationProjectionHead(
                student_channels=self.student_embed_dim,
                teacher_dim=TEACHER_DIM,
            )
            _proj_label = "Conv3x3(48→1280)+BN2d+GELU"

        _log.info(
            "[DistillationOneLayerModule] embed_dim=%d  init=%s  type=%s  proj=%s",
            self.student_embed_dim, encoder_init, distillation_type, _proj_label,
        )

        if distillation_type == "direct":
            self.mse_loss = MSEDistillationLoss()
        else:
            self.kd_loss = KLDistillationLoss(temperature=temperature)
            if distillation_type == "hybrid":
                self.sigreg = _SIGREG_TYPES[sigreg_type]()

        # ── Frozen-encoder mode: only the projection head trains ────────────
        # The FLIM encoder is kept in eval() with requires_grad=False, so its
        # 48d output is constant across epochs — that is why the kNN probe runs
        # on the trainable 1280d projection for this variant (knn_probe).
        if freeze_encoder:
            for p in self.student.parameters():
                p.requires_grad = False
            self.student.eval()
            _log.info("[DistillationOneLayerModule] FLIM encoder FROZEN — only proj_kd trains.")

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

    @torch.no_grad()
    def _teacher_emb(self, student_view: Tensor) -> Tensor:
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

        if self.hparams.distillation_type == "direct":
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

        if self.hparams.distillation_type == "direct":
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
        optimizer    = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler    = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Standalone entry-point ─────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Train one one_layer distillation experiment (Conv3×3 48→1280).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",            required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split",              required=True, type=int)
    p.add_argument("--percentage",         required=True, type=int)
    p.add_argument("--arch-json",          required=True)
    p.add_argument("--run-name",           required=True)
    p.add_argument("--distillation-type",  required=True, choices=list(DISTILLATION_TYPES))
    p.add_argument("--encoder-init",       default="trunc_normal", choices=list(ENCODER_INITS))
    p.add_argument("--flim-weights-path",  default=None,
                   help="Directory with FLIM weight files. Required when --encoder-init flim.")
    p.add_argument("--alpha",              type=float, default=0.5)
    p.add_argument("--temperature",        type=float, default=4.0)
    p.add_argument("--num-classes",        type=int,   default=9)
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
                        "'encoder'=48d student.encode (default; anti-collapse for trainable encoder), "
                        "'projection'=1280d proj_kd (use when --freeze-encoder, since the 48d output is constant).")
    p.add_argument("--knn-train-subsample", type=int, default=3000,
                   help="Memory-bank subsample size for the kNN probe (seeded).")
    p.add_argument("--knn-every-n-epochs", type=int, default=1,
                   help="Run the kNN probe every N validation epochs.")
    p.add_argument("--image-size",         type=int,   default=200)
    p.add_argument("--num-views",          type=int,   default=4)
    p.add_argument("--output-dir",         default=None)
    p.add_argument("--proj-kernel",         type=int, default=3, choices=[1, 3],
                                            help="Kernel size do proj head: 3=Conv3x3, 1=Conv1x1.")
    p.add_argument("--proj-type",          default="",
                   help="proj_type label (ex.: 1x1_init_flim_frozen) usado como tag/sinal no W&B "
                        "e gravado no run_metadata.json.")
    p.add_argument("--wandb",              action="store_true", default=False)
    p.add_argument("--wandb-project",      default="flim-ssl")
    p.add_argument("--wandb-entity",       default="ophira-ai")
    p.add_argument("--seed",               type=int, default=42)
    return p


def _dataset_short_to_parasite_name(dataset: str) -> str:
    return {
        "eggs":      "helminth-eggs_split_2",
        "larvae":    "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }[dataset]


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)

    args = _build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir or os.path.join(_ROOT, "artifacts", "distillation", args.run_name)
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited

    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(args.dataset),
        split=args.split, percentage=args.percentage,
        image_size=args.image_size, V_train=args.num_views, V_eval=1,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        loader="ift_lab",
        imagenet_norm=not args.no_imagenet_norm,
    )

    module = DistillationOneLayerModule(
        arch_json=args.arch_json,
        dataset=args.dataset,
        distillation_type=args.distillation_type,
        encoder_init=args.encoder_init,
        flim_weights_path=args.flim_weights_path,
        alpha=args.alpha, temperature=args.temperature,
        lam_ssl=args.lam_ssl, sigreg_type=args.sigreg_type,
        num_classes=args.num_classes, lr=args.lr,
        weight_decay=args.weight_decay, max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        proj_kernel=args.proj_kernel,
        freeze_encoder=args.freeze_encoder,
        teacher_imagenet_norm=args.teacher_imagenet_norm,
        knn_probe=args.knn_probe,
        knn_train_subsample=args.knn_train_subsample,
        knn_every_n_epochs=args.knn_every_n_epochs,
        seed=args.seed,
    )

    # Two checkpoints: PRIMARY by val/knn_kappa (the MSE val/loss is decoupled
    # from / inverted w.r.t. downstream κ — minimising it collapses the encoder),
    # plus a best-by-loss fallback. Both are evaluated downstream by the SVM.
    checkpoint_knn = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_knn_kappa",
        monitor="val/knn_kappa", mode="max",
        save_last=True, save_top_k=1,
    )
    checkpoint_loss = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_loss",
        monitor="val/loss", mode="min",
        save_last=False, save_top_k=1,
    )

    logger_list: list = []
    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.run_name, save_dir=output_dir,
            )
            # W&B tags sinalizam o tipo do experimento de relance: o proj_type
            # (ex.: "1x1_init_flim_frozen"), além de "frozen"/"no_imagenet_norm".
            _tags = list(wandb_logger.experiment.tags or ())
            if args.proj_type:
                _tags.append(args.proj_type)
            if args.freeze_encoder:
                _tags.append("frozen")
            if args.no_imagenet_norm:
                _tags.append("no_imagenet_norm")
            if _tags:
                wandb_logger.experiment.tags = tuple(dict.fromkeys(_tags))
            _proj_label = f"Conv{args.proj_kernel}x{args.proj_kernel}(48→1280)+BN2d+GELU"
            wandb_logger.log_hyperparams({
                "dataset": args.dataset, "split": args.split,
                "percentage": args.percentage, "distillation_type": args.distillation_type,
                "encoder_init": args.encoder_init,
                "no_imagenet_norm": args.no_imagenet_norm,
                "freeze_encoder": args.freeze_encoder,
                "teacher_imagenet_norm": args.teacher_imagenet_norm,
                "knn_probe": args.knn_probe,
                "proj_type": args.proj_type,
                "arch_json": args.arch_json,
                "teacher": "facebook/ijepa_vith14_1k",
                "proj_head": _proj_label,
                "student_embed_dim": module.student_embed_dim,
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s", _e)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, callbacks=[checkpoint_knn, checkpoint_loss],
        logger=logger_list or False,
        log_every_n_steps=10, enable_progress_bar=True, deterministic=False,
    )

    # ── Resume support (e.g. after a crash). save_last lives on checkpoint_knn.
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0:
        _log.info("[DistillationOneLayerModule] Resuming from checkpoint: %s", resume_ckpt)
    else:
        resume_ckpt = None

    t0 = time.time()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except Exception as exc:
        _log.error("Training failed: %s", exc, exc_info=True)
        _save_metadata(args, module, output_dir, ckpt_dir, status="error", error=str(exc))
        return 1

    elapsed = time.time() - t0
    _save_metadata(args, module, output_dir, ckpt_dir, status="ok",
                   best_ckpt=checkpoint_knn.best_model_path or "",
                   best_loss_ckpt=checkpoint_loss.best_model_path or "",
                   best_knn_kappa=_score(checkpoint_knn),
                   best_val_loss=_score(checkpoint_loss),
                   elapsed=elapsed)

    if args.wandb:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass
    return 0


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
        "distillation_type": args.distillation_type,
        "encoder_init": args.encoder_init,
        "no_imagenet_norm": args.no_imagenet_norm,
        "freeze_encoder": args.freeze_encoder,
        "teacher_imagenet_norm": args.teacher_imagenet_norm,
        "knn_probe": args.knn_probe,
        "proj_type": getattr(args, "proj_type", ""),
        "monitor": "val/knn_kappa",
        "teacher_model": "facebook/ijepa_vith14_1k",
        "teacher_embed_dim": TEACHER_DIM,
        "student_embed_dim": module.student_embed_dim,
        "proj_head": f"Conv{args.proj_kernel}x{args.proj_kernel}(48→1280)+BN2d+GELU",
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
