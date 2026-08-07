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
Unsupervised AutoEncoder pretraining of the FLIM encoder (Experiment Day 5).

No labels enter the loss. The encoder is trained purely by reconstructing its own
input, and the decoder is thrown away afterwards. The question the run answers is
whether the encoder comes out *better* than it went in — measured by a one-vs-one
SVM probe on the pooled embedding, fitted on the labelled train split and scored
**on validation only**. Test is never touched.

Reading the result (see the experiment brief):
  improved  — reconstruction is useful self-supervision for this tiny encoder
  tied      — the FLIM embedding is already saturated
  degraded  — reconstruction pulls the embedding toward low-level information
              (texture, a/b chromaticity, background). The known risk.

Loss: ``BCEWithLogitsLoss`` over LAB in [0, 1] (option 1 of the brief). Cross-entropy
needs a target in [0, 1], so the target is the LAB image **before** ``imagenet_norm``,
recovered by undoing the normalisation. ``ift_lab`` already returns LABNorm2 in [0, 1],
so ``lab_scale`` is the identity — no extra rescaling is invented here.

Best checkpoint is selected by ``val/svm_kappa``, not by ``val/recon_loss``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.svm import SVC
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.classification import compute_metrics
from src.models.autoencoder_resnet import AutoEncoderFLIM
from src.models.models import (
    PROTOZOAN_FLIM_ARCH,
    get_actual_channels_from_weights,
    load_FLIM_encoder,
    load_FLIM_encoder_from_arch_dict,
    override_arch_channels,
    parse_architecture,
)

_log = logging.getLogger(__name__)

# Applied by src/data_modules/datasets/lejepa_dataset.py:44 when imagenet_norm=True.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LABNorm2 (pyift) -> CIE L*a*b*, for W&B visualisation only. Round-trip against the
# source PNG measures MAE ~= 0.011, dominated by the uint8 quantisation the transform
# pipeline already applies (lejepa_dataset.py:33) — not by this mapping.
LAB_L_SCALE = 100.0
LAB_AB_SCALE = 255.0
LAB_AB_OFFSET = 128.0


class AutoEncoderFlimModule(pl.LightningModule):
    """FLIM-init AutoEncoder trained by reconstruction, probed by a one-vs-one SVM.

    Args:
        arch_json:          Path to the FLIM ``architecture.json``.
        dataset:            ``"eggs"`` | ``"larvae"`` | ``"protozoan"``. protozoan loads
                            ``PROTOZOAN_FLIM_ARCH``, matching the convention already used
                            by ``ClassificationFlimModule`` — its trained weights live in
                            ``ch24_32_48_a0.5_f5`` even though the real widths are 24/30/48,
                            which ``get_actual_channels_from_weights`` then corrects.
        flim_weights_path:  Directory with ``conv{n}-kernels.npy`` / ``conv{n}-bias.txt``.
        num_classes:        9 (eggs) / 2 (larvae) / 7 (protozoan). Used by the SVM probe only.
        imagenet_norm:      Whether the dataloader normalised the input. Drives target recovery.
        svm_probe_every:    Run the SVM probe every N validation epochs (1 = every epoch).
    """

    def __init__(
        self,
        arch_json: str,
        dataset: str = "",
        flim_weights_path: Optional[str] = None,
        num_classes: int = 9,
        in_channels: int = 3,
        image_size: int = 200,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        seed: int = 42,
        imagenet_norm: bool = True,
        svm_probe_every: int = 1,
        log_recon_every: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if flim_weights_path is None:
            raise ValueError(
                "flim_weights_path is required — this experiment always initialises "
                "the encoder with FLIM weights (encoder_init=flim)."
            )

        arch = PROTOZOAN_FLIM_ARCH if dataset == "protozoan" else parse_architecture(arch_json)
        channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
        arch = override_arch_channels(arch, channels)

        self.model = AutoEncoderFLIM(
            arch=arch, in_channels=in_channels, out_size=(image_size, image_size)
        )
        self.encoder_out_channels: int = channels[-1]
        self.channels: List[int] = channels

        if dataset == "protozoan":
            load_FLIM_encoder_from_arch_dict(self.model, arch, flim_weights_path, channels)
        else:
            load_FLIM_encoder(self.model, arch_json, flim_weights_path, channels)

        # Encoder stays trainable — moving it without labels is the whole point.
        self.criterion = nn.BCEWithLogitsLoss()

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_norm_mean", mean, persistent=False)
        self.register_buffer("_norm_std", std, persistent=False)

        _log.info(
            "[AutoEncoderFlimModule] FLIM weights from %s | channels=%s | embed_dim=%d | "
            "imagenet_norm=%s | probe_every=%d",
            flim_weights_path, channels, self.encoder_out_channels,
            imagenet_norm, svm_probe_every,
        )

        self._val_emb: List[Tensor] = []
        self._val_labels: List[Tensor] = []
        self._sample_batch: Optional[Tensor] = None
        self._train_loader_cache = None
        self.baseline_metrics: Optional[dict] = None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _first_view(views: Union[List[Tensor], Tensor]) -> Tensor:
        """The datamodule yields ``[B, V, 3, H, W]``; this experiment runs with V=1."""
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0]
        return views

    def _target_from_input(self, x: Tensor) -> Tensor:
        """Recover the LAB image in [0, 1] that BCE needs from the (possibly) normalised input.

        ``ift_lab`` yields LABNorm2 already in [0, 1], so undoing the ImageNet
        normalisation is the only step — there is no separate ``lab_scale``.
        """
        if not self.hparams.imagenet_norm:
            return x.clamp(0.0, 1.0)
        return (x * self._norm_std + self._norm_mean).clamp(0.0, 1.0)

    # ── Forward / training ─────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _recon_loss(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        target = self._target_from_input(x)
        logits = self(x)
        return self.criterion(logits, target), logits, target

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, _ = batch  # labels deliberately discarded — training is unsupervised
        x = self._first_view(views)
        loss, _, _ = self._recon_loss(x)

        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("train/recon_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_emb = []
        self._val_labels = []
        self._sample_batch = None

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        loss, logits, _ = self._recon_loss(x)

        self._val_emb.append(self.model.embed(x).detach().float().cpu())
        self._val_labels.append(y.detach().cpu())
        if batch_idx == 0:
            self._sample_batch = (x[:4].detach().cpu(), logits[:4].detach().float().cpu())

        self.log("val/recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ── SVM probe ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embeddings(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        """One deterministic pass over *loader*, returning pooled embeddings and labels."""
        was_training = self.model.training
        self.model.eval()
        embs, labels = [], []
        for views, y in loader:
            x = self._first_view(views).to(self.device, non_blocking=True)
            embs.append(self.model.embed(x).float().cpu())
            labels.append(y.cpu())
        if was_training:
            self.model.train()
        return torch.cat(embs).numpy(), torch.cat(labels).numpy()

    def _train_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """One deterministic pass over the labelled train split.

        The percentage does not enter the reconstruction loss; it decides how many
        labelled images the probe gets to see here.
        """
        if self._train_loader_cache is None:
            self._train_loader_cache = self.trainer.datamodule.train_dataloader()
        return self._embeddings(self._train_loader_cache)

    def on_fit_start(self) -> None:
        """Probe the untouched FLIM encoder before a single gradient step.

        This is the reference the experiment is judged against. It cannot be taken from
        the existing baseline CSVs: those score the SVM on the **test** split
        (src/evaluate/svm.py:123-137) while this probe scores on **validation**, so the
        two numbers are not comparable. Measuring the baseline here, with the identical
        protocol, splits and probe, makes improved/tied/degraded a sound verdict.
        """
        try:
            X_tr, y_tr = self._train_embeddings()
            X_val, y_val = self._embeddings(self.trainer.datamodule.val_dataloader())
            metrics = self._svm_probe(X_tr, y_tr, X_val, y_val)
        except Exception as exc:
            _log.warning("FLIM baseline probe failed: %s", exc)
            return
        if metrics is None:
            return

        self.baseline_metrics = metrics
        _log.info(
            "[AutoEncoderFlimModule] FLIM-init baseline (no training): "
            "kappa=%.4f acc=%.4f f1=%.4f",
            metrics["kappa"], metrics["acc"], metrics["f1"],
        )
        # self.log() is not allowed this early; write straight to the W&B summary.
        if isinstance(self.logger, WandbLogger):
            for key, value in metrics.items():
                self.logger.experiment.summary[f"baseline/svm_{key}"] = value

    def _svm_probe(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Optional[dict]:
        """One-vs-one linear SVM, fitted on train, scored on validation.

        Hyperparameters mirror ``src/utils/evaluate.py:270-278`` exactly — including the
        absence of a scaler — so this in-training curve stays comparable with the offline
        CSVs produced by the repository's evaluators.
        """
        if len(np.unique(y_tr)) < 2:
            _log.warning("SVM probe skipped: train split has a single class.")
            return None
        clf = SVC(
            max_iter=10000,
            C=1e2,
            degree=3,
            gamma="auto",
            coef0=0,
            decision_function_shape="ovo",
            kernel="linear",
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        return compute_metrics(y_val, y_pred, num_classes=self.hparams.num_classes)

    def on_validation_epoch_end(self) -> None:
        if not self._val_emb or self.trainer.sanity_checking:
            return

        every = max(1, int(self.hparams.svm_probe_every))
        if (self.current_epoch % every) != 0 and self.current_epoch != self.trainer.max_epochs - 1:
            return

        X_val = torch.cat(self._val_emb).numpy()
        y_val = torch.cat(self._val_labels).numpy()

        try:
            X_tr, y_tr = self._train_embeddings()
            t0 = time.time()
            metrics = self._svm_probe(X_tr, y_tr, X_val, y_val)
            fit_s = time.time() - t0
        except Exception as exc:  # a probe failure must not kill the pretraining run
            _log.warning("SVM probe failed at epoch %d: %s", self.current_epoch, exc)
            return

        if metrics is None:
            return

        self.log("val/svm_kappa", metrics["kappa"], prog_bar=True, on_epoch=True)
        self.log("val/svm_acc", metrics["acc"], prog_bar=False, on_epoch=True)
        self.log("val/svm_f1", metrics["f1"], prog_bar=False, on_epoch=True)
        self.log("val/svm_fit_s", fit_s, prog_bar=False, on_epoch=True)

        # Replay the FLIM-init baseline as a flat series so the panel shows, at a glance,
        # whether reconstruction moved the encoder above or below where it started.
        if self.baseline_metrics is not None:
            for key, value in self.baseline_metrics.items():
                self.log(f"baseline/svm_{key}", value, on_epoch=True)
            self.log(
                "val/svm_kappa_delta",
                metrics["kappa"] - self.baseline_metrics["kappa"],
                prog_bar=False, on_epoch=True,
            )

        self._log_reconstruction()

    # ── W&B reconstruction preview ─────────────────────────────────────────

    def _lab01_to_rgb(self, lab01: np.ndarray) -> np.ndarray:
        """``[3,H,W]`` LABNorm2 in [0,1] -> ``[H,W,3]`` sRGB in [0,1].

        Without this the panel would render L/a/b as if they were R/G/B and invite the
        wrong conclusion about reconstruction quality.
        """
        from skimage.color import lab2rgb

        c = np.transpose(lab01, (1, 2, 0)).astype(np.float64)
        lab = np.stack(
            [
                c[..., 0] * LAB_L_SCALE,
                c[..., 1] * LAB_AB_SCALE - LAB_AB_OFFSET,
                c[..., 2] * LAB_AB_SCALE - LAB_AB_OFFSET,
            ],
            axis=-1,
        )
        return np.clip(lab2rgb(lab), 0.0, 1.0)

    def _log_reconstruction(self) -> None:
        if self._sample_batch is None or self.logger is None:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        every = max(1, int(self.hparams.log_recon_every))
        if (self.current_epoch % every) != 0:
            return

        try:
            import wandb

            x, logits = self._sample_batch
            target = self._target_from_input(x.to(self._norm_mean.device)).cpu().numpy()
            recon = torch.sigmoid(logits).numpy()

            images = []
            for i in range(target.shape[0]):
                pair = np.concatenate(
                    [self._lab01_to_rgb(target[i]), self._lab01_to_rgb(recon[i])], axis=1
                )
                images.append(wandb.Image(pair, caption=f"epoch {self.current_epoch} — alvo | recon"))
            self.logger.experiment.log({"val/reconstruction": images}, commit=False)
        except Exception as exc:
            _log.debug("Reconstruction preview skipped: %s", exc)

    # ── Optimizer ──────────────────────────────────────────────────────────

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total = self.hparams.max_epochs or 100
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler = SequentialLR(
            optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Standalone entry-point ─────────────────────────────────────────────────────

NUM_CLASSES = {"eggs": 9, "larvae": 2, "protozoan": 7}
ARCH_TAG = {"eggs": "ch24_32_48", "larvae": "ch24_32_48", "protozoan": "ch24_30_48"}


def _dataset_short_to_parasite_name(dataset: str) -> str:
    return {
        "eggs": "helminth-eggs_split_2",
        "larvae": "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }[dataset]


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Train one unsupervised AutoEncoder (FLIM encoder + ResNet decoder) run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split", required=True, type=int)
    p.add_argument("--percentage", required=True, type=int)
    p.add_argument("--arch-json", required=True)
    p.add_argument("--flim-weights-path", required=True,
                   help="Directory with conv{n}-kernels.npy / conv{n}-bias.txt.")
    p.add_argument("--recon-loss", default="bce_logits", choices=["bce_logits"],
                   help="Reconstruction loss. Cross-entropy over LAB in [0,1] as logits.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=200)
    p.add_argument("--svm-probe-every", type=int, default=1)
    p.add_argument("--log-recon-every", type=int, default=10)
    p.add_argument("--no-imagenet-norm", action="store_true", default=False,
                   help="Disable ImageNet RGB Normalize on ift_lab inputs.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="journal_02_2026_hybrid_FLIM")
    p.add_argument("--wandb-entity", default="ophira-ai")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir or os.path.join(
        _ROOT, "artifacts", "autoencoder_resnet_init_flim", args.run_name
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import (
        ParasiteLejepaDataModuleSplited,
    )

    # V_train=V_eval=1 -> the dataset uses the deterministic test transform
    # (parasite_lejepa.py:109). Reconstruction target == input, and the LeJEPA colour
    # augmentations (ColorJitter/Grayscale/Solarize) never touch the LAB channels.
    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(args.dataset),
        split=args.split, percentage=args.percentage,
        image_size=args.image_size, V_train=1, V_eval=1,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        loader="ift_lab",
        imagenet_norm=not args.no_imagenet_norm,
    )

    module = AutoEncoderFlimModule(
        arch_json=args.arch_json,
        dataset=args.dataset,
        flim_weights_path=args.flim_weights_path,
        num_classes=NUM_CLASSES[args.dataset],
        image_size=args.image_size,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        imagenet_norm=not args.no_imagenet_norm,
        svm_probe_every=args.svm_probe_every,
        log_recon_every=args.log_recon_every,
    )

    # Best checkpoint is the best *encoder*, i.e. highest probe kappa — not lowest recon loss.
    checkpoint_kappa = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best_kappa",
        monitor="val/svm_kappa", mode="max",
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
            wandb_logger.experiment.tags = tuple(dict.fromkeys(
                list(wandb_logger.experiment.tags or ())
                + ["journal_02_2026_hybrid_FLIM", "autoencoder", "flim_init", "unsupervised"]
            ))
            wandb_logger.log_hyperparams({
                "dataset": args.dataset,
                "split": args.split,
                "percentage": args.percentage,
                "encoder_init": "flim",
                "decoder": "resnet",
                "recon_loss": args.recon_loss,
                "architecture": ARCH_TAG[args.dataset],
                "channels": module.channels,
                "color_space": "lab",
                "loader": "ift_lab",
                "lab_scale": "labnorm2_native_01",
                "imagenet_norm": not args.no_imagenet_norm,
                "svm_multiclass": "ovo",
                "svm_kernel": "linear",
                "svm_C": 1e2,
                "svm_probe_every": args.svm_probe_every,
                "svm_probe_split": "fit=train, score=validation",
                "train_augmentation": "none",
                "num_classes": NUM_CLASSES[args.dataset],
                "embed_dim": module.encoder_out_channels,
                "arch_json": args.arch_json,
                "flim_weights_path": args.flim_weights_path,
                "encoder_trainable": True,
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
    if not (os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0):
        resume_ckpt = None

    t0 = time.time()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except Exception as exc:
        _log.error("Training failed: %s", exc, exc_info=True)
        _save_metadata(args, module, output_dir, ckpt_dir, status="error", error=str(exc))
        return 1

    _save_metadata(
        args, module, output_dir, ckpt_dir, status="ok",
        best_ckpt=checkpoint_kappa.best_model_path or "",
        best_kappa=_score(checkpoint_kappa),
        elapsed=time.time() - t0,
    )

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


def _save_metadata(args, module, output_dir, ckpt_dir, status="ok", error="",
                   best_ckpt="", best_kappa=float("nan"), elapsed=0.0) -> None:
    import json
    import subprocess as _sp

    try:
        git_sha = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        git_sha = ""

    meta = {
        "run_name": args.run_name,
        "method": "autoencoder_resnet_init_flim",
        "dataset": args.dataset,
        "split": args.split,
        "percentage": args.percentage,
        "encoder_init": "flim",
        "decoder": "resnet",
        "recon_loss": args.recon_loss,
        "architecture": ARCH_TAG[args.dataset],
        "channels": module.channels,
        "embed_dim": module.encoder_out_channels,
        "num_classes": NUM_CLASSES[args.dataset],
        "color_space": "lab",
        "lab_scale": "labnorm2_native_01",
        "imagenet_norm": not args.no_imagenet_norm,
        "svm_multiclass": "ovo",
        "train_augmentation": "none",
        "arch_json": args.arch_json,
        "flim_weights_path": args.flim_weights_path,
        "ckpt_dir": ckpt_dir,
        "best_ckpt": best_ckpt,
        "best_val_svm_kappa": best_kappa,
        "baseline_flim_svm": module.baseline_metrics,
        "elapsed_s": elapsed,
        "status": status,
        "error": error,
        "git_sha": git_sha,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
