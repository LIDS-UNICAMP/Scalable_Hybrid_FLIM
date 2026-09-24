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

"""
LeJEPA Line Module — Lightning module for LeJEPA pretraining with the FLIM
Encoder backbone and **configurable encoder initializers**.

Merges ``EncoderModels`` (from ``src.models.models``) into the LeJEPA regime,
exposing a settable ``encoder_init`` parameter that selects among:

    - ``"random"``       — default PyTorch initialisation (no action).
    - ``"he"``           — Kaiming / He initialisation.
    - ``"xavier"``       — Xavier / Glorot initialisation.
    - ``"flim"``         — Load FLIM-trained weights from disk.
    - ``"trunc_normal"`` — Truncated normal (timm ViT style, std=0.02).

Training objective (same as LeJEPAModule family):

    loss = lam * sigreg(proj) + (1 - lam) * invariance(proj)

Supports dataset-percentage training and mask application via the
companion ``LejepaLineDataModule`` / ``ParasiteFLIMDataModule``.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    init_weights_he,
    init_weights_xavier,
    init_weights_trunc_normal,
    load_FLIM_encoder,
)
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.losses.lejepa_loss import SimpleSIGReg, RealSIGReg, invariance_loss

_SIGREG_TYPES = {
    "simple": SimpleSIGReg,
    "real":   RealSIGReg,
}

ENCODER_INITS = ("random", "he", "xavier", "flim", "trunc_normal")


class LejepaLineModule(pl.LightningModule):
    """
    Lightning module for LeJEPA pretraining with the FLIM Encoder backbone.

    The FLIM architecture is loaded from an ``architecture.json`` file. The
    encoder can be initialised with one of four strategies via ``encoder_init``.

    Data is expected to come from ``LejepaLineDataModule`` or
    ``ParasiteFLIMDataModule`` — both deliver
    ``(List[Tensor] | Tensor, label)`` batches.

    Args:
        arch_json:         Path to the FLIM architecture JSON file.
        encoder_init:      Initialisation strategy: ``"random"`` | ``"he"`` |
                           ``"xavier"`` | ``"flim"`` | ``"trunc_normal"``.
        flim_weights_path: Directory containing FLIM weight files
                           (``conv{n}-kernels.npy``, ``conv{n}-bias.txt``).
                           **Required** when ``encoder_init="flim"``.
        in_channels:       Number of input image channels. Default: 3.
        proj_dim:          Projection head output dimension.
        proj_hidden:       Projection head hidden dimension.
        lam:               Weight on SIGReg; (1 - lam) on invariance loss.
        sigreg_type:       ``"simple"`` or ``"real"`` (Epps-Pulley).
        lr:                Peak learning rate.
        weight_decay:      AdamW weight decay.
        max_epochs:        Total training epochs (for cosine schedule).
        warmup_epochs:     Epochs of linear LR warmup before cosine decay.
        num_log_images:    Images to log per view in W&B.
        log_img_every_n_epochs: Log input views every N epochs.
    """

    def __init__(
        self,
        arch_json: str,
        encoder_init: str = "random",
        flim_weights_path: Optional[str] = None,
        in_channels: int = 3,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        lam: float = 0.05,
        sigreg_type: str = "simple",
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        num_log_images: int = 8,
        log_img_every_n_epochs: int = 5,
        svm_max_train_samples: int = 3000,
        knn_k: int = 20,
        three_layer_projector: bool = False,
    ) -> None:
        super().__init__()
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )
        if encoder_init not in ENCODER_INITS:
            raise ValueError(
                f"encoder_init must be one of {list(ENCODER_INITS)}, got '{encoder_init}'"
            )
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")

        self.save_hyperparameters()

        # -- Parse architecture ------------------------------------------------
        arch = parse_architecture(arch_json)

        if encoder_init == "flim":
            channels = get_actual_channels_from_weights(
                flim_weights_path, arch, in_channels
            )
            arch = override_arch_channels(arch, channels)
        else:
            channels = get_channels_from_arch(arch, in_channels)

        # -- Build model -------------------------------------------------------
        self.model = LeJEPAFLIMModel(
            arch=arch,
            in_channels=in_channels,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
            three_layer_projector=three_layer_projector,
        )

        # -- Apply encoder initialisation --------------------------------------
        import logging as _logging
        _log = _logging.getLogger(__name__)
        _log.info(
            "[SSL init] encoder_init=%s  arch=%s", encoder_init, arch_json
        )
        if encoder_init == "he":
            init_weights_he(self.model.encoder)
            _log.info("[SSL init] He (Kaiming) initialisation applied.")
        elif encoder_init == "xavier":
            init_weights_xavier(self.model.encoder)
            _log.info("[SSL init] Xavier (Glorot) initialisation applied.")
        elif encoder_init == "trunc_normal":
            init_weights_trunc_normal(self.model.encoder)
            _log.info(
                "[SSL init] Truncated-normal initialisation applied "
                "(timm init_weights_vit_timm, std=0.02, clip=[-0.04,+0.04])."
            )
        elif encoder_init == "flim":
            load_FLIM_encoder(
                self.model, arch_json, flim_weights_path, channels
            )
            _log.info("[SSL init] FLIM weights loaded from %s.", flim_weights_path)
        else:
            # encoder_init == "random" → no action (PyTorch defaults)
            _log.info("[SSL init] Random (PyTorch default) initialisation — no action.")

        self.sigreg = _SIGREG_TYPES[sigreg_type]()

        # SVM probe buffers — reset each epoch. We keep both the raw backbone
        # features (no_proj) and the projector-transformed features (with_proj)
        # so the probe can report κ for both representations.
        self._svm_train_feats:       list[Tensor] = []
        self._svm_train_proj_feats:  list[Tensor] = []
        self._svm_train_labels:      list[Tensor] = []
        self._svm_val_feats:         list[Tensor] = []
        self._svm_val_proj_feats:    list[Tensor] = []
        self._svm_val_labels:        list[Tensor] = []

    # -- Forward / training ----------------------------------------------------

    def forward(self, views: List[Tensor]) -> tuple[Tensor, Tensor]:
        return self.model.forward(views)

    def on_train_epoch_start(self) -> None:
        self._svm_train_feats.clear()
        self._svm_train_proj_feats.clear()
        self._svm_train_labels.clear()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, labels = batch
        emb, proj = self.model.forward(views)

        # 3-layer projector variant: compute the loss on the projector output so
        # the projector actually receives gradient. Default path keeps the loss
        # on the raw backbone embeddings (unchanged legacy behaviour).
        loss_feat = proj if self.hparams.three_layer_projector else emb
        sig_loss = self.sigreg(loss_feat)
        inv_loss = invariance_loss(loss_feat)
        loss = self.hparams.lam * sig_loss + (1 - self.hparams.lam) * inv_loss

        self.log("train/sigreg",    sig_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/invariance", inv_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss",       loss,     prog_bar=True,  on_step=True, on_epoch=True)

        # Accumulate single-view features for SVM probe (no extra forward pass).
        # We store both the raw backbone feature (no_proj) and its projector
        # transform (with_proj) so the probe can report κ for both.
        max_n = self.hparams.svm_max_train_samples
        if sum(f.shape[0] for f in self._svm_train_feats) < max_n:
            with torch.no_grad():
                first_view = views[0] if isinstance(views, (list, tuple)) else views[:, 0]
                feat = self.model.encode(first_view)
                proj_feat = self.model.multi_layer_perceptron(feat)
            self._svm_train_feats.append(feat.detach().cpu())
            self._svm_train_proj_feats.append(proj_feat.detach().cpu())
            self._svm_train_labels.append(labels.detach().cpu())

        return loss

    def validation_step(self, batch: Any, _batch_idx: int) -> Tensor:
        views, labels = batch
        emb, proj = self.model.forward(views)

        # Match training_step: loss on projector output for the 3-layer variant,
        # on raw backbone embeddings otherwise.
        loss_feat = proj if self.hparams.three_layer_projector else emb
        sig_loss = self.sigreg(loss_feat)
        inv_loss = invariance_loss(loss_feat)
        loss = self.hparams.lam * sig_loss + (1 - self.hparams.lam) * inv_loss

        self.log("val/sigreg",    sig_loss, prog_bar=False, on_epoch=True)
        self.log("val/invariance", inv_loss, prog_bar=False, on_epoch=True)
        self.log("val/loss",       loss,     prog_bar=True,  on_epoch=True)

        # Accumulate val features for SVM probe (raw + projector-transformed).
        with torch.no_grad():
            first_view = views[0] if isinstance(views, (list, tuple)) else views[:, 0]
            feat = self.model.encode(first_view)
            proj_feat = self.model.multi_layer_perceptron(feat)
        self._svm_val_feats.append(feat.detach().cpu())
        self._svm_val_proj_feats.append(proj_feat.detach().cpu())
        self._svm_val_labels.append(labels.detach().cpu())

        return loss

    def _svm_kappa(self, X_train, y_train, X_val, y_val) -> Optional[tuple[float, float, float]]:
        """Fit a KNN probe and return (kappa, acc, f1). None on failure.

        Mirrors the distillation pipeline (``KnnKappaProbeMixin``): seeded
        subsample of the train memory bank, per-dimension ``StandardScaler``
        fit on the (subsampled) train features and applied to both splits,
        then a euclidean ``KNeighborsClassifier`` with ``n_neighbors=k``.
        """
        try:
            # Seeded subsample of the train memory bank (matches distillation).
            subsample = int(self.hparams.svm_max_train_samples)
            seed = int(getattr(self.hparams, "seed", 42))
            n_train = X_train.shape[0]
            if n_train > subsample:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_train, size=subsample, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            # Per-dimension standardization, fit on train only.
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val   = scaler.transform(X_val)

            n_neighbors = min(int(self.hparams.knn_k), X_train.shape[0] - 1)
            if n_neighbors < 1:
                return None
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            kappa = float(cohen_kappa_score(y_val, y_pred))
            acc   = float(accuracy_score(y_val, y_pred))
            f1    = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            return kappa, acc, f1
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("KNN probe failed: %s", exc)
            return None

    def on_validation_epoch_end(self) -> None:
        if not self._svm_train_feats or not self._svm_val_feats:
            self._svm_val_feats.clear()
            self._svm_val_proj_feats.clear()
            self._svm_val_labels.clear()
            return

        # Raw backbone features (no_proj) and projector-transformed (with_proj).
        X_train      = torch.cat(self._svm_train_feats).numpy()
        X_train_proj = torch.cat(self._svm_train_proj_feats).numpy()
        y_train      = torch.cat(self._svm_train_labels).numpy()
        X_val        = torch.cat(self._svm_val_feats).numpy()
        X_val_proj   = torch.cat(self._svm_val_proj_feats).numpy()
        y_val        = torch.cat(self._svm_val_labels).numpy()

        self._svm_val_feats.clear()
        self._svm_val_proj_feats.clear()
        self._svm_val_labels.clear()

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            return

        # κ WITHOUT the projector (raw backbone embeddings).
        res_no_proj = self._svm_kappa(X_train, y_train, X_val, y_val)
        if res_no_proj is not None:
            kappa, acc, f1 = res_no_proj
            # Keep the legacy metric name as an alias of the no-proj κ so older
            # dashboards/checkpoints keep working.
            self.log("val/svm_kappa",        kappa, prog_bar=True,  on_epoch=True)
            self.log("val/kappa_no_proj",    kappa, prog_bar=True,  on_epoch=True)
            self.log("val/svm_acc",          acc,   prog_bar=False, on_epoch=True)
            self.log("val/svm_f1",           f1,    prog_bar=False, on_epoch=True)

        # κ THROUGH the projector (projector-transformed embeddings).
        res_with_proj = self._svm_kappa(X_train_proj, y_train, X_val_proj, y_val)
        if res_with_proj is not None:
            kappa_p, acc_p, f1_p = res_with_proj
            self.log("val/kappa_with_proj",     kappa_p, prog_bar=True,  on_epoch=True)
            self.log("val/svm_acc_with_proj",   acc_p,   prog_bar=False, on_epoch=True)
            self.log("val/svm_f1_with_proj",    f1_p,    prog_bar=False, on_epoch=True)

    # -- Optimiser -------------------------------------------------------------

    def configure_optimizers(self):
        warmup_epochs = self.hparams.warmup_epochs or 10
        max_epochs = self.hparams.max_epochs or 100
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-5
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
