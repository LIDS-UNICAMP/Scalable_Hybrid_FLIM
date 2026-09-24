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
Lightning module for LeJEPA pretraining with the FLIM Encoder backbone.

Same training objective as LeJEPAModule / LeJEPACNNModule:
    loss = lam * sigreg(proj) + (1 - lam) * invariance(proj)

Adds a **configurable encoder initializer** via ``encoder_init``:
    - ``"random"``  — default PyTorch initialisation (no action).
    - ``"he"``      — Kaiming / He initialisation.
    - ``"xavier"``  — Xavier / Glorot initialisation.
    - ``"flim"``    — Load FLIM-trained weights from disk.
"""
from __future__ import annotations

from typing import Any, List, Optional

import lightning.pytorch as pl
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    init_weights_he,
    init_weights_xavier,
    load_FLIM_encoder,
)
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.losses.lejepa_loss import SimpleSIGReg, RealSIGReg, invariance_loss

_SIGREG_TYPES = {
    "simple": SimpleSIGReg,
    "real":   RealSIGReg,
}

_ENCODER_INITS = ("random", "he", "xavier", "flim")


class LeJEPAFLIMModule(pl.LightningModule):
    """
    Lightning module for LeJEPA pretraining with the FLIM Encoder backbone.

    The FLIM architecture is loaded from an ``architecture.json`` file. The
    encoder can be initialised with one of four strategies via ``encoder_init``.

    Args:
        arch_json:         Path to the FLIM architecture JSON file.
        encoder_init:      Initialisation strategy: ``"random"`` | ``"he"`` |
                           ``"xavier"`` | ``"flim"``.
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
        num_log_images:           Images to log per view in W&B.
        log_img_every_n_epochs:   Log input views every N epochs.
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
    ) -> None:
        super().__init__()
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )
        if encoder_init not in _ENCODER_INITS:
            raise ValueError(
                f"encoder_init must be one of {list(_ENCODER_INITS)}, got '{encoder_init}'"
            )
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")

        self.save_hyperparameters()

        # ── Parse architecture ──────────────────────────────────────────
        arch = parse_architecture(arch_json)

        # When loading FLIM weights the actual channel counts may differ
        # from the architecture spec (FLIM filter selection may prune kernels).
        if encoder_init == "flim":
            channels = get_actual_channels_from_weights(
                flim_weights_path, arch, in_channels
            )
            arch = override_arch_channels(arch, channels)
        else:
            channels = get_channels_from_arch(arch, in_channels)

        # ── Build model ─────────────────────────────────────────────────
        self.model = LeJEPAFLIMModel(
            arch=arch,
            in_channels=in_channels,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
        )

        # ── Apply encoder initialisation ────────────────────────────────
        if encoder_init == "he":
            init_weights_he(self.model.encoder)
        elif encoder_init == "xavier":
            init_weights_xavier(self.model.encoder)
        elif encoder_init == "flim":
            load_FLIM_encoder(
                self.model, arch_json, flim_weights_path, channels
            )

        self.sigreg = _SIGREG_TYPES[sigreg_type]()

    # ── Forward / training ──────────────────────────────────────────────

    def forward(self, views: List[Tensor]) -> tuple[Tensor, Tensor]:
        return self.model.forward(views)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, _ = batch
        emb, proj = self.model.forward(views)

        sig_loss = self.sigreg(emb)
        inv_loss = invariance_loss(emb)
        loss = self.hparams.lam * sig_loss + (1 - self.hparams.lam) * inv_loss

        self.log("train/sigreg", sig_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/inv",    inv_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss",   loss,     prog_bar=True,  on_step=True, on_epoch=True)

        return loss

    # ── Optimiser ───────────────────────────────────────────────────────

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

