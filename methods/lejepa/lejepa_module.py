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

"""Modulo Lightning do LeJEPA base (encoder TIMM).

Raiz da cadeia ``LeJEPAModule -> LeJEPAFLIMModule -> LejepaLineModule``.
Movido de src/modules/lejepa_module.py sem mudanca de logica: mesmos defaults,
mesma ordem de operacoes, mesmo ``save_hyperparameters``. So os imports do model
e das losses mudaram de caminho.
"""

from __future__ import annotations

from typing import Any, List

import lightning.pytorch as pl
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from core.losses import SimpleSIGReg, RealSIGReg, invariance_loss
from methods.lejepa.lejepa_model import LeJEPAModel

_SIGREG_TYPES = {
    "simple": SimpleSIGReg,
    "real":   RealSIGReg,
}


class LeJEPAModule(pl.LightningModule):
    """
    Lightning module for LeJEPA self-supervised pretraining.

    Expects multi-crop batches: ``(List[Tensor], labels)`` where the list
    contains V view tensors each of shape [B, C, H, W]. Labels are ignored.

    Loss:
        lejepa = lam * sigreg(proj) + (1 - lam) * invariance(proj)

    Where ``proj`` has shape [V, B, proj_dim] and:
    - sigreg:    SIGReg regularization (simple moment-matching or full Epps-Pulley)
    - invariance: MSE between each view's embedding and the global-view mean

    Args:
        arch:         TIMM encoder name.
        pretrained:   Load ImageNet weights.
        proj_dim:     Projection head output dimension.
        proj_hidden:  Projection MLP hidden dimension.
        lam:          Weight on SIGReg; (1 - lam) applied to invariance loss.
                      Paper recommends lam in [1e-3, 1e-1], default 0.05.
        sigreg_type:  Which SIGReg variant to use.
                      ``"simple"`` — moment-matching (mean=0, std=1 per projection).
                      ``"real"``   — full Epps-Pulley characteristic-function test.
        lr:           Peak learning rate.
        weight_decay: AdamW weight decay.
        max_epochs:   Total training epochs (for cosine schedule).
        warmup_epochs: Epochs of linear LR warmup before cosine decay.
        num_log_images:        Number of images to log per view in W&B. Default: 8.
        log_img_every_n_epochs: Log input views to W&B every N epochs. Default: 5.
    """

    def __init__(
        self,
        arch: str = "vit_small_patch16_224",
        pretrained: bool = False,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        lam: float = 0.05,
        sigreg_type: str = "real",
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 1,
        num_log_images: int = 8,
        log_img_every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )
        self.save_hyperparameters()
        self.model = LeJEPAModel(
            arch=arch, pretrained=pretrained, proj_dim=proj_dim, proj_hidden=proj_hidden
        )
        self.sigreg = _SIGREG_TYPES[sigreg_type]()

    def forward(self, views: List[Tensor]) -> tuple[Tensor, Tensor]:
        return self.model(views)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, _ = batch
        _, proj = self.model(views)           # proj: [V, B, proj_dim]

        sig_loss = self.sigreg(proj)
        inv_loss = invariance_loss(proj)
        loss = self.hparams.lam * sig_loss + (1 - self.hparams.lam) * inv_loss

        self.log("train/sigreg", sig_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/inv",    inv_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss",   loss,     prog_bar=True,  on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        warmup_epochs = self.hparams.warmup_epochs or 1
        max_epochs = self.hparams.max_epochs or 100
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs - warmup_epochs),
            eta_min=1e-5,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
