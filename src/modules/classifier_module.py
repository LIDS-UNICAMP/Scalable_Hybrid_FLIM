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

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy

from src.models.encoders import (
    build_encoder,
    freeze_encoder,
    get_embed_dim,
    unfreeze_norm_layers,
)


class ClassificationFinetuneModule(pl.LightningModule):
    """
    Lightning module for supervised fine-tuning on top of a pretrained SSL encoder.

    Loads a checkpoint produced by LeJEPA and attaches a linear
    classification head. The encoder can be frozen or fully fine-tuned.

    Args:
        arch: TIMM encoder name — must match the SSL pretraining architecture.
        num_classes: Number of target classes.
        checkpoint_path: Path to SSL checkpoint (.ckpt). If None, trains from scratch.
        freeze_encoder: Whether to freeze encoder parameters.
        unfreeze_norms: Unfreeze norm layers even when freeze_encoder=True.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Total epochs (for cosine schedule).
    """

    def __init__(
        self,
        arch: str = "resnet50",
        num_classes: int = 10,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        unfreeze_norms: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = build_encoder(arch, pretrained=False)
        embed_dim = get_embed_dim(self.encoder)
        self.head = nn.Linear(embed_dim, num_classes)

        if checkpoint_path is not None:
            self._load_encoder_from_checkpoint(checkpoint_path)

        if freeze_encoder:
            freeze_encoder(self.encoder)
            if unfreeze_norms:
                unfreeze_norm_layers(self.encoder)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _load_encoder_from_checkpoint(self, ckpt_path: str) -> None:
        """Extract encoder weights from an SSL checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Try to load online_encoder or plain encoder weights
        for prefix in ("model.online_encoder.", "model.encoder.", "encoder."):
            encoder_state = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            if encoder_state:
                missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
                print(f"Loaded encoder from '{prefix}' prefix. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                return
        print(f"Warning: No matching encoder weights found in {ckpt_path}")

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        return self.head(features)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        imgs, labels = batch
        logits = self(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
        self.train_acc(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        imgs, labels = batch
        logits = self(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
        self.val_acc(logits, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        params = [{"params": self.head.parameters(), "lr": self.hparams.lr}]
        if not self.hparams.freeze_encoder:
            params.append({
                "params": self.encoder.parameters(),
                "lr": self.hparams.lr * 0.1,
            })
        optimizer = AdamW(params, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
