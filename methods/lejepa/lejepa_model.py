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
Model base do LeJEPA.

Movido de src/models/lejepa.py:67 (LeJEPAModel) sem alteracao de corpo: mesmo
forward, mesmo embed_dim, mesmos nomes de atributo (``encoder``,
``multi_layer_perceptron``) que o checkpoint le.

O encoder TIMM vem de core.blocks.timm_encoder, que e onde build_encoder e
get_embed_dim passaram a viver (antes src/models/encoders.py:25 e :63).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from core.blocks.timm_encoder import build_encoder, get_embed_dim
from .projection_head import ProjectionHead


class LeJEPAModel(nn.Module):
    """
    LeJEPA (Lean Joint-Embedding Predictive Architecture) model.

    Uses a single encoder with a 3-layer MLP projection head. The SIGReg +
    invariance losses are applied externally in the LightningModule.

    forward() accepts a list of V view tensors and returns:
    - emb:  [V*B, embed_dim]  — backbone features for all views (e.g. for a probe)
    - proj: [V, B, proj_dim]  — projected embeddings, views-first

    Args:
        arch: TIMM encoder name.
        pretrained: Load ImageNet weights.
        proj_dim: Output dimension of projection head.
        proj_hidden: Hidden dimension of projection MLP.
    """

    def __init__(
        self,
        arch: str = "vit_small_patch16_224",
        pretrained: bool = False,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(arch, pretrained)
        embed_dim = get_embed_dim(self.encoder)
        self.multi_layer_perceptron = ProjectionHead(embed_dim, proj_hidden, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-view batch. Returns backbone features [B, embed_dim]."""
        return self.encoder(x)

    def forward(
        self, views: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            views: List of V tensors, each [B, C, H, W].

        Returns:
            emb:  [V*B, embed_dim] — backbone features for all views.
            proj: [V, B, proj_dim] — projected embeddings, views-first.
        """
        V = len(views)
        B = views[0].shape[0]
        flat = torch.stack(views, dim=0).flatten(0, 1)  # [V*B, C, H, W]
        emb = self.encode(flat)                          # [V*B, embed_dim]
        proj = self.multi_layer_perceptron(emb).view(V, B, -1)  # [V, B, proj_dim]
        return emb, proj
