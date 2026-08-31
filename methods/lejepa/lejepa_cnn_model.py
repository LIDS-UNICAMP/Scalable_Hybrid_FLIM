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
Model LeJEPA com backbone CNN leve (SimpleCNNModel) no lugar de um encoder TIMM.

Movido de src/models/custom_cnn.py:124-180 (LeJEPACNNModel) sem nenhuma alteracao
de corpo: mesmas camadas, mesma ordem, mesmas chaves de state_dict. So os imports
mudaram para os novos caminhos de methods/lejepa/.

O SimpleCNNModel nao vem junto: ele mora em methods/lejepa/simple_cnn_model.py.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .projection_head import ProjectionHead
from .simple_cnn_model import SimpleCNNModel


class LeJEPACNNModel(nn.Module):
    """
    LeJEPA model backed by SimpleCNNModel instead of a TIMM encoder.

    Implements the same interface as ``LeJEPAModel``:
    - ``encode(x)``  → ``[B, embed_dim]``
    - ``forward(views)`` → ``(emb [V*B, embed_dim], proj [V, B, proj_dim])``

    Args:
        in_channels:   Input image channels. Default: 3.
        conv_channels: 3 conv output-channel sizes. Default: [32, 64, 128].
        linear_dims:   3 linear layer sizes. Default: [1024, 512, 256].
        dropout:       Dropout probability. Default: 0.2.
        proj_dim:      Output dimension of projection head. Default: 256.
        proj_hidden:   Hidden dimension of projection MLP. Default: 2048.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        linear_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder = SimpleCNNModel(
            in_channels=in_channels,
            conv_channels=conv_channels,
            linear_dims=linear_dims,
            dropout=dropout,
        )
        embed_dim = self.encoder.num_features
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
        flat = torch.stack(views, dim=0).flatten(0, 1)   # [V*B, C, H, W]
        emb = self.encoder(flat)                          # [V*B, embed_dim]
        proj = self.multi_layer_perceptron(emb).view(V, B, -1)  # [V, B, proj_dim]
        return emb, proj
