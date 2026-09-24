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
LeJEPA model backed by the FLIM Encoder (from architecture.json).

Same interface as LeJEPAModel / LeJEPACNNModel:
  - encode(x)      → [B, embed_dim]
  - forward(views)  → (emb [V*B, embed_dim], proj [V, B, proj_dim])
"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn

from src.models.models import (
    Encoder,
    parse_architecture,
    get_channels_from_arch,
)
from src.models.lejepa import ProjectionHead, ProjectionHeadHawk


class LeJEPAFLIMModel(nn.Module):
    """
    LeJEPA model using the FLIM Encoder architecture.

    The FLIM Encoder outputs feature maps [B, C_out, H', W']. An adaptive
    average pool collapses the spatial dimensions to produce [B, embed_dim].

    Args:
        arch:        Parsed architecture dict (from architecture.json).
        in_channels: Number of input image channels (3 for LAB/RGB).
        proj_dim:    Projection head output dimension.
        proj_hidden: Projection head hidden dimension.
        three_layer_projector: When True, replace the ProjectionHead with the
            3-Layer Projector variant (Linear→BN→ReLU ×2 + Linear, all dim 48).
    """

    def __init__(
        self,
        arch: dict,
        in_channels: int = 3,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        three_layer_projector: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(arch, in_channels)
        channels = get_channels_from_arch(arch, in_channels)
        self.embed_dim = channels[-1]  # output channels of last conv layer

        # Collapse spatial dims → [B, embed_dim]
        self.pool = nn.AdaptiveAvgPool2d(1)

        if three_layer_projector:
            # 3-Layer Projector variant: keeps embed_dim (48) throughout, with
            # BN+ReLU between linears and no final activation. The projected
            # output feeds the kappa_with_proj downstream probe.
            self.multi_layer_perceptron = nn.Sequential(
                nn.Linear(48, 48), nn.BatchNorm1d(48), nn.ReLU(),
                nn.Linear(48, 48), nn.BatchNorm1d(48), nn.ReLU(),
                nn.Linear(48, 48),
            )
        else:
            self.multi_layer_perceptron = ProjectionHead(
                self.embed_dim, proj_hidden, proj_dim
            )
        # self.multi_layer_perceptron_hawk = ProjectionHeadHawk(
        #     self.embed_dim, proj_hidden, proj_dim
        # )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-view batch. Returns backbone features [B, embed_dim]."""
        feat = self.encoder(x)        # [B, C_out, H', W']
        feat = self.pool(feat)         # [B, C_out, 1, 1]
        return feat.flatten(1)         # [B, embed_dim]

    def forward(
        self, views: Union[Sequence[torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            views:
                Either:
                - Sequence of V tensors, each [B, C, H, W], or
                - A single tensor stacked as [B, V, C, H, W].

        Returns:
            emb:  [V*B, embed_dim]  — backbone features for all views.
            proj: [V, B, proj_dim]  — projected embeddings, views-first.
        """
        if isinstance(views, torch.Tensor):
            # views is [B, V, C, H, W] → transpose to [V, B, C, H, W]
            B, V = views.shape[:2]
            views = views.transpose(0, 1)  # [V, B, C, H, W]
            flat = views.flatten(0, 1)     # [V*B, C, H, W]
        else:
            # views is a sequence of V tensors, each [B, C, H, W]
            V = len(views)
            B = views[0].shape[0]
            flat = torch.stack(views, dim=0).flatten(0, 1)  # [V*B, C, H, W]

        emb = self.encode(flat)  # [V*B, embed_dim]
        emb_reshaped = emb.view(V, B, self.embed_dim)
        proj = self.multi_layer_perceptron(emb).view(V, B, -1)  # [V, B, proj_dim]
        return emb_reshaped, proj


