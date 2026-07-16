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

from typing import List, Optional

import torch
import torch.nn as nn

from src.models.lejepa import ProjectionHead


class SimpleCNNModel(nn.Module):
    """
    Simple CNN encoder: 3 convolutional blocks followed by 3 linear blocks.

    Conv block  = Conv2d → BatchNorm2d → ReLU → Dropout2d → MaxPool2d(2)
    Linear block = Linear → BatchNorm1d → ReLU → Dropout  (last block has no activation)
    After the 3 conv blocks a global average pool collapses spatial dimensions.

    The attribute ``num_features`` exposes the output embedding size (``linear_dims[-1]``),
    making this encoder compatible with ``get_embed_dim`` from ``src.models.encoders``.

    Args:
        in_channels:   Number of input image channels. Default: 3.
        conv_channels: Output channels for each of the 3 Conv2d layers.
                       Default: [32, 64, 128].
        linear_dims:   Neuron counts for each of the 3 Linear layers.
                       Default: [1024, 512, 256].
        dropout:       Dropout probability used in both conv and linear stages.
                       Default: 0.2.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        linear_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if linear_dims is None:
            linear_dims = [1024, 512, 256]
        assert len(conv_channels) == 3, "conv_channels must have exactly 3 elements"
        assert len(linear_dims) == 3, "linear_dims must have exactly 3 elements"

        c1, c2, c3 = conv_channels
        d1, d2, d3 = linear_dims

        self.conv_layers = nn.Sequential(
            # --- Conv block 1 ---
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # --- Conv block 2 ---
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # --- Conv block 3 ---
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Collapse spatial dims regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.linear_layers = nn.Sequential(
            # --- Linear block 1 ---
            nn.Linear(c3, d1, bias=False),
            nn.BatchNorm1d(d1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # --- Linear block 2 ---
            nn.Linear(d1, d2, bias=False),
            nn.BatchNorm1d(d2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # --- Linear block 3 (embedding output — no activation) ---
            nn.Linear(d2, d3, bias=False),
        )

        # Expose for get_embed_dim / LeJEPACNNModel
        self.num_features = d3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, num_features)
        """
        x = self.conv_layers(x)           # (B, c3, H', W')
        x = self.pool(x).flatten(1)       # (B, c3)
        x = self.linear_layers(x)         # (B, d3)
        return x


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
