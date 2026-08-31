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
LeJEPA model backed by the FLIM Encoder (from architecture.json).

Movido de src/models/lejepa_flim.py:41 (LeJEPAFLIMModel). O corpo do forward e
do encode nao mudou; mudou so quem monta o encoder: antes era
``Encoder(arch, in_channels)`` com o arch ja parseado pelo call site
(src/models/lejepa_flim.py:66), agora e a porta unica ``flim.build``, que le o
architecture.json, acerta os canais pelos kernels reais e carrega os pesos FLIM
quando ``init="flim"``. Por isso o construtor recebe ``arch_json``/``init``/
``weights_path`` no lugar do dict ``arch``.

A dimensao do embedding vem de ``encoder.num_features`` (flim/encoder.py:89),
que e o mesmo ``get_channels_from_arch(arch, in_channels)[-1]`` que a linha
src/models/lejepa_flim.py:68 calculava a mao.

Same interface as LeJEPAModel / LeJEPACNNModel:
  - encode(x)      → [B, embed_dim]
  - forward(views)  → (emb [V*B, embed_dim], proj [V, B, proj_dim])
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from flim import build
from .projection_head import ProjectionHead


class LeJEPAFLIMModel(nn.Module):
    """
    LeJEPA model using the FLIM Encoder architecture.

    The FLIM Encoder outputs feature maps [B, C_out, H', W']. An adaptive
    average pool collapses the spatial dimensions to produce [B, embed_dim].

    Args:
        arch_json:   Path to the FLIM ``architecture.json``.
        init:        Init-axis value; only ``"flim"`` loads weights here.
        weights_path: Directory with the FLIM kernels/bias. Required when
            ``init="flim"``, ignored otherwise.
        in_channels: Number of input image channels (3 for LAB/RGB).
        proj_dim:    Projection head output dimension.
        proj_hidden: Projection head hidden dimension.
        three_layer_projector: When True, replace the ProjectionHead with the
            3-Layer Projector variant (Linear→BN→ReLU ×2 + Linear, all dim 48).
    """

    def __init__(
        self,
        arch_json: str,
        init: str,
        weights_path: Optional[str] = None,
        in_channels: int = 3,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        three_layer_projector: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = build(arch_json, init, weights_path, in_channels)
        self.embed_dim = self.encoder.num_features  # output channels of last conv layer

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
