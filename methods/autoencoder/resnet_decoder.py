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
Decoder residual do autoencoder: espelho do encoder FLIM.

Movido de src/models/autoencoder_resnet.py:85 (ResNetDecoder). Mesma sequencia
de canais, mesmo numero de estagios, mesmo forward. Duas mudancas, ambas de
import e nenhuma de numero:

1. ``ResidualUpBlock`` vem de methods/autoencoder/residual_up_block.py.
2. A lista de canais e montada aqui a partir do proprio dict do arch, em vez de
   vir de ``get_channels_from_arch``. A fronteira do pacote ``flim`` so permite
   ``from flim import build``, e ``get_channels_from_arch`` vive em
   flim/arch.py:45 — a conta e a mesma, linha por linha.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_up_block import ResidualUpBlock


class ResNetDecoder(nn.Module):
    """Mirror of the FLIM encoder, built from residual up-blocks.

    Channels follow ``architecture.json`` in reverse (48 → 32 → 24 for eggs/larvae,
    48 → 30 → 24 for protozoan). The last block keeps the widest low-level width
    instead of collapsing straight to 3, so the RGB/LAB projection is a plain conv
    rather than a residual block squeezed to 3 channels.

    ``out_size`` is enforced with a final ``interpolate``: the encoder takes
    200 → 99 → 49 → 24, and three ×2 upsamples give 192, not 200. Resizing at the
    very end (rather than letting the blocks drift) keeps the reconstruction square
    with the target.
    """

    def __init__(
        self,
        arch: dict,
        out_channels: int = 3,
        out_size: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        n_layers = arch["nlayers"]
        # [3, 24, 32, 48] — identico a get_channels_from_arch(arch, out_channels).
        channels: List[int] = [out_channels] + [
            arch[f"layer{n}"]["conv"]["noutput_channels"] for n in range(1, n_layers + 1)
        ]
        self.out_size = out_size

        blocks: List[nn.Module] = []
        for n in range(n_layers, 0, -1):
            scale = arch[f"layer{n}"]["pooling"]["stride"]
            ch_in = channels[n]
            # Last block (n == 1) keeps its width; the projection head handles 3 channels.
            ch_out = channels[n - 1] if n > 1 else channels[1]
            blocks.append(ResidualUpBlock(ch_in, ch_out, scale=scale))
        self.blocks = nn.Sequential(*blocks)

        # Emits logits — BCEWithLogitsLoss applies the sigmoid internally.
        self.to_image = nn.Conv2d(channels[1], out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.blocks(z)
        if out.shape[-2:] != self.out_size:
            out = F.interpolate(out, size=self.out_size, mode="bilinear", align_corners=False)
        return self.to_image(out)
