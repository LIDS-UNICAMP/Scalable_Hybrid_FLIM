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
Residual (ResNet-style) decoder and the AutoEncoder it forms with the FLIM encoder.

Unsupervised counterpart of the distillation / LeJEPA arms: the encoder is trained
by pixel-wise reconstruction, with no labels in the loss. The decoder is a scaffold
that is discarded afterwards — the deliverable is the encoder.

Why not reuse ``models.build_decoder_from_arch``: that decoder is
``Upsample + ConvTranspose2d`` with no residual path, and it ends in ``nn.Sigmoid()``.
This experiment calls for a residual decoder emitting **logits** (the loss is
``BCEWithLogitsLoss``), so the decoder is built here instead. Channel widths still
come from the FLIM ``architecture.json`` via ``get_channels_from_arch`` — nothing is
hardcoded to 32, which matters because protozoan is 24→30→48 while eggs/larvae are
24→32→48.

**No skip connections cross the bottleneck.** The residual paths live strictly inside
each decoder block. A U-Net skip from the encoder would let the decoder reconstruct
around the embedding, removing exactly the pressure this experiment is testing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models import Encoder, get_channels_from_arch


# ─── Residual decoder ─────────────────────────────────────────────────────────


class ResidualUpBlock(nn.Module):
    """Upsample ×``scale``, then two 3×3 convs with an internal residual connection.

    Upsampling is ``nearest`` followed by a conv rather than ``ConvTranspose2d``:
    transposed convolutions with a kernel size not divisible by the stride produce
    the classic checkerboard artefact, and the FLIM kernels here are 5×5 with pool
    stride 2. Nearest+conv (Odena et al., 2016) avoids it by construction.

    The residual is taken *after* the upsample, so it is internal to the block —
    it never reaches back into the encoder.
    """

    def __init__(self, ch_in: int, ch_out: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        # 1×1 projection only when the block changes width.
        self.proj: nn.Module = (
            nn.Conv2d(ch_in, ch_out, 1, bias=False) if ch_in != ch_out else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


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
        channels = get_channels_from_arch(arch, out_channels)  # [3, 24, 32, 48]
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


# ─── AutoEncoder ──────────────────────────────────────────────────────────────


class AutoEncoderFLIM(nn.Module):
    """FLIM encoder + residual decoder, trained by reconstruction without labels.

    ``forward`` returns **logits** over the LAB image in [0, 1]; ``embed`` returns the
    global-average-pooled embedding, the same ``[B, channels[-1]]`` tensor the SVM
    evaluators consume (see ``src/utils/evaluate.py:_encode_pooled``).

    The encoder is a plain ``Encoder``, so ``load_FLIM_encoder`` /
    ``load_FLIM_encoder_from_arch_dict`` find ``model.encoder.conv{n}`` unchanged.
    """

    def __init__(
        self,
        arch: dict,
        in_channels: int = 3,
        out_size: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        self.encoder = Encoder(arch, in_channels)
        self.decoder = ResNetDecoder(arch, in_channels, out_size)
        self.embed_dim: int = get_channels_from_arch(arch, in_channels)[-1]
        self._pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled encoder output, ``[B, embed_dim]``."""
        return self._pool(self.encoder(x)).flatten(start_dim=1)
