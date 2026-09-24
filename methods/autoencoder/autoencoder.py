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
Autoencoder base: encoder FLIM + decoder espelhado com ConvTranspose2d.

Movido de src/models/models.py:514 (AutoEncoder) — NAO de
src/models/autoencoder_resnet.py, que so tem a variante FLIM (a spec aponta o
arquivo errado; ver relatorio). ``forward`` e o mesmo byte a byte.

Duas mudancas, ambas de import e nenhuma de numero:

1. O encoder vem de ``build`` (a unica fronteira de ``flim``) em vez de
   ``Encoder(arch, in_channels)``, que vive em flim/encoder.py e portanto nao
   pode ser importado daqui.
2. ``build_decoder_from_arch`` (flim/arch.py:118, origem src/models/models.py:474)
   esta copiada aqui como ``_build_decoder_from_arch``, privada, byte a byte. Nao
   e reuso porque a fronteira so permite ``from flim import build``: flim/arch.py
   nao pode ser importado por methods/. Divida conhecida, registrada no relatorio.

Este arquivo NAO funde com a variante FLIM: o decoder dali e residual e emite
logits, o daqui e ConvTranspose2d e termina em Sigmoid. Sao dois modelos.
"""

from typing import Optional

import torch
import torch.nn as nn

from flim import build


def _build_decoder_from_arch(arch: dict, out_channels: int = 3) -> nn.Sequential:
    """
    Build a mirrored decoder from the encoder architecture.
    Uses ConvTranspose2d to reverse each encoder block.
    """
    n_layers = arch["nlayers"]
    # [3, 24, 32, 48] — identico a get_channels_from_arch(arch, out_channels).
    channels = [out_channels] + [
        arch[f"layer{n}"]["conv"]["noutput_channels"] for n in range(1, n_layers + 1)
    ]

    layers = []
    for n in range(n_layers, 0, -1):
        layer_desc = arch[f"layer{n}"]
        conv_desc = layer_desc["conv"]
        pool_desc = layer_desc["pooling"]

        ks = conv_desc["kernel_size"]
        kernel_size = (ks[0], ks[1])
        dilation = conv_desc["dilation_rate"]
        dilation_rate = (dilation[0], dilation[1])
        ch_in = channels[n]
        ch_out = channels[n - 1]
        padding = (kernel_size[0] // 2 * dilation_rate[0], kernel_size[1] // 2 * dilation_rate[1])

        pool_stride = pool_desc["stride"]
        pool_size = (pool_desc["size"][0], pool_desc["size"][1])

        # Upsample to reverse pooling
        layers.append(nn.Upsample(scale_factor=pool_stride, mode="bilinear", align_corners=False))
        # Transposed conv to reverse the conv layer
        layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation_rate))
        if n > 1 and layer_desc.get("relu", False):
            layers.append(nn.ReLU(inplace=True))

    # Final sigmoid to output in [0, 1]
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    """Autoencoder using the FLIM encoder architecture with a mirrored decoder."""

    def __init__(self, arch_json: str, init: str, weights_path: Optional[str] = None,
                 in_channels: int = 3):
        super().__init__()
        self.encoder = build(arch_json, init, weights_path, in_channels)
        self.decoder = _build_decoder_from_arch(self.encoder.arch, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        # Crop/pad to match input size if needed due to pooling rounding
        if reconstructed.shape != x.shape:
            reconstructed = nn.functional.interpolate(
                reconstructed, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return reconstructed
