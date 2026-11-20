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
Leitura e interpretacao do architecture.json do FLIM.

Toda a traducao "arch_json -> lista de canais -> blocos torch" mora aqui:
parse, extracao de canais, correcao dos canais pelos pesos que o encoder de
fato carrega, e a construcao do encoder e do decoder espelhado.

Modulo interno da fronteira `flim`: so `flim/build.py`, `flim/encoder.py` e
`flim/flim_residual_encoder.py` importam daqui. `methods/*` fala com `build()`.
"""

import copy
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


def parse_architecture(arch_json: str) -> dict:
    """Parse an architecture JSON file and return a structured description."""
    with open(arch_json, "r") as f:
        arch = json.load(f)
    return arch


def get_channels_from_arch(arch: dict, in_channels: int = 3) -> List[int]:
    """Extract the channel list [in_channels, layer1_out, layer2_out, ...] from arch dict."""
    channels = [in_channels]
    for n in range(1, arch["nlayers"] + 1):
        channels.append(arch[f"layer{n}"]["conv"]["noutput_channels"])
    return channels


def get_actual_channels_from_weights(weights_path: str, arch: dict, in_channels: int = 3) -> List[int]:
    """
    Read actual kernel counts from FLIM bias files.
    The FLIM filter selection may produce fewer kernels than requested in the
    architecture when the dataset has fewer classes (nkernels = nclasses * nkernels_per_marker,
    capped by nkernels_per_image).
    """
    channels = [in_channels]
    weights_dir = Path(weights_path)
    for n in range(1, arch["nlayers"] + 1):
        # quem abre arquivo de peso e flim.weights; aqui so contamos os kernels.
        # import local: flim.weights importa parse_architecture daqui (ciclo)
        from flim.weights import get_bias

        channels.append(len(get_bias(weights_dir / f"conv{n}-bias.txt")))
    return channels


def override_arch_channels(arch: dict, channels: List[int]) -> dict:
    """Return a copy of arch with noutput_channels overridden by actual values."""
    arch = copy.deepcopy(arch)
    for n in range(1, arch["nlayers"] + 1):
        arch[f"layer{n}"]["conv"]["noutput_channels"] = channels[n]
    return arch


class Branches(nn.ModuleList):
    """Camada Inception (`conv.branches` no arch): um conv por ramo sobre a mesma entrada, padding same, cat nos canais.

    No formato FLIM os ramos sao UM conv KxK (K = maior kernel; os menores com zeros em volta, a mesma conta);
    `load_full` recorta cada ramo desse KxK.
    """

    def forward(self, x):
        return torch.cat([conv(x) for conv in self], 1)

    def load_full(self, weight, bias) -> None:
        start = 0
        for conv in self:
            n, m = conv.out_channels, (weight.shape[-1] - conv.kernel_size[0]) // 2
            conv.weight.data = weight[start:start + n, :, m:weight.shape[-1] - m, m:weight.shape[-1] - m].contiguous()
            conv.bias.data = bias[start:start + n]
            start += n


class Whiten(nn.Module):
    """`layerwise_spf.whiten_between`: whitening FIXO da entrada da conv n, x -> G (x - mu), no fim do bloco n-1.

    mu e G sao os do `eval.spf_grade.map_whitener` (ZCA dos canais com shrinkage, sem rotulo, imagens de treino)
    medidos quando a conv n foi montada. Buffers, nao parametros: nao treinam (nem com tudo solto no _ft) e viajam
    no ckpt e no FLIM (`conv{n}-whiten.npz`). No fim do bloco anterior, o conv segue no `[0]` de todo bloco.
    """

    def __init__(self, shift: torch.Tensor, gain: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("shift", shift)
        self.register_buffer("gain", gain)

    def forward(self, x):
        return torch.einsum("dc,bchw->bdhw", self.gain, x - self.shift.view(1, -1, 1, 1))


def build_encoder_from_arch(arch: dict, in_channels: int = 3) -> nn.ModuleDict:
    """
    Build encoder blocks from architecture JSON.
    Returns an nn.ModuleDict with keys 'conv1', 'conv2', ... each being a Sequential
    of Conv2d + ReLU + MaxPool2d (following the arch spec).
    """
    blocks = nn.ModuleDict()
    ch_in = in_channels
    n_layers = arch["nlayers"]

    for n in range(1, n_layers + 1):
        layer_desc = arch[f"layer{n}"]
        conv_desc = layer_desc["conv"]
        pool_desc = layer_desc["pooling"]

        ks = conv_desc["kernel_size"]
        kernel_size = (ks[0], ks[1])
        dilation = conv_desc["dilation_rate"]
        dilation_rate = (dilation[0], dilation[1])
        ch_out = conv_desc["noutput_channels"]
        padding = (kernel_size[0] // 2 * dilation_rate[0], kernel_size[1] // 2 * dilation_rate[1])

        pool_size = (pool_desc["size"][0], pool_desc["size"][1])
        pool_stride = pool_desc["stride"]

        layers = [
            Branches(nn.Conv2d(ch_in, b["noutput_channels"], b["kernel_size"], padding=b["kernel_size"] // 2)
                     for b in conv_desc["branches"]) if "branches" in conv_desc else
            nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation_rate),
        ]
        if layer_desc.get("relu", False):
            layers.append(nn.ReLU(inplace=True))
        if pool_desc["type"] == "max_pool":
            layers.append(nn.MaxPool2d(pool_size, stride=pool_stride))

        blocks[f"conv{n}"] = nn.Sequential(*layers)
        ch_in = ch_out

    return blocks


def build_decoder_from_arch(arch: dict, out_channels: int = 3) -> nn.Sequential:
    """
    Build a mirrored decoder from the encoder architecture.
    Uses ConvTranspose2d to reverse each encoder block.
    """
    n_layers = arch["nlayers"]
    channels = get_channels_from_arch(arch, out_channels)

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
