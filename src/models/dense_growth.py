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
"""
Dense network (skip + concat) grown one block at a time by SPiFiL.

Block types, all ``conv -> ReLU`` (the SPiFiL block's own convention, so the folded
z-score bias of each filter keeps meaning the cut it was built for):

  stem        conv k x k, stride s                        out = out_channels
  dense       concat(x, relu(conv_kxk(x)))                out = in + growth
  transition  conv 1 x 1, stride 2, no concat             out = out_channels
  head        conv 1 x 1, no concat                       out = out_channels

The dense block's skip has no weight: the input is passed through untouched and
concatenated with the ``growth`` new channels, so freezing everything before a block
freezes its skip as well. The embedding is the GAP of the last block.

``architecture.json`` (written by :func:`save_stage`) is the list of block specs with
their resolved ``in_channels`` / ``out_channels`` / ``out_size``;
:func:`load_stage` rebuilds the exact same network from it plus ``encoder.pt``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """``relu(conv(x))``; with ``concat=True`` it returns ``cat([x, relu(conv(x))])``."""

    def __init__(self, spec: dict, in_channels: int) -> None:
        super().__init__()
        self.spec = dict(spec)
        self.concat = spec["type"] == "dense"
        new = spec["growth"] if self.concat else spec["out_channels"]
        k = spec["kernel_size"]
        self.conv = nn.Conv2d(in_channels, new, k, stride=spec["stride"], padding=k // 2)
        self.in_channels = in_channels
        self.out_channels = in_channels + new if self.concat else new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        return torch.cat([x, y], dim=1) if self.concat else y


class DenseGrowthNet(nn.Module):
    """The stack of blocks grown so far; ``forward`` returns the last feature map."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()

    @property
    def out_channels(self) -> int:
        return self.blocks[-1].out_channels if len(self.blocks) else self.in_channels

    @property
    def stride(self) -> int:
        s = 1
        for b in self.blocks:
            s *= b.spec["stride"]
        return s

    def add_block(self, spec: dict) -> ConvBlock:
        block = ConvBlock(spec, self.out_channels)
        self.blocks.append(block)
        return block

    def forward(self, x: torch.Tensor, upto: Optional[int] = None) -> torch.Tensor:
        for b in self.blocks[:upto]:
            x = b(x)
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """GAP of the last block, ``[B, out_channels]``."""
        return self.forward(x).mean(dim=(2, 3))


def check_table(net: DenseGrowthNet, expected: Dict[str, List[int]], image_size: int) -> None:
    """Raise unless every block has the (channels, resolution) of the expected table."""
    x = torch.zeros(1, net.in_channels, image_size, image_size,
                    device=next(net.parameters()).device)
    with torch.no_grad():
        for b in net.blocks:
            x = b(x)
            got = [x.shape[1], x.shape[-1]]
            want = expected[b.spec["name"]]
            if got != list(want) or x.shape[-2] != x.shape[-1]:
                raise AssertionError(f"{b.spec['name']}: got {tuple(x.shape[1:])}, table says {want}")


def save_stage(net: DenseGrowthNet, out_dir: str, image_size: int, meta: dict) -> None:
    """Encoder only: ``encoder.pt`` (state_dict) + ``architecture.json`` + ``stage_metadata.json``."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()},
               os.path.join(out_dir, "encoder.pt"))
    size, layers = image_size, []
    for b in net.blocks:
        size = (size - 1) // b.spec["stride"] + 1
        layers.append({**b.spec, "in_channels": b.in_channels,
                       "out_channels": b.out_channels, "out_size": size,
                       "residual": "concat" if b.concat else None})
    arch = {"format": "spifil_dense_growth/1", "in_channels": net.in_channels,
            "image_size": image_size, "embed_dim": net.out_channels, "layers": layers}
    with open(os.path.join(out_dir, "architecture.json"), "w") as f:
        json.dump(arch, f, indent=2)
    with open(os.path.join(out_dir, "stage_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_stage(stage_dir: str, map_location: str = "cpu") -> DenseGrowthNet:
    """Rebuild a saved stage: the encoder the evaluators consume."""
    with open(os.path.join(stage_dir, "architecture.json")) as f:
        arch = json.load(f)
    net = DenseGrowthNet(arch["in_channels"])
    for layer in arch["layers"]:
        spec = {k: v for k, v in layer.items()
                if k not in ("in_channels", "out_size", "residual")}
        if spec["type"] == "dense":
            spec.pop("out_channels")
        net.add_block(spec)
    net.load_state_dict(torch.load(os.path.join(stage_dir, "encoder.pt"),
                                   map_location=map_location))
    return net
