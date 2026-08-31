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
FLIM encoder with output-side residual concatenation (no new weights).

conv1/conv2/conv3 stay byte-identical to the plain FLIM encoder — no new or
retrained weights anywhere. The residual variants only differ in which
intermediate activation gets concatenated onto conv3's own (untouched)
output to form the final embedding: no summation, no gradient, no learning.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .arch import build_encoder_from_arch, get_channels_from_arch, parse_architecture
from .weights import load_FLIM_encoder


class _StashConv(nn.Module):
    """Wraps a conv block and stashes its output into a shared dict.

    Needed because the SVM evaluators (``src/utils/evaluate.py``) call
    ``model.conv1`` / ``model.conv2`` / ``model.conv3`` *individually*, so the
    residual conv3 cannot receive conv1's feature map through ``forward``.
    ``conv1`` therefore stashes its output so the residual conv3 can read it
    (only needed for ``mode == "1_3"``; conv2's output is already conv3's own
    input, so no stash is needed for ``mode == "2_3"``).
    ``__getitem__`` delegates to the wrapped block so ``load_FLIM_encoder`` can
    still access ``model_block[0]`` transparently.
    """

    def __init__(self, block: nn.Module, store: dict, key: str):
        super().__init__()
        self.block = block
        self._store = store
        self._key = key

    def __getitem__(self, idx):
        return self.block[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        self._store[self._key] = out
        return out


class _ResidualConv3(nn.Module):
    """Wraps the unmodified, pretrained FLIM conv3 block.

    Runs the exact original conv3 weights on its normal input (o2) to produce
    o3, then concatenates a skip activation onto o3's *output* — never into
    the convolution's input, so the pretrained kernel is never touched:

    * ``mode == "1_3"``: concat(o3, o1) → 48 + 24 = 72 channels. o1 comes from
      the shared stash written by conv1 (:class:`_StashConv`).
    * ``mode == "2_3"``: concat(o3, o2) → 48 + 32 = 80 channels. o2 is this
      block's own input, already available.

    ``__getitem__`` delegates to the wrapped block so ``load_FLIM_encoder``
    can load the pretrained conv3 kernels exactly like the other layers.
    """

    def __init__(self, block: nn.Module, store: dict, mode: str):
        super().__init__()
        self.block = block
        self._store = store
        self._mode = mode

    def __getitem__(self, idx):
        return self.block[idx]

    def forward(self, o2: torch.Tensor) -> torch.Tensor:
        o3 = self.block(o2)
        skip = self._store["o1"] if self._mode == "1_3" else o2
        skip_resized = nn.functional.adaptive_avg_pool2d(skip, o3.shape[-2:])
        return torch.cat([o3, skip_resized], dim=1)


class FLIMResidualEncoder(nn.Module):
    """FLIM encoder with output-side residual concatenation (no new weights).

    conv1/conv2/conv3 are the UNMODIFIED, pretrained FLIM blocks — loaded from
    disk exactly like the plain FLIM baseline, with no reinstantiation and no
    random initialization anywhere. The "residual" variants differ only in
    which intermediate activation gets concatenated onto conv3's own,
    untouched output to form the final embedding fed to the SVM:

    * ``mode == "1_3"``: final = concat(conv3_out[48], conv1_out[24]) → 72 ch.
    * ``mode == "2_3"``: final = concat(conv3_out[48], conv2_out[32]) → 80 ch.

    ``forward`` returns the (un-pooled) concatenated feature map.
    """

    def __init__(self, arch: dict, mode: str, in_channels: int = 3):
        super().__init__()
        if mode not in ("1_3", "2_3"):
            raise ValueError(f"mode must be '1_3' or '2_3', got {mode!r}")
        self.arch = arch
        self.mode = mode
        self.n_layers = arch["nlayers"]
        self._store: dict = {}

        blocks = build_encoder_from_arch(arch, in_channels)

        # conv1 must stash its output only when the skip needs o1 (mode 1_3).
        if mode == "1_3":
            self.conv1 = _StashConv(blocks["conv1"], self._store, "o1")
        else:
            self.conv1 = blocks["conv1"]
        self.conv2 = blocks["conv2"]
        self.conv3 = _ResidualConv3(blocks["conv3"], self._store, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.conv1(x)   # stashes o1 into self._store when mode == "1_3"
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)  # unmodified FLIM conv3 + output-side concat
        return o3


def build_flim_residual_encoder(
    mode: str,
    arch_json_path: str,
    weights_path: str,
    channels: Optional[List[int]] = None,
    in_channels: int = 3,
) -> "FLIMResidualEncoder":
    """Instantiate a :class:`FLIMResidualEncoder` and load FLIM weights.

    All layers (conv1, conv2, conv3) load the pretrained FLIM kernels
    unchanged — the residual concatenation happens on conv3's output, so its
    input channel count (and therefore its pretrained kernel shape) is never
    altered.
    """
    arch = parse_architecture(arch_json_path)
    if channels is None:
        channels = get_channels_from_arch(arch, in_channels)
    encoder = FLIMResidualEncoder(arch, mode, in_channels)
    load_FLIM_encoder(encoder, arch_json_path, weights_path, channels)
    return encoder
