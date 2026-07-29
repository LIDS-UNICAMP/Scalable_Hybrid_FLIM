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
Neural network models for parasite classification experiments.

Builds encoder architectures dynamically from FLIM architecture JSON files,
provides MLP classification heads, and autoencoder decoder for pretraining.
"""

import json
import math
from pathlib import Path, PosixPath
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Weight loading utilities ─────────────────────────────────────────────────


def get_bias(bias_path: PosixPath) -> np.ndarray:
    with open(bias_path, "r") as file:
        n_kernels = file.readline()
        bias = file.readline().strip().split(" ")
    return np.array(bias).astype(np.float32)


def shift_weights(weights: np.ndarray, kernel_size: tuple, in_channels: int) -> np.ndarray:
    nkernels = weights.shape[1]
    shifted_weights = np.zeros((nkernels, in_channels, kernel_size[1], kernel_size[0]))
    for k in range(nkernels):
        for channel in range(in_channels):
            i = channel
            for row in range(kernel_size[1]):
                for col in range(kernel_size[0]):
                    shifted_weights[k][channel][row][col] = weights[i][k]
                    i = i + in_channels
    return shifted_weights


def get_weights(kernel_path: PosixPath, kernel_size: tuple, in_channels: int) -> np.ndarray:
    weights = np.load(kernel_path)
    return shift_weights(weights, kernel_size, in_channels)


# ─── Architecture parsing ─────────────────────────────────────────────────────


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
        bias_path = weights_dir / f"conv{n}-bias.txt"
        with open(bias_path, "r") as f:
            n_kernels = int(f.readline().strip())
        channels.append(n_kernels)
    return channels


def override_arch_channels(arch: dict, channels: List[int]) -> dict:
    """Return a copy of arch with noutput_channels overridden by actual values."""
    import copy
    arch = copy.deepcopy(arch)
    for n in range(1, arch["nlayers"] + 1):
        arch[f"layer{n}"]["conv"]["noutput_channels"] = channels[n]
    return arch


# ─── Encoder ──────────────────────────────────────────────────────────────────


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
            nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, dilation=dilation_rate),
        ]
        if layer_desc.get("relu", False):
            layers.append(nn.ReLU(inplace=True))
        if pool_desc["type"] == "max_pool":
            layers.append(nn.MaxPool2d(pool_size, stride=pool_stride))

        blocks[f"conv{n}"] = nn.Sequential(*layers)
        ch_in = ch_out

    return blocks


class Encoder(nn.Module):
    """Convolutional encoder built from FLIM architecture JSON."""

    def __init__(self, arch: dict, in_channels: int = 3):
        super().__init__()
        self.arch = arch
        self.n_layers = arch["nlayers"]
        self.blocks = build_encoder_from_arch(arch, in_channels)
        # Register blocks as named attributes so load_FLIM_encoder can use getattr
        for name, block in self.blocks.items():
            setattr(self, name, block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in range(1, self.n_layers + 1):
            x = self.blocks[f"conv{n}"](x)
        return x


# ─── Residual FLIM encoder ────────────────────────────────────────────────────
#
# conv1/conv2/conv3 stay byte-identical to the plain FLIM encoder — no new or
# retrained weights anywhere. The residual variants only differ in which
# intermediate activation gets concatenated onto conv3's own (untouched)
# output to form the final embedding: no summation, no gradient, no learning.


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


# ─── MLP classification head ─────────────────────────────────────────────────


class MLPHead(nn.Module):
    """
    MLP classification head with adaptive average pooling to handle variable
    spatial dimensions from the encoder output.
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Full classification model ────────────────────────────────────────────────


class ClassificationModel(nn.Module):
    """Encoder + MLP head for classification."""

    def __init__(self, arch: dict, num_classes: int, in_channels: int = 3,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = Encoder(arch, in_channels)
        channels = get_channels_from_arch(arch, in_channels)
        encoder_out_channels = channels[-1]
        self.head = MLPHead(encoder_out_channels, num_classes, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# ─── Two-layer Sigmoid classification head ────────────────────────────────────


class TwoLayerSigmoidHead(nn.Module):
    """
    Minimal 2-layer classification head with a Sigmoid hidden activation:

        AdaptiveAvgPool2d(1) -> flatten -> Linear(in, hidden) -> Sigmoid
            -> Linear(hidden, num_classes) -> Softmax(dim=1)

    Unlike ``MLPHead`` (3 Linear layers, ReLU, Dropout), this is exactly the
    2-layer Sigmoid architecture used by the FLIM-init classification
    experiment: hidden defaults to ``in_features // 2`` (e.g. 48 -> 24).

    With ``output_relu=True`` a ReLU is inserted between the output Linear and
    the Softmax (``... -> Linear(hidden, num_classes) -> ReLU -> Softmax``),
    clamping negative logits to 0 before normalisation.

    ``forward`` returns post-Softmax class probabilities, not raw logits —
    callers must use ``NLLLoss`` on ``log(probs)`` rather than
    ``CrossEntropyLoss`` (which expects logits and applies its own softmax).
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: Optional[int] = None,
                 output_relu: bool = False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_features // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.output_relu = nn.ReLU() if output_relu else None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.layer1(x))
        x = self.layer2(x)
        if self.output_relu is not None:
            x = self.output_relu(x)
        return self.softmax(x)


class TwoLayerSoftplusHead(nn.Module):
    """
    Variant of ``TwoLayerSigmoidHead`` with a Softplus output activation:

        AdaptiveAvgPool2d(1) -> flatten -> Linear(in, hidden) -> Sigmoid
            -> Linear(hidden, num_classes) -> Softplus -> Softmax(dim=1)

    Softplus replaces the ReLU of ``TwoLayerSigmoidHead(output_relu=True)``.
    ReLU clamps every negative logit to exactly 0 and its derivative there is
    also exactly 0, so the whole negative half-space stops receiving gradient
    and training dies. ``Softplus(z) = log(1 + e^z)`` has derivative
    ``sigmoid(z)``, which is ~0.47 in the operating range actually observed in
    these models (z ≈ -0.13) against 0.0 for ReLU; it is strictly positive
    everywhere, so no unit is ever permanently frozen. It is also nearly
    transparent once the model gains confidence, since ``softplus(z) ≈ z`` for
    large z.

    Submodule names are deliberately identical to ``TwoLayerSigmoidHead``
    (``pool``, ``layer1``, ``sigmoid``, ``layer2``, ``softmax``). Softplus has
    no parameters, so the ``state_dict`` of both heads matches exactly and
    checkpoints remain interchangeable for comparative analysis.

    ``forward`` returns post-Softmax class probabilities, not raw logits —
    callers must use ``NLLLoss`` on ``log(probs)`` rather than
    ``CrossEntropyLoss`` (which expects logits and applies its own softmax).
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_features // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.output_softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.layer1(x))
        x = self.layer2(x)
        x = self.output_softplus(x)
        return self.softmax(x)


class SigmoidClassificationModel(nn.Module):
    """
    FLIM Encoder + two-layer Sigmoid classification head.

    The output activation of the head is selected by two mutually exclusive
    flags: ``output_relu=True`` uses ``TwoLayerSigmoidHead`` with a ReLU before
    the Softmax, ``output_softplus=True`` swaps the head for
    ``TwoLayerSoftplusHead``, and with both ``False`` the plain
    ``TwoLayerSigmoidHead`` is used. Either way the attribute is ``self.head``
    and the ``state_dict`` keys are the same across all three variants.
    """

    def __init__(self, arch: dict, num_classes: int, in_channels: int = 3,
                 hidden_dim: Optional[int] = None, output_relu: bool = False,
                 output_softplus: bool = False):
        super().__init__()
        if output_relu and output_softplus:
            raise ValueError(
                "output_relu and output_softplus are mutually exclusive: "
                "pick at most one output activation for the classification head."
            )
        self.encoder = Encoder(arch, in_channels)
        channels = get_channels_from_arch(arch, in_channels)
        encoder_out_channels = channels[-1]
        if output_softplus:
            self.head = TwoLayerSoftplusHead(encoder_out_channels, num_classes, hidden_dim)
        else:
            self.head = TwoLayerSigmoidHead(encoder_out_channels, num_classes, hidden_dim,
                                            output_relu=output_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# ─── Decoder (for autoencoder) ────────────────────────────────────────────────


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


class AutoEncoder(nn.Module):
    """Autoencoder using the FLIM encoder architecture with a mirrored decoder."""

    def __init__(self, arch: dict, in_channels: int = 3):
        super().__init__()
        self.encoder = Encoder(arch, in_channels)
        self.decoder = build_decoder_from_arch(arch, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        # Crop/pad to match input size if needed due to pooling rounding
        if reconstructed.shape != x.shape:
            reconstructed = nn.functional.interpolate(
                reconstructed, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return reconstructed


class AutoEncoderClassifier(nn.Module):
    """
    Pretrained autoencoder encoder + MLP head.
    The decoder is discarded after pretraining.
    """

    def __init__(self, arch: dict, num_classes: int, in_channels: int = 3,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = Encoder(arch, in_channels)
        channels = get_channels_from_arch(arch, in_channels)
        encoder_out_channels = channels[-1]
        self.head = MLPHead(encoder_out_channels, num_classes, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# ─── FLIM weight loading ─────────────────────────────────────────────────────


def load_FLIM_encoder(model: nn.Module, arch_json: str, weights_path: str,
                      channels: List[int], skip_layers: Optional[set] = None) -> None:
    """
    Load FLIM-trained weights into an encoder.
    The model must have attributes conv1, conv2, ... (each a Sequential starting with Conv2d).

    ``skip_layers`` is an optional set of 1-indexed layer numbers whose FLIM
    weights should NOT be loaded (e.g. a reinstantiated residual conv3 whose
    input channel count no longer matches the pretrained kernels). The default
    ``None`` preserves the original behavior (load every layer).
    """
    with open(arch_json, "r") as f:
        arch_description = json.load(f)

    n_layers = arch_description["nlayers"]
    in_channels = channels[0]

    print("[INFO] Loading FLIM Encoder")
    for n in range(1, n_layers + 1):
        if skip_layers is not None and n in skip_layers:
            print(f"[INFO] Skipping Layer {n} (FLIM weights not loaded)")
            in_channels = channels[n]
            continue
        out_channels = channels[n]
        print(f"[INFO] Loading Layer {n} weights")

        # Get the encoder block (handle both Encoder wrapper and direct model)
        if hasattr(model, "encoder"):
            model_block = getattr(model.encoder, f"conv{n}")
        else:
            model_block = getattr(model, f"conv{n}")

        weights_dir = Path(weights_path)

        # Load bias
        bias_path = weights_dir / f"conv{n}-bias.txt"
        bias = get_bias(bias_path)
        bias = nn.Parameter(torch.from_numpy(bias).float())
        model_block[0].bias.data = bias.to(DEVICE)

        # Load kernel weights
        kernel_path = weights_dir / f"conv{n}-kernels.npy"
        kernel_size = arch_description[f"layer{n}"]["conv"]["kernel_size"]
        weights = get_weights(kernel_path, (kernel_size[0], kernel_size[1]), in_channels)
        weights = nn.Parameter(torch.from_numpy(weights).float())
        model_block[0].weight.data = weights.to(DEVICE)

        in_channels = channels[n]

    print("[INFO] FLIM Encoder loaded successfully")


# ─── Protozoan FLIM architecture (ch24_32_48) ─────────────────────────────────
# Used when encoder_init='flim' for the protozoan dataset.
# The trained FLIM weights for protozoan live in ch24_32_48_a0.5_f5 (layer2=32ch),
# not in ch24_30_48_a0.5_f5 (layer2=30ch) which has no models/ directory.
PROTOZOAN_FLIM_ARCH: dict = {
    "stdev_factor": 0.01,
    "nlayers": 3,
    "apply_intrinsic_atrous": False,
    "layer1": {
        "conv": {"kernel_size": [5, 5, 0], "nkernels_per_marker": 24, "dilation_rate": [1, 1, 0],
                 "nkernels_per_image": 24, "noutput_channels": 24},
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
    "layer2": {
        "conv": {"kernel_size": [5, 5, 0], "nkernels_per_marker": 32, "dilation_rate": [1, 1, 0],
                 "nkernels_per_image": 32, "noutput_channels": 32},
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
    "layer3": {
        "conv": {"kernel_size": [5, 5, 0], "nkernels_per_marker": 48, "dilation_rate": [1, 1, 0],
                 "nkernels_per_image": 48, "noutput_channels": 48},
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
}


def load_FLIM_encoder_from_arch_dict(
    model: nn.Module, arch: dict, weights_path: str, channels: List[int]
) -> None:
    """Load FLIM-trained weights using an arch dict instead of an arch_json path.
    Identical to load_FLIM_encoder but takes the already-parsed arch dict directly.
    """
    n_layers = arch["nlayers"]
    in_channels = channels[0]

    print("[INFO] Loading FLIM Encoder")
    for n in range(1, n_layers + 1):
        print(f"[INFO] Loading Layer {n} weights")

        if hasattr(model, "encoder"):
            model_block = getattr(model.encoder, f"conv{n}")
        else:
            model_block = getattr(model, f"conv{n}")

        weights_dir = Path(weights_path)

        bias_path = weights_dir / f"conv{n}-bias.txt"
        bias = get_bias(bias_path)
        bias = nn.Parameter(torch.from_numpy(bias).float())
        model_block[0].bias.data = bias.to(DEVICE)

        kernel_path = weights_dir / f"conv{n}-kernels.npy"
        kernel_size = arch[f"layer{n}"]["conv"]["kernel_size"]
        weights = get_weights(kernel_path, (kernel_size[0], kernel_size[1]), in_channels)
        weights = nn.Parameter(torch.from_numpy(weights).float())
        model_block[0].weight.data = weights.to(DEVICE)

        in_channels = channels[n]

    print("[INFO] FLIM Encoder loaded successfully")


def freeze_encoder(model: nn.Module) -> None:
    """Freeze all encoder parameters."""
    encoder = model.encoder if hasattr(model, "encoder") else model
    for param in encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all encoder parameters."""
    encoder = model.encoder if hasattr(model, "encoder") else model
    for param in encoder.parameters():
        param.requires_grad = True


# ─── Weight initialization ────────────────────────────────────────────────────


def init_weights_he(model: nn.Module) -> None:
    """Initialize model weights using He (Kaiming) initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_weights_xavier(model: nn.Module) -> None:
    """Initialize model weights using Xavier (Glorot) initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_weights_trunc_normal(model: nn.Module) -> None:
    """Initialize model weights using truncated normal (timm ViT style).

    Applies ``init_weights_vit_timm`` from timm to the whole model, which
    initialises ``nn.Linear`` weights with a truncated normal distribution:
    mean=0, std=0.02, truncation interval [-2σ, +2σ] = [-0.04, +0.04].

    After applying, this function verifies that every ``nn.Linear`` layer in
    the model has no weights outside [-0.04, +0.04].  If any violation is
    found a warning is emitted — the run is not aborted because timm may skip
    layers that are not part of a ViT attention block, but the caller should
    be aware of which layers were not covered.

    Raises:
        ImportError: if ``timm`` is not installed.
    """
    import warnings

    try:
        from timm.models.vision_transformer import init_weights_vit_timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for the 'trunc_normal' initializer. "
            "Install with: pip install timm"
        ) from exc

    model.apply(init_weights_vit_timm)

    # ── Criterion verification ─────────────────────────────────────────────
    # nn.Linear weights must be truncated normal: mean=0, std=0.02, [-2σ,+2σ]
    _EXPECTED_STD = 0.02
    _BOUND = 2.0 * _EXPECTED_STD  # 0.04

    violations: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if (module.weight.data.abs() > _BOUND).any().item():
                violations.append(name)

    if violations:
        warnings.warn(
            f"trunc_normal init criterion check: {len(violations)} nn.Linear "
            f"layer(s) have weights outside [-{_BOUND}, +{_BOUND}] after "
            f"init_weights_vit_timm was applied. This may mean timm did not "
            f"cover those layers (e.g. plain Conv-based backbone). "
            f"Affected: {violations[:5]}"
            + (" ..." if len(violations) > 5 else ""),
            stacklevel=2,
        )


# ─── Main (testing) ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = "/mnt/arquivos_linux/LIBRARY/PHD_planning/hawk/scalable_FLIM_self_supervised"
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # Path to one of the architecture.json files
    arch_json_path = Path(ROOT) / "data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json"

    print(f"Loading architecture from: {arch_json_path}\n")
    arch = parse_architecture(str(arch_json_path))

    print("Architecture JSON:")
    print(json.dumps(arch, indent=2))
    print()

    # Build encoder
    in_channels = 3  # RGB or LAB
    encoder = Encoder(arch, in_channels=in_channels)

    print("="*80)
    print("ENCODER MODEL STRUCTURE")
    print("="*80)
    print(encoder)
    print()

    # Show channel progression
    channels = get_channels_from_arch(arch, in_channels)
    print("="*80)
    print("CHANNEL PROGRESSION")
    print("="*80)
    print(f"Input channels: {channels[0]}")
    for i, ch in enumerate(channels[1:], 1):
        print(f"After layer{i}: {ch} channels")
    print()

    # Test forward pass with a dummy input
    dummy_input = torch.randn(2, in_channels, 200, 200)
    print("="*80)
    print("FORWARD PASS TEST")
    print("="*80)
    print(f"Input shape: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        output = encoder(dummy_input)

    print(f"Output shape: {tuple(output.shape)}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("="*80)
    print("PARAMETERS")
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Test ClassificationModel
    num_classes = 9  # 3 classes from parasito dataset
    clf_model = ClassificationModel(arch, num_classes=num_classes, in_channels=in_channels)

    print("="*80)
    print("CLASSIFICATION MODEL (Encoder + MLP Head)")
    print("="*80)
    print(clf_model)
    print()

    with torch.no_grad():
        logits = clf_model(dummy_input)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Num classes: {num_classes}")
    print()

    total_clf_params = sum(p.numel() for p in clf_model.parameters())
    print(f"Total classification model parameters: {total_clf_params:,}")

    # ── Protozoan 30-channel architecture ─────────────────────────────────────
    protozoan_arch_path = Path(ROOT) / "data/to_mateus/model/ch24_30_48_a0.5_f5/protozoan/train1/architecture.json"
    print()
    print("="*80)
    print("PROTOZOAN 30-CHANNEL MODEL (ch24_30_48)")
    print("="*80)
    if protozoan_arch_path.exists():
        protozoan_arch = parse_architecture(str(protozoan_arch_path))
        protozoan_channels = get_channels_from_arch(protozoan_arch, in_channels)
        print(f"Channel progression: {protozoan_channels}")
        protozoan_encoder = Encoder(protozoan_arch, in_channels=in_channels)
        with torch.no_grad():
            protozoan_out = protozoan_encoder(dummy_input)
        print(f"Input shape:  {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(protozoan_out.shape)}")
        protozoan_params = sum(p.numel() for p in protozoan_encoder.parameters())
        print(f"Total parameters: {protozoan_params:,}")
        protozoan_clf = ClassificationModel(protozoan_arch, num_classes=7, in_channels=in_channels)
        with torch.no_grad():
            protozoan_logits = protozoan_clf(dummy_input)
        print(f"ClassificationModel logits shape (7 classes): {tuple(protozoan_logits.shape)}")
    else:
        print(f"[SKIP] arch JSON not found: {protozoan_arch_path}")

    # ── init_weights_trunc_normal ──────────────────────────────────────────────
    print()
    print("="*80)
    print("INIT_WEIGHTS_TRUNC_NORMAL (timm, std=0.02, clip=[-0.04,+0.04])")
    print("="*80)
    import warnings
    test_encoder = Encoder(arch, in_channels=in_channels)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        init_weights_trunc_normal(test_encoder)
    if caught:
        for w in caught:
            print(f"[WARN] {w.message}")
    else:
        print("Criterion check passed: all nn.Linear weights within [-0.04, +0.04].")
    linear_layers = [(n, m) for n, m in test_encoder.named_modules() if isinstance(m, torch.nn.Linear)]
    if linear_layers:
        for name, lm in linear_layers:
            w = lm.weight.data
            print(f"  {name}: mean={w.mean():.5f}  std={w.std():.5f}  "
                  f"min={w.min():.5f}  max={w.max():.5f}")
    else:
        print("  (no nn.Linear layers in encoder — Conv-only backbone)")
        print("  Verifying Conv2d weights are not overwritten by timm init...")
        conv_layers = [(n, m) for n, m in test_encoder.named_modules() if isinstance(m, torch.nn.Conv2d)]
        for name, cm in conv_layers[:3]:
            w = cm.weight.data
            print(f"  {name}: mean={w.mean():.5f}  std={w.std():.5f}")

    print("\n✓ All models built and tested successfully!")

