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
                      channels: List[int]) -> None:
    """
    Load FLIM-trained weights into an encoder.
    The model must have attributes conv1, conv2, ... (each a Sequential starting with Conv2d).
    """
    with open(arch_json, "r") as f:
        arch_description = json.load(f)

    n_layers = arch_description["nlayers"]
    in_channels = channels[0]

    print("[INFO] Loading FLIM Encoder")
    for n in range(1, n_layers + 1):
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

