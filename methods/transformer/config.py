# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
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

"""Hyperparameters of the FLIM transformer. Pure dataclass, no torch import."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TOKENIZERS",
    "READOUTS",
    "CLASSIFIERS",
    "TOKEN_STATS",
    "FlimTransformerConfig",
    "SUPERPIXEL_METHODS",
    "HEADS",
    "CROSS_READOUTS",
    "CrossTransformerConfig",
]

TOKENIZERS: tuple[str, ...] = ("grid", "superpixel")
READOUTS: tuple[str, ...] = ("gap", "attention_weighted")
CLASSIFIERS: tuple[str, ...] = ("logreg", "linear_svm")
TOKEN_STATS: tuple[str, ...] = ("mean", "mean_std", "mean_max")


@dataclass(frozen=True)
class FlimTransformerConfig:
    # Kernels of the one SPiFiL layer (its out_channels); SPiFiL still picks which ones.
    n_kernels: int = 48
    tokenizer: str = "grid"
    window: int = 5
    token_stats: str = "mean_std"
    min_marked_frac: float = 0.2
    patch_size: int = 3
    neighborhood: int = 3
    delta_percentile: float = 25.0
    global_topk: int = 8
    tau: float = 0.1
    residual_alpha: float = 0.5
    readout: str = "gap"
    classifier: str = "logreg"
    eps: float = 1e-6
    seed: int = 42
    # Q/K builder from methods.transformer.qk.QK_REGISTRY ("shared_patch", "content_context",
    # "cross_scale", "prototype_profile"); qk_params go to its constructor.
    qk_builder: str = "shared_patch"
    qk_params: dict = field(default_factory=dict)
    chunk_threshold: int = 2048
    chunk_size: int = 1024


SUPERPIXEL_METHODS: tuple[str, ...] = ("slic", "disf")
HEADS: tuple[str, ...] = ("appearance", "texture", "context", "per_image")
CROSS_READOUTS: tuple[str, ...] = ("weighted", "area")


@dataclass(frozen=True)
class CrossTransformerConfig:
    """Superpixel tokens, graph self-attention, multi-head cross-attention to the SPiFiL prototypes."""

    tokenizer: str = "superpixel"  # "superpixel" | "grid"
    superpixel_method: str = "slic"
    n_superpixels: int = 200
    min_mask_frac: float = 0.5
    window: int = 5  # grid only
    neighborhood: int = 3  # grid only
    num_layers: int = 2
    self_attention: bool = True
    delta_percentile: float = 25.0
    global_topk: int = 8
    tau_self: float = 0.1
    alpha: float = 0.5
    heads: tuple[str, ...] = ("appearance", "texture", "context")
    tau_cross: float = 0.5
    gate: bool = True
    gate_percentile: float = 10.0
    readout: str = "weighted"  # "weighted" (conf * area) | "area"
    classifier: str = "logreg"
    n_kernels: int = 48
    chunk_threshold: int = 2048
    chunk_size: int = 1024
    eps: float = 1e-6
    seed: int = 42
