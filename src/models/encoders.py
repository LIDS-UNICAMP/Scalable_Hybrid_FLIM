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

from __future__ import annotations

import timm
import torch.nn as nn


def build_encoder(
    arch: str,
    pretrained: bool = False,
    in_chans: int = 3,
    features_only: bool = False,
) -> nn.Module:
    """
    Build a TIMM encoder with the classification head removed.

    Args:
        arch: TIMM model name (e.g. ``"resnet50"``, ``"vit_base_patch16_224"``).
        pretrained: Load ImageNet pretrained weights.
        in_chans: Number of input channels (3 for RGB).
        features_only: Return multi-scale feature maps (CNNs only).

    Returns:
        Encoder ``nn.Module`` with ``num_classes=0`` (no head).
    """
    is_vit = any(k in arch for k in ("vit_"))
    model = timm.create_model(
        arch,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=0,
        features_only=features_only if _supports_features_only(arch) else False,
        **({"dynamic_img_size": True} if is_vit else {}),
    )
    return model


def _supports_features_only(arch: str) -> bool:
    try:
        timm.create_model(arch, features_only=True)
        return True
    except Exception:
        return False


def get_embed_dim(encoder: nn.Module) -> int:
    """Return the output embedding dimension of an encoder."""
    if hasattr(encoder, "num_features"):
        return encoder.num_features
    if hasattr(encoder, "embed_dim"):
        return encoder.embed_dim
    raise AttributeError(f"Cannot infer embed_dim from {type(encoder).__name__}")


def freeze_encoder(model: nn.Module) -> nn.Module:
    """Freeze all parameters in `model`."""
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze_norm_layers(model: nn.Module) -> nn.Module:
    """Unfreeze all normalization layers (BN, LN, GN, IN) for fine-tuning."""
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d,
    )
    for module in model.modules():
        if isinstance(module, norm_types):
            for p in module.parameters():
                p.requires_grad = True
    return model


def unfreeze_encoder(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters in `model`."""
    for p in model.parameters():
        p.requires_grad = True
    return model
