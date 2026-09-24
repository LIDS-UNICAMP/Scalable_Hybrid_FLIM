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
Encoder unico do pacote flim.

Funde os backbones que viviam em src/models/encoders.py (TIMM) e em
src/models/custom_cnn.py (CNN simples) com o encoder FLIM montado a partir do
architecture.json, e absorve em ``from_flim`` o loader de arch/pesos que ate
aqui era colado a mao em cada call site.

Importado so por flim/build.py.
"""

from __future__ import annotations

from typing import List, Optional

import timm
import torch
import torch.nn as nn

from flim.arch import (
    build_encoder_from_arch,
    get_actual_channels_from_weights,
    get_channels_from_arch,
    override_arch_channels,
    parse_architecture,
)
from flim.weights import load_FLIM_encoder, load_FLIM_encoder_from_arch_dict

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d,
)


def _supports_features_only(arch: str) -> bool:
    try:
        timm.create_model(arch, features_only=True)
        return True
    except Exception:
        return False


class Encoder(nn.Module):
    """
    Backbone unico do pacote, em tres sabores:

    - FLIM: ``Encoder(arch, in_channels)`` monta conv1..convN direto do arch, e
      ``Encoder.from_flim(...)`` faz o caminho completo (le o json, acerta os
      canais pelos kernels reais e carrega os pesos FLIM).
    - TIMM: ``Encoder.from_timm(...)``, modelo sem cabeca de classificacao.
    - CNN simples: ``Encoder.from_cnn(...)``, 3 blocos conv + 3 lineares.

    ``forward`` devolve o que cada backbone ja devolvia antes: mapa
    [B, C, H, W] no FLIM, vetor [B, num_features] nos outros dois.

    No sabor FLIM os blocos ficam em ``blocks`` (ModuleDict com chaves
    ``conv1``..``convN``) e tambem como atributos ``conv1``..``convN`` no topo:
    o loader de pesos os procura por ``getattr`` e o curriculo de crescimento
    indexa ``blocks[f"conv{n_layers}"]``.
    """

    def __init__(self, arch: Optional[dict] = None, in_channels: int = 3,
                 blocks: Optional[nn.Module] = None,
                 num_features: Optional[int] = None) -> None:
        """``arch`` monta o encoder FLIM; ``blocks``/``num_features`` recebem um
        backbone ja pronto (TIMM ou CNN) e sao para uso dos construtores abaixo."""
        super().__init__()
        self.arch = arch
        self.n_layers = arch["nlayers"] if arch is not None else 0
        self.blocks = build_encoder_from_arch(arch, in_channels) if arch is not None else blocks
        self.num_features = get_channels_from_arch(arch, in_channels)[-1] if arch is not None else num_features
        for n in range(1, self.n_layers + 1):
            setattr(self, f"conv{n}", self.blocks[f"conv{n}"])

    # ── Construtores ──────────────────────────────────────────────────────────

    @classmethod
    def from_flim(cls, arch_json: str = "", in_channels: int = 3,
                  weights_path: Optional[str] = None,
                  arch: Optional[dict] = None) -> "Encoder":
        """
        Encoder FLIM a partir do ``architecture.json`` — ou de ``arch`` ja
        parseado, caso do protozoan, cujo arch vive em codigo e nao em disco.

        Com ``weights_path`` os canais vem dos kernels que o FLIM realmente
        produziu (podem ser menos que os pedidos no json) e os pesos entram no
        encoder recem-montado. Sem ele fica so a arquitetura, canais do json.
        """
        if arch is None:
            arch = parse_architecture(arch_json)

        if weights_path:
            channels = get_actual_channels_from_weights(weights_path, arch, in_channels)
            arch = override_arch_channels(arch, channels)
        else:
            channels = get_channels_from_arch(arch, in_channels)

        encoder = cls(arch, in_channels)

        if weights_path:
            if arch_json:
                load_FLIM_encoder(encoder, arch_json, weights_path, channels)
            else:
                load_FLIM_encoder_from_arch_dict(encoder, arch, weights_path, channels)
        return encoder

    @classmethod
    def from_timm(cls, arch: str, pretrained: bool = False, in_chans: int = 3,
                  features_only: bool = False) -> "Encoder":
        """Modelo TIMM (ex.: ``resnet50``, ``vit_base_patch16_224``) sem cabeca."""
        model = timm.create_model(
            arch,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            features_only=features_only if _supports_features_only(arch) else False,
            **({"dynamic_img_size": True} if "vit_" in arch else {}),
        )
        for attr in ("num_features", "embed_dim"):
            if hasattr(model, attr):
                return cls(blocks=model, num_features=getattr(model, attr))
        raise AttributeError(f"Cannot infer embed_dim from {type(model).__name__}")

    @classmethod
    def from_cnn(cls, in_channels: int = 3,
                 conv_channels: Optional[List[int]] = None,
                 linear_dims: Optional[List[int]] = None,
                 dropout: float = 0.2) -> "Encoder":
        """
        CNN leve: 3x (Conv-BN-ReLU-Dropout2d-MaxPool), pool global, 3 lineares.
        O ultimo linear e a saida do embedding e nao tem ativacao.
        """
        conv_channels = [32, 64, 128] if conv_channels is None else conv_channels
        linear_dims = [1024, 512, 256] if linear_dims is None else linear_dims
        assert len(conv_channels) == 3, "conv_channels must have exactly 3 elements"
        assert len(linear_dims) == 3, "linear_dims must have exactly 3 elements"

        layers: List[nn.Module] = []
        dim_in = in_channels
        for ch_out in conv_channels:
            layers += [
                nn.Conv2d(dim_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            dim_in = ch_out

        # colapsa o espacial seja qual for a resolucao de entrada
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]

        for dim_out in linear_dims[:-1]:
            layers += [
                nn.Linear(dim_in, dim_out, bias=False),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, linear_dims[-1], bias=False))

        return cls(blocks=nn.Sequential(*layers), num_features=linear_dims[-1])

    # ── Uso ───────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers:
            for n in range(1, self.n_layers + 1):
                x = self.blocks[f"conv{n}"](x)
            return x
        return self.blocks(x)

    def freeze(self, except_last: bool = False) -> "Encoder":
        """
        Congela tudo. ``except_last`` deixa treinavel so o bloco de maior indice
        (``conv{n_layers}``): o estagio 3 do crescimento treina apenas a camada
        recem-nascida, com todo o resto fixo.
        """
        for param in self.parameters():
            param.requires_grad = False
        if except_last:
            for param in self.blocks[f"conv{self.n_layers}"].parameters():
                param.requires_grad = True
        return self

    def unfreeze(self) -> "Encoder":
        """Descongela tudo."""
        for param in self.parameters():
            param.requires_grad = True
        return self

    def unfreeze_norms(self) -> "Encoder":
        """Descongela so as normalizacoes (BN, LN, GN, IN) para fine-tuning."""
        for module in self.modules():
            if isinstance(module, _NORM_TYPES):
                for param in module.parameters():
                    param.requires_grad = True
        return self
