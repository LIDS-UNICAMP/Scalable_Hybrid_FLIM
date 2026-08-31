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
Variante FLIM do autoencoder: encoder FLIM + decoder residual.

Movido de src/models/autoencoder_resnet.py:132 (AutoEncoderFLIM). ``forward`` e
``embed`` sao os mesmos byte a byte. Muda so a porta de entrada do encoder:

- Antes: ``Encoder(arch, in_channels)``, com o arch ja parseado e com os canais
  ja sobrescritos pelo call site (src/modules/autoencoder_flim_module.py:252-258)
  e ``load_FLIM_encoder`` chamado logo depois (linha 261).
- Agora: ``build(arch_json, init, weights_path, in_channels)``, a unica fronteira
  do pacote ``flim``. ``Encoder.from_flim`` (flim/encoder.py:95) faz exatamente as
  mesmas tres etapas na mesma ordem — parse do json, canais reais lidos dos
  ``conv{n}-bias.txt``, override no arch — e carrega os pesos. Este arquivo nao
  abre json nem arquivo de peso por conta propria.

O decoder continua espelhando o MESMO arch que o encoder usou: ele vem de
``self.encoder.arch`` (flim/encoder.py:86), ja com os canais reais. E o
``embed_dim`` vem de ``self.encoder.num_features`` (flim/encoder.py:89), que e o
mesmo ``channels[-1]`` de antes sem precisar do parser de arch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from flim import build

from .resnet_decoder import ResNetDecoder


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
        arch_json: str,
        init: str,
        weights_path: Optional[str] = None,
        in_channels: int = 3,
        out_size: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        self.encoder = build(arch_json, init, weights_path, in_channels)
        self.decoder = ResNetDecoder(self.encoder.arch, in_channels, out_size)
        self.embed_dim: int = self.encoder.num_features
        self._pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled encoder output, ``[B, embed_dim]``."""
        return self._pool(self.encoder(x)).flatten(start_dim=1)
