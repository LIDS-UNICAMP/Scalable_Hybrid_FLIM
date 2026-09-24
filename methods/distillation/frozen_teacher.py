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
Wrapper do professor I-JEPA permanentemente congelado.

Movido de src/models/distillation.py:99 (FrozenTeacher) sem nenhuma alteracao de
corpo: mesmo submodulo `_encoder`, mesmos defaults, mesmo comportamento de
`train()` e mesmas chaves de state_dict.

NAO usa core.mixins.FrozenTeacherCheckpointMixin de proposito: aquele mixin e um
hook de LightningModule (on_save_checkpoint / on_load_checkpoint) que poda o
prefixo "teacher." do checkpoint do MODULO. Aqui nao existe logica de checkpoint
nenhuma para deduplicar — esta classe e um nn.Module e so congela pesos.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..lejepa.ijepa_encoder import IJEPAEncoder
from .teacher_constants import TEACHER_DIM, TEACHER_IMAGE_SIZE


class FrozenTeacher(nn.Module):
    """IJEPAEncoder wrapper that enforces a permanently frozen state.

    The model is placed in eval mode and all parameters have
    ``requires_grad=False`` on construction.  The ``register_load_state_dict_post_hook``
    re-applies the freeze after any state_dict restore, guaranteeing the teacher
    never accumulates gradients.

    Args:
        model_id: HuggingFace model identifier for I-JEPA.  Defaults to
            ``"facebook/ijepa_vith14_1k"``.
    """

    def __init__(self, model_id: str = "facebook/ijepa_vith14_1k",
                 frozen: bool = True) -> None:
        super().__init__()
        self._encoder = IJEPAEncoder(model_id=model_id)
        self.frozen: bool = frozen
        if frozen:
            self._freeze()
        else:
            self._encoder.requires_grad_(True)
        self.embed_dim: int = TEACHER_DIM
        self.image_size: int = TEACHER_IMAGE_SIZE

    def _freeze(self) -> None:
        self._encoder.eval()
        self._encoder.requires_grad_(False)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Extract I-JEPA embeddings.

        Args:
            pixel_values: ``(B, 3, 224, 224)`` ImageNet-normalised tensors.

        Returns:
            ``(B, 1280)`` mean-pooled patch embeddings — on CPU when frozen,
            on the teacher device (grad-tracked) when unfrozen.
        """
        if self.frozen:
            with torch.no_grad():
                return self._encoder.extract_features(pixel_values)
        # Unfrozen: bypass extract_features, whose @torch.no_grad() and .cpu()
        # would sever the graph.
        return self._encoder._model(pixel_values.to(self._encoder.device))

    def train(self, mode: bool = True):
        """Stay in eval mode while frozen; follow the trainer once unfrozen."""
        return super().train(mode and not self.frozen)
