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
Cabeca de duas camadas com ativacao de saida Softplus.

Movida de src/models/models.py:392 (TwoLayerSoftplusHead) sem alteracao de corpo.
Nao compartilha base abstrata com TwoLayerSigmoidHead: a duplicacao aparente e
o que mantem os dois ramos do eixo de comparacao independentes.
"""

from typing import Optional

import torch
import torch.nn as nn


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
