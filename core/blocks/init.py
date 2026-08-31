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

"""Congelamento de encoder e as inicializacoes de peso do eixo ``init``.

Um unico dicionario ``INIT_FNS`` mapeia nome -> funcao; e ele que faz o eixo
``init: [flim, he, xavier, random, trunc_normal]`` existir. Consumido pelo YAML
via ``init_args`` com caminho explicito para este modulo. Sem registry, sem
decorator, sem plugin, sem if/elif espalhado por modulo de metodo.

``flim`` e ``random`` mapeiam para ``None`` de proposito -- nao sao funcoes
daqui:
  * ``flim``   -> resolvido pela porta do pacote ``flim`` (``flim/build.py::build``),
                  que hoje vive em src/models/models.py::load_FLIM_encoder (L555) e
                  load_FLIM_encoder_from_arch_dict (L636).
  * ``random`` -> default do torch; nenhuma acao, os pesos ficam como o
                  ``nn.Module.reset_parameters`` deixou.

Movido byte-a-byte de src/models/models.py: freeze_encoder (L672),
unfreeze_encoder (L687), init_weights_he (L697), init_weights_xavier (L710),
init_weights_trunc_normal (L723).

Desvio registrado da arvore normativa: ``unfreeze_norm_layers`` nao esta listada
nela e ficou orfa (nao existe em core/blocks/timm_encoder.py). Como e da familia
freeze/unfreeze, foi copiada VERBATIM de src/models/encoders.py:79 para ca, para
que methods/classification/classification_finetune_module.py pare de importar de
src. A copia em src/ continua onde estava (src/ e read-only nesta refatoracao) e
segue servindo src/modules/classifier_module.py:82.
"""

from __future__ import annotations

import torch.nn as nn


def freeze_encoder(model: nn.Module, except_last: bool = False) -> None:
    """Freeze all encoder parameters.

    ``except_last`` keeps the highest-indexed block (``conv{n_layers}``) trainable: the
    growth curriculum's stage 3 trains ONLY the freshly grown layer, with every older
    block fixed. Default False is the whole-encoder freeze every other caller expects.
    """
    encoder = model.encoder if hasattr(model, "encoder") else model
    for param in encoder.parameters():
        param.requires_grad = False
    if except_last:
        for param in encoder.blocks[f"conv{encoder.n_layers}"].parameters():
            param.requires_grad = True


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all encoder parameters."""
    encoder = model.encoder if hasattr(model, "encoder") else model
    for param in encoder.parameters():
        param.requires_grad = True


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

    # ── Criterion verification ─────────────────────────────────────────
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


# O eixo `init` do YAML. None = a chave e valida mas nao e resolvida aqui.
INIT_FNS = {
    "he": init_weights_he,
    "xavier": init_weights_xavier,
    "trunc_normal": init_weights_trunc_normal,
    "flim": None,
    "random": None,
}
