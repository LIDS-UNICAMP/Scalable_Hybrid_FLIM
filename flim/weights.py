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
Carregamento dos pesos FLIM treinados fora deste repositorio.

Le os arquivos que o treino FLIM deixa no diretorio ``models/`` de cada run
(``conv{n}-bias.txt`` e ``conv{n}-kernels.npy``), converte o layout do .npy para
o layout de peso do ``nn.Conv2d`` e escreve os tensores num encoder ja
construido.

Formato esperado, para cada camada ``n`` (1-indexada), dentro de
``weights_path``:

  conv{n}-bias.txt     texto. Linha 1 = numero de kernels; linha 2 = os bias
                       separados por espaco. Lido como float32.
  conv{n}-kernels.npy  array ``(kh * kw * in_channels, nkernels)``, float32,
                       com os canais intercalados: a linha de indice
                       ``(row * kw + col) * in_channels + channel`` guarda o
                       peso daquele canal naquela posicao espacial.

Nao interpreta arch (isso vive em ``flim/arch.py``) e nao decide inicializacao:
so carrega e transforma tensor.
"""

from pathlib import Path, PosixPath
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from flim.arch import parse_architecture

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
    arch_description = parse_architecture(arch_json)

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
