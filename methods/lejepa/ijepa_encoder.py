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

"""ijepa_encoder.py — Minimal ViT-H/14 I-JEPA encoder.

Loads ``facebook/ijepa_vith14_1k`` weights directly from the HuggingFace cache
using ``safetensors``, without going through ``transformers.AutoModel``.
This avoids the PyTorch >= 2.4 version gate introduced in transformers >= 4.47.

Usage::

    from methods.lejepa.ijepa_encoder import IJEPAEncoder

    encoder = IJEPAEncoder(device=torch.device("cuda"))
    # batch: (B, 3, 224, 224) float32 ImageNet-normalised tensors
    feats = encoder.extract_features(batch)  # (B, 1280)
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_HF_MODEL_ID = "facebook/ijepa_vith14_1k"
_HF_CACHE_SUBDIR = "models--facebook--ijepa_vith14_1k"


# ── Minimal ViT-H/14 building blocks ──────────────────────────────────────────
# Weight names mirror HuggingFace IJepaModel exactly so safetensors loads
# with strict=True without any key remapping.
#
# HF key layout per block:
#   encoder.layer.N.attention.attention.{query,key,value}.{weight,bias}
#   encoder.layer.N.attention.output.dense.{weight,bias}
#   encoder.layer.N.intermediate.dense.{weight,bias}
#   encoder.layer.N.output.dense.{weight,bias}
#   encoder.layer.N.layernorm_{before,after}.{weight,bias}

class _SelfAttention(nn.Module):
    """encoder.layer.N.attention.attention — Q, K, V projections + scaled dot-product."""
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key   = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.query(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.key(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.value(x).reshape(B, N, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, N, C)


class _AttentionOutput(nn.Module):
    """encoder.layer.N.attention.output — output projection."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)


class _AttentionWrapper(nn.Module):
    """encoder.layer.N.attention — wraps SelfAttention + output projection."""
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention = _SelfAttention(hidden_size, num_heads)
        self.output    = _AttentionOutput(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.attention(x))


class _Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.layernorm_before = nn.LayerNorm(hidden_size, eps=eps)
        self.layernorm_after  = nn.LayerNorm(hidden_size, eps=eps)
        self.attention        = _AttentionWrapper(hidden_size, num_heads)
        self.intermediate     = _MLP1(hidden_size, mlp_hidden)
        self.output           = _MLP2(mlp_hidden, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.layernorm_before(x))
        h = self.layernorm_after(x)
        x = x + self.output(self.intermediate(h))
        return x


class _MLP1(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.dense(x))


class _MLP2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)


class _PatchEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)            # (B, C, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, C)


class _Embeddings(nn.Module):
    def __init__(self, hidden_size: int, num_patches: int, patch_size: int) -> None:
        super().__init__()
        self.patch_embeddings   = _PatchEmbeddings(hidden_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_embeddings(x) + self.position_embeddings


class _Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.layer = nn.ModuleList([
            _Block(hidden_size, num_heads, mlp_ratio, eps)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.layer:
            x = blk(x)
        return x


class _IJepaViT(nn.Module):
    """Minimal ViT-H/14 matching facebook/ijepa_vith14_1k weight layout."""

    def __init__(
        self,
        image_size:   int   = 224,
        patch_size:   int   = 14,
        hidden_size:  int   = 1280,
        num_layers:   int   = 32,
        num_heads:    int   = 16,
        mlp_ratio:    float = 4.0,
        eps:          float = 1e-6,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        self.embeddings = _Embeddings(hidden_size, num_patches, patch_size)
        self.encoder    = _Encoder(hidden_size, num_layers, num_heads, mlp_ratio, eps)
        self.layernorm  = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.layernorm(x)
        return x.mean(dim=1)  # (B, hidden_size) — no CLS token in I-JEPA


# ── Weight loading ─────────────────────────────────────────────────────────────

def _find_safetensors_path(model_id: str) -> str:
    """Locate the safetensors file in the HuggingFace cache."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    subdir  = "models--" + model_id.replace("/", "--")
    snaps   = os.path.join(hub_dir, subdir, "snapshots")

    if os.path.isdir(snaps):
        for snap in os.listdir(snaps):
            candidate = os.path.join(snaps, snap, "model.safetensors")
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"Could not find cached model for '{model_id}' in {hub_dir}. "
        "Run: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{model_id}')\" to download it."
    )


def _load_weights(model: _IJepaViT, path: str) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys when loading I-JEPA weights: {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading I-JEPA weights: {unexpected[:5]}")


# ── Public interface ───────────────────────────────────────────────────────────

class IJEPAEncoder(nn.Module):
    """ViT-H/14 I-JEPA encoder loaded from the local HuggingFace cache.

    Accepts batches of ImageNet-normalised tensors with shape
    ``(B, 3, 224, 224)`` and returns mean-pooled patch embeddings of shape
    ``(B, 1280)``.

    Note:
        The model expects inputs pre-processed to 224×224 with ImageNet
        normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
        Use ``_build_test(224)`` from ``lejepa_dataset`` to prepare images.
    """

    EMBEDDING_DIM: int = 1280  # ViT-H hidden size
    IMAGE_SIZE:    int = 224

    def __init__(
        self,
        model_id: str = _HF_MODEL_ID,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[IJEPAEncoder] Building ViT-H/14 from cached weights for '{model_id}'...")
        path = _find_safetensors_path(model_id)
        self._model = _IJepaViT()
        _load_weights(self._model, path)
        self._model.eval()
        self._model.to(self.device)
        print(f"[IJEPAEncoder] Model loaded on {self.device}.")

    @torch.no_grad()
    def extract_features(self, batch: Tensor) -> Tensor:
        """Forward pass — returns mean-pooled patch embeddings.

        Args:
            batch: Float32 tensor of shape ``(B, 3, 224, 224)``,
                   ImageNet-normalised.

        Returns:
            Tensor of shape ``(B, 1280)`` on CPU.
        """
        return self._model(batch.to(self.device)).cpu()

    @torch.no_grad()
    def forward(self, batch: Tensor) -> Tensor:
        """Alias for ``extract_features`` — enables use as nn.Module."""
        return self.extract_features(batch)
