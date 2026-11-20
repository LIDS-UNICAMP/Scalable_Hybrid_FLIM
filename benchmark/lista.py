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
"""Lista de benchmarks do usuario (08/11): 19 modelos congelados, pesos oficiais do HF, mesmo protocolo de `common`.

Cada entrada e (repositorio, revisao, pre-processamento do timm, carregador). Nenhum peso vem do timm: o timm so
da o pre-processamento e, para AIMv2 e ConvNeXt XXLarge, a arquitetura onde os pesos oficiais sao carregados.
Bases: as 14 de ORGANIZED sem food101. Uma divisao combinada: grandes no jaci, o resto no athena.

`python -m benchmark.lista extract <nome> ...` grava os embeddings (GPU); `python -m benchmark.lista svm <nome> ...`
roda o SVM oficial a 50%. `python -m benchmark.lista nomes` lista as entradas.
"""
import sys

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import os

from benchmark import common
from benchmark.common import extract, svm
from core.constants import ORGANIZED

if not os.access("/exps/moliveira", os.W_OK):  # jaci: /exps/moliveira nao existe; tudo em /dados/home/moliveira/exps
    JACI_EXPS = "/dados/home/moliveira/exps"
    common.EMB_DIR = f"{JACI_EXPS}/benchmark/emb"
    common.STATUS = f"{JACI_EXPS}/queues/logs/bench.status"
    common.SVM_LOCK = f"{JACI_EXPS}/queues/svm.lock"
    common.GPU_CAP_MIB = 80000  # GPU de 96 GB; o benchmark so entra depois que a fila SPiFiL do jaci acabar
else:
    common.GPU_CAP_MIB = 15000  # athena: no maximo 15 GB nossos por GPU (regra do usuario desde 07/11)

DATASETS = [d for d in ORGANIZED if d != "food101"]
AIM_NAMES = {"rms_norm1": "norm1", "rms_norm2": "norm2", "attention.out_proj": "attn.proj",
             "ffn.gate_proj": "mlp.fc1_g", "ffn.up_proj": "mlp.fc1_x", "ffn.down_proj": "mlp.fc2"}


def aimv2(repo: str, rev: str, arch: str):
    """Pesos oficiais da Apple renomeados para o ViT do timm, como em `benchmark/aimv2.py`; profundidade lida dos pesos."""
    def load():
        sd = load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
        depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
        out = {"pos_embed": sd["embeddings.position_embedding.weight"][None],
               "patch_embed.proj.weight": sd["embeddings.patch_embed.weight"],
               "patch_embed.proj.bias": sd["embeddings.patch_embed.bias"],
               "patch_embed.norm.weight": sd["embeddings.rms_norm.weight"], "norm.weight": sd["rms_norm.weight"]}
        for i in range(depth):
            old = f"encoder.layers.{i}."
            out |= {f"blocks.{i}.{b}.weight": sd[f"{old}{a}.weight"] for a, b in AIM_NAMES.items()}
            out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([sd[f"{old}attention.{p}_proj.weight"] for p in "qkv"])
        net = timm.create_model(arch, pretrained=False, num_classes=0)
        net.load_state_dict(out)
        return net, net
    return load


def dinov2(repo: str, rev: str):
    def load():
        from transformers import Dinov2Model
        net = Dinov2Model.from_pretrained(repo, revision=rev)
        return net, lambda x: net(x).pooler_output
    return load


def convnext(repo: str, rev: str):
    def load():
        from transformers import ConvNextModel
        net = ConvNextModel.from_pretrained(repo, revision=rev)
        return net, lambda x: net(x).pooler_output
    return load


def convnext_clip(repo: str, rev: str):
    """Torre visual do CLIP oficial da LAION (formato open_clip) carregada no ConvNeXt do timm, sem a projecao."""
    def load():
        sd = torch.load(hf_hub_download(repo, "open_clip_pytorch_model.bin", revision=rev), map_location="cpu")
        trunk = {k[len("visual.trunk."):]: v for k, v in sd.items() if k.startswith("visual.trunk.")}
        net = timm.create_model("convnext_xxlarge", pretrained=False, num_classes=0)
        missing, unexpected = net.load_state_dict(trunk, strict=False)
        if missing or [k for k in unexpected if not k.startswith("head.fc")]:
            raise RuntimeError(f"convnext_xxlarge: faltam {missing[:5]}, sobram {unexpected[:5]}")
        return net, net
    return load


def ijepa(repo: str, rev: str, **arch):
    def load():
        from methods.lejepa.ijepa_encoder import _IJepaViT, _load_weights
        net = _IJepaViT(**arch)
        _load_weights(net, hf_hub_download(repo, "model.safetensors", revision=rev))
        return net, net
    return load


def resnet50(repo: str, rev: str):
    """Pesos oficiais do timm no HF; num_classes=0 deixa so o pooling medio global, 2048-d."""
    def load():
        sd = load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
        net = timm.create_model("resnet50", pretrained=False, num_classes=0)
        net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
        return net, net
    return load


A = "apple/aimv2-{}-patch14-{}"
MODELS = {
    "aimv2_large_224": (A.format("large", 224), "fcb5093a9c5b3efb7db5acee213849967fd18210", "aimv2_large_patch14_224.apple_pt"),
    "aimv2_large_336": (A.format("large", 336), "7819fc7a5cf07b51cc14794056d08eebb7880bcc", "aimv2_large_patch14_336.apple_pt"),
    "aimv2_large_448": (A.format("large", 448), "6ef0b3123a5ec9a1c0482773e261d96161c64927", "aimv2_large_patch14_448.apple_pt"),
    "aimv2_huge_224": (A.format("huge", 224), "172051d9df825cda55607cfe9016da30f249a133", "aimv2_huge_patch14_224.apple_pt"),
    "aimv2_huge_336": (A.format("huge", 336), "d8d604d9ddbdf1af1d7302cc16f1b4d882e7b73c", "aimv2_huge_patch14_336.apple_pt"),
    "aimv2_huge_448": (A.format("huge", 448), "f723839533d3bbdc969f541c864789f531ec0e5c", "aimv2_huge_patch14_448.apple_pt"),
    "dinov2_small": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2_base": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "vit_base_patch14_dinov2.lvd142m"),
    "dinov2_large": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", "vit_large_patch14_dinov2.lvd142m"),
    "dinov2_giant": ("facebook/dinov2-giant", "611a9d42f2335e0f921f1e313ad3c1b7178d206d", "vit_giant_patch14_dinov2.lvd142m"),
    "convnext_base": ("facebook/convnext-base-224-22k", "a801e9ad0e52947a4583a14696afbe20decbf89e", "convnext_base.fb_in22k"),
    "convnext_large": ("facebook/convnext-large-224-22k", "4d5c8052c18a254f587ef463b5199d4bfd198a38", "convnext_large.fb_in22k"),
    "convnext_xlarge": ("facebook/convnext-xlarge-224-22k", "fc6b9974e8ce68f2880a5ff3589a90556e34e332", "convnext_xlarge.fb_in22k"),
    "convnext_xxlarge_clip": ("laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup", "9f3e8ee3f383c672388d9178afe70af9e63ac9df",
                              "convnext_xxlarge.clip_laion2b_soup"),
    "ijepa_vith14_1k": ("facebook/ijepa_vith14_1k", "f157467ea509bc356ff9f61fd3c0d840eec5e04e", "vit_huge_patch14_gap_224.in1k_ijepa"),
    "ijepa_vith16_1k": ("facebook/ijepa_vith16_1k", "43ea3b33a17addfa9a32261f6fde003fd97f38d9", "vit_huge_patch16_gap_448.in1k_ijepa"),
    "ijepa_vith14_22k": ("facebook/ijepa_vith14_22k", "ba3c4513ca2b0f0c010f80ae2265b3dbe1083039", "vit_huge_patch14_gap_224.in22k_ijepa"),
    "ijepa_vitg16_22k": ("facebook/ijepa_vitg16_22k", "2306205e956b753a39dae4fd38ec1c74145257d3", "vit_giant_patch16_gap_224.in22k_ijepa"),
    "resnet50": ("timm/resnet50.a1_in1k", "767268603ca0cb0bfe326fa87277f19c419566ef", "resnet50.a1_in1k"),
}
LOADERS = {
    **{k: aimv2(*v[:2], v[2].split(".")[0]) for k, v in MODELS.items() if k.startswith("aimv2")},
    **{k: dinov2(*v[:2]) for k, v in MODELS.items() if k.startswith("dinov2")},
    **{k: convnext(*v[:2]) for k, v in MODELS.items() if k in ("convnext_base", "convnext_large", "convnext_xlarge")},
    "convnext_xxlarge_clip": convnext_clip(*MODELS["convnext_xxlarge_clip"][:2]),
    "ijepa_vith14_1k": ijepa(*MODELS["ijepa_vith14_1k"][:2]),
    "ijepa_vith16_1k": ijepa(*MODELS["ijepa_vith16_1k"][:2], image_size=448, patch_size=16),
    "ijepa_vith14_22k": ijepa(*MODELS["ijepa_vith14_22k"][:2]),
    "ijepa_vitg16_22k": ijepa(*MODELS["ijepa_vitg16_22k"][:2], patch_size=16, hidden_size=1408, num_layers=40, mlp_ratio=48 / 11),
    "resnet50": resnet50(*MODELS["resnet50"][:2]),
}
PENDENTE = {"aimv2_large_native": "apple/aimv2-large-patch14-native: resolucao variavel, sem pre-processamento fixo no timm"}
JACI = ["ijepa_vitg16_22k", "dinov2_giant", "convnext_xxlarge_clip", "aimv2_huge_224", "aimv2_huge_336", "aimv2_huge_448"]
ATHENA = [k for k in MODELS if k not in JACI]


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "nomes":
        print("jaci:", *JACI, "\nathena:", *ATHENA, "\npendente:", *PENDENTE)
    for name in names:
        repo, rev, pre = MODELS[name]
        if cmd == "extract":
            extract(repo, rev, pre, LOADERS[name], datasets=DATASETS)
        elif cmd == "svm":
            for dataset in DATASETS:
                svm(repo, dataset, 50)
