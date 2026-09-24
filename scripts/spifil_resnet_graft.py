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
"""spifil_resnet_graft.py — enxerta um bloco SPiFiL no MEIO de uma ResNet-18.

O superpixel é calculado UMA VEZ, na imagem LAB, e nunca mais. O que viaja para
as camadas seguintes são as **sementes**, reposicionadas por divisao inteira de
coordenada conforme o grid encolhe. Este script respeita isso: o LAB continua
sendo quem define onde estao as sementes, e a ResNet entra so como fornecedora
das features que os kernels recortam.

    imagem 224x224
      -> LabNorm              -> superpixels -> sementes   grid (224, 224)
      -> before_graft         -> features (64, 56, 56)     ResNet conv1..layer1
      -> sementes.project(4)                               grid (56, 56)
      -> SpifilNet            -> 2 camadas aprendidas sem backprop
      -> after_graft                                       ResNet layer3..fc

O fator 4 nao e escolhido: e `224 // 56`, o downsample do proprio trecho de
ResNet (conv1 stride 2, maxpool stride 2). Ele e calculado de um forward de
sonda, nao escrito na mao.

Por que nao basta passar a ResNet como ColorTransform
-----------------------------------------------------
Porque ai o superpixel seria calculado em cima do mapa de 64 canais da ResNet,
e nao no LAB. Mecanicamente funciona, mas e outro metodo. Para manter o
superpixel no LAB, a `state[0]` do Learner e montada a mao (`build_layer0`) e o
`prepare()` a aceita em vez de recomputar — mesmo gancho que o callback
`Checkpoint` usa para retomar um fit.

O `Trunk` continua sendo passado como `color=` porque o `Learner` le
`color.out_channels` para dimensionar o primeiro bloco, e o `export()` grava
`describe_color(color)` no `architecture.json`. Deixar o `LabNorm` default ali
gravaria metadado errado: as features fitadas sao da ResNet, nao do LAB.

O que este script NAO faz, de proposito
---------------------------------------
Nao treina o `after_graft`. Os filtros SPiFiL emitem resposta de similaridade
z-scored (spifil/nn/builder.py), nao ativacao de ResNet, e o `layer3` foi
treinado contra a distribuicao do `layer2`, que e outra. O modelo sai montado e
com shape correto; a acuracia so aparece depois de um fine-tune do `after_graft`.

Tambem nao ha conexao residual dentro do trecho enxertado: SpifilConvBlock e
conv -> ReLU -> pool, sem skip. E uma mudanca de arquitetura, nao so de pesos.

Pre-requisitos
--------------
    # ja instalado no env scalable_FLIM; a linha abaixo so em env novo
    pip install -e /dados/home/moliveira/SPiFiL

Uso
---
    python scripts/spifil_resnet_graft.py --images /caminho/images --out runs/graft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.transform import resize as sk_resize

from spifil import ArchSpec, LayerSpec, Learner, SpifilDataset
from spifil import preparation
from spifil.color import LabNorm
from spifil.learner import LayerState
from spifil.superpixels.centers import Medoids
from spifil.superpixels.slic import SLIC

# O `before_graft` e um backbone pre-treinado da torchvision e espera normalizacao
# ImageNet — nao a normalizacao do resto deste repo.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# A ResNet pre-treinada so aceita este tamanho, e e nele que o LAB e o superpixel
# rodam — as coordenadas das sementes nascem neste grid.
IMAGE_SIZE = 224


class Resized(SpifilDataset):
    """Entrega imagem e mascara ja em IMAGE_SIZE x IMAGE_SIZE.

    O redimensionamento tem que acontecer ANTES do LAB, senao as coordenadas das
    sementes nascem num grid que nao e multiplo do grid da ResNet e a projecao
    deixa de ser uma divisao inteira exata.

    `from_folders` e um classmethod que faz `cls(samples)`, entao ele devolve
    esta subclasse sem precisar de nada a mais — desde que nao se acrescente
    parametro obrigatorio no __init__.
    """

    def load_image(self, index: int) -> np.ndarray:
        image = super().load_image(index)
        scaled = sk_resize(
            image, (IMAGE_SIZE, IMAGE_SIZE), order=1, preserve_range=True,
            anti_aliasing=True,
        )
        return np.ascontiguousarray(scaled.astype(np.uint8))

    def load_mask(self, index: int) -> np.ndarray | None:
        mask = super().load_mask(index)
        if mask is None:
            return None
        # order=0: vizinho mais proximo. Mascara e rotulo, nao intensidade —
        # interpolar inventaria rotulo que nao existe.
        scaled = sk_resize(
            mask, (IMAGE_SIZE, IMAGE_SIZE), order=0, preserve_range=True,
            anti_aliasing=False,
        )
        return np.ascontiguousarray(scaled.astype(np.int32))


class Trunk:
    """O pedaco da ResNet antes do enxerto, no formato ColorTransform do SPiFiL.

    Ele NAO calcula superpixel — isso e do LabNorm, em `build_layer0`. Aqui ele
    so produz as features de onde os kernels serao recortados, e serve de
    metadado para o `export()`/`load_encoder`.

    O protocolo pede dois membros: `out_channels` e `__call__(image) -> (C,H,W)`.
    Os dois saem de um forward de sonda no __init__, entao mover o ponto de
    enxerto (passar outro `before_graft`) nao exige corrigir numero nenhum aqui.
    """

    def __init__(self, before_graft: nn.Module, size: int = IMAGE_SIZE) -> None:
        self.before_graft = before_graft.eval()
        self.size = size

        # Congelar nao e detalhe de higiene: cada filtro SPiFiL e um patch
        # recortado DESTAS ativacoes. BatchNorm em modo train, ou qualquer
        # fine-tune do before_graft depois do fit, invalida o banco em silencio.
        for param in self.before_graft.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            probe = self.before_graft(torch.zeros(1, 3, size, size, device=self.device))
        self.out_channels = int(probe.shape[1])  # 64 para resnet18.layer1
        self.grid = int(probe.shape[-1])         # 56 para entrada 224

    @property
    def device(self) -> torch.device:
        return next(self.before_graft.parameters()).device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = (image[..., :3] / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)[None]
        # O dataset ja entrega 224, mas em inferencia via load_encoder a imagem
        # pode chegar em qualquer tamanho; aqui vira no-op no caminho do fit.
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        feats = self.before_graft(x.to(self.device))[0].cpu().numpy()
        return np.ascontiguousarray(feats, dtype=np.float32)


def build_layer0(
    data: SpifilDataset,
    trunk: Trunk,
    n_superpixels: int,
    device: str,
) -> LayerState:
    """Superpixel no LAB, features na ResNet, sementes projetadas entre os dois.

    E a unica parte do script que existe porque o metodo exige — o resto e
    encanamento. O `Learner.prepare()` faria isto sozinho, mas calcularia o
    superpixel em cima das features que o `color` devolvesse, e o `color` aqui e
    a ResNet. Montando a `state[0]` a mao, o superpixel fica onde deve.
    """
    prepared = list(
        preparation.prepare(
            data,
            superpixels=SLIC(),
            seed_extractor=Medoids(),
            n_superpixels=n_superpixels,
            color=LabNorm(),          # <- o superpixel roda AQUI, no LAB, em 224
        )
    )
    if not prepared:
        raise ValueError("dataset vazio: nenhuma imagem encontrada")

    lab_grid = int(prepared[0].seeds.grid[0])
    stride, rest = divmod(lab_grid, trunk.grid)
    if rest:
        raise ValueError(
            f"grid do LAB ({lab_grid}) nao e multiplo do grid da ResNet "
            f"({trunk.grid}); a projecao de sementes precisa de divisao exata"
        )

    features, seeds = [], []
    for index, image in enumerate(prepared):
        features.append(
            torch.from_numpy(trunk(data.load_image(index))).to(device)
        )
        # A transformacao de coordenada: coords // stride, grid ceil(g/stride).
        # Sementes que caem no mesmo pixel se fundem em uma.
        seeds.append(image.seeds.project(stride))

    print(f"[layer 0] superpixel no LAB {lab_grid}x{lab_grid} -> sementes "
          f"projetadas por {stride} para {trunk.grid}x{trunk.grid} "
          f"({sum(len(s) for s in seeds)} sementes em {len(seeds)} imagens)")

    return LayerState(
        features=features,
        seeds=seeds,
        # So inspecao — nada no fit le este campo. Guardado porque nao da para
        # recuperar a partir das sementes.
        superpixel_labels=[image.superpixel_labels for image in prepared],
    )


def build_graft(
    images_dir: Path,
    masks_dir: Path | None,
    out_dir: Path,
    device: str = "cuda",
    n_superpixels: int = 100,
) -> nn.Module:
    """Fita as camadas SPiFiL e devolve a ResNet remontada com elas no meio."""
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().to(device)

    # O corte. `before_graft` produz as features da layer 0; `after_graft` e o
    # que continua depois do enxerto.
    before_graft = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    after_graft = nn.Sequential(
        resnet.layer3, resnet.layer4, resnet.avgpool, nn.Flatten(), resnet.fc
    )
    # Quantos canais o `after_graft` exige na entrada (128 na resnet18). Lido do
    # modelo em vez de escrito na mao, para o corte poder mudar sem quebrar isto.
    need = int(resnet.layer3[0].conv1.in_channels)

    color = Trunk(before_graft)
    data = Resized.from_folders(images_dir, masks_dir)

    # Atencao ao enxertar mais fundo: o patch da camada 1 tem
    # in_channels * kernel^2 dimensoes (64*9 = 576 aqui). A Mahalanobis estima
    # uma covariancia DxD e quer N > D, com N = total de sementes ranqueadas.
    # 8 imagens x ~100 sementes = 800 contra 576 e pouca folga; cortar depois do
    # layer2 (128 canais, D=1152) exigiria mais imagens ou menos superpixels.
    arch = ArchSpec(
        layers=[
            # Camada 1: preserva o grid (pool_type="none"), 56x56 -> 56x56.
            LayerSpec(kernel_size=3, out_channels=color.out_channels,
                      pool_type="none"),
            # Camada 2: faz o downsample que o layer2 fazia, 56x56 -> 28x28.
            LayerSpec(kernel_size=3, out_channels=need,
                      pool_type="max", pool_stride=2),
        ]
    )
    # superpixels/seed_extractor/n_superpixels NAO vao para o Learner: quem os
    # consome e o build_layer0. O Learner so precisa do `color` para dimensionar
    # o primeiro bloco e para o metadado do export.
    learn = Learner(data, arch, color=color, device=device)

    learn.state[0] = build_layer0(data, color, n_superpixels, device)
    learn.prepare()  # ve state[0] pronto, pula a preparacao, so pontua

    # AQUI e o "camada a camada". learn.fit() seria exatamente este laco; rodar
    # na mao e o que deixa inspecionar — ou refazer, com learn.refit_from(L) —
    # uma camada antes de seguir para a proxima.
    for layer in range(1, learn.n_layers + 1):
        learn.fit_layer(layer)
        bank = learn.state[layer].bank
        print(f"[layer {layer}] {len(bank)} filtros | classes {bank.labels.tolist()}")

    # Os filtros SPiFiL nao sao pesos treinaveis: sao patches recortados. Mas o
    # torch cria todo Conv2d com requires_grad=True e o set_filters nao mexe
    # nisso — entao um Adam(model.parameters()) para ajustar o after_graft (que
    # e o proximo passo pretendido) reescreveria os filtros em silencio, sem
    # erro e sem shape mismatch. O notebook oficial sempre chama
    # SpifilEncoder.freeze(); SpifilNet nao tem esse metodo, entao vai na mao.
    for param in learn.model.parameters():
        param.requires_grad_(False)

    learn.export(out_dir)

    # UniformAllocator distribui out_channels // n_classes por classe, entao o
    # total encolhe quando `need` nao e multiplo do numero de classes: 128 com 6
    # classes da 126, e o layer3 recusa. Um 1x1 e a ponte mais barata.
    got = int(learn.model.block(learn.n_layers).out_channels)
    if got == need:
        adapter: nn.Module = nn.Identity()
    else:
        print(f"[graft] {got} filtros != {need} exigidos pelo after_graft; "
              f"1x1 inserido")
        adapter = nn.Conv2d(got, need, kernel_size=1).to(device)

    return nn.Sequential(before_graft, learn.model, adapter, after_graft)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, type=Path,
                        help="pasta de imagens; a classe e o prefixo inteiro do nome")
    parser.add_argument("--masks", type=Path, default=None,
                        help="pasta de mascaras; omitir usa a imagem inteira")
    parser.add_argument("--out", type=Path, default=Path("runs/spifil-resnet"))
    parser.add_argument("--n-superpixels", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42,
                        help="mesmo default do spifil-fit (conf/config.yaml)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)   # unico determinismo que o spifil-fit fixa

    model = build_graft(args.images, args.masks, args.out, args.device,
                        args.n_superpixels)
    print(model)

    # O bundle guarda SO o miolo SPiFiL (Trunk como metadado + os 2 blocos). O
    # adapter 1x1 e o after_graft ficam de fora, entao recarregar o bundle nao
    # reconstroi este modelo — remonte os dois na mao, ou salve o Sequential.
    # E o `color=` e obrigatorio: resolve_color se recusa a reconstruir um
    # ColorTransform que tem argumento de construtor, e o Trunk tem.
    print(f"\nbundle SPiFiL (so o miolo) em {args.out}\n"
          f"  encoder:  load_encoder({str(args.out)!r}, color=Trunk(before_graft))\n"
          f"  completo: torch.save(model, {str(args.out / 'grafted.pt')!r})")


if __name__ == "__main__":
    main()
