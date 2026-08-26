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
"""spifil_grow.py — cresce o encoder FLIM em UMA camada, via SPiFiL, sem backprop.

Uma rodada de crescimento. Le o encoder JA TREINADO de um checkpoint, deixa o
SPiFiL recortar UMA camada nova das features que esse encoder produz, e grava o
resultado no formato de pesos do proprio FLIM:

    <ckpt> -> encoder treinado -> features (48, 24, 24)
                                       |
    imagens do split de treino -> LAB -> superpixels -> sementes (200, 200)
                                       |
                                 sementes reescaladas para (24, 24)
                                       |
                              SPiFiL fita 1 camada  ->  conv4-kernels.npy
                                                        conv4-bias.txt
                                                        architecture.json (nlayers 4)

Por que gravar em formato FLIM em vez de guardar um `nn.Sequential`
-------------------------------------------------------------------
Porque assim o modelo crescido NAO e um modelo novo. O `build_encoder_from_arch`
(models.py:113) monta conv1..conv4 a partir do mesmo `architecture.json`, o
`ResNetDecoder` (autoencoder_resnet.py:99) espelha esse mesmo arquivo e ja forca
o `out_size` com um interpolate final, e o `load_FLIM_encoder` (models.py:555) le
os quatro `conv{n}-kernels.npy` sem saber que o ultimo veio do SPiFiL. O treino
do estagio 3 e o mesmo comando do estagio 1 apontando para o novo diretorio.

O `nn.Sequential(encoder_treinado, learn.model)` do enunciado existe aqui: e
exatamente a funcao que estes arquivos reconstroem. A unica diferenca mensuravel
seria o pooling — o `SpifilConvBlock` usa `padding=pool_size//2` e o FLIM usa 0 —
e ela desaparece porque a camada nova nasce com `--pool-stride 1`, isto e, sem
pooling nenhum dos dois lados. Com `--pool-stride 2` a diferenca volta (12 contra
11 pixels) e o decoder absorve, porque ele reescala para o tamanho da imagem de
qualquer jeito.

Por que a camada nova sai do backbone treinado, e nao das imagens
-----------------------------------------------------------------
Um filtro SPiFiL e um patch recortado das ativacoes. Recortar da imagem crua daria
a primeira camada de novo. O que se quer e a camada que vem DEPOIS do que o
modelo ja aprendeu, entao os patches tem que sair das features do encoder no
estado em que o estagio 2 (ou a rodada anterior) o deixou.

O orcamento de covariancia
--------------------------
A Mahalanobis inverte uma covariancia `D x D` com `D = in_channels * kernel^2`.
`D` cresce a cada rodada, `N` (as sementes) nao. O script imprime os dois e
RECUSA (exit 3) quando `N <= D`: o Ledoit-Wolf mantem a metrica *definida*, o que
nao e a mesma coisa que informativa.

Como as imagens da camada nova sao escolhidas
---------------------------------------------
`--one-per-class` (default) e o protocolo do artigo: UMA imagem por classe. A
escolha e HIBRIDA — o id que o artigo publicou (em `spifil_paper_seeds.json`,
ao lado deste arquivo) quando ele cai neste split de treino, senao um sorteio
semeado por `--seed`. Nao da para forcar os ids do artigo: dos 48, so 3 caem no
treino de 5% e 23 no de 50%; puxa-los para dentro quebraria o protocolo de
escassez que o percentual define. Cada escolha sai no log com a origem.

Onde o superpixel roda
----------------------
`--spifil-in-image` (default, e o do artigo) segmenta a imagem LAB e so viaja as
sementes para a grade do encoder. `--spifil-in-feature` segmenta a propria grade
de features, com teto de superpixels.

AVISO: medido, `--spifil-in-feature` nao fecha o orcamento de covariancia junto
com `--one-per-class` — a mascara cobre ~3,2% do quadro, o que em 24x24 vira uma
mediana de 12 posicoes de 576. O `seeds_in_features` tem os numeros.

`--impurities` e a saida para isso: sem mascara a semente cai no quadro inteiro,
e o que ha fora do parasita e impureza. A segmentacao passa a usar as 576
posicoes e o orcamento fecha (N=1287 contra D=432 em eggs/split1/5%, contra N=39
com mascara). O default e `--no-impurities`, que mantem a mascara; o preco de
abrir mao dela esta em `Cropped.load_mask` — leia antes de usar.

Uso
---
    python scripts/spifil_grow.py \
        --ckpt artifacts/.../checkpoints/best_kappa.ckpt \
        --arch-json data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/architecture.json \
        --flim-weights-path data/to_mateus/model/ch24_32_48_a0.5_f5/eggs/train1/models \
        --dataset eggs --split 1 --percentage 100 \
        --out artifacts/spifil_growth/eggs_s1_p100/round1_grow
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from constants import PROJECT_ROOT as _ROOT
# Mesmo valor que o laco le como GROW_EXHAUSTED: um contrato de codigo de saida
# nao pode ter duas fontes. Aliasado para o nome de quem PRODUZ o codigo.
from constants import GROW_EXHAUSTED as EXIT_BUDGET

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from spifil import ArchSpec, LayerSpec, Learner, SpifilDataset
from spifil import preparation
from spifil.color import LabNorm
from spifil.learner import LayerState
from spifil.superpixels.centers import Medoids
from spifil.superpixels.slic import SLIC
from spifil.types import Seeds

from config import get_single_parasite_paths
from src.data_modules.datasets.dataset import ift_lab_loader
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.models.models import parse_architecture
from src.modules.autoencoder_flim_module import (
    AutoEncoderFlimModule,
    _dataset_short_to_parasite_name,
)

# Uma imagem por classe, por split, por dataset — os ids publicados no artigo do
# SPiFiL. Vive num JSON no repositorio para o caminho de crescimento nao depender
# de um clone do SPiFiL nesta maquina; o `_about` de la diz como regenerar.
PAPER_SEEDS_JSON = Path(__file__).with_name("spifil_paper_seeds.json")

# Teto de superpixels quando a segmentacao roda na grade de features: no minimo
# 4 posicoes por superpixel. Abaixo disso o "superpixel" e um pixel isolado, o
# medoide devolve o proprio ponto e o SLIC deixa de reduzir nada — vira uma copia
# da grade. 4 e o menor bloco (2x2) que ainda tem do que escolher.
MIN_POSITIONS_PER_SUPERPIXEL = 4


class BudgetExhausted(RuntimeError):
    """N <= D: nao ha semente que sustente a covariancia que a metrica inverte."""


class GrowOneLayer:
    """`FitStrategy` que visita so a camada nova — e recusa quando N <= D.

    O `SequentialStrategy` do SPiFiL varre `1..n_layers`. Como o `ArchSpec` desta
    rodada carrega UMA `LayerSpec` (uma camada por rodada, nunca duas), "a camada
    nova" e `learn.n_layers`; o que esta estrategia acrescenta ao laco e o
    relatorio N/D e a recusa.

    Ela e injetada por `strategy=` — o gancho que o SPiFiL ja expoe justamente
    para isto. Nada no `Learner` precisa mudar, e nada aqui precisa entrar no
    repositorio do SPiFiL: a politica de parada e deste experimento, nao da
    biblioteca.
    """

    def run(self, learn: Learner) -> None:
        layer = learn.n_layers
        patches = learn.state[layer - 1].patches
        n, d = len(patches), int(patches.feats.shape[1])
        print(f"[grow] N={n} D={d}")
        if n <= d:
            raise BudgetExhausted(
                f"N={n} <= D={d}: a covariancia fica deficiente de posto. O "
                f"shrinkage a mantem inversivel, nao informativa — acrescente "
                f"imagens (--n-images) ou pare de crescer."
            )
        learn.fit_layer(layer)


class Cropped(SpifilDataset):
    """Entrega imagem e mascara com a MESMA geometria que o treino usa.

    `Resize(lado menor)` + `CenterCrop`, que e o `_build_test` do repo sem a
    conversao final para float. Tem que acontecer ANTES do LAB: as sementes
    nascem no grid que o encoder de fato enxerta, e recortar depois as poria em
    cima de pixels que o encoder nunca viu.
    """

    def __init__(self, samples, size: int, *, use_mask: bool = True) -> None:
        super().__init__(samples)
        self.use_mask = use_mask
        self._image = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size), v2.CenterCrop(size),
        ])
        # Mascara e rotulo, nao intensidade: interpolar inventaria rotulo.
        self._mask = v2.Compose([
            v2.ToImage(),
            v2.Resize(size, interpolation=v2.InterpolationMode.NEAREST),
            v2.CenterCrop(size),
        ])

    def load_image(self, index: int) -> np.ndarray:
        image = self._image(super().load_image(index)[..., :3])
        return np.ascontiguousarray(image.permute(1, 2, 0).numpy())

    def load_mask(self, index: int) -> np.ndarray | None:
        # `--impurities` desliga a mascara nos DOIS caminhos de superpixel de uma
        # vez, porque este metodo e o unico lugar por onde ela chega ao SPiFiL:
        # o `preparation.prepare` (caminho da imagem) e o `seeds_in_features`
        # (caminho das features) leem os dois daqui.
        #
        # O CUSTO: sem mascara a semente cai em qualquer lugar do quadro, e como
        # o fundo e ~97% dele a maioria dos candidatos vira recorte de fundo. O
        # ranking de Fisher TENDE a punir esses candidatos — um patch de fundo e
        # parecido em todas as classes, entao separa mal —, mas isso FAVORECE,
        # nao GARANTE; a mascara garantia. O artigo usa mascara exatamente por
        # isso: "They keep every candidate filter on the class of interest
        # rather than on background."
        if not self.use_mask:
            return None
        mask = super().load_mask(index)
        if mask is None:
            return None
        out = self._mask((mask > 0).astype(np.uint8)[..., None])
        return np.ascontiguousarray(out[0].numpy().astype(np.int32))


class Trunk:
    """O encoder FLIM ja treinado, no formato `ColorTransform` do SPiFiL.

    O `Learner` le daqui apenas `out_channels`, para dimensionar o bloco novo.
    As features entram pela `state[0]` montada a mao, entao `__call__` nunca e
    chamado no fit — e por isso ele levanta em vez de adivinhar: a entrada certa
    e o tensor LAB do repo, e passar RGB cru por aqui daria features que o
    encoder nunca viu.

    cada filtro SPiFiL e um patch recortado DESTAS
    ativacoes. Um gradiente no backbone entre o fit e o uso invalida o banco em
    silencio.
    """

    def __init__(self, encoder: torch.nn.Module, size: int, device: str) -> None:
        self.encoder = encoder.eval().to(device)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, 3, size, size, device=device))
        self.out_channels = int(probe.shape[1])
        self.grid = int(probe.shape[-1])

    def __call__(self, image):
        raise NotImplementedError(
            "Trunk so existe para o out_channels; as features vem da state[0]."
        )


def flim_kernels(weight: torch.Tensor) -> np.ndarray:
    """Peso de conv do torch `(k, C, kh, kw)` -> layout `.npy` do FLIM `(C*kh*kw, k)`.

    Inverso exato de `shift_weights` (models.py:48), que le
    `weights[c + C*(row*kw + col)][k]` — canal variando mais rapido. O permute
    poe os eixos nessa ordem e o reshape achata; o `.T` fecha.
    """
    n_kernels, channels, kh, kw = weight.shape
    if kh != kw:
        raise ValueError(f"kernel precisa ser quadrado, veio {kh}x{kw}")
    flat = weight.detach().cpu().permute(0, 2, 3, 1).reshape(n_kernels, kh * kw * channels)
    return np.ascontiguousarray(flat.numpy().T)


def rescale(seeds: Seeds, grid: tuple[int, int]) -> Seeds:
    """Reposiciona as sementes no grid do encoder, proporcionalmente.

    `Seeds.project` divide por um stride inteiro, e aqui nao ha um: o
    `MaxPool2d(3, stride=2, padding=0)` do FLIM leva 200 -> 99 -> 49 -> 24, e
    200/24 nao e inteiro. A conta proporcional e a unica que fecha nas duas
    pontas. Sementes que caem no mesmo pixel viram candidatas duplicadas —
    estado que o proprio `project` admite ("duplicates and all") e que o seletor
    por diversidade descarta.
    """
    old = torch.tensor(seeds.grid, dtype=torch.int64)
    new = torch.tensor(grid, dtype=torch.int64)
    return Seeds(
        coords=seeds.coords * new // old,
        labels=seeds.labels, ranks=seeds.ranks, grid=tuple(grid),
    )


def train_samples(dataset: str, split: int, percentage: int, n_images: int,
                  *, one_per_class: bool, seed: int):
    """Amostras do split de TREINO: uma por classe, ou a fatia de passo uniforme.

    So treino em qualquer um dos dois modos: recortar filtro de imagem de
    validacao vazaria o conjunto que o kappa mede.

    `one_per_class=False` e o comportamento antigo — passo uniforme sobre a lista
    ordenada, que cobre todas as classes porque o nome do arquivo comeca pelo id
    da classe. Da 87 a 200 imagens.

    `one_per_class=True` (o default) e o protocolo do artigo: UMA imagem por
    classe presente no treino, escolhida de forma HIBRIDA — o id publicado no
    artigo quando ele cai neste split de treino, senao um sorteio. Forcar os ids
    do artigo nao e opcao: so 3 dos 48 caem no treino de 5% (23 no de 50%), e
    puxa-los para dentro quebraria a escassez que o percentual define.
    """
    parasite = _dataset_short_to_parasite_name(dataset)
    info = get_single_parasite_paths(parasite, split, percentage)[0]
    names = set(json.loads(Path(info["split_json"]).read_text())["train"])

    everything = SpifilDataset.from_folders(info["images_dir"], info["masks_dir"])
    chosen = [s for s in everything if s.image_path.name in names]
    if not chosen:
        raise ValueError(f"nenhuma imagem de treino em {info['images_dir']}")
    if not one_per_class:
        return chosen[:: max(1, len(chosen) // n_images)][:n_images]

    # Gerador LOCAL, nao o `torch.manual_seed` global: este e o primeiro sorteio
    # de verdade do caminho de crescimento (o SPiFiL nao tem RNG nenhum), e ele
    # tem que reproduzir a partir de --seed sozinho, sem depender de quantas
    # vezes o RNG global foi consumido antes dele.
    rng = random.Random(seed)
    paper = json.loads(PAPER_SEEDS_JSON.read_text(encoding="utf-8"))[dataset]
    paper = paper.get(str(split), {})
    picked = []
    for label in sorted({s.label for s in chosen}):
        same = [s for s in chosen if s.label == label]
        sample = {s.name: s for s in same}.get(paper.get(str(label)))
        origin = "artigo"
        if sample is None:
            sample, origin = rng.choice(same), "sorteio"
        picked.append(sample)
        print(f"[grow] classe {label}: {sample.name} ({origin})")
    print(f"[grow] --one-per-class: {len(picked)} imagens para {len(picked)} classes")
    return picked


def encoder_features(samples, encoder, size, imagenet_norm, device):
    """As features exatamente como o treino as ve: ift_lab + o transform de teste."""
    transform = _build_test(size, imagenet_norm=imagenet_norm)
    out = []
    with torch.no_grad():
        for sample in samples:
            lab = np.ascontiguousarray(ift_lab_loader(sample.image_path))
            x = transform(torch.from_numpy(lab).permute(2, 0, 1).float())
            out.append(encoder(x[None].to(device))[0])
    return out


def seeds_in_features(features, data, n_superpixels, grid):
    """`--spifil-in-feature`: segmenta a PROPRIA grade de features, sem reprojetar.

    O caminho default segmenta a imagem LAB de 200x200 e so viaja as sementes.
    Aqui a segmentacao inteira acontece na grade que o encoder JA produz —
    `trunk.grid`, hoje 24x24 = 576 posicoes. (`--pool-stride` nao muda isso: ele
    dimensiona a SAIDA da camada nova, e portanto so encolhe a grade da RODADA
    SEGUINTE.) Tres coisas mudam por causa disso, e as tres estao no laco abaixo:

    1. **Teto de superpixels.** O numero pedido e limitado pela area da MASCARA
       na grade nova, a `MIN_POSITIONS_PER_SUPERPIXEL` posicoes por superpixel.
       Pedir 100 superpixels num objeto que ocupa ~150 das 576 posicoes daria
       regiao de 1,5 pixel — o medoide devolveria o proprio pixel e o SLIC
       deixaria de reduzir qualquer coisa. O teto efetivo sai no log.
    2. **Mascara reduzida por `nearest`.** Sem isso nao ha como restringir a
       segmentacao ao objeto. Mascara e rotulo: bilinear inventaria valor entre
       dentro e fora, que e o mesmo motivo pelo qual o `Cropped._mask` ja usa
       NEAREST.
    3. **Features normalizadas por imagem.** O `compactness=10` do SLIC pesa
       distancia de cor contra distancia espacial, e foi calibrado para o
       `LabNorm`, que entrega ~[0,1]. Ativacao de ReLU nao tem escala fixa: sem
       normalizar, o mesmo compactness significaria outra coisa a cada rodada, e
       a comparacao entre rodadas deixaria de existir.

    Medido, e o motivo pelo qual isto quase nunca vai fechar
    --------------------------------------------------------
    A mascara destes datasets cobre ~3,2% do quadro. Reduzida para 24x24 sobram
    de 5 a 60 posicoes, mediana 12 — de 576. Nao e o teto que mata: mesmo com o
    teto REMOVIDO (uma semente por posicao mascarada, que ja e o limite duro do
    SLIC) o maximo em eggs/split1/5% com `--one-per-class` e 185 sementes, e
    D=432. Ou seja, COM MASCARA `--spifil-in-feature` + `--one-per-class` NAO TEM
    COMO passar no orcamento de covariancia nesta grade: o `GrowOneLayer` recusa
    com EXIT_BUDGET, corretamente. Com `--no-one-per-class` (124 imagens) o teto
    de 4 posicoes da N=496 contra D=432 — passa raspando, contra N=12313 do
    caminho da imagem. Medido em 2026-08-24; refaca a conta se a mascara, a grade
    ou D mudarem.

    Com `--impurities` a conta e outra: a area vira 576, o teto sobe para os 100
    pedidos e saem 143 sementes por imagem — N=1287 contra D=432 com as mesmas 9
    imagens do `--one-per-class`. E a unica configuracao em que este caminho
    fecha o orcamento sem inchar o numero de imagens.
    """
    slic, medoids = SLIC(), Medoids()
    seeds, regions, caps = [], [], []
    for index, feature in enumerate(features):

        band = feature.detach().cpu().float().numpy()

        band = np.ascontiguousarray(band / max(float(np.abs(band).max()), 1e-8))

        mask = data.load_mask(index)

        if mask is not None:
            mask = F.interpolate(torch.from_numpy(mask)[None, None].float(),
                                 size=grid, mode="nearest")[0, 0].numpy().astype(np.int32)
            
        area = int((mask != 0).sum()) if mask is not None else grid[0] * grid[1]

        cap = max(1, min(n_superpixels, area // MIN_POSITIONS_PER_SUPERPIXEL))

        caps.append(cap)

        labels = slic(band, mask, cap)

        regions.append(labels)

        seeds.append(Seeds.from_coords(
            torch.from_numpy(medoids(labels, band)), label=data[index].label, grid=grid,
        ))

    per = [len(s) for s in seeds]
    print(f"[grow] superpixel NAS FEATURES {grid}: {n_superpixels} pedidos -> teto "
          f"{min(caps)}..{max(caps)} por imagem (>= {MIN_POSITIONS_PER_SUPERPIXEL} "
          f"posicoes por superpixel) | {sum(per)} sementes em {len(per)} imagens, "
          f"{min(per)}..{max(per)} por imagem | mascara "
          f"{'ON' if data.use_mask else 'OFF (--impurities)'}")
    return seeds, regions


def build_layer0(data, trunk, samples, n_superpixels, image_size, imagenet_norm,
                 device, in_feature):
    """Superpixel no LAB, features no encoder treinado, sementes entre os dois.

    E a unica parte do script que existe porque o metodo exige. O `prepare()`
    faria isto sozinho, mas rodaria o superpixel em cima das features do encoder;
    no caminho default o superpixel pertence ao LAB, e so as sementes viajam.

    `in_feature=True` e justamente o que o `prepare()` faria — e por isso ele
    tambem tem que aparecer aqui, com o teto de superpixels que a grade pequena
    exige. Veja `seeds_in_features`.
    """
    grid = (trunk.grid, trunk.grid)
    features = encoder_features(samples, trunk.encoder, image_size, imagenet_norm, device)
    if in_feature:
        seeds, regions = seeds_in_features(features, data, n_superpixels, grid)
    else:
        prepared = list(preparation.prepare(
            data, superpixels=SLIC(), seed_extractor=Medoids(),
            n_superpixels=n_superpixels, color=LabNorm(),
        ))
        seeds = [rescale(image.seeds, grid) for image in prepared]
        regions = [image.superpixel_labels for image in prepared]
        per = [len(s) for s in seeds]
        print(f"[grow] superpixel no LAB {prepared[0].seeds.grid} -> sementes em {grid} "
              f"({sum(per)} sementes em {len(per)} imagens, {min(per)}..{max(per)} por "
              f"imagem | mascara {'ON' if data.use_mask else 'OFF (--impurities)'})")
    return LayerState(features=features, seeds=seeds, superpixel_labels=regions)


def write_weights(out: Path, source: Path, arch: dict, layer: int, block) -> None:
    """Grava o diretorio de pesos completo do modelo crescido, em formato FLIM.

    Completo de proposito: o `AutoEncoderFlimModule` le TODAS as camadas deste
    diretorio no `__init__`, e so depois o `--init-ckpt` sobrescreve as antigas
    com as ja treinadas. Um diretorio so com a camada nova quebraria o
    `get_actual_channels_from_weights`, que abre `conv{n}-bias.txt` de 1 a N.
    """
    out.mkdir(parents=True, exist_ok=True)
    for n in range(1, layer):
        for name in (f"conv{n}-kernels.npy", f"conv{n}-bias.txt"):
            shutil.copy2(source / name, out / name)

    np.save(out / f"conv{layer}-kernels.npy", flim_kernels(block.conv.weight))
    bias = block.conv.bias.detach().cpu().numpy()
    (out / f"conv{layer}-bias.txt").write_text(
        f"{len(bias)}\n" + " ".join(f"{b:.8f}" for b in bias) + "\n"
    )
    (out / "architecture.json").write_text(json.dumps(arch, indent=4) + "\n")


def grown_arch(arch: dict, kernel_size: int, out_channels: int, pool_stride: int) -> dict:
    """`arch` + uma camada. O decoder espelha este dicionario, entao ele basta.

    `ResNetDecoder` le `pooling.stride` de cada camada para escolher o fator de
    upsample (autoencoder_resnet.py:112) e forca o tamanho da imagem no fim
    (:124) — a camada nova do decoder sai daqui, nao escrita a mao. Com
    `pool_stride=1` o tipo vira `"none"`: `build_encoder_from_arch` so acrescenta
    `MaxPool2d` quando o tipo e `"max_pool"`, e o upsample de fator 1 e no-op.
    """
    grown = json.loads(json.dumps(arch))
    layer = grown["nlayers"] + 1
    grown["nlayers"] = layer
    grown[f"layer{layer}"] = {
        "conv": {
            "kernel_size": [kernel_size, kernel_size, 0],
            "nkernels_per_marker": out_channels,
            "dilation_rate": [1, 1, 0],
            "nkernels_per_image": out_channels,
            "noutput_channels": out_channels,
        },
        "relu": True,
        "pooling": {
            "type": "max_pool" if pool_stride > 1 else "none",
            "size": [3, 3, 0],
            "stride": pool_stride,
        },
    }
    return grown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="checkpoint do estagio anterior; a camada nova sai DESTE encoder")
    parser.add_argument("--arch-json", required=True, type=Path)
    parser.add_argument("--flim-weights-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["eggs", "larvae", "protozoan"])
    parser.add_argument("--split", required=True, type=int)
    parser.add_argument("--percentage", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path,
                        help="diretorio de pesos do modelo crescido (arch + conv{1..N+1})")
    parser.add_argument("--out-channels", type=int, default=0,
                        help="0 = a largura do proprio backbone, que mantem D constante entre rodadas")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-stride", type=int, default=1,
                        help="1 = sem pooling. O grid ja e pequeno, e cada pooling funde sementes (derruba N)")
    parser.add_argument("--n-superpixels", type=int, default=100)
    parser.add_argument("--n-images", type=int, default=200,
                        help="imagens de treino que doam patches; e o unico jeito de aumentar N")
    parser.add_argument("--image-size", type=int, default=200)
    parser.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=False)
    # Default = o comportamento de hoje, que e tambem o do artigo: o superpixel
    # roda uma vez na imagem de entrada e as camadas seguintes reusam a posicao.
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--spifil-in-image", action="store_true", default=False,
                       help="DEFAULT: superpixel na imagem LAB, sementes reprojetadas para a grade do encoder")
    where.add_argument("--spifil-in-feature", action="store_true", default=False,
                       help="recalcula os superpixels na grade de features da ultima camada (com teto)")
    parser.add_argument("--impurities", action=argparse.BooleanOptionalAction, default=False,
                        help="--impurities solta a semente no quadro inteiro (sem mascara): fora do "
                             "parasita e impureza. DEFAULT --no-impurities, com mascara "
                             "(veja Cropped.load_mask)")
    parser.add_argument("--one-per-class", action=argparse.BooleanOptionalAction, default=True,
                        help="UMA imagem por classe: id do artigo se ele cair no treino, senao sorteio por --seed")
    parser.add_argument("--random-layer", action="store_true", default=False,
                        help="ABLACAO DE CONTROLE: mantem todo o fluxo (superpixel, orcamento, "
                             "shape) e troca so os VALORES dos filtros por ruido de mesma norma. "
                             "Responde 'a camada SPiFiL ajuda, ou qualquer camada deste tamanho "
                             "ajudaria?'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    module = AutoEncoderFlimModule.load_from_checkpoint(
        args.ckpt, map_location=args.device
    )
    trunk = Trunk(module.model.encoder, args.image_size, args.device)

    arch = parse_architecture(args.arch_json)

    layer = arch["nlayers"] + 1

    out_channels = args.out_channels or trunk.out_channels

    print(f"[grow] backbone de {args.ckpt} | {trunk.out_channels} canais em "
          f"{trunk.grid}x{trunk.grid} | crescendo a camada {layer}")

    samples = train_samples(args.dataset, args.split, args.percentage, args.n_images,
                            one_per_class=args.one_per_class, seed=args.seed)
    
    data = Cropped(samples, args.image_size, use_mask=not args.impurities)

    
    learn = Learner(
        data,
        ArchSpec(layers=[LayerSpec(
            kernel_size=args.kernel_size, out_channels=out_channels,
            pool_type="max" if args.pool_stride > 1 else "none",
            pool_stride=args.pool_stride,
        )]),
        color=trunk, strategy=GrowOneLayer(), device=args.device,
    )
    learn.state[0] = build_layer0(
        data, trunk, samples, args.n_superpixels, args.image_size,
        args.imagenet_norm, args.device, args.spifil_in_feature,
    )
    try:
        learn.fit()  # prepare() ve a state[0] pronta, so pontua; a estrategia fita
    except BudgetExhausted as exhausted:
        print(f"[grow] RECUSADO: {exhausted}")
        return EXIT_BUDGET

    block = learn.model.block(learn.n_layers)
    # A largura vem do modelo, nunca do pedido: o UniformAllocator distribui
    # out_channels // n_classes por classe e joga o resto fora (48 em 9 classes
    # da 45). Um decoder dimensionado pelo pedido nao falha — ele erra calado.
    got = int(block.out_channels)
    if got != out_channels:
        print(f"[grow] {out_channels} pedidos -> {got} filtros (resto do allocator)")

    if args.random_layer:
        # Troca so os VALORES, no ultimo instante: shape, largura do allocator e
        # architecture.json continuam vindo do mesmo `block`, entao a familia aleatoria e
        # identica a SPiFiL em tudo menos nos pesos. Como o orcamento (N <= D) ja foi
        # cobrado la em cima, ela tambem recusa os MESMOS bracos — o controle sai pareado.
        # A norma e copiada FILTRO A FILTRO do proprio tensor, nao normalizada para 1: o
        # SPiFiL faz `patch_zscored / desvio`, o que espalha a norma (medido: 0.36 a 1.19,
        # media 0.54). Sortear com escala propria mediria magnitude de ativacao em vez de
        # estrutura, entao o ruido herda a escala exata do filtro que substitui — o que
        # varia e so a DIRECAO, que e o que o superpixel escolhe.
        # O bias do SPiFiL e -(K . media do patch); sem patch ele perde sentido e vai a zero.
        with torch.no_grad():
            w = block.conv.weight
            g = torch.Generator().manual_seed(args.seed)
            noise = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
            escala = (w.flatten(1).norm(dim=1) / noise.flatten(1).norm(dim=1))
            w.copy_(noise * escala.view(-1, *([1] * (w.dim() - 1))))
            if block.conv.bias is not None:
                block.conv.bias.zero_()
        print(f"[grow] --random-layer: {got} filtros sorteados (seed={args.seed}, "
              f"norma por filtro preservada, bias=0)")

    write_weights(args.out, args.flim_weights_path,
                  grown_arch(arch, args.kernel_size, got, args.pool_stride), layer, block)
    
    learn.export(args.out / "spifil_bundle")  # labels{L}.txt: de que classe veio cada filtro

    print(f"[grow] camada {layer} com {got} filtros em {args.out}\n"
          f"  --arch-json {args.out / 'architecture.json'}\n"
          f"  --flim-weights-path {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
