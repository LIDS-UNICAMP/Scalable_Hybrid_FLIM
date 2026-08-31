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

"""spifil.py — SPiFiL sem backprop: `grow` cresce o encoder FLIM, `graft` enxerta na ResNet.

Duas funcoes puras, uma por protocolo. Nenhuma delas le linha de comando, abre
processo ou fala com um agendador: quem faz isso e o runner de orquestracao
(`experiments/ray/runners/growth.py`), que chama `grow` uma vez por rodada.

    grow  — UMA camada nova recortada das features do encoder FLIM ja treinado
    graft — DUAS camadas novas recortadas do miolo de uma ResNet-18 pre-treinada


Handoff: o que `experiments/ray/runners/growth.py` tem que passar
=================================================================
`grow` NAO abre checkpoint. Quem monta o braco e que carrega o modulo e entrega
o encoder pronto:

    module = AutoEncoderFlimModule.load_from_checkpoint(ckpt, map_location=device)
    codigo = grow(encoder=module.model.encoder, ckpt=ckpt, ...)

E de proposito. O `AutoEncoderFlimModule` mora do lado dos metodos, e a fronteira
do projeto so tem uma direcao: `methods/` importa `flim`, `flim` nunca importa
`methods/`. Carregar o checkpoint aqui dentro inverteria isso — por import no
topo ou por import tardio, tanto faz, o segundo so esconde a inversao do grep.
Por isso `grep -nE '^\s*(import|from)\s+(src|methods)\b' flim/spifil.py` tem que
continuar VAZIO.

Tres consequencias praticas para o runner:

1. `encoder` e o UNICO objeto que `grow` precisa do lado dos metodos: um
   `nn.Module` que aceita `(1, 3, image_size, image_size)`. Nao passe o
   LightningModule inteiro — `grow` nao le mais nada dele.
2. `random_layer_classic=True` nao tem backbone nenhum e sai antes de qualquer
   coisa pesada. Passe `encoder=None` nesse braco e NAO carregue checkpoint: era
   exatamente isso que o `return` antes do load ja economizava.
3. `ckpt` continua na assinatura, mas agora e so ETIQUETA — o caminho que sai no
   log de procedencia. Passe o mesmo de onde o encoder veio, senao o log mente.


grow — uma rodada de crescimento
================================
Recebe o encoder JA TREINADO (carregado pelo chamador), deixa o SPiFiL recortar
UMA camada nova das features que esse encoder produz, e grava o resultado no
formato de pesos do proprio FLIM:

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
monta conv1..conv4 a partir do mesmo `architecture.json`, o `ResNetDecoder`
(autoencoder_resnet.py:99) espelha esse mesmo arquivo e ja forca o `out_size` com
um interpolate final, e o `load_FLIM_encoder` le os quatro `conv{n}-kernels.npy`
sem saber que o ultimo veio do SPiFiL. O treino do estagio 3 e o mesmo comando do
estagio 1 apontando para o novo diretorio.

O `nn.Sequential(encoder_treinado, learn.model)` do enunciado existe aqui: e
exatamente a funcao que estes arquivos reconstroem. A unica diferenca mensuravel
seria o pooling — o `SpifilConvBlock` usa `padding=pool_size//2` e o FLIM usa 0 —
e ela desaparece porque a camada nova nasce com `pool_stride=1`, isto e, sem
pooling nenhum dos dois lados. Com `pool_stride=2` a diferenca volta (12 contra
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
`D` cresce a cada rodada, `N` (as sementes) nao. A funcao imprime os dois e
RECUSA (devolve `EXIT_BUDGET`, o mesmo 3 que o laco le como `GROW_EXHAUSTED`)
quando `N <= D`: o Ledoit-Wolf mantem a metrica *definida*, o que nao e a mesma
coisa que informativa.

Isto vale tambem para `random_layer=True`, que e o MESMO caminho: o orcamento e
cobrado antes da troca dos pesos. So `random_layer_classic=True` escapa dele,
porque nao fita nada — sem fit nao ha N, nao ha covariancia e nao ha recusa.

Os dois controles: `random_layer` e `random_layer_classic`
----------------------------------------------------------
Sao perguntas diferentes, por isso sao dois parametros (mutuamente exclusivos).

`random_layer` (PAREADO) responde "a camada do SPiFiL e melhor que uma aleatoria
DO MESMO SHAPE?". Roda o pipeline inteiro — checkpoint, features, SLIC, sementes,
ranking, allocator, orcamento — e troca so os VALORES dos pesos no ultimo instante,
logo antes do `write_weights`, preservando a norma FILTRO A FILTRO. Consequencias
que definem o pareamento: a camada sai com os 45 canais do allocator (nao 48), o
`architecture.json` fica byte-identico ao do caminho SPiFiL, e ela RECUSA os mesmos
bracos (larvae sai com `EXIT_BUDGET`). Mesmo shape, mesmos bracos, mesmo protocolo:
a unica diferenca e a DIRECAO dos filtros.

`random_layer_classic` responde "acrescentar uma camada qualquer ajuda?". Le a
`architecture.json` e os pesos de entrada, sorteia `kaiming_normal_` (fan_out, relu)
com bias zero e grava no mesmo formato. Sem checkpoint, sem dado, sem superpixel, sem
semente, sem allocator — e portanto sem orcamento de covariancia. Consequencias: a
camada sai com o numero de canais PEDIDO (48, e nao os 45 do allocator) e cresce em
braco onde o SPiFiL recusa por `N <= D`. Shape e bracos DIFERENTES da variante SPiFiL,
por isso ela nao serve de controle pareado.

Como as imagens da camada nova sao escolhidas
---------------------------------------------
`one_per_class=True` (default) e o protocolo do artigo: UMA imagem por classe. A
escolha e HIBRIDA — o id que o artigo publicou (em `spifil_paper_seeds.json`)
quando ele cai neste split de treino, senao um sorteio semeado por `seed`. Nao da
para forcar os ids do artigo: dos 48, so 3 caem no treino de 5% e 23 no de 50%;
puxa-los para dentro quebraria o protocolo de escassez que o percentual define.
Cada escolha sai no log com a origem.

Onde o superpixel roda
----------------------
`spifil_in_image=True` (default de fato, e o do artigo) segmenta a imagem LAB e
so viaja as sementes para a grade do encoder. `spifil_in_feature=True` segmenta a
propria grade de features, com teto de superpixels.

AVISO: medido, `spifil_in_feature` nao fecha o orcamento de covariancia junto
com `one_per_class` — a mascara cobre ~3,2% do quadro, o que em 24x24 vira uma
mediana de 12 posicoes de 576. O `seeds_in_features` tem os numeros.

`impurities=True` e a saida para isso: sem mascara a semente cai no quadro inteiro,
e o que ha fora do parasita e impureza. A segmentacao passa a usar as 576
posicoes e o orcamento fecha (N=1287 contra D=432 em eggs/split1/5%, contra N=39
com mascara). O default e `impurities=False`, que mantem a mascara; o preco de
abrir mao dela esta em `Cropped.load_mask` — leia antes de usar.


graft — enxerto de um bloco SPiFiL no MEIO de uma ResNet-18
===========================================================
O superpixel e calculado UMA VEZ, na imagem LAB, e nunca mais. O que viaja para
as camadas seguintes sao as **sementes**, reposicionadas por divisao inteira de
coordenada conforme o grid encolhe. Este caminho respeita isso: o LAB continua
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
superpixel no LAB, a `state[0]` do Learner e montada a mao (`graft_layer0`) e o
`prepare()` a aceita em vez de recomputar — mesmo gancho que o callback
`Checkpoint` usa para retomar um fit.

O `GraftTrunk` continua sendo passado como `color=` porque o `Learner` le
`color.out_channels` para dimensionar o primeiro bloco, e o `export()` grava
`describe_color(color)` no `architecture.json`. Deixar o `LabNorm` default ali
gravaria metadado errado: as features fitadas sao da ResNet, nao do LAB.

O que `graft` NAO faz, de proposito
-----------------------------------
Nao treina o `after_graft`. Os filtros SPiFiL emitem resposta de similaridade
z-scored (spifil/nn/builder.py), nao ativacao de ResNet, e o `layer3` foi
treinado contra a distribuicao do `layer2`, que e outra. O modelo sai montado e
com shape correto; a acuracia so aparece depois de um fine-tune do `after_graft`.

Tambem nao ha conexao residual dentro do trecho enxertado: SpifilConvBlock e
conv -> ReLU -> pool, sem skip. E uma mudanca de arquitetura, nao so de pesos.

O bundle que `graft` grava guarda SO o miolo SPiFiL (o `GraftTrunk` como metadado
+ os 2 blocos). O adapter 1x1 e o `after_graft` ficam de fora, entao recarregar o
bundle nao reconstroi o modelo devolvido — remonte os dois na mao, ou salve o
`nn.Sequential`. E o `color=` e obrigatorio na volta: `resolve_color` se recusa a
reconstruir um `ColorTransform` que tem argumento de construtor, e o `GraftTrunk`
tem.

Pre-requisitos
--------------
    # ja instalado no env scalable_FLIM; a linha abaixo so em env novo
    pip install -e /dados/home/moliveira/SPiFiL

Uso
---
    from flim.spifil import grow, graft

    codigo = grow(encoder=module.model.encoder, ckpt=..., arch_json=...,
                  flim_weights_path=..., dataset="eggs", split=1,
                  percentage=100, out=...)
    model = graft(images_dir=..., out_dir=...)
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.transform import resize as sk_resize
from torchvision.transforms import v2

from spifil import ArchSpec, LayerSpec, Learner, SpifilDataset
from spifil import preparation
from spifil.color import LabNorm
from spifil.learner import LayerState
from spifil.superpixels.centers import Medoids
from spifil.superpixels.slic import SLIC
from spifil.types import Seeds

from config import get_single_parasite_paths
from core.data.loaders import ift_lab_loader
from core.data.transforms import build_test

from .arch import get_actual_channels_from_weights, parse_architecture
# Mesmo valor que o laco le como GROW_EXHAUSTED: um contrato de codigo de saida
# nao pode ter duas fontes. Aliasado para o nome de quem PRODUZ o codigo.
from .constants import GROW_EXHAUSTED as EXIT_BUDGET

# Uma imagem por classe, por split, por dataset — os ids publicados no artigo do
# SPiFiL. Vive num JSON no repositorio para o caminho de crescimento nao depender
# de um clone do SPiFiL nesta maquina; o `_about` de la diz como regenerar.
# O JSON continua em `scripts/` porque esta fase nao move arquivo de dado; quando
# ele vier para o lado do pacote isto volta a ser um `with_name`.
PAPER_SEEDS_JSON = Path(__file__).resolve().parent.parent / "scripts" / "spifil_paper_seeds.json"

# Chave curta -> nome registrado do parasita, que e o que `get_single_parasite_paths`
# exige (ele casa "helminth-eggs", nao "eggs"). Vem de core/constants.py: o mapa
# nao e do SPiFiL nem de um metodo, e o vocabulario de dataset do projeto.
from core.constants import PARASITE_NAME

# Teto de superpixels quando a segmentacao roda na grade de features: no minimo
# 4 posicoes por superpixel. Abaixo disso o "superpixel" e um pixel isolado, o
# medoide devolve o proprio ponto e o SLIC deixa de reduzir nada — vira uma copia
# da grade. 4 e o menor bloco (2x2) que ainda tem do que escolher.
MIN_POSITIONS_PER_SUPERPIXEL = 4

# O `before_graft` e um backbone pre-treinado da torchvision e espera normalizacao
# ImageNet — nao a normalizacao do resto deste repo.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# A ResNet pre-treinada so aceita este tamanho, e e nele que o LAB e o superpixel
# rodam — as coordenadas das sementes nascem neste grid. Grandeza do `graft`, nao
# do `grow`: la o tamanho e parametro (`image_size`, default 200).
IMAGE_SIZE = 224


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
                f"imagens (n_images) ou pare de crescer."
            )
        learn.fit_layer(layer)


class Cropped(SpifilDataset):
    """Entrega imagem e mascara com a MESMA geometria que o treino usa.

    `Resize(lado menor)` + `CenterCrop`, que e o `build_test` do repo sem a
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
        # `impurities=True` desliga a mascara nos DOIS caminhos de superpixel de
        # uma vez, porque este metodo e o unico lugar por onde ela chega ao SPiFiL:
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
    parasite = PARASITE_NAME[dataset]
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
    # tem que reproduzir a partir de `seed` sozinho, sem depender de quantas
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
    print(f"[grow] one_per_class: {len(picked)} imagens para {len(picked)} classes")
    return picked


def encoder_features(samples, encoder, size, imagenet_norm, device):
    """As features exatamente como o treino as ve: ift_lab + o transform de teste."""
    transform = build_test(size, imagenet_norm=imagenet_norm)
    out = []
    with torch.no_grad():
        for sample in samples:
            lab = np.ascontiguousarray(ift_lab_loader(sample.image_path))
            x = transform(torch.from_numpy(lab).permute(2, 0, 1).float())
            out.append(encoder(x[None].to(device))[0])
    return out


def seeds_in_features(features, data, n_superpixels, grid):
    """`spifil_in_feature`: segmenta a PROPRIA grade de features, sem reprojetar.

    O caminho default segmenta a imagem LAB de 200x200 e so viaja as sementes.
    Aqui a segmentacao inteira acontece na grade que o encoder JA produz —
    `trunk.grid`, hoje 24x24 = 576 posicoes. (`pool_stride` nao muda isso: ele
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
    SLIC) o maximo em eggs/split1/5% com `one_per_class` e 185 sementes, e
    D=432. Ou seja, COM MASCARA `spifil_in_feature` + `one_per_class` NAO TEM
    COMO passar no orcamento de covariancia nesta grade: o `GrowOneLayer` recusa
    com EXIT_BUDGET, corretamente. Com `one_per_class=False` (124 imagens) o teto
    de 4 posicoes da N=496 contra D=432 — passa raspando, contra N=12313 do
    caminho da imagem. Medido em 2026-08-24; refaca a conta se a mascara, a grade
    ou D mudarem.

    Com `impurities=True` a conta e outra: a area vira 576, o teto sobe para os
    100 pedidos e saem 143 sementes por imagem — N=1287 contra D=432 com as
    mesmas 9 imagens do `one_per_class`. E a unica configuracao em que este
    caminho fecha o orcamento sem inchar o numero de imagens.
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
          f"{'ON' if data.use_mask else 'OFF (impurities)'}")
    return seeds, regions


def build_layer0(data, trunk, samples, n_superpixels, image_size, imagenet_norm,
                 device, in_feature):
    """Superpixel no LAB, features no encoder treinado, sementes entre os dois.

    E a unica parte do `grow` que existe porque o metodo exige. O `prepare()`
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
              f"imagem | mascara {'ON' if data.use_mask else 'OFF (impurities)'})")
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


def _grow_random(arch_json: Path, flim_weights_path: Path, out: Path,
                 out_channels: int, kernel_size: int, pool_stride: int, seed: int) -> int:
    """`random_layer_classic`: acrescenta a camada como se o SPiFiL nao existisse.

    O caminho CLASSICO. Nao carrega checkpoint, nao le imagem, nao roda superpixel
    nem semente, nao instancia `Learner`: le a arquitetura de entrada, sorteia a
    camada e grava. Por isso ele NAO PODE sair com EXIT_BUDGET — o orcamento
    `N <= D` e uma condicao da Mahalanobis que fita os filtros SPiFiL, e sem fit
    nao existe N para comparar. Na pratica o controle cresce ate onde o SPiFiL
    recusa (larvae), e a comparacao vira "SPiFiL onde ele cabe" contra "qualquer
    camada deste tamanho".

    Tambem nao ha allocator, entao nao ha o arredondamento por classe: a camada
    sai com os `out_channels` pedidos (48, nao os 45 do `UniformAllocator`).

    `in_channels` sai do `conv{N}-bias.txt` do proprio diretorio de pesos, nao da
    `architecture.json`: quando o FLIM tem menos classes que canais pedidos os
    dois discordam, e quem manda e o arquivo que o encoder de fato carrega.
    """
    arch = parse_architecture(arch_json)
    layer = arch["nlayers"] + 1
    in_channels = get_actual_channels_from_weights(flim_weights_path, arch)[-1]
    out_channels = out_channels or in_channels

    # kaiming_normal_ fan_out/relu: a inicializacao padrao de conv+ReLU. Bias zero
    # porque o do SPiFiL e -(K . media do patch) e aqui nao ha patch nenhum.
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    torch.nn.init.zeros_(conv.bias)

    print(f"[grow] random_layer_classic: camada {layer} SEM SPiFiL (sem dado, sem semente, "
          f"sem orcamento) | {in_channels} -> {out_channels} canais, k={kernel_size}, "
          f"pool_stride={pool_stride} | kaiming_normal_ fan_out/relu seed={seed}, bias=0")

    # `write_weights` so quer `.conv`; sem `Learner` nao ha bloco de onde tira-lo.
    write_weights(out, flim_weights_path,
                  grown_arch(arch, kernel_size, out_channels, pool_stride),
                  layer, SimpleNamespace(conv=conv))

    print(f"[grow] camada {layer} com {out_channels} filtros em {out}\n"
          f"  arch_json: {out / 'architecture.json'}\n"
          f"  flim_weights_path: {out}")
    return 0


def grow(
    *,
    encoder: torch.nn.Module | None,
    ckpt: Path,
    arch_json: Path,
    flim_weights_path: Path,
    dataset: str,
    split: int,
    percentage: int,
    out: Path,
    out_channels: int = 0,
    kernel_size: int = 3,
    pool_stride: int = 1,
    n_superpixels: int = 100,
    n_images: int = 200,
    image_size: int = 200,
    imagenet_norm: bool = False,
    spifil_in_image: bool = False,
    spifil_in_feature: bool = False,
    impurities: bool = False,
    one_per_class: bool = True,
    random_layer: bool = False,
    random_layer_classic: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> int:
    """UMA rodada de crescimento. Devolve 0, ou `EXIT_BUDGET` quando N <= D.

    O CONTRATO: `grow` nao abre checkpoint. `encoder` chega PRONTO, carregado
    pelo chamador (`experiments/ray/runners/growth.py`), porque quem sabe abrir
    esse checkpoint e o `AutoEncoderFlimModule`, que mora do lado dos metodos —
    e `flim/` nunca importa `methods/`. O objeto pedido e o MINIMO que a funcao
    usa: um `nn.Module` que aceita `(1, 3, image_size, image_size)`, isto e,
    `module.model.encoder`, e nao o LightningModule inteiro.

    `encoder=None` so faz sentido com `random_layer_classic=True`, o unico
    caminho sem backbone — e ele sai antes de tocar no encoder, entao nesse braco
    o chamador nao deve carregar checkpoint nenhum. Em qualquer outro caminho,
    `None` levanta.

    Os demais parametros sao os mesmos knobs de sempre, com os mesmos defaults:

    `ckpt` NAO e mais aberto aqui: e so a etiqueta de procedencia que sai no log.
    Passe o caminho de onde `encoder` veio, senao o log mente sobre o backbone.
    `out` e o diretorio de pesos do modelo crescido (arch + conv{1..N+1}).
    `out_channels=0` significa "a largura do proprio backbone", que mantem D
    constante entre rodadas. `pool_stride=1` e sem pooling: o grid ja e pequeno e
    cada pooling funde sementes, o que derruba N. `n_images` sao as imagens de
    treino que doam patches — o unico jeito de aumentar N.

    `spifil_in_image` e `spifil_in_feature` sao exclusivos (o default e o do
    artigo: superpixel na imagem LAB, sementes reprojetadas para a grade do
    encoder). `random_layer` e `random_layer_classic` tambem sao exclusivos, e
    pelo mesmo motivo: sao perguntas diferentes, nao intensidades da mesma — uma
    mantem o protocolo e troca a direcao, a outra troca o protocolo inteiro.
    """
    # As duas exclusividades que o parser do script garantia; sem elas o par
    # passaria calado e produziria uma combinacao que hoje e impossivel.
    if spifil_in_image and spifil_in_feature:
        raise ValueError("spifil_in_image e spifil_in_feature sao exclusivos")
    if random_layer and random_layer_classic:
        raise ValueError("random_layer e random_layer_classic sao exclusivos")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Antes de tocar no backbone de proposito: o caminho classico nao usa nada do
    # que vem abaixo, e carregar encoder e dataset para joga-los fora custa minutos
    # por braco. Como quem carrega o checkpoint agora e o chamador, a economia so
    # existe se ele tambem nao carregar: passe `encoder=None` neste braco.
    # So o CLASSICO sai por aqui: no pareado o superpixel roda de verdade, entao ele
    # segue o fluxo normal e so troca os valores dos pesos la embaixo.
    if random_layer_classic:
        return _grow_random(arch_json, flim_weights_path, out,
                            out_channels, kernel_size, pool_stride, seed)

    # Daqui para baixo o backbone e obrigatorio. Sem a guarda o `None` so
    # apareceria la dentro do `Trunk`, como AttributeError, e o erro apontaria
    # para o lugar errado: o contrato quebrado e do chamador.
    if encoder is None:
        raise ValueError(
            "grow precisa do encoder ja carregado: so random_layer_classic=True "
            "dispensa backbone. Carregue o checkpoint no chamador e passe "
            "encoder=module.model.encoder."
        )
    trunk = Trunk(encoder, image_size, device)

    arch = parse_architecture(arch_json)

    layer = arch["nlayers"] + 1

    out_channels = out_channels or trunk.out_channels

    print(f"[grow] backbone de {ckpt} | {trunk.out_channels} canais em "
          f"{trunk.grid}x{trunk.grid} | crescendo a camada {layer}")

    samples = train_samples(dataset, split, percentage, n_images,
                            one_per_class=one_per_class, seed=seed)

    data = Cropped(samples, image_size, use_mask=not impurities)

    learn = Learner(
        data,
        ArchSpec(layers=[LayerSpec(
            kernel_size=kernel_size, out_channels=out_channels,
            pool_type="max" if pool_stride > 1 else "none",
            pool_stride=pool_stride,
        )]),
        color=trunk, strategy=GrowOneLayer(), device=device,
    )
    learn.state[0] = build_layer0(
        data, trunk, samples, n_superpixels, image_size,
        imagenet_norm, device, spifil_in_feature,
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

    if random_layer:
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
            g = torch.Generator().manual_seed(seed)
            noise = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
            escala = (w.flatten(1).norm(dim=1) / noise.flatten(1).norm(dim=1))
            w.copy_(noise * escala.view(-1, *([1] * (w.dim() - 1))))
            if block.conv.bias is not None:
                block.conv.bias.zero_()
        print(f"[grow] random_layer: {got} filtros sorteados (seed={seed}, "
              f"norma por filtro preservada, bias=0)")

    write_weights(out, flim_weights_path,
                  grown_arch(arch, kernel_size, got, pool_stride), layer, block)

    learn.export(out / "spifil_bundle")  # labels{L}.txt: de que classe veio cada filtro

    print(f"[grow] camada {layer} com {got} filtros em {out}\n"
          f"  arch_json: {out / 'architecture.json'}\n"
          f"  flim_weights_path: {out}")
    return 0


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


class GraftTrunk:
    """O pedaco da ResNet antes do enxerto, no formato ColorTransform do SPiFiL.

    Chamava-se `Trunk` no script de origem; renomeado porque o `grow` deste mesmo
    modulo tem o SEU `Trunk` (o encoder FLIM), e os dois nao cabem sob um nome so.

    Ele NAO calcula superpixel — isso e do LabNorm, em `graft_layer0`. Aqui ele
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


def graft_layer0(
    data: SpifilDataset,
    trunk: GraftTrunk,
    n_superpixels: int,
    device: str,
) -> LayerState:
    """Superpixel no LAB, features na ResNet, sementes projetadas entre os dois.

    Chamava-se `build_layer0` no script de origem; renomeado porque o `grow` deste
    modulo tem o seu `build_layer0`, que monta a state[0] do encoder FLIM.

    E a unica parte do `graft` que existe porque o metodo exige — o resto e
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


def graft(
    *,
    images_dir: Path,
    masks_dir: Path | None = None,
    out_dir: Path = Path("runs/spifil-resnet"),
    device: str | None = None,
    n_superpixels: int = 100,
    seed: int = 42,
) -> nn.Module:
    """Fita as camadas SPiFiL e devolve a ResNet remontada com elas no meio.

    `images_dir` e a pasta de imagens; a classe e o prefixo inteiro do nome.
    `masks_dir=None` usa a imagem inteira. `seed=42` e o mesmo default do
    `spifil-fit` (conf/config.yaml) e e o unico determinismo que ele fixa.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

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

    color = GraftTrunk(before_graft)
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
    # consome e o graft_layer0. O Learner so precisa do `color` para dimensionar
    # o primeiro bloco e para o metadado do export.
    learn = Learner(data, arch, color=color, device=device)

    learn.state[0] = graft_layer0(data, color, n_superpixels, device)
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
