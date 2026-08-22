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
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from constants import PROJECT_ROOT as _ROOT

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

# Exit code proprio para "o orcamento de covariancia acabou". O laco de
# crescimento distingue isto de uma falha de verdade e encerra limpo.
EXIT_BUDGET = 3


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

    def __init__(self, samples, size: int) -> None:
        super().__init__(samples)
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


def train_samples(dataset: str, split: int, percentage: int, n_images: int):
    """Amostras do split de TREINO, na ordem do disco, subamostradas por passo.

    So treino: recortar filtro de imagem de validacao vazaria o conjunto que o
    kappa mede. O passo uniforme sobre a lista ordenada cobre todas as classes,
    porque o nome do arquivo comeca pelo id da classe.
    """
    parasite = _dataset_short_to_parasite_name(dataset)
    info = get_single_parasite_paths(parasite, split, percentage)[0]
    names = set(json.loads(Path(info["split_json"]).read_text())["train"])

    everything = SpifilDataset.from_folders(info["images_dir"], info["masks_dir"])
    chosen = [s for s in everything if s.image_path.name in names]
    if not chosen:
        raise ValueError(f"nenhuma imagem de treino em {info['images_dir']}")
    return chosen[:: max(1, len(chosen) // n_images)][:n_images]


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


def build_layer0(data, trunk, samples, n_superpixels, image_size, imagenet_norm, device):
    """Superpixel no LAB, features no encoder treinado, sementes entre os dois.

    E a unica parte do script que existe porque o metodo exige. O `prepare()`
    faria isto sozinho, mas rodaria o superpixel em cima das features do encoder;
    o superpixel pertence ao LAB, e so as sementes viajam.
    """
    prepared = list(preparation.prepare(
        data, superpixels=SLIC(), seed_extractor=Medoids(),
        n_superpixels=n_superpixels, color=LabNorm(),
    ))
    grid = (trunk.grid, trunk.grid)
    seeds = [rescale(image.seeds, grid) for image in prepared]
    print(f"[grow] superpixel no LAB {prepared[0].seeds.grid} -> sementes em {grid} "
          f"({sum(len(s) for s in seeds)} sementes em {len(seeds)} imagens)")
    return LayerState(
        features=encoder_features(samples, trunk.encoder, image_size, imagenet_norm, device),
        seeds=seeds,
        superpixel_labels=[image.superpixel_labels for image in prepared],
    )


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

    samples = train_samples(args.dataset, args.split, args.percentage, args.n_images)
    data = Cropped(samples, args.image_size)

    # UMA LayerSpec por rodada. Sem after_graft nao ha ninguem exigindo contagem
    # de canal a jusante, entao nao existe `need` para reconciliar nem adapter.
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
        args.imagenet_norm, args.device,
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

    write_weights(args.out, args.flim_weights_path,
                  grown_arch(arch, args.kernel_size, got, args.pool_stride), layer, block)
    learn.export(args.out / "spifil_bundle")  # labels{L}.txt: de que classe veio cada filtro

    print(f"[grow] camada {layer} com {got} filtros em {args.out}\n"
          f"  --arch-json {args.out / 'architecture.json'}\n"
          f"  --flim-weights-path {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
