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
"""spifil_ext.py — o que o GA precisa a mais do SPiFiL, só por herança e composição."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from spifil import (DISF, SLIC, ArchSpec, Callback, Centroids, DistanceSumScorer, DiversitySelector,
                    Euclidean, FisherScorer, GeodesicCenters, LabNorm, LayerSpec, Learner, Mahalanobis,
                    Medoids, SequentialStrategy, SpifilConvBlock, SpifilEncoder, TopN, UniformAllocator)

from flim.spifil import BudgetExhausted, Cropped, seeds_in_features


@dataclass
class ActLayerSpec(LayerSpec):
    """LayerSpec com ativação escolhível (relu | leaky_relu | tanh | none)."""

    activation: str = "relu"
    leaky_slope: float = 0.01


class ActivationBlock(SpifilConvBlock):
    """conv -> ativação(spec.activation) -> pool."""

    def act(self, x):
        act = self.spec.activation
        if act == "leaky_relu":
            return F.leaky_relu(x, self.spec.leaky_slope)
        return x if act == "none" else getattr(torch, act)(x)  # relu | tanh

    def forward(self, x):
        x = self.act(self.conv(x))
        return x if self.pool is None else self.pool(x)


class FeatureSuperpixels(Callback):
    """spifil_in=features: refaz as sementes da camada sobre as próprias features.

    `n_superpixels[l]` é o da camada l+1 (a que vai consumir estas sementes); None = não age.
    Roda no after_encode, antes do score_layer da mesma camada.
    """

    def __init__(self, n_superpixels):
        self.n_superpixels = n_superpixels

    def after_encode(self, learn):
        layer = learn.layer
        if layer >= learn.n_layers or self.n_superpixels[layer] is None:
            return
        state = learn.state[layer]
        grid = tuple(state.features[0].shape[-2:])
        state.seeds, state.superpixel_labels = seeds_in_features(
            state.features, learn.data, self.n_superpixels[layer], grid)


class BudgetSequential(SequentialStrategy):
    """Sequencial, mas recusa a camada l quando N (patches da camada l-1) <= D."""

    def __init__(self, budget_check=True):
        self.budget_check = budget_check

    def run(self, learn):
        for layer in range(1, learn.n_layers + 1):
            if learn.state[layer - 1].patches is None:
                learn.extract_layer_patches(layer - 1)
            patches = learn.state[layer - 1].patches
            n, d = len(patches), int(patches.feats.shape[1])
            if self.budget_check and n <= d:
                raise BudgetExhausted(f"camada {layer}: N={n} <= D={d}")
            learn.fit_layer(layer)


def build_learner(cfg, samples, *, device, leaky_slope, block_factory=ActivationBlock):
    """Learner do SPiFiL a partir da config ativa (§3 do contrato)."""
    sp, sel = cfg["superpixels"], cfg["selector"]
    layers = [ActLayerSpec(**{k: v for k, v in L.items() if k not in ("n_superpixels", "channels") and v is not None},
                           relu=L["activation"] == "relu", leaky_slope=leaky_slope)
              for L in cfg["layers"]]
    return Learner(
        Cropped(samples, cfg["r"], use_mask=not cfg["impurities"]),
        ArchSpec(layers=layers, stdev_factor=cfg["stdev_factor"]),
        superpixels=(SLIC(sp["compactness"], sp["sigma"]) if sp["name"] == "slic"
                     else DISF(sp["n_init_seeds"], sp["adjacency_radius"])),
        seed_extractor={"medoid": Medoids, "centroid": Centroids,
                        "geodesic": GeodesicCenters}[cfg["seed_extractor"]](),
        metric=Mahalanobis(cfg["estimator"]) if cfg["metric"] == "mahalanobis" else Euclidean(),
        scorer={"fisher": FisherScorer, "distance_sum": DistanceSumScorer}[cfg["scorer"]](),
        selector=(DiversitySelector(sel["alpha"], sel["pool_factor"]) if sel["name"] == "diversity"
                  else TopN()),
        allocator={"uniform": UniformAllocator}[cfg["allocator"]](),
        color={"labnorm": LabNorm}[cfg["color"]](),
        block_factory=block_factory,
        strategy=BudgetSequential(cfg["budget_check"]),
        cbs=[FeatureSuperpixels([L.get("n_superpixels") for L in cfg["layers"]])],
        n_superpixels=cfg["n_superpixels"],
        device=device,
    )


_LAB = {}  # por processo: (r, nome) -> imagem ja em LAB (3, r, r); a cor e fixa, entao so muda com r


def encode(learn, samples, r, *, batch):
    """Features (n, C, H, W) na CPU, com a mesma geometria do treino e sem máscara.

    Ler, redimensionar e converter para LAB era ~85% do tempo de cada avaliacao e dava
    sempre o mesmo resultado; cada imagem e convertida uma vez por processo e por r.
    """
    enc = SpifilEncoder(learn.color, learn.model).eval()
    data = Cropped(samples, r, use_mask=False)
    for j, s in enumerate(samples):
        if (r, s.name) not in _LAB:
            _LAB[(r, s.name)] = enc.transform([data.load_image(j)])[0].cpu()
    dev = next(learn.model.parameters()).device
    with torch.no_grad():
        return torch.cat([enc(torch.stack([_LAB[(r, s.name)] for s in samples[i:i + batch]]).to(dev)).cpu()
                          for i in range(0, len(samples), batch)])
