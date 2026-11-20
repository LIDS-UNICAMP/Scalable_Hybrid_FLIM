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
"""Escolhe um numero fixo de filtros entre candidatos SPiFiL, sem rotulo e sem backprop.

`select_filters` recebe os candidatos e devolve indices e metricas do banco escolhido:
- D1 `fps_*`, `kmedoids_*`: `spatial` usa a posicao do patch na imagem (0 a 1), `visual` os
  pixels crus do patch. Duas imagens na mesma posicao contam como o mesmo lugar.
- D2 `kernel_random`, `kernel_distance` (FPS em 1 - cosseno), `kernel_diversity` (menor soma
  de cosseno positivo ao quadrado com os ja escolhidos).
- D3 `activation`: a mesma gulosa da `kernel_diversity`, na correlacao das ativacoes.

Similaridade COM SINAL, como em `kernel_redundancy`: com ReLU, k e -k servem os dois. D2 e D3
decidem na dobra sobre todos os candidatos; as metricas medem a dobra do banco escolhido.
Filtro morto tem correlacao 0 com todos e a gulosa o pegaria primeiro: D3 o tira da disputa.

`features` sao os mapas das imagens de TREINO, no mesmo espaco de `patches.feats` (sem whiten
ou patch_norm por fora). Tudo aqui e matriz N x N: serve para milhares de candidatos.
"""
import torch
import torch.nn.functional as F

from flim.kernel_redundancy import activation_sample, measure, similarity

METHODS = ("fps_spatial", "fps_visual", "kmedoids_spatial", "kmedoids_visual",
           "kernel_random", "kernel_distance", "kernel_diversity", "activation")


def greedy(cost, n, first, merge) -> torch.Tensor:
    """A cada passo pega quem tem o menor custo somado por `merge` com os ja escolhidos."""
    chosen = [first]
    taken = torch.zeros(len(cost), dtype=torch.bool, device=cost.device)
    total = cost[first].clone()
    for _ in range(n - 1):
        taken[chosen[-1]] = True
        chosen.append(int(total.masked_fill(taken, float("inf")).argmin()))
        total = merge(total, cost[chosen[-1]])
    return torch.tensor(chosen, device=cost.device)


def k_medoids(dist, medoids, iters=100) -> torch.Tensor:
    """Park & Jun 2009. Cada medoide fica no proprio grupo: nenhum grupo esvazia ou repete."""
    groups = torch.arange(len(medoids), device=dist.device)
    for _ in range(iters):
        nearest = dist[:, medoids].argmin(1)
        nearest[medoids] = groups
        moved = medoids.clone()
        for group in groups:
            members = (nearest == group).nonzero(as_tuple=True)[0]
            moved[group] = members[dist[members][:, members].sum(0).argmin()]
        if moved.equal(medoids):
            break
        medoids = moved
    return medoids


def low_redundancy(sim, n, first) -> torch.Tensor:
    return greedy(F.relu(sim).square(), n, first, torch.add)


def pick(patches, spec, sample, n, method, coords, seed) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(seed)
    total = len(patches)
    first = int(torch.randint(total, (1,), generator=generator))
    if method == "kernel_random":
        return torch.randperm(total, generator=generator)[:n], {}
    if method.startswith(("fps", "kmedoids")):
        points = patches.feats if method.endswith("visual") else coords.to(patches.feats)
        dist = torch.cdist(points, points)
        chosen = greedy(-dist, n, first, torch.maximum)
        return (k_medoids(dist, chosen) if method.startswith("kmedoids") else chosen), {}
    _, cos, corr = similarity(patches, spec, sample)
    if method == "kernel_distance":
        return greedy(cos, n, first, torch.maximum), {}
    if method == "kernel_diversity":
        return low_redundancy(cos, n, first), {}
    alive = corr.diagonal().nonzero(as_tuple=True)[0]
    if len(alive) < n:
        raise ValueError(f"so {len(alive)} candidatos vivos para {n} filtros")
    chosen = low_redundancy(corr[alive][:, alive], n, first % len(alive))
    return alive[chosen], {"dead_candidates": total - len(alive)}


def select_filters(patches, spec, features, n, method, coords=None, seed=0):
    """Devolve (indices em `patches`, na ordem da escolha; metricas do banco escolhido).

    `coords` (N, 2) so para os metodos `*_spatial`. GPU se couber, senao CPU, com aviso.
    """
    if method not in METHODS:
        raise ValueError(f"metodo {method!r} nao existe; use um de {METHODS}")
    if not 2 <= n <= len(patches):
        raise ValueError(f"{n} filtros pedidos de {len(patches)} candidatos")
    if method.endswith("spatial") and coords is None:
        raise ValueError(f"{method} precisa de coords")
    sample = activation_sample(features, spec)

    def run(device):
        moved = patches.to(device)
        chosen, extra = pick(moved, spec, sample, n, method, coords, seed)
        chosen = chosen.to(device)
        return chosen.cpu(), {"method": method, "n": n, "candidates": len(patches),
                              "device": str(device), **extra,
                              **measure(moved.select(chosen), spec, sample)}

    if torch.cuda.is_available():
        try:
            return run(torch.device("cuda"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[diversity] {method}: {len(patches)} candidatos nao couberam na GPU, vai CPU")
    return run(torch.device("cpu"))


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    import cv2
    from spifil import LayerSpec
    from spifil.color import LabNorm
    from spifil.data import Sample
    from spifil.patches import extract_patches
    from spifil.preparation import prepare_sample
    from spifil.superpixels.centers import Medoids
    from spifil.superpixels.disf import DISF
    from spifil.types import PatchSet

    from flim.spifil import Cropped

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)

    def candidates(feats):
        n = len(feats)
        return PatchSet(feats, torch.zeros(n, dtype=torch.long), torch.arange(n),
                        torch.zeros(n, dtype=torch.long))

    spec = LayerSpec(kernel_size=1, out_channels=8)
    features = [torch.randn(8, 32, 32)]

    # 4 grupos separados: 1 de cada, nos dois espacos e nos dois metodos de D1.
    centers = torch.eye(8)[:4] * 50
    feats = (centers[:, None] + 0.1 * torch.randn(4, 30, 8)).flatten(0, 1)
    coords = (torch.tensor([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])[:, None]
              + 0.01 * torch.randn(4, 30, 2)).flatten(0, 1)
    for method in METHODS[:4]:
        chosen, metrics = select_filters(candidates(feats), spec, features, 4, method, coords)
        assert sorted((chosen // 30).tolist()) == [0, 1, 2, 3], method
        assert metrics["n"] == 4 and "effective_rank" in metrics

    # D2: 8 direcoes, cada uma 5 vezes. Sem repetir, o banco de 8 tem posto efetivo cheio.
    feats = torch.eye(8).repeat(5, 1) * torch.linspace(1, 2, 40)[:, None]
    for method in ("kernel_distance", "kernel_diversity"):
        chosen, metrics = select_filters(candidates(feats), spec, features, 8, method)
        assert sorted((chosen % 8).tolist()) == list(range(8)), method

    # Numero fixo mesmo com n acima das direcoes distintas, e sem repetir indice.
    for method in METHODS[2:]:
        chosen = select_filters(candidates(feats), spec, features, 20, method, torch.rand(40, 2))[0]
        assert len(chosen) == len(chosen.unique()) == 20, method

    # D3 enxerga o que D2 nao ve: canal 7 sempre zero nas imagens, entao k e k + canal 7 sao
    # o mesmo filtro funcional com cosseno baixo.
    blind = torch.randn(8, 32, 32)
    blind[7] = 0
    base = torch.randn(12, 8)
    base[:, 7] = 0
    twins = base.clone()
    twins[:, 7] = 8.0
    both = torch.cat([base, twins])
    chosen, metrics = select_filters(candidates(both), spec, [blind], 12, "activation")
    assert sorted((chosen % 12).tolist()) == list(range(12))
    assert metrics["corr_max"] < 0.99

    # Filtro morto: na media em tudo menos no canal zerado, fica negativo em toda janela.
    outlier = torch.cat([base, base.mean(0, keepdim=True)])
    outlier[12, 7] = 1000.0
    chosen, metrics = select_filters(candidates(outlier), spec, [blind], 6, "activation")
    assert metrics["dead_candidates"] == 1 and 12 not in chosen.tolist()

    for method in METHODS:
        args = candidates(feats), spec, features, 8, method, torch.rand(40, 2).round(decimals=3)
        assert select_filters(*args, seed=3)[0].equal(select_filters(*args, seed=3)[0]), method
    for bad in ((1, "activation"), (41, "activation"), (8, "x"), (8, "fps_spatial")):
        try:
            select_filters(candidates(feats), spec, features, *bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    print("ok sintetico: D1 acha os grupos; D2 sem repetir; D3 ve gemeo funcional e tira o morto")

    root = Path("/dados/home/moliveira/datasets/organized/flowers")
    train = json.loads((root / "splits/split1.json").read_text())["train"]
    paths = [root / "orig" / name for name in train[::len(train) // 40][:40]]
    data = Cropped([Sample(path.stem, path, None, 0) for path in paths], 200, use_mask=False)
    samples = [prepare_sample(data, i, superpixels=DISF(), seed_extractor=Medoids(),
                              n_superpixels=50, color=LabNorm()) for i in range(len(paths))]
    spec = LayerSpec(kernel_size=5, out_channels=64)
    features = [torch.as_tensor(sample.features) for sample in samples]
    patches = PatchSet.cat([extract_patches(feature, sample.seeds, spec, image_id=i) for
                            i, (feature, sample) in enumerate(zip(features, samples))])
    coords = torch.cat([sample.seeds.coords / torch.tensor(sample.seeds.grid)
                        for sample in samples])
    print(f"flowers: {len(paths)} imagens de treino, {len(patches)} candidatos, 64 filtros")
    print(f"{'metodo':18}{'cos_near':>9}{'cos_max':>9}{'corr_near':>10}{'corr_max':>9}"
          f"{'posto':>7}{'seg':>6}  device")
    for method in METHODS:
        start = time.perf_counter()
        chosen, m = select_filters(patches, spec, features, 64, method, coords)
        assert len(chosen.unique()) == 64
        print(f"{method:18}{m['cos_nearest_mean']:9.3f}{m['cos_max']:9.3f}"
              f"{m['corr_nearest_mean']:10.3f}{m['corr_max']:9.3f}{m['effective_rank']:7.1f}"
              f"{time.perf_counter() - start:6.1f}  {m['device']} "
              f"{m.get('dead_candidates', '')}")
