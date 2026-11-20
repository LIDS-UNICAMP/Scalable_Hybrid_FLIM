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
"""Tira kernels SPiFiL redundantes entre a escolha e a inicializacao do modelo.

Um `Selector` por fora do que o `Learner` ja usa: o de dentro escolhe como sempre, este
mede o banco, fica com um representante por grupo e (`replace`) repoe com o proximo
candidato nao redundante, ate a verba do alocador. Sem candidato a camada fica menor. O
`fill_random` roda depois do `Learner`, entao os aleatorios nunca passam por aqui.

Redundante = cosseno OU correlacao acima do limiar, COM SINAL: a conv tem ReLU, entao k e
-k respondem a coisas opostas e os dois servem. O z-score do filtro depende do banco, e o
banco muda ao podar; a decisao usa a dobra sobre todos os candidatos, que nao muda. O
`after` do JSON mede o banco real, entao pode passar um pouco do limiar.
"""
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from spifil import learner as spifil_learner  # pelo modulo: o a3 troca build_filters la
from spifil.patches import extract_patches
from spifil.selection import candidate_order
from spifil.types import Seeds

SAMPLE_PER_IMAGE = 1024


def merge_seeds(state) -> tuple[int, int]:
    """Funde, em cada imagem, sementes no mesmo pixel. Devolve o total antes e depois.

    `Seeds.project` so funde com stride > 1: na grade dobrada, stride 2 volta para a mesma
    grade e herda a regra dele (melhor rank, rotulo do primeiro).
    """
    before = sum(len(seeds) for seeds in state.seeds)
    state.seeds = [
        Seeds(seeds.coords * 2, seeds.labels, seeds.ranks,
              tuple(2 * size for size in seeds.grid)).project(2)
        for seeds in state.seeds
    ]
    return before, sum(len(seeds) for seeds in state.seeds)


def activation_sample(features, spec) -> torch.Tensor:
    """Janelas da conv numa grade regular de cada imagem de treino, sem sorteio."""
    rows = []
    for feature in features:
        h, w = feature.shape[1:]
        step = max(1, math.isqrt(h * w // SAMPLE_PER_IMAGE))
        coords = torch.cartesian_prod(torch.arange(0, h, step), torch.arange(0, w, step))
        rows.append(extract_patches(feature, Seeds.from_coords(coords, 0, (h, w)), spec).feats)
    return torch.cat(rows)


def similarity(patches, spec, sample):
    """Kernels dobrados de norma 1, cosseno entre eles e correlacao das ativacoes."""
    bank = spifil_learner.build_filters(patches, spec)
    weight = bank.weight.flatten(1)
    unit = F.normalize(weight, dim=1)
    active = F.relu(sample.to(weight) @ weight.T + bank.bias)
    n = len(patches)
    return unit, unit @ unit.T, torch.corrcoef(active.T).nan_to_num().reshape(n, n)


def effective_rank(kernels: torch.Tensor) -> float:
    """exp da entropia dos valores singulares (Roy & Vetterli 2007)."""
    s = torch.linalg.svdvals(kernels)
    p = s[s > 0] / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def measure(patches, spec, sample) -> dict:
    unit, cos, corr = similarity(patches, spec, sample)
    self_pair = torch.eye(len(unit), dtype=torch.bool, device=unit.device)
    nearest_cos = cos.masked_fill(self_pair, -1).amax(1)
    nearest_corr = corr.masked_fill(self_pair, -1).amax(1)
    return {
        "cos_max": float(nearest_cos.max()), "cos_nearest_mean": float(nearest_cos.mean()),
        "corr_max": float(nearest_corr.max()), "corr_nearest_mean": float(nearest_corr.mean()),
        "effective_rank": effective_rank(unit),
    }


def prune(hit, picked, rest, labels, budget, replace):
    """`picked` e `rest` vem do melhor rank para o pior. Devolve kept, grupos e repostos."""
    kept, groups = [], {}
    for i in picked:
        twin = hit[i, kept].nonzero()
        if len(twin):
            groups.setdefault(kept[int(twin[0])], []).append(i)
        else:
            kept.append(i)
    added = []
    if replace:
        # Pela verba, nao por quem saiu: o teto do pool do DiversitySelector deixa candidato
        # bom de fora quando `merge_seeds` encolhe N (eggs L4: 160 sementes, 132 escolhidas).
        missing = Counter(budget) - Counter(labels[i] for i in kept)
        for i in rest:  # primeiro a mesma classe: a verba do alocador continua valendo
            if missing[labels[i]] and not hit[i, kept].any():
                kept.append(i)
                added.append(i)
                missing[labels[i]] -= 1
        spare = missing.total()
        for i in rest:  # classe sem candidato cede a vaga: a largura vale mais
            if spare and i not in added and not hit[i, kept].any():
                kept.append(i)
                added.append(i)
                spare -= 1
    return kept, groups, added


class RedundancySelector:
    def __init__(self, inner, spec, features, layer, path, cos, corr, replace, seeds):
        self.inner, self.spec, self.features = inner, spec, features
        self.layer, self.path = layer, Path(path)
        self.cos, self.corr, self.replace, self.seeds = cos, corr, replace, seeds

    def __call__(self, patches, ranks, n_per_class):
        ranks = ranks.to(patches.feats.device)
        mask = self.inner(patches, ranks, n_per_class)
        sample = activation_sample(self.features, self.spec)
        _, cos, corr = similarity(patches, self.spec, sample)
        hit = torch.zeros_like(cos, dtype=torch.bool)
        if self.cos:
            hit |= cos > self.cos
        if self.corr:
            hit |= corr > self.corr
        order = torch.argsort(ranks, stable=True)
        labels = patches.labels.tolist()
        picked, rest = order[mask[order]].tolist(), order[~mask[order]].tolist()
        kept, groups, added = prune(hit.cpu(), picked, rest, labels, n_per_class, self.replace)
        result = torch.zeros_like(mask)
        result[kept] = True

        def who(i):
            return {"image": int(patches.image_ids[i]), "seed_row": int(patches.seed_rows[i]),
                    "class": labels[i], "rank": int(ranks[i])}

        # A mesma ordem do `Learner`: a linha de `kernels` e o filtro da conv.
        rows = result.nonzero(as_tuple=True)[0]
        rows = rows[candidate_order(patches.select(rows), ranks[rows])].tolist()
        entry = {
            "seeds_before": self.seeds[0], "seeds_after": self.seeds[1],
            "cos": self.cos, "corr": self.corr, "replace": self.replace,
            "requested": sum(n_per_class.values()), "entered": len(picked),
            "removed": sum(map(len, groups.values())), "replaced": len(added),
            "kept": len(kept), "short": sum(n_per_class.values()) - len(kept),
            "classes": {str(c): {"entered": sum(labels[i] == c for i in picked),
                                 "kept": sum(labels[i] == c for i in kept)}
                        for c in sorted(n_per_class)},
            "before": measure(patches.select(mask), self.spec, sample),
            "after": measure(patches.select(result), self.spec, sample),
            "kernels": [who(i) for i in rows],
            "groups": [{"kept": who(k), "removed": [who(i) for i in removed]}
                       for k, removed in groups.items()],
        }
        report = json.loads(self.path.read_text()) if self.path.exists() else {}
        report[str(self.layer)] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[grow] redundancia L{self.layer}: {entry['entered']} entraram, "
              f"{entry['removed']} sairam, {entry['replaced']} repostos, {entry['kept']} ficaram")
        return result


def attach(learn, layer, path, cos=0.0, corr=0.0, replace=False, merge=False) -> None:
    """Liga a poda num `Learner` com `state[0]` pronto. Tudo desligado: nao toca em nada."""
    if not (cos or corr or merge):
        return
    if learn.strategy.whiten or learn.strategy.patch_norm:
        raise ValueError("redundancia nao combina com whiten/patch_norm: o kernel do selector "
                         "vive em outro espaco que o das features")
    state = learn.state[0]
    total = sum(len(seeds) for seeds in state.seeds)
    learn.selector = RedundancySelector(
        learn.selector, learn.spec(learn.n_layers), state.features, layer, path,
        cos, corr, replace, merge_seeds(state) if merge else (total, total))


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    from spifil import LayerSpec
    from spifil.selection import DiversitySelector, TopN
    from spifil.types import PatchSet

    torch.manual_seed(0)
    spec = LayerSpec(kernel_size=1, out_channels=8)
    features = [torch.randn(8, 16, 16)]
    path = Path(tempfile.mkdtemp()) / "kernel_redundancy.json"

    def run(feats, budget, inner=TopN(), cos=0.99, corr=0.0, replace=False):
        n = len(feats)
        patches = PatchSet(feats, torch.zeros(n, dtype=torch.long), torch.arange(n),
                           torch.zeros(n, dtype=torch.long))
        select = RedundancySelector(inner, spec, features, 1, path, cos, corr, replace, (n, n))
        mask = select(patches, torch.arange(1, n + 1), {0: budget})
        return mask, json.loads(path.read_text())["1"]

    eye = torch.eye(8)
    twins = torch.cat([eye[:3], eye[:1], eye[3:]])  # linha 3 = linha 0
    for cos, corr in ((0.99, 0.0), (0.0, 0.99)):
        assert run(twins, 4, cos=cos, corr=corr)[1]["kept"] == 3
        entry = run(twins, 4, cos=cos, corr=corr, replace=True)[1]
        assert (entry["removed"], entry["replaced"], entry["kept"], entry["short"]) == (1, 1, 4, 0)
        entry = run(eye[[0, 1, 2, 0, 0]], 4, cos=cos, corr=corr, replace=True)[1]
        assert (entry["replaced"], entry["kept"], entry["short"]) == (0, 3, 1)
        assert run(eye, 8, DiversitySelector(), cos=cos, corr=corr)[1]["kept"] == 8

    inner = DiversitySelector()
    patches = PatchSet(twins, torch.zeros(9, dtype=torch.long), torch.arange(9),
                       torch.zeros(9, dtype=torch.long))
    same = inner(patches, torch.arange(1, 10), {0: 4})
    assert torch.equal(run(twins, 4, inner, cos=0.0)[0], same)
    learn = SimpleNamespace(selector=inner)
    attach(learn, 1, path)
    assert learn.selector is inner

    state = SimpleNamespace(seeds=[Seeds.from_coords(
        torch.tensor([[1, 2], [1, 2], [3, 4]]), 0, (5, 7))])
    assert merge_seeds(state) == (3, 2) and state.seeds[0].grid == (5, 7)
    assert state.seeds[0].coords.tolist() == [[1, 2], [3, 4]]
    print("ok: iguais viram um; ortogonais ficam; desligado sai o mesmo; sementes fundidas")
