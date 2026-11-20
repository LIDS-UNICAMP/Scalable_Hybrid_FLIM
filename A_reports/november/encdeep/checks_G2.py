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
"""checks_G2.py — §14 da integracao (eval/encdeep.py + hook em eval/spf_grade.py), em CPU com tensores minusculos.

Rodar com CUDA_VISIBLE_DEVICES="" na raiz do worktree. O check 1 compara os CSVs da celula minuscula
antes/depois (rodada a parte na GPU, ver G2_integracao.md); os demais sao sinteticos.
"""
import ast
import copy
import csv
import inspect
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from spifil.config import LayerSpec
from spifil.patches import extract_patches
from spifil.types import Seeds

from eval import encdeep, spf_grade
from eval.spf_grade import folded, map_whitener, whitened
from experiments.ray.schema import check_encdeep
from flim import spifil
from flim.arch import build_encoder_from_arch
from flim.build import build

torch.set_num_threads(2)
torch.manual_seed(0)
assert str(encdeep.DEVICE) == "cpu", "rode com CUDA_VISIBLE_DEVICES=''"
SMOKE = Path("/exps/moliveira/eval/encdeep_smoke")
TIMING = {"select_s", "extract_s", "fit_s", "gpu_mib", "rss_mib"}

# 1. desligado == antes (celula real minuscula, CSVs sem as colunas de tempo)
if (SMOKE / "before.csv").exists() and (SMOKE / "after.csv").exists():
    a, b = (list(csv.DictReader(open(SMOKE / f))) for f in ("before.csv", "after.csv"))
    head = [open(SMOKE / f).readline() for f in ("before.csv", "after.csv")]
    strip = lambda r: {k: v for k, v in r.items() if k not in TIMING}
    assert head[0] == head[1] and len(a) == len(b) and all(strip(x) == strip(y) for x, y in zip(a, b))
    print(f"1 ok: desligado, {len(a)} linhas iguais ao antes (cabecalho identico; sem {sorted(TIMING)})")
spec_off = {"depth": 3, "draws": [42], "kernels": [5], "widths": [32], "methods": ["fps"], "controls": ["random_z"]}
assert spf_grade.cells(spec_off) == [("random_z", (32,) * 3, 5, 42), ("fps", (32,) * 3, 5, 42)]
assert encdeep.deep_mechanism(spec_off, 3, ["hebb"]) is None
yml = yaml.safe_load(open("experiments/autoencoder/disc_md_encdeep_mech_off.yaml"))["runner_args"]["amostra"]
check_encdeep(yml)
for bad in ({"encdeep": yml["encdeep"] | {"xyz": 1}}, {"encdeep": yml["encdeep"] | {"mecanismos": ["hebbx"]}},
            {"encdeep": yml["encdeep"] | {"hebb": yml["encdeep"]["hebb"] | {"lrr": 1}}},
            {"encdeep": yml["encdeep"] | {"kernels": [3, 3]}}, {"encdeep": yml["encdeep"] | {"dim_eq": 512}}):
    try:
        check_encdeep(yml | bad)
    except ValueError:
        continue
    raise AssertionError(f"schema aceitou {bad}")
arms = [c[0] for c in spf_grade.cells(yml)]
assert "random_z~g1" in arms and "fps~g1~clapp2" in arms and "fps~g1~hebb~nw" in arms and len(arms) == len(set(arms))
print("1 ok: cells/deep_mechanism sem encdeep inalterados; schema aceita o YAML e recusa chave, mecanismo,"
      f" subchave, kernels e dim_eq invalidos; bracos: {arms}")

# tronco sintetico L1..L3 (k5, pool 2) gravado e montado pelo caminho da grade
work = Path(tempfile.mkdtemp())
arch, source, C, size = {"nlayers": 0, "stdev_factor": 0.01}, work, 6, 48
for n in range(3):
    arch = spifil.grown_arch(arch, 5, C, 2)
    cin = 3 if n == 0 else C
    spifil.write_weights(work / f"L{n + 1}", source, arch, n + 1, SimpleNamespace(conv=SimpleNamespace(
        weight=torch.randn(C, cin, 5, 5), bias=torch.randn(C))))
    source = work / f"L{n + 1}"
encoder = build(str(source / "architecture.json"), "flim", str(source)).eval()
before = copy.deepcopy(encoder.state_dict())
hash3 = encdeep.trunk_hash(source, 3)
train = torch.randint(0, 256, (12, 3, size, size), dtype=torch.uint8)
to_float = lambda x: x.float() / 255

# 4. unfold dilatado == extract_patches do SPiFiL == conv; whitening explicito == dobrado, bordas incluidas
for k, dil in ((3, 1), (3, 2)):
    span = 2 * (k // 2) + 1
    x = to_float(train[:4])
    raw = encoder(x)
    H, W = raw.shape[-2:]
    seeds = Seeds.from_coords(torch.tensor([[0, 0], [H - 1, W - 1], [2, 3], [H // 2, 0]]), 0, (H, W))
    pf = extract_patches(raw[0], seeds, LayerSpec(kernel_size=k, dilation=dil)).feats
    cols = F.unfold(raw[:1], span, dilation=dil, padding=dil * (k // 2))[0].T
    assert torch.equal(pf, cols[[0, H * W - 1, 2 * W + 3, (H // 2) * W]])
    ref = cols[torch.randperm(len(cols))[:200]]
    fold = map_whitener(ref, span**2, 0.1)
    white = whitened(ref, *fold, span**2)
    mean, stdev = white.mean(0), white.std(0, unbiased=False) + 0.01
    u, b_v = F.normalize(torch.randn(5, C * span**2), dim=1), torch.randn(5)
    v, grid = encdeep.layer_rows(encoder, x, fold, mean, stdev, k, dil)
    explicit = (v @ u.T + b_v).transpose(1, 2).reshape(4, 5, H, W)
    weight = u / stdev
    weight, bias = folded(weight, -(weight * mean).sum(1) + b_v, *fold, span**2)
    a4 = spifil.grown_arch(arch, k, 5, 1, dil)
    spifil.write_weights(work / f"L4_{dil}", source, a4, 4, SimpleNamespace(conv=SimpleNamespace(
        weight=weight.view(5, -1, span, span), bias=bias)))
    net = build(str(work / f"L4_{dil}" / "architecture.json"), "flim", str(work / f"L4_{dil}")).eval()
    conv = net.conv4[0](raw)
    border = torch.ones(H, W, dtype=torch.bool)
    border[dil:-dil, dil:-dil] = False
    err, err_b = (explicit - conv).abs().max(), (explicit - conv)[..., border].abs().max()
    assert grid == (H, W) and err < 1e-4 * explicit.abs().max(), err
    assert net.conv4[0].dilation == (dil, dil) and net.conv4[0].padding == (dil, dil)
    print(f"4 ok: k{k} dil{dil}: extract_patches == unfold; explicito vs conv dobrada max|erro| {err:.1e}, borda {err_b:.1e}")

# 2. sem rotulo e sem autograd
names = {"y", "label", "labels", "mask", "masks", "target", "targets"}
for path in ("eval/encdeep.py", "flim/local_plasticity.py", "flim/local_predictive.py", "flim/slow_features.py"):
    tree = ast.parse(Path(path).read_text())
    hit = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in names}
    hit |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg) and n.arg in names}
    hit |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in names}
    assert not hit or path == "eval/encdeep.py" and hit == {"y"}, (path, hit)  # `y` so em save_errors (pos-SVM)
learn_src = inspect.getsource(encdeep.learn)
assert not {"y", "label", "labels", "mask"} & {n.id for n in ast.walk(ast.parse(learn_src)) if isinstance(n, ast.Name)}
assert 'train=images["train"][0]' in inspect.getsource(spf_grade.chain)
with encdeep.no_autograd():
    assert not torch.is_grad_enabled()
    for call in (lambda: torch.ones(1, requires_grad=True).sum().backward(),
                 lambda: torch.autograd.grad(torch.ones(1, requires_grad=True).sum(), [])):
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError("backward passou")
assert torch.Tensor.backward is not None and torch.is_grad_enabled()
torch.ones(1, requires_grad=True).sum().backward()  # restaurado
print("2 ok: nenhum rotulo/mascara no caminho encdeep (AST); hook so recebe images['train'][0];"
      " no_grad e backward/grad recusados dentro, restaurados fora")

# 3. + 8. cada mecanismo: so a camada nova muda, L1-L3 e a inferencia intactas, exportacao sem auxiliar
deep = yml["encdeep"] | {k: yml["encdeep"][k] | {"steps": 3, "batch": 3, "images": 12} for k in ("hebb", "clapp", "sfa")}
deep["sfa"] = deep["sfa"] | {"lentos": 20, "residuo": 20}
deep["fu"] = {"steps": 2, "batch": 2, "images": 12}
k, dil, span = 3, 2, 3
for mech in encdeep.ENCDEEP_MECHS:
    fn = encdeep.deep_mechanism({"encdeep": deep}, 3, ["g2", mech])
    u = F.normalize(torch.randn(48, C * span**2), dim=1)
    with encdeep.no_autograd():
        W, b, cols = fn(u=u, encoder=encoder, fold=fold, mean=mean, stdev=stdev, k=k, dil=dil, train=train,
                        to_float=to_float, seed=7)
    assert all(torch.equal(before[key], val) for key, val in encoder.state_dict().items()), mech
    assert W.shape[1] == u.shape[1] and len(b) == len(W) and not W.requires_grad and torch.isfinite(W).all()
    assert (mech == "z0") == torch.equal(W, u), mech
    weight = W / stdev
    weight, bias = folded(weight, -(weight * mean).sum(1) + b, *fold, span**2)
    out = work / f"L4_{mech}"
    a4 = spifil.grown_arch(arch, k, len(W), 1, dil)
    spifil.write_weights(out, source, a4, 4, SimpleNamespace(conv=SimpleNamespace(weight=weight.view(len(W), -1, span, span), bias=bias)))
    net = build(str(out / "architecture.json"), "flim", str(out)).eval()
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(["architecture.json"] + [f"conv{i}-{t}" for i in range(1, 5) for t in ("bias.txt", "kernels.npy")])
    assert encdeep.trunk_hash(out, 3) == hash3
    deploy = sum(p.numel() for p in net.parameters())
    assert deploy == 3 * 25 * C + 2 * C * C * 25 + 3 * C + len(W) * (C * 9 + 1)
    assert set(net.state_dict()) == {f"{pre}conv{i}.0.{t}" for pre in ("", "blocks.") for i in range(1, 5)
                                     for t in ("weight", "bias")}  # `blocks.` e o mesmo tensor, apelido do Encoder
    frozen = copy.deepcopy(net.state_dict())
    with torch.no_grad():
        net(to_float(train[:2]))
    assert all(torch.equal(frozen[key], val) for key, val in net.state_dict().items())
    print(f"3/8 ok: {mech:6s} L1-L3 intactas (hash {hash3}), inferencia nao muda pesos, export so conv*"
          f" ({deploy} params implantados, {cols['aux_params']} auxiliares fora), {cols}")

# geometria: H x W, stride e RF a 200 px batem com a tabela do G0 e com o mapa real
pool = [2, 2, 2, 2, 1]
want = {"g0": ("99-49-24-11-11", "43", "107"), "g1": ("99-49-24-24-24", "43", "75"), "g2": ("99-49-24-24-24", "43", "91")}
for g, (maps, rf3, rf5) in want.items():
    ks, ds, ps = encdeep.geometry({"encdeep": {"kernels": [5, 5, 5, 3, 3]}}, [g], pool)
    log = encdeep.geo_log([g], 200, ks, ds, ps)
    assert log["maps"] == maps and log["rfs"].split("-")[2] == rf3 and log["rfs"].split("-")[4] == rf5, log
    a = {"nlayers": 0, "stdev_factor": 0.01}
    for kk, dd, pp in zip(ks, ds, ps):
        a = spifil.grown_arch(a, kk, 2, pp, dd)
    blocks, h, sides = build_encoder_from_arch(a), torch.zeros(1, 3, 200, 200), []
    for i in range(1, 6):
        h = blocks[f"conv{i}"](h)
        sides.append(h.shape[-1])
    assert "-".join(map(str, sides)) == maps, (g, sides)
    print(f"geo ok: {g} {log}")
print("TODOS OS CHECKS PASSARAM")
