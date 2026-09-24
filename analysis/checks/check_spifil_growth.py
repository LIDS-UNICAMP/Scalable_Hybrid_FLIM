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
"""check_spifil_growth.py — a UNICA verificacao runnable do protocolo de crescimento.

    python -m analysis.checks.check_spifil_growth

Cobre so as quatro pecas que nao sao obvias por leitura, e nada alem disso:

1. **O decoder crescido continua devolvendo a imagem no tamanho certo.** Toda a
   garantia de forma do autoencoder mora em `ResNetDecoder`, que le
   `arch[f"layer{n}"]["pooling"]["stride"]` como fator de upsample
   (src/models/autoencoder_resnet.py:112) e forca `out_size` num interpolate
   final (:124). Se a camada nova entrar com um pooling que o decoder nao
   espelha, a reconstrucao muda de tamanho e o BCE explode. Testado nas DUAS
   variantes que o grow pode produzir (pooling `none`/stride 1 e `max_pool`/
   stride 2) e tambem na arquitetura de 3 camadas nao crescida, para provar que
   crescer nao mexeu na garantia.

2. **A regra de parada do laco.** `should_stop` decide sozinha quando o
   protocolo acaba; errar nela e treinar rodadas a toa ou parar cedo demais.

3. **O ida-e-volta do layout de peso FLIM.** `flim_kernels` e o inverso exato de
   `flim/weights.py:shift_weights`. Se os dois discordarem, a camada que o
   SPiFiL cortou NAO e a camada que o treinador remonta do `.npy` — e o erro e
   silencioso, porque a forma continua batendo.

4. **O congelamento do estagio 3.** O curriculo manda treinar SO a camada FLIM nova e
   SO o bloco novo do decoder; o resto (encoder antigo, decoder antigo, `to_image`)
   fica parado. Errar isso nao quebra nada visivelmente — o run treina, converge e
   mede kappa, so que estagio 3 vira outra coisa. Testado pelo `__init__` de verdade,
   com um `init_ckpt` sintetico de 3 blocos de decoder (delta = 1).

Sem pytest, sem tests/, sem dependencia nova: um script que imprime uma linha por
verificacao e morre no primeiro assert que falhar. Mesmo estilo de
`analysis/checks/check_refactor_equivalence.py`.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# analysis/checks/ esta a 2 niveis da raiz do repo (o arquivo veio de scripts/, que era 1).
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from src.models.autoencoder_resnet import AutoEncoderFLIM  # noqa: E402
from flim.weights import shift_weights  # noqa: E402

_EGGS_ARCH = os.path.join(_ROOT, "data", "to_mateus", "model",
                          "ch24_32_48_a0.5_f5", "eggs", "train1", "architecture.json")

# As duas unicas variantes de pooling que `spifil_grow.py` sabe escrever.
_POOL_NONE = {"type": "none", "size": [3, 3, 0], "stride": 1}
_POOL_MAX2 = {"type": "max_pool", "size": [3, 3, 0], "stride": 2}


def _load(filename: str, module_name: str):
    """Importa um irmao de scripts/ por caminho — nao existe scripts/__init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── (a) forma do autoencoder crescido ─────────────────────────────────────────

def _grown(arch: dict, pooling: dict) -> dict:
    """Acrescenta a 4a camada como o `spifil_grow.py` a escreve: 3x3, 48 canais, relu."""
    grown = copy.deepcopy(arch)
    n = arch["nlayers"] + 1
    grown["nlayers"] = n
    grown[f"layer{n}"] = {
        "conv": {"kernel_size": [3, 3, 0], "nkernels_per_marker": 48,
                 "dilation_rate": [1, 1, 0], "nkernels_per_image": 48,
                 "noutput_channels": 48},
        "relu": True,
        "pooling": copy.deepcopy(pooling),
    }
    return grown


def _recon_shape(arch: dict) -> tuple:
    model = AutoEncoderFLIM(arch=arch, in_channels=3, out_size=(200, 200))
    with torch.no_grad():
        return tuple(model(torch.zeros(2, 3, 200, 200)).shape)


def check_grown_decoder_shape() -> None:
    with open(_EGGS_ARCH, encoding="utf-8") as f:
        base = json.load(f)
    assert base["nlayers"] == 3, f"a arch de eggs deixou de ter 3 camadas: {base['nlayers']}"

    want = (2, 3, 200, 200)
    got_base = _recon_shape(base)
    assert got_base == want, f"a arch NAO crescida ja reconstruia errado: {got_base} != {want}"

    for label, pooling in (("none/stride1", _POOL_NONE), ("max_pool/stride2", _POOL_MAX2)):
        grown = _grown(base, pooling)
        got = _recon_shape(grown)
        assert got == want, f"arch crescida ({label}) reconstruiu {got}, esperado {want}"
    print(f"  (a) grown decoder OK  {want} para 3 camadas e para 4 nas duas variantes de pooling")


# ── (b) regra de parada ───────────────────────────────────────────────────────

# (kappas, tolerancia, patience, deve_parar, por que)
_STOP_CASES = [
    ([0.50, 0.62],             0.01, 1, False, "ganho claro na rodada 1"),
    ([0.50, 0.62, 0.71],       0.01, 1, False, "dois ganhos claros seguidos"),
    ([0.50, 0.505],            0.01, 1, True,  "ganho menor que a tolerancia = empate"),
    ([0.50, 0.505],            0.01, 2, False, "um empate so nao esgota patience 2"),
    ([0.50, 0.505, 0.508],     0.01, 2, True,  "dois empates seguidos esgotam patience 2"),
    ([0.50, 0.41],             0.01, 1, True,  "regressao"),
    ([0.50, 0.41, 0.62],       0.01, 2, False, "recuperacao depois de uma rodada ruim"),
    ([0.50],                   0.01, 1, False, "so a baseline, nenhuma rodada ainda"),
]


def check_should_stop() -> None:
    should_stop = _load("spifil_growth_loop.py", "spifil_growth_loop").should_stop
    for kappas, tol, patience, expected, why in _STOP_CASES:
        got = should_stop(kappas, tol, patience)
        assert got is expected, \
            f"should_stop({kappas}, {tol}, {patience}) = {got}, esperado {expected} ({why})"
    print(f"  (b) should_stop   OK  {len(_STOP_CASES)} casos, tolerancia 0.01")


# ── (c) ida-e-volta do layout FLIM ────────────────────────────────────────────

def check_flim_kernels_roundtrip() -> None:
    path = os.path.join(_HERE, "spifil_grow.py")
    if not os.path.isfile(path):
        print("  (c) flim layout  SKIP  scripts/spifil_grow.py ainda nao existe")
        return

    flim_kernels = _load("spifil_grow.py", "spifil_grow").flim_kernels
    k, c, kh, kw = 7, 5, 3, 3
    weight = torch.from_numpy(np.random.RandomState(0).randn(k, c, kh, kw).astype(np.float32))

    npy = np.asarray(flim_kernels(weight))
    assert npy.shape == (c * kh * kw, k), f"flim_kernels devolveu {npy.shape}, esperado {(c * kh * kw, k)}"

    back = shift_weights(npy, (kw, kh), c)
    assert back.shape == tuple(weight.shape), f"shift_weights devolveu {back.shape}, esperado {tuple(weight.shape)}"
    assert np.allclose(back, weight.numpy()), "shift_weights(flim_kernels(W)) != W — o layout .npy diverge"
    print(f"  (c) flim layout   OK  W{tuple(weight.shape)} -> .npy{npy.shape} -> W, elemento a elemento")


# ── (d) congelamento do estagio 3 ─────────────────────────────────────────────

def check_stage3_freeze() -> None:
    from methods.autoencoder import AutoEncoderFlimModule  # noqa: E402

    with open(_EGGS_ARCH, encoding="utf-8") as f:
        grown = _grown(json.load(f), _POOL_MAX2)
    n = grown["nlayers"]

    with tempfile.TemporaryDirectory() as tmp:
        arch_json = os.path.join(tmp, "architecture.json")
        with open(arch_json, "w", encoding="utf-8") as f:
            json.dump(grown, f)

        # conv1..conv3 sao os pesos FLIM reais; conv4 e a camada que o SPiFiL teria
        # cortado — 48 kernels 3x3 sobre 48 canais, no layout .npy do FLIM.
        weights = os.path.join(tmp, "weights")
        shutil.copytree(os.path.join(os.path.dirname(_EGGS_ARCH), "models"), weights)
        with open(os.path.join(weights, f"conv{n}-bias.txt"), "w", encoding="utf-8") as f:
            f.write(f"48\n{' '.join(['0.0'] * 48)}\n")
        np.save(os.path.join(weights, f"conv{n}-kernels.npy"),
                np.zeros((48 * 3 * 3, 48), dtype=np.float32))

        # ckpt da rodada anterior: 3 camadas => 3 blocos de decoder => delta = 1.
        ckpt = os.path.join(tmp, "prev.ckpt")
        torch.save({"state_dict": {f"model.decoder.blocks.{i}.conv1.weight": torch.zeros(1)
                                   for i in range(3)}}, ckpt)

        module = AutoEncoderFlimModule(arch_json=arch_json, flim_weights_path=weights,
                                       freeze_encoder_flag=True, init_ckpt=ckpt)

    named = dict(module.model.named_parameters())
    got = {name for name, p in named.items() if p.requires_grad}
    want = {name for name in named
            if name.startswith(f"encoder.blocks.conv{n}.") or name.startswith("decoder.blocks.0.")}
    assert want, "cenario montado errado: nenhum parametro na camada nova"
    assert got == want, (f"estagio 3 treinando o conjunto errado:\n  a mais: {sorted(got - want)}"
                         f"\n  a menos: {sorted(want - got)}")
    for name in ("decoder.to_image.weight", "decoder.to_image.bias", "decoder.blocks.1.conv1.weight"):
        assert not named[name].requires_grad, f"{name} devia estar congelado no estagio 3"
    print(f"  (d) stage3 freeze OK  delta=1, treinam so encoder.blocks.conv{n}.* e "
          f"decoder.blocks.0.* ({len(want)} tensores)")


if __name__ == "__main__":
    print("check_spifil_growth")
    check_grown_decoder_shape()
    check_should_stop()
    check_flim_kernels_roundtrip()
    check_stage3_freeze()
    print("ALL CHECKS PASSED")
