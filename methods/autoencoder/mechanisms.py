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

"""
Mecanismos de treino do encoder SPiFiL, todos desligados por padrao.

Moram dentro de ``conv{n}``, como SE e GRN em autoencoder_flim_module.py: quem chama o bloco
leva tudo junto, inclusive ``Encoder.forward``, o grow e ``eval/svm.py:_encode_pooled``, que
percorre as camadas a mao. ``MechBlock`` e um ``nn.Sequential``: ``[0]`` continua sendo a conv
e as chaves do state_dict nao mudam. So a soma acrescenta uma chave, ``gain``.

A conv nasce do SPiFiL (``flim.spifil.grow``) e o bloco e montado por cima dela a cada
estagio, entao nao ha numero maximo de camadas. EMA nao esta aqui: e o callback nativo
``EMAWeightAveraging`` do Lightning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import drop_block2d, stochastic_depth


class MechBlock(nn.Sequential):
    """``conv{n}`` com soma ``x + gain * f(x)`` e regularizacao depois da soma."""

    def __init__(self, block: nn.Sequential, gain: float = 0.0, drop_path: float = 0.0,
                 dropout: float = 0.0, spatial_dropout: float = 0.0,
                 dropblock: float = 0.0, dropblock_size: int = 5) -> None:
        super().__init__(*block)
        # Um ganho por filtro: o canal i de x e o filtro i do SPiFiL nao tem relacao,
        # cada filtro abre no seu ritmo. Sem zero, a conv recebe gradiente desde o passo 1.
        channels = block[0].out_channels
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), float(gain))) if gain else None
        self.drop_path = drop_path
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout
        self.dropblock = dropblock
        self.dropblock_size = dropblock_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.gain is not None:
            y = x + stochastic_depth(self.gain * y, self.drop_path, "row", self.training)
        if self.dropout:
            y = F.dropout(y, self.dropout, self.training)
        if self.spatial_dropout:
            y = F.dropout2d(y, self.spatial_dropout, self.training)
        if self.dropblock:
            y = drop_block2d(y, self.dropblock, self.dropblock_size, training=self.training)
        return y


def can_sum(block: nn.Sequential) -> bool:
    """Saida do mesmo tamanho da entrada: mesma largura, kernel impar, sem pool."""
    conv = block[0]
    return (conv.in_channels == conv.out_channels
            and all(k % 2 for k in conv.kernel_size)
            and not any(isinstance(m, nn.MaxPool2d) for m in block))


def apply_mechanisms(encoder: nn.Module, dropout: float = 0.0, spatial_dropout: float = 0.0,
                     dropblock: float = 0.0, dropblock_size: int = 5, drop_path: float = 0.0,
                     residual_gain: float = 0.0) -> nn.Module:
    """Monta os mecanismos em ``encoder``; com tudo em zero nao toca em nada.

    A soma entra em toda camada que ``can_sum`` aceita. Dropout, dropout espacial e DropBlock
    so na ultima, antes do decoder. Stochastic depth vai de ``drop_path / N`` na camada 1 ate
    ``drop_path`` na camada N, e so existe onde ha soma.
    """
    if drop_path and not residual_gain:
        raise ValueError("drop_path pula a soma: precisa de residual_gain > 0")
    n_layers = encoder.n_layers
    for n in range(1, n_layers + 1):
        block = encoder.blocks[f"conv{n}"]
        gain = residual_gain if residual_gain and can_sum(block) else 0.0
        reg = (dict(dropout=dropout, spatial_dropout=spatial_dropout, dropblock=dropblock)
               if n == n_layers else {})
        if gain or any(reg.values()):
            new = MechBlock(block, gain, drop_path * n / n_layers,
                            dropblock_size=dropblock_size, **reg)
            encoder.blocks[f"conv{n}"] = new
            setattr(encoder, f"conv{n}", new)
    return encoder


if __name__ == "__main__":
    import copy
    import tempfile

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EMAWeightAveraging

    from eval.svm import _encode_pooled
    from flim.encoder import Encoder
    from flim.spifil import grown_arch
    from methods.autoencoder.resnet_decoder import ResNetDecoder

    def stack(n_layers: int) -> dict:
        """Tres camadas com pool, e o resto de mesma largura, como o grow escreve."""
        arch = {"nlayers": 0}
        for n in range(n_layers):
            arch = grown_arch(arch, 3, 8 * min(n + 1, 3), 2 if n < 3 else 1)
        return arch

    torch.manual_seed(0)
    N = 9  # acima dos 5 de LAYERWISE_SPECS
    arch = stack(N)
    base = Encoder(arch, 3)
    decoder = ResNetDecoder(arch, 3, (64, 64))
    x = torch.rand(4, 3, 64, 64)
    ref = base.eval()(x)

    off = apply_mechanisms(copy.deepcopy(base))
    assert all(type(off.blocks[f"conv{n}"]) is nn.Sequential for n in range(1, N + 1))
    assert off.state_dict().keys() == base.state_dict().keys() and torch.equal(off(x), ref)
    print(f"desligado: forward identico, chaves iguais, N={N}")

    for key, value in (("dropout", 0.3), ("spatial_dropout", 0.3), ("dropblock", 0.3),
                       ("residual_gain", 0.1), ("drop_path", 0.5)):
        kw = {key: value} | ({"residual_gain": 0.1} if key == "drop_path" else {})
        enc = apply_mechanisms(copy.deepcopy(base), **kw)
        enc.train()
        torch.manual_seed(1)
        out = enc(x)
        decoder(out).pow(2).mean().backward()
        grad = enc.conv9[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0 and torch.isfinite(grad).all(), key
        changed = not torch.equal(out, ref)
        same_eval = torch.equal(enc.eval()(x), ref)
        pooled = _encode_pooled(enc, x, "avgpool2d")
        assert torch.allclose(pooled, enc(x).mean((2, 3)), atol=1e-6), key
        assert changed and same_eval == (key not in ("residual_gain", "drop_path")), key
        new_keys = enc.state_dict().keys() - base.state_dict().keys()
        assert all(k.endswith(".gain") for k in new_keys), key
        print(f"{key}={value}: treino muda, grad conv9 {grad.abs().sum():.3g}, "
              f"eval igual ao base={same_eval}, _encode_pooled == forward, chaves novas {len(new_keys)}")

    grown = apply_mechanisms(Encoder(stack(N + 1), 3), residual_gain=0.1)
    old = apply_mechanisms(copy.deepcopy(base), residual_gain=0.1)
    missing, unexpected = grown.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all("conv10" in k for k in missing)
    print(f"crescer {N} -> {N + 1}: so conv10 fica de fora do checkpoint antigo")

    class Tiny(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = apply_mechanisms(Encoder(stack(4), 3), residual_gain=0.1, dropout=0.1)

        def training_step(self, batch, _):
            return self.encoder(batch).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp:
        tiny = Tiny()
        trainer = pl.Trainer(max_steps=4, accelerator="cpu", logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp,
                             callbacks=[EMAWeightAveraging(decay=0.9)])
        start = copy.deepcopy(tiny.state_dict())
        trainer.fit(tiny, torch.utils.data.DataLoader(torch.rand(8, 3, 64, 64), batch_size=2))
        trainer.save_checkpoint(f"{tmp}/e.ckpt")
        ckpt = torch.load(f"{tmp}/e.ckpt", weights_only=False)
        assert any(not torch.equal(v, start[k]) for k, v in ckpt["state_dict"].items())
        print(f"ema: {len(ckpt['state_dict'])} tensores no ckpt, pesos mudaram")
