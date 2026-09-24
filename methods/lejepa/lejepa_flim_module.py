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
Lightning module for LeJEPA pretraining with the FLIM Encoder backbone.

Same training objective as LeJEPAModule / LeJEPACNNModule:
    loss = lam * sigreg(proj) + (1 - lam) * invariance(proj)

Adds a **configurable encoder initializer** via ``encoder_init``:
    - ``"random"``  — default PyTorch initialisation (no action).
    - ``"he"``      — Kaiming / He initialisation.
    - ``"xavier"``  — Xavier / Glorot initialisation.
    - ``"flim"``    — Load FLIM-trained weights from disk.

Movido de src/modules/lejepa_flim_module.py. O parse do ``architecture.json``, a
resolucao de canais e a carga dos pesos FLIM sairam daqui: agora vivem atras da
porta unica ``flim.build``, consumida pelo ``LeJEPAFLIMModel``. A sequencia de
operacoes e a mesma de antes (monta o encoder, depois carrega os pesos), so mudou
de arquivo.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from core.blocks.init import INIT_FNS
from core.losses import invariance_loss
from methods.lejepa.lejepa_module import LeJEPAModule, _SIGREG_TYPES
from methods.lejepa.lejepa_flim_model import LeJEPAFLIMModel

_ENCODER_INITS = ("random", "he", "xavier", "flim")

# Os unicos valores de `encoder_init` que o if/elif historico deste modulo
# resolvia. Ver o bloco `# ponytail:` em __init__.
_HISTORIC_INITS = ("he", "xavier")


class LeJEPAFLIMModule(LeJEPAModule):
    """
    Lightning module for LeJEPA pretraining with the FLIM Encoder backbone.

    The FLIM architecture is loaded from an ``architecture.json`` file. The
    encoder can be initialised with one of four strategies via ``encoder_init``.

    Args:
        arch_json:         Path to the FLIM architecture JSON file.
        encoder_init:      Initialisation strategy: ``"random"`` | ``"he"`` |
                           ``"xavier"`` | ``"flim"``.
        flim_weights_path: Directory containing FLIM weight files
                           (``conv{n}-kernels.npy``, ``conv{n}-bias.txt``).
                           **Required** when ``encoder_init="flim"``.
        in_channels:       Number of input image channels. Default: 3.
        proj_dim:          Projection head output dimension.
        proj_hidden:       Projection head hidden dimension.
        lam:               Weight on SIGReg; (1 - lam) on invariance loss.
        sigreg_type:       ``"simple"`` or ``"real"`` (Epps-Pulley).
        lr:                Peak learning rate.
        weight_decay:      AdamW weight decay.
        max_epochs:        Total training epochs (for cosine schedule).
        warmup_epochs:     Epochs of linear LR warmup before cosine decay.
        num_log_images:           Images to log per view in W&B.
        log_img_every_n_epochs:   Log input views every N epochs.
        use_init_fns:      ``False`` (default) mantem o comportamento historico:
                           so ``he`` e ``xavier`` sao aplicados aqui e qualquer
                           outra funcao de init cai em random com um aviso.
                           ``True`` resolve ``encoder_init`` pelo dicionario
                           unico ``core.blocks.init.INIT_FNS``, que tem a funcao
                           real. Setavel pelo YAML em ``init_args``.
    """

    def __init__(
        self,
        arch_json: str,
        encoder_init: str = "random",
        flim_weights_path: Optional[str] = None,
        in_channels: int = 3,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        lam: float = 0.05,
        sigreg_type: str = "simple",
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        num_log_images: int = 8,
        log_img_every_n_epochs: int = 5,
        use_init_fns: bool = False,
    ) -> None:
        # super de 2 argumentos: pula a construcao do pai — o LeJEPAModel
        # (encoder TIMM) nao entra no state_dict — e, por mencionar o nome
        # `super`, mantem a cell implicita `__class__` no frame, sem a qual o
        # save_hyperparameters abaixo sai vazio (parsing.py:_get_init_args).
        super(LeJEPAModule, self).__init__()
        if sigreg_type not in _SIGREG_TYPES:
            raise ValueError(
                f"sigreg_type must be one of {list(_SIGREG_TYPES)}, got '{sigreg_type}'"
            )
        if encoder_init not in _ENCODER_INITS:
            raise ValueError(
                f"encoder_init must be one of {list(_ENCODER_INITS)}, got '{encoder_init}'"
            )
        if encoder_init == "flim" and flim_weights_path is None:
            raise ValueError("flim_weights_path is required when encoder_init='flim'")

        # `use_init_fns` fica FORA do hparams de proposito: e chave de rota de
        # codigo, nao hiperparametro do run, e sem o ignore o hparams desta
        # classe ganharia uma 15a chave que o modulo antigo nao tem.
        self.save_hyperparameters(ignore="use_init_fns")

        # ── Build model ─────────────────────────────────────────────────
        # O arch_json, os canais reais e os pesos FLIM sao resolvidos dentro do
        # model pela porta unica `flim.build`.
        self.model = LeJEPAFLIMModel(
            arch_json=arch_json,
            init=encoder_init,
            weights_path=flim_weights_path,
            in_channels=in_channels,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
        )

        # ── Apply encoder initialisation ────────────────────────────────
        # `flim` e `random` mapeiam para None no INIT_FNS: o primeiro ja foi
        # carregado dentro do model, o segundo e o default do torch.
        _init_fn = INIT_FNS.get(encoder_init)
        if _init_fn is None:
            pass
        elif use_init_fns or encoder_init in _HISTORIC_INITS:
            _init_fn(self.model.encoder)
        else:
            # ponytail: divida deliberada. O if/elif historico deste modulo
            # (src/modules/lejepa_flim_module.py:140-147) so tinha branch para
            # he/xavier/flim; qualquer outro init aceito pela validacao caia em
            # silencio no random. O comportamento e preservado de proposito para
            # que pesos e checkpoints ja gravados continuem sendo chamados como
            # antes. O caminho correto existe e esta a um `use_init_fns: true`
            # de distancia; quando ninguem mais depender do rotulo antigo, este
            # else morre e o INIT_FNS passa a resolver sozinho.
            logging.getLogger(__name__).warning(
                "[LeJEPAFLIMModule] encoder_init='%s' nao tem branch neste modulo "
                "e esta caindo em RANDOM (comportamento historico preservado). "
                "Na pratica, os runs rotulados '%s' que passaram por aqui sao "
                "random. Para aplicar a inicializacao real, passe "
                "use_init_fns: true em init_args.",
                encoder_init, encoder_init,
            )

        self.sigreg = _SIGREG_TYPES[sigreg_type]()

    # ── Forward / training ──────────────────────────────────────────────

    def forward(self, views: List[Tensor]) -> tuple[Tensor, Tensor]:
        return self.model.forward(views)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, _ = batch
        emb, proj = self.model.forward(views)

        sig_loss = self.sigreg(emb)
        inv_loss = invariance_loss(emb)
        loss = self.hparams.lam * sig_loss + (1 - self.hparams.lam) * inv_loss

        self.log("train/sigreg", sig_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/inv",    inv_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss",   loss,     prog_bar=True,  on_step=True, on_epoch=True)

        return loss

    # ── Optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self):
        warmup_epochs = self.hparams.warmup_epochs or 10
        max_epochs = self.hparams.max_epochs or 100
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-5
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
