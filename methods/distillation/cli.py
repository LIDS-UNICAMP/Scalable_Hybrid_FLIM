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

"""cli.py — zona de transicao do argparse da destilacao (arquivo condenado).

A arvore mantem `methods/distillation/cli.py`, mas a regra global do refactor e
**matar argparse**. Resolucao: `cli.py` nasce como **zona de transicao**. Ele
existe apenas para que o corte de `src/models/distillation.py` seja um MOVE puro,
sem mudanca de comportamento. Em seguida, cada flag vira `init_args` do modulo no
YAML e a funcao correspondente e apagada. **Alvo final:
`methods/distillation/cli.py` deletado.** Enquanto existir, **nenhum entrypoint
novo pode importa-lo** — quem importa `cli.py` e codigo legado em vias de morrer,
nao codigo novo.

Conteudo: as seis funcoes que a spec MD-17 nomeia — `add_student_flags`,
`resolve_student`, `add_distill_flags`, `resolve_distill_flags`,
`derive_flim_paths`, `distill_run_tags` — coladas de `src/models/distillation.py`
sem mudar uma virgula, junto do minimo que elas usam (`_str2bool`, `STUDENTS`, o
`sys.path` de `scripts/constants.py`). A origem continua intacta: este arquivo e
uma copia de transicao, nao um corte.

A tabela `flag -> init_args` que permite apagar este arquivo esta no relatorio
MD-17 do refactor.
"""
from __future__ import annotations

import argparse
import os
import sys


# ── Single point of configuration shared by every distillation module ─────────
#
# All four distillation LightningModules declare the same flags by calling
# ``add_distill_flags`` and resolve them through ``resolve_distill_flags``.  The
# new user-facing flags are a façade over hyper-parameters the modules already
# have (``encoder_init``, ``distillation_type``), so nothing is duplicated per
# model.

# scripts/ nao e pacote importavel a partir de src/, e scripts/constants.py e a
# fonte unica de FLIM_ARCH_BASE/FLIM_WEIGHTS_BASE.
_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from constants import (  # noqa: E402
    ARCH_JSON_FILENAME, FLIM_ARCH_BASE, FLIM_WEIGHTS_BASE, MODELS_SUBDIR, train_dir,
)


def derive_flim_paths(args: "argparse.Namespace") -> None:
    """Preenche --arch-json / --flim-weights-path a partir de --dataset e --split.

    O valor passado na mao manda: so o caminho DERIVADO e checado em disco, entao
    quem ja passa as flags se comporta exatamente como antes. E o protozoan tem
    DUAS arvores (arch em ch24_30_48_a0.5_f5, pesos em ch24_32_48_a0.5_f5) — por
    isso os dois dicts separados, e nao um template de string.
    """
    def _need(path: str, ok: bool, flag: str) -> None:
        if not ok:
            raise SystemExit(
                f"[paths] caminho derivado de --dataset {args.dataset} --split {args.split} "
                f"nao existe: {path}\n        passe {flag} <caminho> na mao."
            )

    if not args.arch_json:
        args.arch_json = os.path.join(
            FLIM_ARCH_BASE[args.dataset], train_dir(args.split), ARCH_JSON_FILENAME)
        _need(args.arch_json, os.path.isfile(args.arch_json), "--arch-json")

    if not args.flim_weights_path and (args.init_flim or args.encoder_init == "flim"):
        args.flim_weights_path = os.path.join(
            FLIM_WEIGHTS_BASE[args.dataset], train_dir(args.split), MODELS_SUBDIR)
        _need(args.flim_weights_path, os.path.isdir(args.flim_weights_path), "--flim-weights-path")


def _str2bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")


# Os quatro students do experimento, medidos (encoder FLIM eggs 24_32_48 + cabeca de
# projecao, so `requires_grad`). Os numeros batem com statistics/tools/compute_cost.csv e
# com a legenda de A_reports/2026/august/details_report/table_result_methods.md.
#
# `module` e o modulo que SABE rodar aquele student, e `proj_kernel` so existe para os dois
# que dividem o mesmo modulo. Este dict e a UNICA fonte do mapa: a flag, a validacao e a
# mensagem de erro saem todas dele.
STUDENTS: dict = {
    "distill_1": {"module": "distillation_onelayer_module", "proj_kernel": 1,
                  "params": 123_504, "head": "OneLayer1x1ConvDistillationProjectionHead"},
    "distill_2": {"module": "distillation_twolayer_module", "proj_kernel": None,
                  "params": 402_544, "head": "TwoLayer1x1ConvBN2dDistillationProjectionHead"},
    "distill_3": {"module": "distillation_onelayer_module", "proj_kernel": 3,
                  "params": 615_024, "head": "OneLayerConvDistillationProjectionHead"},
    "distill_4": {"module": "distillation_conv_module", "proj_kernel": None,
                  "params": 889_200, "head": "ConvDistillationProjectionHead"},
}


def add_student_flags(parser: "argparse.ArgumentParser") -> "argparse.ArgumentParser":
    """Declare `--distill_1..4`: qual dos quatro students o comando esta treinando.

    FACHADA, nao seletor: a flag nao troca de modulo, ela NOMEIA e VALIDA o que o modulo ja
    ia fazer. Por isso e opcional — os lancadores Ray
    (scripts/distillation_conv_ray.py:564) escolhem o student pelo modulo que invocam e nao
    passam flag nenhuma; torna-la obrigatoria quebraria a producao. Quando presente, ela
    fecha a lacuna que motivou sua existencia: ler o comando e saber qual student roda.
    """
    g = parser.add_argument_group("student (qual dos quatro)")
    ex = g.add_mutually_exclusive_group()
    for flag, spec in STUDENTS.items():
        alvo = spec["module"].replace("distillation_", "").replace("_module", "")
        kern = f" --proj-kernel {spec['proj_kernel']}" if spec["proj_kernel"] else ""
        ex.add_argument(f"--{flag}", action="store_true", default=False,
                        help=f"{spec['params']:,} params — {alvo}{kern} "
                             f"({spec['head']}).")
    return parser


def resolve_student(args: "argparse.Namespace", module_name: str) -> dict:
    """Valida a flag de student contra o modulo que esta rodando.

    Devolve `{}` quando nenhuma flag foi passada (invocacao legada: o modulo e o
    `--proj-kernel` continuam mandando). Com flag, RECUSA a combinacao errada em vez de
    treinar outro student calado — que e o unico motivo de a flag existir.

    Returns:
        `{"proj_kernel": int}` quando a flag fixa o kernel, senao `{}`.
    """
    escolhidas = [f for f in STUDENTS if getattr(args, f, False)]
    if not escolhidas:
        return {}
    flag = escolhidas[0]
    spec = STUDENTS[flag]
    if spec["module"] != module_name:
        raise SystemExit(
            f"[student] --{flag} e o student de {spec['params']:,} params e roda em "
            f"src.modules.{spec['module']}, mas voce invocou src.modules.{module_name}.\n"
            f"           use: python -m src.modules.{spec['module']} ... --{flag}")
    return {"proj_kernel": spec["proj_kernel"]} if spec["proj_kernel"] else {}


def add_distill_flags(parser: "argparse.ArgumentParser") -> "argparse.ArgumentParser":
    """Declare the shared distillation flags on *parser*.

    Adds the six flags every distillation model must accept:
    ``--init_flim`` / ``--init_random``, ``--fine_tune``, ``--loss_cos`` /
    ``--loss_mse``, ``--teacher_frozen`` / ``--teacher_unfrozen``.
    """
    g = parser.add_argument_group("distillation (shared)")

    init = g.add_mutually_exclusive_group()
    init.add_argument("--init_flim", action="store_true", default=False,
                      help="Initialise the student encoder with FLIM weights "
                           "(requires --flim-weights-path).")
    init.add_argument("--init_random", action="store_true", default=False,
                      help="Initialise the student encoder randomly.")

    g.add_argument("--fine_tune", type=_str2bool, default=None,
                   metavar="True|False",
                   help="False: train on the teacher embedding only, with "
                        "--loss_cos or --loss_mse. True: train on the label "
                        "with the hybrid kd_loss (CE + KL).")

    emb = g.add_mutually_exclusive_group()
    emb.add_argument("--loss_cos", action="store_true", default=False,
                     help="Embedding loss = 1 - cosine. Only with --fine_tune False.")
    emb.add_argument("--loss_mse", action="store_true", default=False,
                     help="Embedding loss = MSE. Only with --fine_tune False.")

    tch = g.add_mutually_exclusive_group()
    tch.add_argument("--teacher_frozen", action="store_true", default=False,
                     help="Teacher stays in eval with requires_grad=False.")
    tch.add_argument("--teacher_unfrozen", action="store_true", default=False,
                     help="Teacher trains alongside the student.")

    g.add_argument("--kd_temperature", type=float, default=None,
                   help="Temperature T of the kd_loss KL term (--fine_tune True). "
                        "Default: 4.0.")
    g.add_argument("--kd_alpha", type=float, default=None,
                   help="Weight alpha of the kd_loss KL term (--fine_tune True). "
                        "Default: 0.7.")
    return parser


def distill_run_tags(hparams) -> list:
    """W&B tags derived from the shared flags.

    Makes a run's configuration readable at a glance in the W&B run list, which
    the ``run_name`` alone does not guarantee for runs launched by the ray
    scripts.

    Args:
        hparams: the module's ``self.hparams``.

    Returns:
        ``["init_flim", "loss_mse", "teacher_frozen"]`` and the like.
    """
    loss = {"direct": "loss_mse", "direct_cosine": "loss_cos"}.get(
        hparams.distillation_type, hparams.distillation_type)
    return [
        f"init_{hparams.encoder_init}",
        loss,
        "teacher_frozen" if hparams.teacher_frozen else "teacher_unfrozen",
    ]


def resolve_distill_flags(args: "argparse.Namespace") -> dict:
    """Validate the shared flags and resolve them into module kwargs.

    Raises:
        SystemExit: on any invalid combination, with an explicit message —
            an invalid run must fail loudly instead of training something else.

    Returns:
        ``{"encoder_init", "distillation_type", "fine_tune", "teacher_frozen",
        "kd_temperature", "kd_alpha"}``.
    """
    def fail(msg: str):
        raise SystemExit(f"[distill-flags] {msg}")

    # No new flag touched → legacy invocation; --encoder-init / --distillation-type
    # stay in charge and nothing is overridden.
    if not (args.init_flim or args.init_random or args.fine_tune is not None
            or args.loss_cos or args.loss_mse
            or args.teacher_frozen or args.teacher_unfrozen
            or args.kd_temperature is not None or args.kd_alpha is not None):
        return {}

    if args.init_flim == args.init_random:
        fail("choose exactly one of --init_flim / --init_random.")
    if args.teacher_frozen == args.teacher_unfrozen:
        fail("choose exactly one of --teacher_frozen / --teacher_unfrozen.")
    if args.fine_tune is None:
        fail("--fine_tune True|False is required.")

    if args.fine_tune:
        if args.loss_cos or args.loss_mse:
            fail("--loss_cos / --loss_mse are embedding losses and only apply "
                 "to --fine_tune False; with --fine_tune True the loss is kd_loss.")
        distillation_type = "kd_hybrid"
    else:
        if args.loss_cos == args.loss_mse:
            fail("with --fine_tune False, choose exactly one of --loss_cos / --loss_mse.")
        distillation_type = "direct_cosine" if args.loss_cos else "direct"

    if args.init_flim and not getattr(args, "flim_weights_path", None):
        fail("--init_flim requires --flim-weights-path.")

    return {
        "encoder_init":      "flim" if args.init_flim else "random",
        "distillation_type": distillation_type,
        "fine_tune":         bool(args.fine_tune),
        "teacher_frozen":    bool(args.teacher_frozen),
        "kd_temperature":    4.0 if args.kd_temperature is None else float(args.kd_temperature),
        "kd_alpha":          0.7 if args.kd_alpha is None else float(args.kd_alpha),
    }
