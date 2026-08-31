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
"""
Diz se um checkpoint de destilacao esta MAGRO ou GORDO --- sem treinar nem carregar modelo.

GORDO e o ckpt gravado antes do ``FrozenTeacherCheckpointMixin`` entrar nos quatro modulos de
destilacao: leva as ~517 chaves ``teacher._encoder._model.*`` do I-JEPA (~2,4 GB) e os
``optimizer_states``, e por isso pesa ~2,5 GB. MAGRO e o que os ``ModelCheckpoint`` gravam
hoje (``save_weights_only=True``): so ``student.``, ``proj_kd.``, ``cls_head.`` e
``teacher_cls_head.``. O teacher congelado volta do cache do HuggingFace no load, entao nada
se perde --- ver ``src/models/distillation.py`` (``FrozenTeacherCheckpointMixin``).

Um teacher DESTREINADO (``--unfreeze_teacher``) tem peso treinado e por isso e gravado de
proposito: ali ``teacher.*`` no disco e o esperado, e o script diz isso em vez de acusar gordura.

Este script so INSPECIONA. Quem reescreve ckpt antigo continua sendo
``scripts/strip_teacher_only.py``, que remove ``teacher.*`` preservando o resto.

Uso:
  python -m analysis.checks.check_ckpt_slim  # glob default de artifacts/distillation
  # os antigos flags sao parametros de check_ckpt_slim(): ckpt, self_test
"""
from __future__ import annotations

import glob as globlib
import os
import sys
from types import SimpleNamespace

import torch

# analysis/checks/ esta a 2 niveis da raiz do repo (o arquivo veio de tools/, que era 1).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# strip_teacher_only.py foi para experiments/ckpt/ no refactor; scripts/ fica no
# path so enquanto os launchers antigos ainda viverem la.
for _p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "experiments", "ckpt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import (ARCH_JSON_FILENAME, ARTIFACTS_DISTILLATION_DIR,   # noqa: E402
                       FLIM_ARCH_BASE, TEACHER_PREFIX)
from strip_teacher_only import human                               # noqa: E402


def _nbytes(obj) -> int:
    """Bytes de tensor somados recursivamente (optimizer_states e dict aninhado)."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _report(path: str) -> None:
    size = os.path.getsize(path)
    # mmap: le so o indice do zip, nao os 2,4 GB de peso do teacher.
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("state_dict", {})
    hp = ck.get("hyper_parameters", {}) or {}

    groups: dict[str, list] = {}
    for k, v in sd.items():
        groups.setdefault(k.split(".")[0], []).append(v)

    teacher_b = _nbytes({k: v for k, v in sd.items() if k.startswith(TEACHER_PREFIX)})
    optim_b   = _nbytes(ck.get("optimizer_states", []))
    frozen    = hp.get("teacher_frozen", "?")
    fat       = bool(teacher_b) or bool(optim_b)

    print(f"\n{path}")
    print(f"  tamanho          {human(size)}")
    for g in sorted(groups):
        print(f"  {g + '.':<18} {len(groups[g]):>4} chaves  {human(_nbytes(groups[g]))}")
    print(f"  optimizer_states {'sim' if optim_b else 'nao'}  {human(optim_b)}")
    print(f"  teacher_frozen   {frozen}    teacher_model_id  {hp.get('teacher_model_id', '?')}")

    if not fat:
        print("  -> MAGRO")
    elif teacher_b and frozen is False:
        # Teacher treinado: gravar e correto. So o optimizer_states sobra.
        print(f"  -> teacher DESTREINADO: teacher.* e esperado; economizaria {human(optim_b)}")
    else:
        print(f"  -> GORDO: economizaria {human(teacher_b + optim_b)} "
              f"({human(size - teacher_b - optim_b)} restantes)")


# ── Self-test: prova que o ckpt magro volta identico ──────────────────────────

_TEACHER_SD_PREFIX = TEACHER_PREFIX + "_encoder._model."
_PROBE_KEYS = ("embeddings.patch_embeddings.projection.weight",
               "encoder.layer.0.layernorm_before.weight")


def _self_test() -> int:
    """Round-trip do ckpt magro, sem treinar: salva, recarrega e compara tensores.

    Prova (1) que nenhuma chave do teacher vai para o disco, (2) que o teacher
    recarregado bate bit a bit com o ``model.safetensors`` de origem e (3) que
    ``student.``/``proj_kd.`` batem com o modulo que gerou o ckpt.
    """
    import subprocess           # noqa: PLC0415
    import tempfile             # noqa: PLC0415
    from safetensors.torch import load_file   # noqa: PLC0415

    from methods.lejepa.ijepa_encoder import _find_safetensors_path  # noqa: PLC0415
    from methods.distillation import DistillationConvModule  # noqa: PLC0415

    arch = os.path.join(FLIM_ARCH_BASE["eggs"], "train1", ARCH_JSON_FILENAME)
    kw = dict(arch_json=arch, distillation_type="direct", encoder_init="trunc_normal")
    fails = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        fails += not ok
        assert ok, f"{name}: {detail}"

    print(f"\n== self-test == arch_json={arch}")
    src_mod = DistillationConvModule(**kw)

    # (1) Round-trip frozen: on_save_checkpoint tira o teacher do state_dict.
    full = src_mod.state_dict()
    ckpt = {"state_dict": dict(full), "hyper_parameters": dict(src_mod.hparams)}
    src_mod.on_save_checkpoint(ckpt)
    saved = ckpt["state_dict"]

    n_teacher_full = sum(k.startswith(_TEACHER_SD_PREFIX) for k in full)
    check("teacher fora do disco",
          not any(k.startswith(TEACHER_PREFIX) for k in saved),
          f"{n_teacher_full} chaves {_TEACHER_SD_PREFIX}* no modulo vivo, "
          f"{sum(k.startswith(TEACHER_PREFIX) for k in saved)} no ckpt")
    for pref in ("student.", "proj_kd."):
        check(f"{pref}* preservado",
              any(k.startswith(pref) for k in saved),
              f"{sum(k.startswith(pref) for k in saved)} chaves")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "slim.ckpt")
        torch.save(ckpt, path)
        print(f"  ckpt magro em disco: {human(os.path.getsize(path))} "
              f"({len(saved)} chaves)")

        dst_mod = DistillationConvModule(**kw)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        dst_mod.on_load_checkpoint(loaded)
        dst_mod.load_state_dict(loaded["state_dict"], strict=True)

    # (2) Teacher recarregado == safetensors de origem.
    origin = load_file(_find_safetensors_path(src_mod.hparams.teacher_model_id))
    got = dst_mod.state_dict()
    for k in _PROBE_KEYS:
        a, b = got[_TEACHER_SD_PREFIX + k].cpu(), origin[k]
        check(f"teacher {k}", torch.equal(a, b),
              f"shape={tuple(a.shape)} max|d|={(a - b).abs().max().item():.3e}")

    # (3) student./proj_kd. recarregados == modulo original, bit a bit.
    trained = [k for k in saved if k.startswith(("student.", "proj_kd."))]
    diffs = [k for k in trained if not torch.equal(got[k].cpu(), full[k].cpu())]
    check("student./proj_kd. bit a bit", not diffs,
          f"{len(trained)} tensores iguais, {len(diffs)} diferentes")

    # (4) Caso negativo: sem o safetensors de origem o erro tem que ser ALTO.
    with tempfile.TemporaryDirectory() as empty:
        env = {**os.environ, "HF_HOME": empty, "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run(
            [sys.executable, "-c",
             "from methods.lejepa import IJEPAEncoder; IJEPAEncoder()"],
            cwd=_ROOT, env=env, capture_output=True, text=True)
    msg = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    check("falha alta sem o peso de origem",
          r.returncode != 0 and "FileNotFoundError" in msg,
          f"rc={r.returncode} :: {msg}")

    print(f"  -> {'TODOS OS ASSERTS PASSARAM' if not fails else f'{fails} FALHA(S)'}")
    return int(bool(fails))


def check_ckpt_slim(ckpt=(), self_test: bool = False) -> int:
    # O corpo abaixo continua lendo `a.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    a = SimpleNamespace(**locals())
    # Sem ckpt nenhum vale o mesmo default de antes: o glob de artifacts/distillation.
    a.ckpt = list(a.ckpt) or sorted(globlib.glob(os.path.join(
        ARTIFACTS_DISTILLATION_DIR, "*", "checkpoints", "*.ckpt")))
    if a.self_test:
        return _self_test()
    if not a.ckpt:
        print("Nenhum checkpoint encontrado.")
        return 1
    for p in a.ckpt:
        try:
            _report(p)
        except Exception as exc:
            # Ckpt truncado por disco cheio existe no repo; nao pode matar o lote.
            print(f"\n{p}\n  ILEGIVEL: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_ckpt_slim())
