#!/usr/bin/env python3
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
"""strip_teacher_from_ckpt.py — encolhe checkpoints removendo o teacher congelado.

NÃO-DESTRUTIVO: lê <best>.ckpt e grava <best>.student.ckpt (novo arquivo).
Mantém apenas o necessário para inferência/SVM:
  - state_dict com chaves student.* e proj_kd.*  (descarta teacher.*)
  - hyper_parameters  (arch_json, encoder_init, flim_weights_path, etc.)
  - epoch / global_step / versão do Lightning
Descarta optimizer_states, lr_schedulers, callbacks (não precisam p/ inferência).

O formato de saída continua sendo um .ckpt Lightning normal — os avaliadores
(svm_distillation_conv / svm_distill_with_projection) carregam sem qualquer mudança,
pois já filtram apenas student.*/proj_kd.*.

Uso (da raiz do repositorio; o argparse morreu no refactor, os parametros sao
defaults nomeados de ``strip_teacher_from_ckpt``):

  # dry-run com os defaults (lista o que faria, nao grava nada):
  python -m experiments.ckpt.strip_teacher_from_ckpt

  # qualquer outro parametro: chamada direta da funcao
  python -c "from experiments.ckpt.strip_teacher_from_ckpt import strip_teacher_from_ckpt as f; \
             f(from_metadata=True, apply=True)"
"""
import glob as globlib, json, os, time

from core.constants import RUN_METADATA_FILENAME
from experiments.constants import KEEP_STUDENT_PREFIXES as KEEP_PREFIXES

KEEP_TOP = ("hyper_parameters", "epoch", "global_step",
            "pytorch-lightning_version", "loops", "state_dict")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}"
        n /= 1024


def bar(done, total, width=24):
    filled = int(width * done / total) if total else width
    return "[" + "█" * filled + "░" * (width - filled) + f"] {100*done//max(total,1)}%"


def collect_from_metadata():
    out = []
    # Glob RELATIVO ao cwd de propósito: absolutizar via ARTIFACTS_DISTILLATION_DIR
    # faria o script passar a achar checkpoints rodado de fora da raiz — bug fix, não refactor.
    for m in globlib.glob(os.path.join("artifacts", "distillation", "*", RUN_METADATA_FILENAME)):
        try:
            best = json.load(open(m)).get("best_checkpoint", "")
        except Exception:
            continue
        if best and os.path.exists(best):
            out.append(best)
    return sorted(set(out))


def strip_teacher_from_ckpt(
    glob_pattern: str | None = None,
    from_metadata: bool = False,
    apply: bool = False,
    overwrite: bool = False,
    suffix: str = ".student.ckpt",
    skip_running: str = "protozoan",
    shard: str | None = None,
) -> int:
    """Encolhe checkpoints de destilacao guardando so student + proj.

    Os defaults reproduzem o `ap.error("informe --glob ou --from-metadata")` de
    antes: sem parametro nenhum o script recusa e explica. Um parametro so
    muda por chamada de funcao.

    glob_pattern  padrao glob de checkpoints (exclusivo com from_metadata)
    from_metadata usa o best_checkpoint de cada run_metadata.json
    apply         grava os arquivos (senao dry-run)
    overwrite     SOBRESCREVE o proprio .ckpt original (in-place, atomico via
                  .tmp+replace). DESTRUTIVO: o teacher e perdido.
    suffix        sufixo do arquivo de saida (modo nao-overwrite)
    skip_running  pula caminhos que contenham esta substring E
                  'no_imagenet_norm' (treino vivo)
    shard         "I/N": processa so os ckpts com indice i%N==I
    """
    import torch  # importado só agora p/ dry-run não exigir o env

    if from_metadata and glob_pattern:
        raise SystemExit("strip_teacher_from_ckpt: informe glob_pattern OU from_metadata, nao os dois")
    if from_metadata:
        ckpts = collect_from_metadata()
    elif glob_pattern:
        ckpts = sorted(globlib.glob(glob_pattern))
    else:
        raise SystemExit("strip_teacher_from_ckpt: informe glob_pattern ou from_metadata")

    if shard:
        i, n = (int(x) for x in shard.split("/"))
        ckpts = [c for idx, c in enumerate(ckpts) if idx % n == i]
        print(f"[shard {i}/{n}] {len(ckpts)} checkpoint(s) neste worker\n")

    # pula treino vivo e arquivos já encolhidos
    def skip(p):
        if p.endswith(suffix):
            return "já é saída"
        if skip_running and skip_running in p and "no_imagenet_norm" in p:
            # só pula se o run ainda não tem metadata (sinal de treino em andamento)
            run = os.path.dirname(os.path.dirname(p))
            if not os.path.exists(os.path.join(run, RUN_METADATA_FILENAME)):
                return "treino em andamento"
        return None

    print(f"{'APPLY' if apply else 'DRY-RUN'} — {len(ckpts)} checkpoint(s)\n")
    tot_old = tot_new = 0
    done = 0
    for i, src in enumerate(ckpts, 1):
        reason = skip(src)
        rel = os.path.relpath(src)
        if reason:
            print(f"{bar(i, len(ckpts))}  {i}/{len(ckpts)}  SKIP ({reason}): {rel}")
            continue
        if overwrite:
            dst = src
        else:
            dst = src[:-len(".ckpt")] + suffix if src.endswith(".ckpt") else src + suffix
        old = os.path.getsize(src)
        t0 = time.time()
        ck = torch.load(src, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", {})
        new_sd = {k: v for k, v in sd.items() if k.startswith(KEEP_PREFIXES)}
        dropped = len(sd) - len(new_sd)
        out = {k: ck[k] for k in KEEP_TOP if k in ck}
        out["state_dict"] = new_sd
        out["_stripped_teacher"] = True  # marcador do novo formato
        est = sum(v.numel() for v in new_sd.values() if hasattr(v, "numel"))
        if apply:
            if overwrite:
                tmp = src + ".tmp_strip"   # escrita atômica: não corrompe se interromper
                torch.save(out, tmp)
                os.replace(tmp, src)
            else:
                torch.save(out, dst)
            new = os.path.getsize(dst)
        else:
            new = est * 4  # estimativa (~fp32)
        tot_old += old
        tot_new += new
        done += 1
        print(f"{bar(i, len(ckpts))}  {i}/{len(ckpts)}  {rel}\n"
              f"        {human(old)} → {human(new)}  | params mantidos={est/1e3:.0f}k "
              f"| chaves teacher removidas={dropped} | {time.time()-t0:.1f}s"
              + ("" if apply else "  (estimativa, dry-run)"))

    print(f"\n{'='*60}\nconvertidos: {done}/{len(ckpts)}")
    print(f"antes:  {human(tot_old)}")
    print(f"depois: {human(tot_new)}   (economia ~{human(tot_old-tot_new)})")
    if not apply:
        print("\n(dry-run — nada gravado; passe apply=True para gerar os .student.ckpt)")
    print("NENHUM checkpoint original foi removido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(strip_teacher_from_ckpt())
