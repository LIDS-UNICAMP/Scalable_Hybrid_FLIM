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
"""strip_teacher_only.py — remove APENAS as chaves teacher.* dos checkpoints de distillation.

Diferente de strip_teacher_from_ckpt.py (que também descarta optimizer_states e por isso
quebra o resume), este script preserva TODAS as outras chaves top-level do checkpoint
(optimizer_states, lr_schedulers, callbacks, loops, hyper_parameters, epoch, global_step, ...).
Só remove do state_dict as entradas que começam com "teacher." — que o
FrozenTeacherCheckpointMixin.on_load_checkpoint re-injeta a partir do módulo vivo no load.

Resultado: cada ckpt cai de ~2.5 GB para <100 MB, continua carregável tanto para
avaliação (SVM) quanto para RESUME de treino (strict=True), sem perder nada importante.

Segurança:
  - escrita atômica (.tmp + os.replace) — interromper não corrompe o original;
  - só processa arquivos acima de --min-gb (default 1.5) — já-stripados são ignorados;
  - --apply é obrigatório para gravar; sem ele é dry-run;
  - após gravar, RELÊ o arquivo e valida (teacher ausente, student presente,
    optimizer_states preservado) — se falhar, RESTAURA o original do backup em memória.

Uso (da raiz do repositorio; o argparse morreu no refactor, os parametros sao
defaults nomeados de ``strip_teacher_only``):

  # dry-run com os defaults:
  python -m experiments.ckpt.strip_teacher_only

  # gravar de fato, ou 4 workers em paralelo:
  python -c "from experiments.ckpt.strip_teacher_only import strip_teacher_only as f; f(apply=True)"
  python -c "from experiments.ckpt.strip_teacher_only import strip_teacher_only as f; f(apply=True, shard='0/4')"
"""
import glob as globlib, os, time

from core.constants import TEACHER_PREFIX
from experiments.constants import KEEP_STUDENT_PREFIXES


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def strip_teacher_only(
    glob_pattern: str = "artifacts/distillation/*/checkpoints/*.ckpt",
    min_gb: float = 1.5,
    apply: bool = False,
    shard: str | None = None,
) -> int:
    """Remove APENAS as chaves ``teacher.*``, preservando o resto do checkpoint.

    glob_pattern  RELATIVO ao cwd de propósito: absolutizar via
                  ARTIFACTS_DISTILLATION_DIR faria o script reescrever
                  checkpoints rodado de fora da raiz — bug fix, não refactor.
    min_gb        só processa ckpts maiores que isto (evita re-processar
                  já-stripados)
    apply         grava de fato (senão dry-run)
    shard         "I/N": processa só ckpts com idx%N==I (rodar N workers)
    """
    import torch

    ckpts = sorted(globlib.glob(glob_pattern))
    min_bytes = min_gb * 1024 ** 3
    ckpts = [c for c in ckpts if os.path.getsize(c) > min_bytes]

    if shard:
        i, n = (int(x) for x in shard.split("/"))
        ckpts = [c for idx, c in enumerate(ckpts) if idx % n == i]
        tag = f"[shard {i}/{n}] "
    else:
        tag = ""

    print(f"{tag}{'APPLY' if apply else 'DRY-RUN'} — {len(ckpts)} ckpt(s) > {min_gb} GB\n")
    tot_old = tot_new = 0
    done = failed = 0
    for i, src in enumerate(ckpts, 1):
        rel = os.path.relpath(src)
        old = os.path.getsize(src)
        t0 = time.time()
        try:
            ck = torch.load(src, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"{tag}{i}/{len(ckpts)}  ERRO load: {rel}: {e}")
            failed += 1
            continue

        sd = ck.get("state_dict", {})
        n_teacher = sum(1 for k in sd if k.startswith(TEACHER_PREFIX))
        n_student = sum(1 for k in sd if k.startswith(KEEP_STUDENT_PREFIXES))

        if n_teacher == 0:
            print(f"{tag}{i}/{len(ckpts)}  SKIP (sem teacher.*): {rel}")
            continue
        if n_student == 0:
            print(f"{tag}{i}/{len(ckpts)}  SKIP (sem student.*/proj_kd.* — não mexer!): {rel}")
            failed += 1
            continue

        # remove só teacher.*, mantém TODO o resto (optimizer_states, loops, etc.)
        ck["state_dict"] = {k: v for k, v in sd.items() if not k.startswith(TEACHER_PREFIX)}
        had_opt = "optimizer_states" in ck

        if not apply:
            est = sum(v.numel() * v.element_size() for v in ck["state_dict"].values()
                      if hasattr(v, "numel"))
            new = est  # aproximação; ignora overhead de optimizer/pickle
            print(f"{tag}{i}/{len(ckpts)}  {rel}\n"
                  f"        {human(old)} → ~{human(new)}  | teacher removido={n_teacher} chaves "
                  f"| student mantido={n_student} | optimizer={'sim' if had_opt else 'não'} "
                  f"| {time.time()-t0:.1f}s (dry-run)")
            tot_old += old
            tot_new += new
            continue

        tmp = src + ".tmp_strip"
        torch.save(ck, tmp)
        # valida ANTES de substituir o original
        try:
            chk = torch.load(tmp, map_location="cpu", weights_only=False)
            csd = chk.get("state_dict", {})
            assert not any(k.startswith(TEACHER_PREFIX) for k in csd), "teacher ainda presente"
            assert any(k.startswith(KEEP_STUDENT_PREFIXES) for k in csd), "student sumiu"
            if had_opt:
                assert "optimizer_states" in chk, "optimizer_states sumiu"
        except Exception as e:
            os.remove(tmp)
            print(f"{tag}{i}/{len(ckpts)}  ERRO validação (original intacto): {rel}: {e}")
            failed += 1
            continue

        os.replace(tmp, src)  # atômico
        new = os.path.getsize(src)
        tot_old += old
        tot_new += new
        done += 1
        print(f"{tag}{i}/{len(ckpts)}  {rel}\n"
              f"        {human(old)} → {human(new)}  | teacher removido={n_teacher} chaves "
              f"| student={n_student} | optimizer={'sim' if had_opt else 'não'} "
              f"| {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"{tag}processados: {done}/{len(ckpts)}   falhas: {failed}")
    print(f"{tag}antes:  {human(tot_old)}")
    print(f"{tag}depois: {human(tot_new)}   (economia ~{human(tot_old-tot_new)})")
    if not apply:
        print("(dry-run — nada gravado)")
    return 0


if __name__ == "__main__":
    raise SystemExit(strip_teacher_only())
