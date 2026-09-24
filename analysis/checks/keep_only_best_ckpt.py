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
"""keep_only_best_ckpt.py — deixa UM unico checkpoint por experimento: o melhor.

Cada run dir de destilacao acumula best.ckpt, best-v1.ckpt, last.ckpt, last-v1.ckpt...
Este script escolhe, por diretorio, o ckpt com o melhor `best_model_score` gravado pelo
ModelCheckpoint (respeitando `mode` min/max) e apaga os demais. Nao e "um grupo de bests":
sobra exatamente um arquivo por experimento.

Empate: `best*` vence `last*` (ambos gravam o mesmo `best_model_score`).
Empate remanescente ou score ausente: desempata pelo mtime mais novo.
Ckpt ilegivel (truncado por disco cheio) e sempre lixo — entra na lista de remocao.

Seguranca:
  - dry-run por padrao; apply=True e obrigatorio para apagar;
  - pula diretorios escritos nos ultimos skip_recent_min minutos (run ainda vivo);
  - le com mmap, nao carrega os 2.4 GB de peso para inspecionar.

Uso:
  python -m analysis.checks.keep_only_best_ckpt
  # os antigos flags sao parametros de keep_only_best(): glob, apply, skip_recent_min
  # apply=False por padrao: sem ele o script so lista o que apagaria
"""
import glob as globlib, os, time
from types import SimpleNamespace

import torch
from tqdm import tqdm


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def score_of(path):
    """(mode, score) gravados pelo ModelCheckpoint; (None, None) se ilegivel/ausente."""
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return None, None
    for key, state in (ck.get("callbacks") or {}).items():
        if "ModelCheckpoint" not in str(key):
            continue
        sc = state.get("best_model_score")
        if sc is None:
            continue
        return ("max" if "'max'" in str(key) or state.get("mode") == "max" else "min"), float(sc)
    return None, None


def keep_only_best(glob: str = "artifacts/distillation/*/checkpoints/*.ckpt",
                   apply: bool = False, skip_recent_min: float = 30.0):
    # O corpo abaixo continua lendo `args.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    dirs = {}
    for p in globlib.glob(args.glob):
        dirs.setdefault(os.path.dirname(p), []).append(p)

    cutoff = time.time() - args.skip_recent_min * 60
    freed = kept = skipped = 0
    doomed = []

    for d, files in tqdm(sorted(dirs.items()), desc="run dirs", unit="dir"):
        if len(files) < 2:
            kept += len(files)
            continue
        if max(os.path.getmtime(f) for f in files) > cutoff:
            skipped += 1
            continue

        ranked = []
        for f in files:
            mode, sc = score_of(f)
            ranked.append((sc is not None, sc if mode != "min" else (-sc if sc is not None else None),
                           os.path.basename(f).startswith("best"), os.path.getmtime(f), f))
        # tem score vence sem score; entre os com score, maior chave (ja invertida p/ min);
        # empate: best* vence last* (gravam o mesmo score); depois, mtime mais novo
        ranked.sort(key=lambda r: (r[0], r[1] if r[1] is not None else float("-inf"), r[2], r[3]))
        best = ranked[-1][4]
        kept += 1
        for *_, f in ranked[:-1]:
            freed += os.path.getsize(f)
            doomed.append(f)

    for f in doomed:
        print(f"{'APAGA ' if args.apply else 'apagaria'} {human(os.path.getsize(f)):>9}  {f}")
        if args.apply:
            os.remove(f)

    verb = "Liberado" if args.apply else "Liberaria"
    print(f"\n{len(doomed)} arquivos · {verb} {human(freed)} · {kept} mantidos · {skipped} dirs pulados (ativos)")
    if not args.apply and doomed:
        print("dry-run. Repita com --apply para apagar.")


if __name__ == "__main__":
    keep_only_best()
