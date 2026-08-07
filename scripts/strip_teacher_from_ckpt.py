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

Uso:
  # dry-run (lista o que faria, não grava nada):
  python scripts/strip_teacher_from_ckpt.py --glob 'artifacts/distillation/*/checkpoints/best*.ckpt'
  # gravar de fato:
  python scripts/strip_teacher_from_ckpt.py --glob '...' --apply
  # só os best referenciados pelo run_metadata.json (recomendado):
  python scripts/strip_teacher_from_ckpt.py --from-metadata --apply
"""
import argparse, glob as globlib, json, os, sys, time

KEEP_PREFIXES = ("student.", "proj_kd.")
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
    for m in globlib.glob("artifacts/distillation/*/run_metadata.json"):
        try:
            best = json.load(open(m)).get("best_checkpoint", "")
        except Exception:
            continue
        if best and os.path.exists(best):
            out.append(best)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=None, help="padrão glob de checkpoints")
    ap.add_argument("--from-metadata", action="store_true",
                    help="usa o best_checkpoint de cada run_metadata.json")
    ap.add_argument("--apply", action="store_true", help="grava os arquivos (senão dry-run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="SOBRESCREVE o próprio .ckpt original (in-place, atômico via .tmp+replace). "
                         "DESTRUTIVO: o teacher é perdido (redownloadável p/ treino novo).")
    ap.add_argument("--suffix", default=".student.ckpt", help="sufixo do arquivo de saída (modo não-overwrite)")
    ap.add_argument("--skip-running", default="protozoan",
                    help="pula caminhos que contenham esta substring E 'no_imagenet_norm' (treino vivo)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="processa só os ckpts com índice i%%N==I (para rodar N workers em paralelo)")
    args = ap.parse_args()

    import torch  # importado só agora p/ dry-run não exigir o env

    if args.from_metadata:
        ckpts = collect_from_metadata()
    elif args.glob:
        ckpts = sorted(globlib.glob(args.glob))
    else:
        ap.error("informe --glob ou --from-metadata")

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        ckpts = [c for idx, c in enumerate(ckpts) if idx % n == i]
        print(f"[shard {i}/{n}] {len(ckpts)} checkpoint(s) neste worker\n")

    # pula treino vivo e arquivos já encolhidos
    def skip(p):
        if p.endswith(args.suffix):
            return "já é saída"
        if args.skip_running and args.skip_running in p and "no_imagenet_norm" in p:
            # só pula se o run ainda não tem metadata (sinal de treino em andamento)
            run = os.path.dirname(os.path.dirname(p))
            if not os.path.exists(os.path.join(run, "run_metadata.json")):
                return "treino em andamento"
        return None

    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — {len(ckpts)} checkpoint(s)\n")
    tot_old = tot_new = 0
    done = 0
    for i, src in enumerate(ckpts, 1):
        reason = skip(src)
        rel = os.path.relpath(src)
        if reason:
            print(f"{bar(i, len(ckpts))}  {i}/{len(ckpts)}  SKIP ({reason}): {rel}")
            continue
        if args.overwrite:
            dst = src
        else:
            dst = src[:-len(".ckpt")] + args.suffix if src.endswith(".ckpt") else src + args.suffix
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
        if args.apply:
            if args.overwrite:
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
              + ("" if args.apply else "  (estimativa, dry-run)"))

    print(f"\n{'='*60}\nconvertidos: {done}/{len(ckpts)}")
    print(f"antes:  {human(tot_old)}")
    print(f"depois: {human(tot_new)}   (economia ~{human(tot_old-tot_new)})")
    if not args.apply:
        print("\n(dry-run — nada gravado; adicione --apply para gerar os .student.ckpt)")
    print("NENHUM checkpoint original foi removido.")


if __name__ == "__main__":
    sys.exit(main())
