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
"""Gera o md dos testes do athena (all_untested_test.jsonl) a cada 10 min: por base, modelos ordenados pelo kappa de VAL
(so2), com kappa de TESTE em gap / so2 / tta8. O teste nunca alimenta selecao; a ordem e so pela val."""
import json, time
from collections import defaultdict
from pathlib import Path
SRC = Path(__file__).resolve().parent / "results" / "all_untested_test.jsonl"
OUT = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM/A_reports/november/testes_todos_athena_20261118.md")
BASES = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees",
         "eggs", "larvae", "protozoan", "eggs_raw", "larvae_raw", "protozoan_raw"]
H = {"gap1280": "gap", "so2_4096_pca": "so2", "so2_4096_pca_tta8": "tta8"}
f2 = lambda v: "" if v is None else f"{v:.3f}"
while True:
    rows = defaultdict(dict)
    for l in [x for f in (SRC, SRC.parent / "jaci_all_untested_test.jsonl") if f.exists() for x in open(f)]:   # athena + Jaci
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("head") in H:
            m = rows[(r["dataset"], r["backbone_id"])]
            m[H[r["head"]]] = r
    ic = {}
    if (SRC.parent / "init_class.jsonl").exists():
        for l in open(SRC.parent / "init_class.jsonl"):
            try:
                r = json.loads(l); ic[r["backbone_id"]] = r.get("init_class", "")
            except Exception:
                pass
    out = [f"# Testes de todos os modelos (athena) — atualizado {time.strftime('%d/%m %H:%M')}", "",
           "SVM original (C=100, linear, ovo, sem scaler) no TESTE. Ordem por κ de VAL (so2); o teste nunca entra na seleção. "
           f"Fontes: `{SRC}` (athena) e `jaci_all_untested_test.jsonl` (Jaci, sincronizado a cada 15 min).", "", f"Modelos testados: **{len(rows)}**", ""]
    for b in BASES:
        it = [(bid, m) for (d, bid), m in rows.items() if d == b]
        if not it:
            continue
        it.sort(key=lambda x: -((x[1].get("so2") or x[1].get("gap") or {}).get("kappa_val") or 0))
        out += [f"## {b} ({len(it)})", "", "| modelo | init | κ val so2 | κ teste gap | κ teste so2 | κ teste tta8 |", "|---|---|---|---|---|---|"]
        for bid, m in it:
            g = lambda k, f: (m.get(k) or {}).get(f)
            out.append(f"| {bid} | {ic.get(bid, '?')} | {f2(g('so2', 'kappa_val'))} | {f2(g('gap', 'kappa_test'))} | {f2(g('so2', 'kappa_test'))} | {f2(g('tta8', 'kappa_test'))} |")
        out.append("")
    OUT.write_text("\n".join(out))
    time.sleep(600)
