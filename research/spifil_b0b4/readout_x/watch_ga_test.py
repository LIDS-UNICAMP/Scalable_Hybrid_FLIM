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
"""watch_ga_test.py -- vigia de TESTE de uma busca modehb_v4 / pnas_cs_v4 (nunca decide nada na busca).

Fica armado esperando <state>/search_state.json. A cada bracket fechado (nova entrada em
brackets_log) e no fim da busca (done=True), escolhe PELA VALIDACAO, na regua da busca
(result.json:fitness_kappa = so2_4096 + SVM C=100 sem scaler), o melhor (cand, fid) de cada
banda/estrato (B0-B4 / P0-P4) entre os jobs ate ali (select_ga_best.results) e mede o TESTE com
run_ga_best.py -> eval_backbone_head.py na CPU (CUDA_VISIBLE_DEVICES=''): SVM original
(core.svm.fit_svm, linear, C=100, ovo, sem scaler) nas leituras gap1280 / so2_4096_pca /
so2_4096_pca_tta8. O teste e calculado uma vez por (cand_id, fid, sha256 do encoder.pt).

Saida ao lado do state: test_best_per_band.jsonl (uma linha por bracket x banda),
test_best_raw.jsonl (bruto do eval), test_best_list.json, test_best_logs/.

  python research/spifil_b0b4/readout_x/watch_ga_test.py --state artifacts/pnas_cs_v4/ham10000/state/seed42
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_ga_best import REPO, best, results  # noqa: E402

OUT, RAW = "test_best_per_band.jsonl", "test_best_raw.jsonl"
HEADS = {"gap": "gap1280", "so2": "so2_4096_pca", "tta8": "so2_4096_pca_tta8"}


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_pass(st, search, ds, bracket, upto, a):
    done = {(r["bracket"], r["band"]) for r in jl(st / OUT)}
    _, rows = results(search, ds, st=st, upto=upto)
    picks = []
    for band in sorted({r["stratum"] for r in rows}):
        if (bracket, band) not in done:
            m = dict(best([r for r in rows if r["stratum"] == band]))
            m["sha256"] = sha256(REPO / m["ckpt"])
            m["id"] = f"{ds}_{m['cand']}_f{m['fid']}_{m['sha256'][:12]}"   # chave (cand_id, fid, sha)
            picks.append((band, m))
    if not picks:
        return
    uniq = list({m["id"]: m for _, m in picks}.values())
    (st / "test_best_list.json").write_text(json.dumps({"bases": {ds: {"models": uniq}}}, indent=1))
    subprocess.run([sys.executable, str(HERE / "run_ga_best.py"), "--list", str(st / "test_best_list.json"),
                    "--out", str(st / RAW), "--logs", str(st / "test_best_logs"), "--bases", ds,
                    "--jobs", str(a.jobs), "--threads", str(a.threads)], cwd=REPO, check=False)
    raw = {}
    for r in jl(st / RAW):
        raw.setdefault(r["backbone_id"], {})[r["head"]] = r
    with open(st / OUT, "a") as f:
        for band, m in picks:
            got = raw.get(m["id"], {})
            if not all(h in got for h in HEADS.values()):
                print(f"[watch] {bracket}/{band} {m['id']}: eval incompleto, tenta no proximo ciclo", flush=True)
                continue
            rec = {"bracket": bracket, "band": band, "search": search, "dataset": ds, "cand_id": m["cand"],
                   "fid": m["fid"], "sha256": m["sha256"], "params": m["params"], "ckpt": m["ckpt"],
                   "kappa_val_search": m["kappa_val_search"], "selection": "max kappa val (regua da busca)",
                   **{f"kappa_val_{k}": got[h]["kappa_val"] for k, h in HEADS.items()},
                   **{f"kappa_test_{k}": got[h]["kappa_test"] for k, h in HEADS.items()},
                   **{f"f1_test_{k}": got[h]["f1_test"] for k, h in HEADS.items()},
                   "svm_protocol": got[HEADS["so2"]]["svm_protocol"], "n_test": got[HEADS["so2"]]["n_test"],
                   "never_feeds_fitness": True, "used_for_selection": False, "time": time.time()}
            f.write(json.dumps(rec) + "\n")
            print(f"[watch] {bracket}/{band} {m['cand']}/f{m['fid']} kval={m['kappa_val_search']:.4f} "
                  f"k_test so2={rec['kappa_test_so2']:.4f} tta8={rec['kappa_test_tta8']:.4f} "
                  f"gap={rec['kappa_test_gap']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="pasta do state, relativa ao repo (artifacts/...)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    st = REPO / a.state
    search, ds = Path(a.state).parts[1], Path(a.state).parts[2]
    print(f"[watch] armado: {st} ({search}/{ds})", flush=True)
    while True:
        try:
            s = json.loads((st / "search_state.json").read_text())
        except Exception:            # ainda nao comecou / escrita em andamento
            time.sleep(a.poll)
            continue
        for b in s.get("brackets_log", []):
            run_pass(st, search, ds, b["bracket"], b["n_jobs"], a)
        if s.get("done"):
            run_pass(st, search, ds, "final", None, a)
            if {r["band"] for r in jl(st / OUT) if r["bracket"] == "final"} >= \
                    {r["stratum"] for r in results(search, ds, st=st)[1]}:
                print("[watch] busca terminada, teste final feito", flush=True)
                return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
