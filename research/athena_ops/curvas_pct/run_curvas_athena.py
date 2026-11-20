# Curvas kappa de TESTE x % de treino (1,5,10,25,50,75,100) no athena, SO GPUs 2/3 (memoria real), pedido 19/11.
# Encoder congelado; SVM original (C=100) em p% do treino = descritor ∩ treino da busca; PCA da so2 so no p%;
# val/teste da busca; split1. Cabecas gap1280, so2_4096_pca, so2_4096_pca_tta8. Saida: readout_x/results/curvas_pct.jsonl
import csv, json, os, subprocess, time
from pathlib import Path
D = Path(__file__).resolve().parent
RX = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/readout_x"); REPO = RX.parents[2]
OUT = RX / "results/curvas_pct.jsonl"
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"; LIB = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib"
GC = "/dados/home/moliveira/exps/growth/code"
NODES = f"{GC}/artifacts/hmodehb_b0b4/growth/athena_gpu3/ham10000_fix5_nossl/seed42/nodes"
OCT = "/dados/home/moliveira/artifacts/spifil_growth/disc_bp01w64c1280"
PRAW = "/dados/home/moliveira/artifacts/modehb_hybrid/protozoan_raw/state/seed42/candidates/c00004/f200/encoder.pt"
PROT = "protocolo do %: treino_p = descritor ∩ treino da busca; PCA so2 e SVM so no p%; val/teste da busca; split1; encoder congelado"
JOBS = [  # id, kind, ckpt, init, code_root, nota
 ("curva_ham10000_best", "growth", f"{NODES}/B1g0c06/f15/backbone.pt", None, GC,
  "melhor ham10000 = no de crescimento B1g0c06 f15 (growth/athena_gpu3/ham10000_fix5_nossl)"),
 ("curva_ham10000_spifil_sem_bp", "growth", f"{NODES}/seed00/backbone.pt", None, GC,
  "gemeo spifil_sem_bp = B0 SPiFiL da propria cadeia (seed00, pai do B1g0c06, fid 0), sem AE/backprop: convencao da Jaci p/ nos de crescimento"),
 ("curva_protozoan_spifil_sem_bp", "hybrid", f"{OCT}/protozoan_split1_pct100/L4/checkpoints/best_kappa.ckpt", "spifil", GC,
  "gemeo do AE de outubro disc_bp01w64c1280 L4 (split1, o mesmo split do melhor): mesma arquitetura, SPiFiL camada a camada "
  "(AutoEncoderFlimModule sem load_state_dict, spifil_init.py), sem AE; 188.468 params = treinado; descritor perc100 da discovery tem val == treino -> kappa_val e de TREINO"),
 ("curva_protozoan_raw_best", "modehb", PRAW, None, str(REPO),
  "melhor protozoan_raw = GA hibrido modehb_hybrid c00004 f200 (kappa 0,663 no marcados/models.csv); nomes <label:06d>_<raw> -> descritor raw"),
 ("curva_protozoan_raw_spifil_sem_bp", "modehb", PRAW, "spifil", str(REPO),
  "gemeo spifil_sem_bp do c00004 f200: spifil_init.py (Evaluator._grow com ae_off, softhebb off, sem heranca); nomes <label:06d>_<raw> -> descritor raw"),
]
NOTE = {j[0]: j[5] for j in JOBS}
def merge(rows):
    have = set()
    if OUT.is_file():
        for l in open(OUT):
            try: r = json.loads(l); have.add((r["backbone_id"], r["head"]))
            except Exception: pass
    with open(OUT, "a") as fh:
        for r in rows:
            if (r["backbone_id"], r["head"]) not in have: fh.write(json.dumps(r, default=str) + "\n"); have.add((r["backbone_id"], r["head"]))
# protozoan melhor: so conversao do CSV de outubro (split1), sem rodar
rows = []
for r in csv.DictReader(open(REPO / "results/svm_disc_bp01w64c1280_pcts_test_gemttapwr.csv")):
    if r["dataset_short"] != "protozoan" or r["split"] != "1" or r["stage_label"] != "L4": continue
    p = int(r["percentage"])
    rows.append({"backbone_id": f"curva_protozoan_best_p{p}", "kind": "hybrid", "dataset": "protozoan", "head": "gem_tta_pw",
                 "kappa_test": float(r["kappa"]), "acc_test": float(r["acc"]), "f1_test": float(r["f1"]), "kappa_val": None,
                 "dim": int(r["dim"]), "pct": p, "split": 1, "ckpt": str(REPO / r["ckpt"]), "ckpt_epoch": r["ckpt_epoch"],
                 "svm_protocol": "SVM original (core.svm.fit_svm, linear C=100 ovo, sem scaler)", "host": "athena",
                 "fonte": "results/svm_disc_bp01w64c1280_pcts_test_gemttapwr.csv (convertido, nada rodado)",
                 "nota": "melhor protozoan = AE de outubro disc_bp01w64c1280 L4. LEITURA GeM+TTA+whitening (gem_tta_pw), PROTOCOLO "
                         "DE OUTUBRO: o encoder e TREINADO em cada p% (um checkpoint por p), nao congelado; sem p10; "
                         "nao e gap/so2/so2_tta8, nao comparar direto com as curvas do readout_x",
                 "never_feeds_fitness": True})
merge(sorted(rows, key=lambda r: r["pct"])); print(f"[CONV] protozoan best: {len(rows)} linhas (p {sorted(r['pct'] for r in rows)})", flush=True)
# os 5 testes: GPU 2/3 pela memoria real, pico 9 GB + margem 2
def free():
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(i): int(f) / 1024 for i, f in (l.split(", ") for l in o.strip().splitlines())}
env = dict(os.environ, READOUT_SO2_DEVICE="cuda", LD_LIBRARY_PATH=LIB, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4",
           PYTHONWARNINGS="ignore", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
todo, run, t0 = list(JOBS), [], time.time()
while todo or run:
    fr = free()
    for g in (2, 3):                                         # NUNCA 0/1
        res = sum(9.0 for x in run if x[2] == g and time.time() - x[3] < 180)
        if todo and fr.get(g, 0) - res >= 11.0:
            i, kind, ck, init, code, _n = todo.pop(0)
            cmd = ["nice", "-n", "10", PY, "-u", str(RX / "eval_backbone_head.py"), "--code-root", code,
                   "--item", f"id={i},kind={kind},ckpt={ck}" + (",init=spifil,variant=spifil_init" if init else ""),
                   "--heads", "gap1280", "so2_4096_pca", "--tta8", "--train-pcts", "1", "5", "10", "25", "50", "75", "100",
                   "--device", "cuda", "--batch-size", "32", "--num-workers", "4", "--data-cache", str(REPO / "artifacts/readout_x_cache"),
                   "--out", str(D / "parts" / f"{i}.jsonl"), "--table-md", os.devnull]
            p = subprocess.Popen(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=open(D / "logs" / f"{i}.log", "a"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            run.append((p, i, g, time.time())); print(f"[START] {i} gpu{g} (free {fr[g]:.1f} GB)", flush=True); break
    time.sleep(2)                                         # 2 s: ganha do runner (10 s) a memoria que liberar = prioridade
    for x in list(run):
        if x[0].poll() is None: continue
        run.remove(x); part = D / "parts" / f"{x[1]}.jsonl"
        rows = [json.loads(l) for l in open(part)] if part.is_file() else []
        for r in rows: r.update(nota=NOTE[x[1]], protocolo=PROT)
        merge(rows)
        print(f"[END] {x[1]} rc={x[0].returncode} {len(rows)} linhas {(time.time() - x[3]) / 60:.1f} min", flush=True)
print(f"[FIM] {(time.time() - t0) / 60:.1f} min", flush=True)
