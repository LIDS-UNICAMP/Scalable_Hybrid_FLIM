"""run_curvas_pct.py -- curvas κ × % de treino (1,5,10,25,50,75,100) do MELHOR modelo de cada base e do seu gêmeo
"SPiFiL sem backprop" (research/tools/spifil_twin.py). Encoder congelado; SVM original C=100 em p% do treino
(descritor ∩ treino da busca); val/teste da busca; 2ª ordem + TTA8 (+ TTA2h em flowers). Saída results/curvas_pct.jsonl."""
import json, os, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; R = H.parents[2]; A = R / "artifacts"
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
OUT = H / "results" / "curvas_pct.jsonl"; LOG = H / "results" / "curvas_pct_logs"; LOG.mkdir(parents=True, exist_ok=True)
PCTS = [1, 5, 10, 25, 50, 75, 100]
BEST = {  # base: (sigla, kind, ckpt) -- melhores marcados nos gráficos (host jaci)
 "flowers": ("GA-H c00064", "modehb", A / "modehb_hybrid/flowers/state/seed42/candidates/c00064/f5/encoder.pt"),
 "aid": ("GA-H c00091", "modehb", A / "modehb_hybrid/aid/state/seed42/candidates/c00091/f45/encoder.pt"),
 "coconut_trees": ("GA-H c00015", "modehb", A / "modehb_hybrid/coconut_trees/state/seed42/candidates/c00015/f5/encoder.pt"),
 "eggs": ("GA-L c00032", "modehb", A / "modehb_lemonade/eggs/state/seed42/candidates/c00032/f15/encoder.pt"),
 "larvae": ("GA-L c00078", "modehb", A / "modehb_lemonade/larvae/state/seed42/candidates/c00078/f15/encoder.pt"),
 "eggs_raw": ("GA-H c00027", "modehb", A / "modehb_hybrid/eggs_raw/state/seed42/candidates/c00027/f5/encoder.pt"),
 "larvae_raw": ("GA-H c00023", "modehb", A / "modehb_hybrid/larvae_raw/state/seed42/candidates/c00023/f15/encoder.pt"),
 "remote_sensing": ("GA-LS c00002", "modehb", A / "modehb_lemonade_seeded/remote_sensing/state/seed42/candidates/c00002/f200/encoder.pt"),
}
env = dict(os.environ, READOUT_SO2_DEVICE=os.environ.get("READOUT_SO2_DEVICE", "cpu"), OMP_NUM_THREADS="4")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
done = set()
if OUT.is_file():
    for l in OUT.read_text().splitlines():
        try: r = json.loads(l); done.add(r["backbone_id"])
        except Exception: pass
ONLY = sys.argv[2:]  # bases a rodar (vazio = todas)
for base, (sig, kind, ck) in BEST.items():
    if ONLY and base not in ONLY: continue
    twin = A / "spifil_twin" / base / f"{ck.parts[-8].replace('modehb_', '')}_{ck.parts[-3]}_{ck.parts[-2]}" / "encoder.pt"
    if not twin.is_file():
        subprocess.run([PY, str(R / "research/tools/spifil_twin.py"), str(ck), str(twin.parent), "--device", dev], cwd=R, env=env,
                       stdout=open(LOG / f"twin_{base}.log", "w"), stderr=subprocess.STDOUT)
    for role, path in (("best", ck), ("spifil_sem_bp", twin)):
        if not Path(path).is_file(): print("FALTA", base, role, flush=True); continue
        for p in PCTS:
            i = f"curva_{base}_{role}_p{p}"
            if i in done: continue
            cmd = [PY, str(H / "eval_backbone_head.py"), "--code-root", str(R), "--item", f"id={i},kind={kind},ckpt={path}",
                   "--heads", "gap1280", "so2_4096_pca", "--tta8", *(["--tta2h"] if base == "flowers" else []),
                   *(["--train-pct", str(p)] if p != 100 else []), "--device", dev, "--batch-size", "16", "--num-workers", "4",
                   "--data-cache", str(A / "readout_x_cache"), "--out", str(OUT), "--table-md", str(LOG / "_t.md")]
            rc = subprocess.run(cmd, cwd=R, env=env, stdout=open(LOG / f"{i}.log", "w"), stderr=subprocess.STDOUT).returncode
            print(f"{base} {role} p{p} rc={rc}", flush=True)
print("FIM", flush=True)
