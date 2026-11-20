"""A1b — extrai TODOS os jobs dos search_state.json (Jaci + bundle athena) para A1b_data.csv.

Só leitura. Cada linha = 1 job (candidato × fidelidade). Colunas rastreáveis: source_state, job, cand_id,
fidelity, checkpoint. Junta o teste (ORIG/COV) de test_svm_C100.jsonl (Jaci + bundle) por (run, dataset, cand, fid).
Uso: python extract_jobs.py  (CPU, sem GPU)
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM")
BUNDLE = Path("/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena")
OUT = ROOT / "research/spifil_b0b4/agent1/parts/A1b_data.csv"

SOURCES = []  # (path, run, kind, fitness_sig, machine)
for p in sorted(glob.glob(str(ROOT / "artifacts/modehb_hybrid/*/state/seed42/search_state.json"))):
    SOURCES.append((p, "artifacts/modehb_hybrid", "hybrid", "C100_kw1", "jaci"))
for p in sorted(glob.glob(str(ROOT / "artifacts/modehb_lemonade/*/state/seed42/search_state.json"))):
    SOURCES.append((p, "artifacts/modehb_lemonade", "lemonade", "C100_kw1", "jaci"))
for p, run, kind in [
    (ROOT / "artifacts/modehb_hybrid_C01_final/remote_sensing/state/seed42/search_state.json", "artifacts/modehb_hybrid_C01_final", "hybrid"),
    (ROOT / "artifacts/modehb_lemonade_C01_final/remote_sensing/state/seed42/search_state.json", "artifacts/modehb_lemonade_C01_final", "lemonade"),
]:
    SOURCES.append((str(p), run, kind, "C0.1_kw0.75", "jaci"))
for p in sorted(glob.glob(str(ROOT / "artifacts/modehb_hybrid_C01_2049/*/state/seed42/search_state.json"))):
    SOURCES.append((p, "artifacts/modehb_hybrid_C01_2049", "hybrid", "C0.1_kw0.75", "jaci"))
SOURCES.append((str(BUNDLE / "pnas_cs/search_state.json"), "artifacts/pnas_cs", "pnas_cs", "C100_kw1", "athena_bundle"))

LF = ["effective_rank", "effective_rank_ratio", "rankme", "alignment", "uniformity", "cov_condition_number",
      "mean_abs_offdiag_corr", "norm_mean", "norm_std", "var_mean", "var_min", "knn_view_consistency",
      "clustering_stability", "frac_near_constant_dims", "collapse_dimensional", "collapse_total"]


def dataset_of(path, run):
    if run == "artifacts/pnas_cs":
        return "remote_sensing"  # pnas_cs athena GPU2 = remote_sensing (test jsonl do bundle)
    return Path(path).parts[-4]


def main():
    rows = []
    for path, run, kind, sig, mach in SOURCES:
        s = json.load(open(path))
        ds = dataset_of(path, run)
        jobs = s["jobs"]
        byjob = {j["job"]: j for j in jobs}
        for j in jobs:
            r = j.get("result") or {}
            g = j.get("genome") or {}
            op = j.get("op") or {}
            m = r.get("metrics") or {}
            layers = r.get("layers") or []
            ae = [L.get("ae") or {} for L in layers if isinstance(L, dict)]
            ae = [a for a in ae if a]
            wk = r.get("wise_kappas") or {}
            res_from = j.get("resumed_from")
            parent_fid = byjob[res_from]["fidelity"] if res_from is not None and res_from in byjob else None
            row = {
                "source_state": os.path.relpath(path, ROOT) if str(path).startswith(str(ROOT)) else str(path),
                "machine": mach, "run": run, "kind": kind, "fitness_sig": sig, "dataset": ds,
                "job": j.get("job"), "cand_id": j.get("cand_id"), "fidelity": j.get("fidelity"),
                "rung": j.get("rung"), "role": j.get("role"), "op_type": op.get("type"),
                "op_from_fidelity": op.get("from_fidelity"), "op_exploratory": op.get("exploratory"),
                "op_target_stratum": op.get("target_stratum") or op.get("stratum"),
                "resumed_from_job": res_from, "resumed_from_fid": parent_fid, "parent_id": j.get("parent_id"),
                "status": j.get("status"), "stratum": j.get("stratum"),
                "params": r.get("params"), "params_est": j.get("params_est"), "flops": r.get("flops"),
                "depth": g.get("depth"), "embed_dim": g.get("embed_dim"), "pooling": g.get("pooling"),
                "ssl": g.get("ssl"), "reg": g.get("reg"), "protect": g.get("protect"),
                "ae_schedule": g.get("ae_schedule"), "ae_target": g.get("ae_target"),
                "projector": g.get("projector"), "spifil_ipc": g.get("spifil_images_per_class"),
                "ae_mode": r.get("ae_mode"),
                "fitness_kappa": r.get("fitness_kappa"), "fitness_f1": r.get("fitness_macro_f1"),
                "quality": r.get("quality"), "quality_label_free": r.get("quality_label_free"),
                "wise_alpha": r.get("wise_alpha"),
                **{f"wise_k_{a}": wk.get(a) for a in ("0.0", "0.25", "0.5", "0.75", "1.0")},
                "ssl_final_loss": r.get("ssl_final_loss"), "ssl_epochs": r.get("ssl_epochs"),
                "ssl_steps": r.get("ssl_steps"), "ae_epochs_per_layer": r.get("ae_epochs_per_layer"),
                "n_layers_logged": len(layers),
                "ae_recon_ratio_mean": (sum(a.get("recon_ratio", float("nan")) for a in ae) / len(ae)) if ae else None,
                "ae_eff_rank_ratio_mean": (sum(a.get("effective_rank_ratio", float("nan")) for a in ae) / len(ae)) if ae else None,
                "ae_eff_rank_ratio_last": ae[-1].get("effective_rank_ratio") if ae else None,
                "gpu_hours": r.get("gpu_hours"), "checkpoint": r.get("checkpoint"),
                **{f"lf_{k}": m.get(k) for k in LF},
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    # teste (ORIG/COV) — só leitura posterior, nunca seleção
    tests = []
    for f in [ROOT / "ga_search/modehb/reports/test_svm_C100.jsonl", BUNDLE / "reports/test_svm_C100.jsonl"]:
        for line in open(f):
            if not line.strip():
                continue
            t = json.loads(line)
            if "test_kappa" not in t:
                continue
            tests.append({"run": t["run"], "dataset": t["dataset"], "cand_id": t["cand_id"],
                          "fidelity": float(t["fidelity"]), "test_orig_kappa": t.get("test_kappa"),
                          "test_orig_f1": t.get("test_f1"), "test_cov_kappa": t.get("cov_kappa"),
                          "test_cov_f1": t.get("cov_f1"), "test_val_kappa_logged": t.get("val_kappa"),
                          "test_src": "jaci" if str(f).startswith(str(ROOT)) else "athena_bundle",
                          "test_time": t.get("time")})
    T = pd.DataFrame(tests).sort_values("test_time").drop_duplicates(["run", "dataset", "cand_id", "fidelity"], keep="last")
    df["fidelity"] = df["fidelity"].astype(float)
    df = df.merge(T.drop(columns=["test_time"]), on=["run", "dataset", "cand_id", "fidelity"], how="left")
    df.to_csv(OUT, index=False)
    T.to_csv(OUT.with_name("A1b_tests_all.csv"), index=False)
    print(len(df), "jobs;", df["test_orig_kappa"].notna().sum(), "com teste;", len(T), "linhas de teste únicas")


if __name__ == "__main__":
    main()
