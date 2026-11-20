"""Congelamento E0–E8 (auditoria CHECK C3/C4/C6/C7): UM passo gera, juntos e com md5,
    * ``checkpoints_E0_E8_<tag>.csv``      lista final (1 linha por dataset x kind x estrato);
    * ``lists_<tag>/<dataset>.txt``         caminhos do staging no athena;
    * ``rsync_files_from_<tag>.txt``        encoder.pt + config_resolved.yaml (relativos a artifacts/);
    * ``fila_gpu3_attention_<tag>.json``    fila da GPU 3 (formato agent5), com ``codigo_md5``;
    * ``MANIFEST_<tag>.md5`` + ``freeze_<tag>.log``.

Modos:
    --from-csv X.csv   parte de uma lista ja congelada (ex. ``checkpoints_E0_E8_congelado_20261117_0159.csv``)
                       e a VALIDA contra o ``search_state.json`` (fonte de verdade): linha cujo job nao
                       esta ``ok`` e SUBSTITUIDA pelo melhor candidato ok do mesmo estrato com
                       ``job_time <= cutoff`` (cutoff = mtime do CSV de origem, ou ``--cutoff``);
                       sem substituto, e EXCLUIDA.  Tudo registrado no log.
    (sem --from-csv)   seleciona do zero (``select_checkpoints``) com cutoff = agora.

    python freeze_E0_E8.py --artifacts <repo>/artifacts --from-csv <csv> --tag <tag> --out-dir <agent6>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as SC  # noqa: E402

AGENT6 = Path(__file__).resolve().parents[1]
REPO_COPY = Path(__file__).resolve().parents[4]
ROOT = "/dados/home/moliveira/exps/attn6"
STG, CODE = ROOT + "/ckpts", ROOT + "/code"
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
ENV = ("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 "
       "MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
ENV_CPU = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2"
GPU3_UUID_PREFIX = "GPU-3c72f1c7"          # PCI AF:00.0 (auditoria CHECK C7)
ORDER = ["remote_sensing", "flowers", "ham10000", "larvae", "larvae_raw", "eggs", "eggs_raw",
         "protozoan", "protozoan_raw", "coconut_trees", "aid"]
IMGS = {"aid": 5502, "coconut_trees": 6038, "eggs": 5903, "eggs_raw": 5903, "flowers": 2334,
        "ham10000": 4878, "larvae": 1358, "larvae_raw": 1358, "protozoan": 9607,
        "protozoan_raw": 9607, "remote_sensing": 841}
SMOKE = ["artifacts/modehb_hybrid/remote_sensing/state/seed42/candidates/c00009/f5/encoder.pt",
         "artifacts/modehb_lemonade/remote_sensing/state/seed42/candidates/c00030/f5/encoder.pt"]
CODE_FILES = ["ga_search/modehb/fixed_attention.py", "tests/modehb/test_fixed_attention.py",
              "research/spifil_b0b4/agent6/tools/eval_fixed_attention.py",
              "research/spifil_b0b4/agent6/tools/select_checkpoints.py",
              "research/spifil_b0b4/agent6/tools/freeze_E0_E8.py",
              "research/spifil_b0b4/agent6/tools/ckpt_compat_check.py",
              "research/spifil_b0b4/agent6/tools/paired_bootstrap.py"]


def stg(ck: str) -> str:
    return STG + "/" + ck.removeprefix("artifacts/")


def build_fila(rows, tag, md5s):
    by: dict = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    order = [d for d in ORDER if d in by] + sorted(set(by) - set(ORDER))
    jobs_b = []
    for ds in order:
        rs = by[ds]
        n, im = len(rs), IMGS.get(ds, 5000)
        gpu_h = n * im * (6 + 4 + 8 + 3.125) * 2.5e-3 / 3600
        svm_h = n * 9 * (20.0 if im > 3000 else 5.0) / 3600
        out = f"{ROOT}/out/B-attn-E0E8-{ds}"
        jobs_b.append({
            "job_id": f"B-attn-E0E8-{ds}", "status": "queued", "fase": "B", "gate_requerido": "A",
            "depende_de": ["A-attn-03"], "tipo": "avaliacao_sem_treino", "never_feeds_fitness": True,
            "descricao": f"E0-E5 (validacao) + E6-E8 (TTA do melhor E0-E5 pela validacao) em {n} checkpoints de {ds}",
            "checkpoints": [{"kind": r["kind"], "estrato": r["stratum"], "cand": r["cand"], "fid": r["fid"],
                             "params": int(r["params"]), "kappa_val_busca": round(float(r["kappa_val_search"]), 4),
                             "md5": r["md5"], "path": stg(r["checkpoint"])} for r in rs],
            "output_dir": out, "lock": out + "/.controller.lock",
            "cmd": (f"mkdir -p {out} && cd {CODE} && flock -n {out}/.controller.lock env {ENV} nice -n 10 {PY} "
                    f"research/spifil_b0b4/agent6/tools/eval_fixed_attention.py --checkpoint-list "
                    f"research/spifil_b0b4/agent6/lists_{tag}/{ds}.txt --experiments E0,E1,E2,E3,E4,E5,E6,E7,E8 "
                    f"--split validation --device cuda --bs 32 --num-workers 8 --out {out}/val.jsonl "
                    f"--cache {out}/data_cache"),
            "verificacao_gpu": f"nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv | grep {GPU3_UUID_PREFIX}",
            "retomada": "relancar o mesmo cmd: linhas (checkpoint, exp) ja gravadas sao reaproveitadas; "
                        "checkpoint com linha selected_by_validation e pulado",
            "saidas": "val.jsonl (1 linha por exp + 1 selected_by_validation por checkpoint) + preds/*.npz "
                      "(sample_id, y, pred; md5 na linha)",
            "mem_est_gb": 6, "gpu_h_est": round(gpu_h, 2), "svm_cpu_h_est": round(svm_h, 2),
            "imagens_por_ckpt": im})
    test = {
        "job_id": "B-attn-test-report", "status": "queued", "fase": "B", "gate_requerido": "A",
        "depende_de": [j["job_id"] for j in jobs_b], "tipo": "relato_teste",
        "descricao": ("SO depois de TODOS os B-attn-E0E8 completed e da selecao congelada: teste com E0 + vencedor "
                      "sem TTA + vencedor congelado. used_for_selection=false, reporting_only=true. Nada e "
                      "reajustado no teste. Relato com as ressalvas do A1c (plano §8): HAM 26,2% do teste com "
                      "lesao em train/val; sensibilidade sem recapturas (eggs/larvae); duplicatas flowers/aid; "
                      "pre/raw so contra FM do mesmo dominio; regua ORIG_sub4000."),
        "cmd": (f"mkdir -p {ROOT}/out/B-attn-test-report && cd {CODE} && for ds in {' '.join(order)}; do env {ENV} "
                f"nice -n 10 {PY} research/spifil_b0b4/agent6/tools/eval_fixed_attention.py --checkpoint-list "
                f"research/spifil_b0b4/agent6/lists_{tag}/$ds.txt --split test --frozen-from "
                f"{ROOT}/out/B-attn-E0E8-$ds/val.jsonl --device cuda --bs 32 --num-workers 8 --out "
                f"{ROOT}/out/B-attn-test-report/test.jsonl --cache {ROOT}/out/B-attn-test-report/data_cache; done"),
        "gpu": True, "mem_est_gb": 6, "tempo_est_h": 2.0}
    jobs_a = [
        {"job_id": "A-attn-00b", "status": "queued", "fase": "A", "gate_requerido": None,
         "tipo": "staging_somente_leitura",
         "descricao": (f"delta do congelamento {tag}: rsync (sem --delete) dos arquivos de rsync_files_from_{tag}.txt "
                       f"ainda ausentes em {STG}; md5 contra a coluna md5; recopiar o codigo congelado (patch v2) "
                       f"para {CODE} e conferir codigo_md5"),
         "cmd": (f"rsync -a --ignore-existing --files-from=rsync_files_from_{tag}.txt "
                 f"jaci:/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/ {STG}/"),
         "gpu": False, "tempo_est_h": 0.1},
        {"job_id": "A-attn-01b", "status": "queued", "fase": "A", "gate_requerido": None,
         "depende_de": ["A-attn-00b"], "tipo": "testes",
         "descricao": "suite no athena (torch 2.2.2), CPU, com o patch v2",
         "cmd": f"cd {CODE} && env {ENV_CPU} nice -n 15 {PY} -m pytest -q tests/ -p no:cacheprovider",
         "gpu": False, "criterio": "0 failed", "tempo_est_h": 0.05},
        {"job_id": "A-attn-03", "status": "queued", "fase": "A", "gate_requerido": None,
         "depende_de": ["A-attn-01b"], "tipo": "compat_torch_2.7_para_2.2 (C7)",
         "descricao": ("carrega OS N checkpoints da lista no athena (torch 2.2.2, weights_only=False), forward E0 numa "
                       "entrada sintetica fixa e compara com a referencia da Jaci (torch 2.7.1): cos >= 0.9999 e "
                       "erro relativo L2 <= 1e-3, TF32 desligado; confere que o processo esta na GPU-3c72f1c7 (PCI AF:00.0)"),
         "cmd": (f"cd {CODE} && env {ENV} nice -n 10 {PY} research/spifil_b0b4/agent6/tools/ckpt_compat_check.py "
                 f"--csv research/spifil_b0b4/agent6/checkpoints_E0_E8_{tag}.csv --root {STG} --device cuda "
                 f"--compare research/spifil_b0b4/agent6/compat_ref_{tag}.npz --expect-gpu-uuid {GPU3_UUID_PREFIX} "
                 f"--out {ROOT}/out/A-attn-03/compat.jsonl"),
         "gpu": True, "mem_est_gb": 3, "tempo_est_h": 0.1, "criterio": "0 divergencias; uuid confere"},
    ]
    return {
        "host": "athena", "gpu_fisica": 3, "cuda_visible_devices": "3", "cuda_device_order": "PCI_BUS_ID",
        "gpu_uuid_esperado": GPU3_UUID_PREFIX + "… (PCI AF:00.0)", "dono": "moliveira",
        "pesquisa": "SPiFiL Progressivo B0-B4 -- Agente 6: atencao fixa sem pesos (E0-E8)",
        "congelamento": tag, "gerado_em": dt.datetime.now().isoformat(timespec="seconds"),
        "gerado_por": "agente6 freeze_E0_E8.py (jaci)",
        "regras": {
            "gpu2": "NAO MEXER", "jaci": "NAO usar GPU da Jaci (so leitura dos checkpoints para o staging)",
            "gate": "fase seguinte so roda com gate aprovado (gates/gate_<fase>.approved na Jaci, assinado pelo CHECK)",
            "estados": ["queued", "running", "checkpointed", "failed", "completed"],
            "idempotencia": "job_id unico; relancar completed e no-op; running exige flock em <output_dir>/.controller.lock",
            "hpc": ("CUDA_DEVICE_ORDER=PCI_BUS_ID em TODO comando; OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 "
                    "MKL_NUM_THREADS=8; OOM -> diagnostico, --bs 32->16->8 registrado, sem repeticao infinita"),
            "protocolo": ("SVM core.svm.fit_svm linear C=100 sem scaler; treino = subconjunto estratificado max_train "
                          "da busca (regua ORIG_sub4000); selecao SO na validacao; teste so no B-attn-test-report"),
            "unidades": "FLOPs = 2*MAC em codigo e documentos; o campo flops da busca e MAC (coluna flops_mac)",
            "paralelismo": "ate 4 jobs B simultaneos (~6 GB cada) na GPU 3",
            "codigo": "copia congelada em " + CODE + " (patch v2); a arvore viva nao e usada"},
        "codigo_md5": md5s["code"], "artefatos_md5": md5s["artifacts"],
        "fases": {
            "A": {"descricao": "testes de implementacao (atencao fixa); A-attn-00/01/02 ja completed (agent5 "
                               "attn_fase_A); 00b/01b/03 = delta do patch v2 e compat C7",
                  "gate_requerido": None, "status": "aguardando_jobs", "jobs": jobs_a},
            "B": {"descricao": "E0-E8 nos mesmos checkpoints (sem treino) + relato de teste",
                  "gate_requerido": "A", "status": "bloqueada", "jobs": jobs_b + [test]},
            "B2": {"descricao": "2a ablacao: operador dentro do AE local e do LeJEPA+SIGReg (so aprovados)",
                   "gate_requerido": "B", "status": "bloqueada", "jobs": []},
            "D": {"descricao": "genes de atencao no MO-DEHB hierarquico (Agente 4), so aprovados",
                  "gate_requerido": "C", "status": "bloqueada", "jobs": []}},
        "estimativa": {"gpu_h_B": round(sum(j["gpu_h_est"] for j in jobs_b), 1),
                       "svm_cpu_h_B": round(sum(j["svm_cpu_h_est"] for j in jobs_b), 1),
                       "wall_h_B_4_paralelos": round(sum(j["gpu_h_est"] + j["svm_cpu_h_est"] for j in jobs_b) / 4 + 0.5, 1)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", default="/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts")
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--cutoff", type=float, default=None, help="epoch; padrao = mtime do --from-csv (ou agora)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(AGENT6))
    args = ap.parse_args()
    art = Path(args.artifacts).resolve()
    root = art.parent
    out = Path(args.out_dir)
    log = []
    say = lambda m: (log.append(m), print(m))
    cutoff = args.cutoff or (os.path.getmtime(args.from_csv) if args.from_csv else dt.datetime.now().timestamp())
    say(f"# freeze {args.tag}  cutoff={cutoff} ({dt.datetime.fromtimestamp(cutoff).isoformat()})  artifacts={art}")
    cands = SC.candidates(art, cutoff=cutoff)
    best = {(r["dataset"], r["kind"], r["stratum"]): r for r in SC.select(cands)}
    if args.from_csv:
        say(f"origem: {args.from_csv} md5 {SC.md5(args.from_csv)}")
        rows = []
        for r in csv.DictReader(open(args.from_csv)):
            st, err = SC.job_status(art, r["checkpoint"])
            key = (r["dataset"], r["kind"], r["stratum"])
            if st == "ok":
                c = next((c for c in cands if c["checkpoint"] == r["checkpoint"]), None)
                row = dict(c or r)
                row.setdefault("flops_mac", r.get("flops"))
                row["md5"] = r["md5"]
                if SC.md5(root / r["checkpoint"]) != r["md5"]:
                    say(f"ERRO md5 divergente {r['checkpoint']}")
                    sys.exit(2)
                rows.append(row)
                continue
            sub = best.get(key)
            if sub and sub["checkpoint"] != r["checkpoint"]:
                say(f"C4 SUBSTITUIDO {key}: {r['cand']} {r['fid']} (search_state status={st}; {err}) -> "
                    f"{sub['cand']} {sub['fid']} (kappa_val {sub['kappa_val_search']:.4f}, job {sub['job']}, "
                    f"params {sub['params']}; melhor ok do estrato com job_time <= cutoff)")
                rows.append(dict(sub))
            else:
                say(f"C4 EXCLUIDO {key}: {r['cand']} {r['fid']} (search_state status={st}; {err}); sem substituto ok")
        # o que a selecao no cutoff daria de diferente (so registro)
        for key, b in sorted(best.items()):
            cur = next((x for x in rows if (x["dataset"], x["kind"], x["stratum"]) == key), None)
            if cur is None:
                say(f"nota: estrato {key} existe no cutoff mas nao na lista de origem (mantida a origem)")
            elif cur["checkpoint"] != b["checkpoint"]:
                say(f"nota: no cutoff o melhor de {key} seria {b['cand']} {b['fid']} ({b['kappa_val_search']:.4f}); "
                    f"mantido {cur['cand']} {cur['fid']} da lista congelada")
    else:
        rows = list(best.values())
    rows.sort(key=lambda r: (r["dataset"], r["kind"], r["stratum"]))
    csv_p = out / f"checkpoints_E0_E8_{args.tag}.csv"
    SC.write_csv(rows, csv_p, root)
    ldir = out / f"lists_{args.tag}"
    ldir.mkdir(parents=True, exist_ok=True)
    by: dict = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    for ds, rs in by.items():
        (ldir / f"{ds}.txt").write_text(f"# {ds} ({args.tag}): melhor por validacao de cada (kind, estrato)\n"
                                        + "".join(stg(r["checkpoint"]) + "\n" for r in rs))
    files = []
    for ck in [r["checkpoint"] for r in rows] + SMOKE:
        p = Path(ck)
        files += [str(p.relative_to("artifacts")), str((Path(*p.parts[:5]) / "config_resolved.yaml").relative_to("artifacts"))]
    ff = out / f"rsync_files_from_{args.tag}.txt"
    ff.write_text("".join(f + "\n" for f in sorted(set(files))))
    code = {f: SC.md5(REPO_COPY / f) for f in CODE_FILES if (REPO_COPY / f).exists()}
    arts = {p.name if p.parent == out else str(p.relative_to(out)): SC.md5(p)
            for p in [csv_p, ff, *sorted(ldir.glob("*.txt")),
                      *[q for q in [out / f"compat_ref_{args.tag}.npz"] if q.exists()]]}
    fila = build_fila(rows, args.tag, {"code": code, "artifacts": arts})
    fila_p = out / f"fila_gpu3_attention_{args.tag}.json"
    fila_p.write_text(json.dumps(fila, indent=1, ensure_ascii=False))
    arts[fila_p.name] = SC.md5(fila_p)
    man = out / f"MANIFEST_{args.tag}.md5"
    man.write_text("".join(f"{v}  {k}\n" for k, v in {**arts, **{"code/" + k: v for k, v in code.items()}}.items()))
    say(f"{len(rows)} checkpoints; {len(by)} datasets; {len(set(files))} arquivos p/ rsync")
    for k, v in arts.items():
        say(f"md5 {v}  {k}")
    for k, v in code.items():
        say(f"md5 {v}  code/{k}")
    (out / f"freeze_{args.tag}.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
