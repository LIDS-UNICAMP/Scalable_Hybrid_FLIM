#!/usr/bin/env python
"""Varre artifacts/_staging_gpu3 (buscas migradas da GPU 3 do athena), valida o resume das que têm marcador
<kind>_<base>.ok (validate_resume.py, CPU, sem treino) e reescreve:
  research/spifil_b0b4/agent5/resume_validation_report.md
  research/spifil_b0b4/agent5/jaci_resume_queue.json
NÃO publica, NÃO lança nada: só gera os comandos (publicação + waiter) para o coordenador.

Uso: python staging_watch.py            (uma passada)
     python staging_watch.py --loop 300 --max-hours 3 --exit-on-new
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM")
STAGING = REPO / "artifacts/_staging_gpu3"
AG5 = REPO / "research/spifil_b0b4/agent5"
VAL = AG5 / "validation"
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
TOOL = AG5 / "tools/validate_resume.py"
KINDS = ["modehb_lemonade_seeded", "modehb_lemonade", "modehb_seeded", "modehb_hybrid", "pnas_cs"]
SHORT = {"modehb_hybrid": "hyb", "modehb_lemonade": "lem", "modehb_seeded": "seedhyb",
         "modehb_lemonade_seeded": "seedlem", "pnas_cs": "pnas"}
ALGO = {"modehb_hybrid": "MO-DEHB hybrid (SPiFiL)", "modehb_lemonade": "LEMONADE",
        "modehb_seeded": "MO-DEHB hybrid semeada", "modehb_lemonade_seeded": "LEMONADE semeada", "pnas_cs": "PNAS-CS"}
# esperadas (pedido do coordenador): prioridade menor = antes
EXPECTED = [("modehb_hybrid", "eggs", 42, 1), ("modehb_hybrid", "eggs_raw", 42, 2),
            ("modehb_lemonade", "flowers", 42, 3), ("modehb_lemonade", "coconut_trees", 42, 4),
            ("modehb_hybrid", "coconut_trees", 42, 5),
            ("modehb_hybrid", "flowers", 43, 20), ("modehb_hybrid", "larvae", 43, 21)]
PRIO_KIND = {"modehb_hybrid": 1, "modehb_lemonade": 4, "modehb_seeded": 10, "modehb_lemonade_seeded": 12, "pnas_cs": 30}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def parse_marker(stem):
    """-> (kind real, base, nome do kind no staging). Aceita '<kind>_seedNN_<base>' (staging
    artifacts/_staging_gpu3/<kind>_seedNN/<base>, destino artifacts/<kind>/<base>/state/seedNN)."""
    for k in KINDS:
        m = re.match(rf"^{k}_seed(\d+)_(.+)$", stem)
        if m:
            return k, m.group(2), f"{k}_seed{m.group(1)}"
        if stem.startswith(k + "_"):
            return k, stem[len(k) + 1:], k
    return None, None, None


def running_controllers(out_rel):
    """PIDs de run_search (python) que usam o mesmo output_dir."""
    pat = rf"^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir={re.escape(out_rel)}( |$)"
    return [p for p in sh(f"pgrep -f {shlex.quote(pat)}").split() if p]


def waiters_touching(path_rel):
    """Sessões tmux cujo comando (bash -c) cita o caminho (ex.: seedq_* que vão criar a mesma busca)."""
    hits = []
    for line in sh("tmux list-panes -a -F '#{session_name} #{pane_pid}'").splitlines():
        try:
            s, pid = line.split()
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            continue
        if path_rel in cmd:
            hits.append(s)
    return hits


def waiter_cmd(kind, base, seed, run_cmd, gate_mib, name, mode="resume", parent_state=None, seed_ckpt=None):
    out = f"artifacts/{kind}/{base}/state/seed{seed}"
    log = f"artifacts/{kind}/{base}/logs/search_s{seed}.log"
    sets = " ".join(f"--set {s}" for s in run_cmd["sets"])
    lock = f"artifacts/migration_markers/{kind}_{base}_s{seed}.ctrl.lock"
    script = f"""cd /dados/home/moliveira/Scalable_Hybrid_FLIM
NAME={name}; OUT={out}; LOG={log}; GATE={gate_mib}
{{pre}}
ok=0
while true; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f" -ge $GATE ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ] && flock -n artifacts/migration_markers/.last_start.flock -c 'age=$(( $(date +%s) - $(stat -c %Y artifacts/migration_markers/.last_start) )); [ $age -ge 900 ] && touch artifacts/migration_markers/.last_start'; then break; fi
  sleep $((90 + RANDOM % 60))
done
echo "START $NAME $(date)" >> artifacts/migration_from_athena.log
ulimit -n 65536
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$(dirname $LOG)"
for i in 1 2 3 4 5; do
  if pgrep -f "^[^ ]*python[^ ]* -m ga_search.modehb.run_search .*output_dir=$OUT( |\\$)" >/dev/null; then
    echo "DUP $NAME: outro run_search já usa $OUT (tentativa $i adiada) $(date)" >> artifacts/migration_from_athena.log; sleep 600; continue
  fi
  flock -n {lock} {PY} -m ga_search.modehb.run_search --config {run_cmd['config']} --dataset {base} --seed {seed} {sets} >> $LOG 2>&1 && break
  echo "RETRY $NAME tentativa $i rc=$? $(date)" >> artifacts/migration_from_athena.log
  sleep 300
done
echo "END $NAME $(date)" >> artifacts/migration_from_athena.log"""
    if mode == "resume":
        pre = ('test -f "$OUT/search_state.json" || { echo "ABORT $NAME: $OUT/search_state.json ausente (publicar antes) '
               '$(date)" >> artifacts/migration_from_athena.log; exit 1; }')
    else:
        conds = []
        if parent_state:
            conds.append(f"{PY} -c \"import json,sys; sys.exit(0 if json.load(open('{parent_state}')).get('done') else 1)\" 2>/dev/null")
        if seed_ckpt:
            conds.append(f"[ -f {seed_ckpt} ]")
        pre = (f"# busca NOVA (sem estado no athena): espera {'o done da busca-mãe na Jaci + ' if parent_state else ''}"
               f"{'o ckpt da semente' if seed_ckpt else 'só o gate'}\n"
               + (f"until {' && '.join(conds)}; do sleep 300; done" if conds else ":"))
    script = script.replace("{pre}", pre)
    return f"tmux new-session -d -s {name} bash -c \"$(cat <<'G3EOF'\n{script}\nG3EOF\n)\""


def publish_cmd(kind, base):
    src = f"artifacts/_staging_gpu3/{kind}/{base}"
    dst = f"artifacts/{kind}/{base}"
    return (f"cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test -f artifacts/_staging_gpu3/{kind}_{base}.ok "
            f"&& test ! -e {dst} && mkdir -p artifacts/{kind} && mv -T --no-clobber {src} {dst} "
            f"&& ls {dst}/state/seed*/search_state.json >/dev/null && touch artifacts/migration_markers/{kind}_{base}.gpu3.ok "
            f"&& echo \"PUBLISH {kind}/{base} $(date)\" >> artifacts/migration_from_athena.log "
            f"|| echo 'NAO PUBLICADO: destino existe ou staging incompleto'")


def publish_seed_only_cmd(kind, base, seed, skind=None):
    src = f"artifacts/_staging_gpu3/{skind or kind}/{base}"
    dst = f"artifacts/{kind}/{base}"
    return (f"cd /dados/home/moliveira/Scalable_Hybrid_FLIM && test ! -e {dst}/state/seed{seed} "
            f"&& mv -T --no-clobber {src}/state/seed{seed} {dst}/state/seed{seed} "
            f"&& mkdir -p {dst}/logs && {{ test ! -e {src}/logs/search_s{seed}.log || "
            f"mv --no-clobber {src}/logs/search_s{seed}.log {dst}/logs/search_s{seed}.log; }} "
            f"&& echo \"PUBLISH {kind}/{base}/seed{seed} $(date)\" >> artifacts/migration_from_athena.log "
            f"|| echo 'NAO PUBLICADO: destino seed{seed} existe'")


def du_gb(p):
    try:
        return round(int(sh(f"du -sb {json.dumps(str(p))} 2>/dev/null").split()[0]) / 1e9, 2)
    except Exception:
        return None


def one_pass(force=False):
    VAL.mkdir(parents=True, exist_ok=True)
    markers = sorted(STAGING.glob("*.ok")) if STAGING.exists() else []
    seen, new = set(), []
    entries = []
    for m in markers:
        kind, base, skind = parse_marker(m.stem)
        if kind is None:
            entries.append({"id": m.stem, "migration_status": "marcador_desconhecido", "estado": "failed",
                            "errors": [f"marcador {m.name} não casa com nenhum kind {KINDS}"]})
            continue
        src = STAGING / skind / base
        seeds = sorted(int(p.name[4:]) for p in (src / "state").glob("seed*") if p.name[4:].isdigit()) \
            if (src / "state").exists() else []
        if not seeds and (REPO / f"artifacts/{kind}/{base}/state").exists():
            # já publicada pelo coordenador: reporta estado vivo (cache da validação, se houver)
            seed_dirs = sorted(p for p in (REPO / f"artifacts/{kind}/{base}/state").glob("seed*")
                               if (VAL / f"{kind}_{base}_s{p.name[4:]}.json").exists())
            for sdir in seed_dirs:
                seed = int(sdir.name[4:])
                seen.add((kind, base, seed))
                r = json.loads((VAL / f"{kind}_{base}_s{seed}.json").read_text())
                e = make_entry(r, kind, base, seed, m, skind)
                out_rel = f"artifacts/{kind}/{base}/state/seed{seed}"
                e["migration_status"] = "publicada"
                e["estado"] = "running" if running_controllers(out_rel) else "checkpointed"
                try:
                    stj = json.loads((sdir / "search_state.json").read_text())
                    e["n_jobs_atual"] = len(stj["jobs"])
                    e["estado"] = "completed" if stj.get("done") else e["estado"]
                except Exception:
                    pass
                e.pop("cmd_publicar", None)
                e["conflitos"] = [c for c in e.get("conflitos", []) if "JÁ EXISTE" not in c and "run_search ativo" not in c]
                entries.append(e)
            if seed_dirs:
                seen.add((kind, base))
                continue
        if not seeds:
            entries.append({"id": f"{kind}/{base}", "kind": kind, "dataset": base, "migration_status": "marcador_sem_state",
                            "estado": "failed", "errors": [f"{src}/state/seed*/search_state.json ausente"]})
            seen.add((kind, base))
            continue
        for seed in seeds:
            seen.add((kind, base, seed))
            sp = src / "state" / f"seed{seed}" / "search_state.json"
            if not sp.exists():
                dst = REPO / f"artifacts/{kind}/{base}"
                entries.append({"id": f"{kind}/{base}/seed{seed}", "origem": f"athena GPU 3 -> {src.relative_to(REPO)}",
                                "marcador": str(m.relative_to(REPO)), "marcador_texto": m.read_text()[:300],
                                "kind": kind, "dataset": base, "algoritmo": ALGO.get(kind), "seed": seed,
                                "prioridade": next((p for k, b, s_, p in EXPECTED if (k, b, s_) == (kind, base, seed)), 50),
                                "n_jobs": 0, "migration_status": "sem_state (0 jobs: resume = começar do zero)",
                                "resume_status": "nao_aplicavel", "estado": "queued",
                                "disk_staging_gb": du_gb(src),
                                "conflitos": ([f"destino artifacts/{kind}/{base} JÁ EXISTE na Jaci"] if dst.exists() else [])
                                + [f"waiters/sessões que citam o caminho: {w}" for w in
                                   [sorted(set(waiters_touching(f'artifacts/{kind}/{base}/state/seed{seed}')))] if w],
                                "obs": "busca nova de fato (réplica seed43, F4 = reserva): decidir com o coordenador se entra; "
                                       "não há o que retomar"})
                continue
            key = f"{m.stat().st_mtime:.0f}:{sp.stat().st_mtime if sp.exists() else 0:.0f}"
            cache = VAL / f"{kind}_{base}_s{seed}.json"
            r = None
            if cache.exists() and not force:
                r = json.loads(cache.read_text())
                if r.get("_key") != key:
                    r = None
            if r is None:
                new.append(f"{kind}/{base}/s{seed}")
                tmpj = VAL / f".{kind}_{base}_s{seed}.tmp.json"
                subprocess.run(["nice", "-n", "15", PY, str(TOOL), "--src", str(src.relative_to(REPO)), "--kind", kind,
                                "--base", base, "--seed", str(seed), "--json-out", str(tmpj), "--marker", str(m)], cwd=REPO,
                               capture_output=True, text=True)
                try:
                    r = json.loads(tmpj.read_text())[0]
                except Exception as exc:
                    r = {"kind": kind, "base": base, "seed": seed, "verdict": "BLOQUEADA",
                         "errors": [f"validador não gerou saída: {exc}"]}
                r["_key"] = key
                r["disk_staging_gb"] = du_gb(src)
                cache.write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
            entries.append(make_entry(r, kind, base, seed, m, skind))
    # esperadas que ainda não chegaram
    for kind, base, seed, prio in EXPECTED:
        if (kind, base, seed) in seen or (kind, base) in seen:
            continue
        src = STAGING / kind / base
        st = "em_transito (dir sem marcador .ok)" if src.exists() else "aguardando_staging"
        dst = REPO / f"artifacts/{kind}/{base}"
        entries.append({"id": f"{kind}/{base}/seed{seed}", "origem": f"athena GPU 3 -> {src.relative_to(REPO)}",
                        "kind": kind, "dataset": base, "algoritmo": ALGO[kind], "seed": seed, "prioridade": prio,
                        "migration_status": st, "resume_status": "pendente", "estado": "queued",
                        "conflito_destino": dst.exists(),
                        "obs": ("destino artifacts/{0}/{1} JÁ EXISTE na Jaci (seed42 local): publicação por "
                                "subdiretório state/seed{2} exige OK do coordenador").format(kind, base, seed)
                        if dst.exists() else ""})
    # staging dirs sem marcador e não esperados
    if STAGING.exists():
        for kd in STAGING.iterdir():
            if kd.is_dir() and (kd.name in KINDS or re.match(r"^(%s)_seed\d+$" % "|".join(KINDS), kd.name)):
                for bd in kd.iterdir():
                    if bd.is_dir() and not (STAGING / f"{kd.name}_{bd.name}.ok").exists() and \
                            not any(e.get("kind") == kd.name and e.get("dataset") == bd.name for e in entries):
                        entries.append({"id": f"{kd.name}/{bd.name}", "kind": kd.name, "dataset": bd.name,
                                        "algoritmo": ALGO.get(kd.name), "migration_status": "em_transito (sem .ok)",
                                        "resume_status": "pendente", "estado": "queued",
                                        "prioridade": PRIO_KIND.get(kd.name, 50)})
    entries += new_entries()
    for e in entries:      # larvae seed43: busca nova (0 jobs) -> waiter opcional, com OK
        if e.get("id") == "modehb_hybrid/larvae/seed43" and "cmd_waiter" not in e:
            cr = STAGING / "modehb_hybrid_seed43/larvae/state/seed43/config_resolved.yaml"
            try:
                import yaml
                rc = yaml.safe_load(cr.read_text())
                bud = (rc.get("search") or {}).get("stratified", {}).get("memory_budget_gb", 12)
                cap = rc.get("gpu_process_cap_gb", 14)
            except Exception:
                bud, cap = 12, 14
            run_cmd = {"config": "ga_search/config/modehb/search_v3.yaml", "dataset": "larvae", "seed": 43,
                       "sets": ["output_dir=artifacts/modehb_hybrid/larvae/state/seed43",
                                f"search.stratified.memory_budget_gb={bud}", f"gpu_process_cap_gb={cap}"]}
            e["config"] = run_cmd
            e["recursos"] = {"memory_budget_gb": bud, "gpu_process_cap_gb": cap, "gate_free_mib": 22528}
            e["cmd_waiter_OPCIONAL_com_OK"] = waiter_cmd("modehb_hybrid", "larvae", 43, run_cmd, 22528,
                                                         "g3_hyb_larvae_s43", mode="new")
    entries.sort(key=lambda e: (e.get("prioridade", 99), e.get("id", "")))
    write_outputs(entries)
    return new, entries


NEW_SEEDED = [("modehb_seeded", "eggs", "modehb_hybrid", 10), ("modehb_seeded", "eggs_raw", "modehb_hybrid", 11),
              ("modehb_lemonade_seeded", "flowers", "modehb_lemonade", 12),
              ("modehb_lemonade_seeded", "coconut_trees", "modehb_lemonade", 13),
              ("modehb_seeded", "coconut_trees", "modehb_hybrid", 14)]
SEEDED_CFG = {"modehb_seeded": "ga_search/config/modehb/search_v3_seeded.yaml",
              "modehb_lemonade_seeded": "ga_search/config/modehb/search_v3_lemonade_seeded.yaml"}


def new_entries():
    out = []
    for kind, base, parent, prio in NEW_SEEDED:
        dst = f"artifacts/{kind}/{base}"
        out_rel = f"{dst}/state/seed42"
        seed_dir = f"artifacts/external_ssl_pequeno/{base}_split1_pct100"
        ck = f"{seed_dir}/lejepa/checkpoints/best_kappa.ckpt"
        arch = f"{seed_dir}/L0/architecture.json"
        parent_state = f"artifacts/{parent}/{base}/state/seed42/search_state.json"
        name = f"g3_{SHORT[kind]}_{base}_s42"
        probs = []
        if (REPO / dst).exists():
            probs.append(f"destino {dst} JÁ EXISTE na Jaci")
        if running_controllers(out_rel):
            probs.append("run_search ativo no mesmo output_dir")
        w = sorted(set(waiters_touching(out_rel)) - {name})
        if w:
            probs.append(f"waiters/sessões da Jaci que citam {out_rel}: {w}")
        errs = [f"semente ausente: {p}" for p in (ck, arch) if not (REPO / p).exists()]
        run_cmd = {"config": SEEDED_CFG[kind], "dataset": base, "seed": 42,
                   "sets": [f"output_dir={out_rel}", "search.stratified.memory_budget_gb=24", "gpu_process_cap_gb=28"]}
        e = {"id": f"{kind}/{base}/seed42", "origem": "athena GPU 3 (waiter seedq_* que nunca iniciou; sem estado)",
             "kind": kind, "dataset": base, "algoritmo": ALGO[kind], "seed": 42, "prioridade": prio, "n_jobs": 0,
             "migration_status": "nova (nada a migrar)", "resume_status": "nao_aplicavel", "estado": "queued",
             "config": run_cmd, "semente": {"checkpoint": ck, "realpath": os.path.realpath(REPO / ck),
                                             "arch": arch},
             "depende_de": f"done de {parent_state} (busca-mãe migrada, regra F3: semeada só depois da atual)",
             "recursos": {"memory_budget_gb": 24, "gpu_process_cap_gb": 28, "gate_free_mib": 28672},
             "logs": {"jaci": f"{dst}/logs/search_s42.log", "waiter": "artifacts/migration_from_athena.log"},
             "errors": errs, "conflitos": probs, "tmux_waiter": name,
             "obs": "budget 24/28 = padrão F3 (athena: 'todas 24/28'); o comando original está em athena:"
                    f"/exps/moliveira/queues/gpu3_migracao/waiters/seedq_{kind}_{base}.txt — conferir antes"}
        if not errs and not probs:
            e["cmd_waiter"] = waiter_cmd(kind, base, 42, run_cmd, 28672, name, mode="new",
                                         parent_state=parent_state, seed_ckpt=ck)
        out.append(e)
    return out


def make_entry(r, kind, base, seed, marker, skind=None):
    skind = skind or kind
    C = r.get("checks", {})
    stt = C.get("state", {})
    ck = C.get("checkpoints", {})
    dst = f"artifacts/{kind}/{base}"
    out_rel = f"{dst}/state/seed{seed}"
    dst_exists = (REPO / dst).exists()
    seed_dst_exists = (REPO / out_rel).exists()
    running = running_controllers(out_rel)
    published = seed_dst_exists and not (STAGING / skind / base / "state" / f"seed{seed}").exists()
    prio = next((p for k, b, s, p in EXPECTED if (k, b, s) == (kind, base, seed)), PRIO_KIND.get(kind, 50))
    name = f"g3_{SHORT.get(kind, kind)}_{base}_s{seed}"
    touch = sorted(set(waiters_touching(out_rel)) - {name})
    conflict = []
    if dst_exists:
        conflict.append(f"destino {dst} JÁ EXISTE na Jaci" + (f" (inclusive {out_rel})" if seed_dst_exists else
                                                               f" (sem state/seed{seed})"))
    if running:
        conflict.append(f"run_search ativo no mesmo output_dir: PIDs {running}")
    if touch:
        conflict.append(f"waiters/sessões da Jaci que citam {out_rel}: {touch} (2 controladores no mesmo state se ambos rodarem)")
    verdict = r.get("verdict")
    if published:
        mig = "publicada"
    elif verdict == "VALIDADA" and not conflict:
        mig = "validada_pronta_para_publicar"
    elif verdict == "VALIDADA":
        mig = "validada_CONFLITO"
    else:
        mig = "bloqueada"
    rp = C.get("replay", {})
    e = {"id": f"{kind}/{base}/seed{seed}", "origem": f"athena GPU 3 -> artifacts/_staging_gpu3/{skind}/{base}",
         "marcador": str(marker.relative_to(REPO)), "kind": kind, "dataset": base, "algoritmo": ALGO.get(kind),
         "seed": seed, "prioridade": prio,
         "ultimo_job": stt.get("last_job"), "n_jobs": stt.get("n_jobs"), "status_jobs": stt.get("status"),
         "state_sha256": stt.get("sha256"), "state_updated": stt.get("updated"),
         "jobs_failed": stt.get("failed_jobs"),
         "brackets_concluidos": stt.get("brackets_done"), "done": stt.get("done"), "gpu_hours": stt.get("gpu_hours"),
         "fidelidade_max": ck.get("max_fidelity"), "n_ckpt_fid_max": ck.get("n_max_fidelity"),
         "checkpoint": f"{out_rel}/candidates/<cand>/f<fid>/encoder.pt",
         "checkpoints_carregados_cpu": ck.get("loaded"), "checkpoints_ausentes": ck.get("missing"),
         "config": r.get("run_cmd"), "controller_id_jaci": (C.get("controller") or {}).get("controller_id_jaci"),
         "fit_sig": (C.get("controller") or {}).get("fit_sig_jaci"),
         "migration_status": mig, "resume_status": ("replay_ok" if rp.get("ok") else ("replay_FALHOU" if rp else "nao_testado")),
         "proximo_job": {k: rp.get(k) for k in ("next_job", "next_fidelity", "next_role", "next_cand")} if rp else None,
         "recursos": r.get("resources"), "disk_staging_gb": r.get("disk_staging_gb"),
         "logs": {"staging": f"artifacts/_staging_gpu3/{skind}/{base}/logs/", "jaci": f"{dst}/logs/search_s{seed}.log",
                  "waiter": "artifacts/migration_from_athena.log"},
         "estado": "running" if running and published else ("completed" if stt.get("done") else "checkpointed"),
         "errors": r.get("errors"), "warnings": r.get("warnings"), "conflitos": conflict,
         "validado_em": r.get("validated_at"), "tmux_waiter": name}
    if r.get("run_cmd") and verdict == "VALIDADA":
        gate = (r.get("resources") or {}).get("gate_free_mib", 22528)
        e["cmd_waiter"] = waiter_cmd(kind, base, seed, r["run_cmd"], gate, name)
        if not dst_exists and skind == kind:
            e["cmd_publicar"] = publish_cmd(kind, base)
        elif not seed_dst_exists:
            e["cmd_publicar_ALTERNATIVO_so_seed"] = publish_seed_only_cmd(kind, base, seed, skind)
    return e


def fmt_list(x):
    return "; ".join(str(i) for i in x) if x else "—"


def write_outputs(entries):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    free = sh("df -h /dados | tail -1").split()
    gpu = sh("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits").strip()
    q = {"gerado_em": now, "host": "jaci", "staging": str(STAGING.relative_to(REPO)),
         "regras": ["não publicar sem o coordenador", "publicar só se o destino não existir",
                    "1 controlador por search_state (pgrep + flock no waiter)",
                    "gate de memória + flock .last_start com 15 min entre começos", "5 tentativas",
                    "OPENBLAS=2 OMP=8 MKL=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
         "jobs": entries}
    tmp = AG5 / ".jaci_resume_queue.json.tmp"
    tmp.write_text(json.dumps(q, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, AG5 / "jaci_resume_queue.json")

    L = [f"# Validação de resume — buscas migradas da GPU 3 do athena → Jaci",
         "",
         f"_Atualizado em {now} (Jaci). Gerado por `research/spifil_b0b4/agent5/tools/staging_watch.py` "
         f"(validador: `tools/validate_resume.py`, CPU, sem treino). Disco /dados livre: {free[3] if len(free) > 3 else '?'}; "
         f"GPU usada/livre (MiB): {gpu}._",
         "",
         "**Nada foi publicado nem lançado.** Os comandos abaixo são para o coordenador executar, na ordem de prioridade, "
         "um por vez (o waiter já respeita o gate de memória e o espaçamento de 15 min).",
         "",
         *(([VAL.joinpath("notes.md").read_text().rstrip(), ""]) if VAL.joinpath("notes.md").exists() else []),
         "## O que cada validação checa",
         "1. `search_state.json` carrega; nº de jobs, status, brackets, último job.",
         "2. Manifesto (marcador `.ok`, `*manifest*` do staging, `gpu3_migration_manifest.csv`): nº de jobs e sha256 (se houver).",
         "3. Checkpoints de TODOS os jobs ok existem; `torch.load` em CPU (`weights_only=False`) de 5 amostras + todos os de fidelidade máxima.",
         "4. Config viva da Jaci (`search_v3*.yaml` + mesmos `--set`) = `state[\"config\"]` (seção search); `config_resolved.yaml` comparado.",
         "5. Dataset na Jaci: yaml, imagens, split (sha256), **identidade de dados** (hash de todas as imagens de treino/val, a mesma do Evaluator) = `data_sha256` dos jobs; `spifil_pool_sizes` iguais.",
         "6. Semente externa (semeadas): arquivos presentes; sha do encoder convertido na Jaci = o do athena.",
         "7. Assinatura do fitness (C100_kw1) e id do controller W&B que a Jaci vai usar.",
         "8. **Replay a seco**: `MODEHB.run(resume=True)` com avaliador-stub sobre uma cópia do state — tem de reproduzir todos os jobs (sem `ReplayMismatch`) e parar no próximo job novo.",
         "",
         "## Resumo",
         "",
         "| prio | busca | migração | resume | jobs | fid máx | último job | erros/conflitos |",
         "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        lj = e.get("ultimo_job") or {}
        ljs = f"{lj.get('job')} {lj.get('cand_id')} f{lj.get('fidelity')}" if lj else "—"
        probs = (e.get("errors") or []) + (e.get("conflitos") or [])
        L.append(f"| {e.get('prioridade', '')} | {e['id']} | {e.get('migration_status')} | {e.get('resume_status', '')} | "
                 f"{e.get('n_jobs', '—')} | {e.get('fidelidade_max', '—')} | {ljs} | "
                 f"{fmt_list(probs)[:300].replace('|', '/')} |")
    L.append("")
    for e in entries:
        if e.get("migration_status") in ("aguardando_staging",) or "n_jobs" not in e:
            continue
        if e.get("n_jobs") == 0:
            L += [f"## {e['id']} — {e.get('algoritmo')} — **{e.get('migration_status')}**", ""]
            for k in ("origem", "depende_de", "semente", "config", "recursos", "obs"):
                if e.get(k):
                    L.append(f"- {k}: `{e[k]}`" if k in ("config", "semente", "recursos") else f"- {k}: {e[k]}")
            if e.get("marcador_texto"):
                L.append(f"- marcador: \"{e['marcador_texto'].strip()}\"")
            if e.get("errors"):
                L.append(f"- **ERROS**: {fmt_list(e['errors'])}")
            if e.get("conflitos"):
                L.append(f"- **CONFLITO — PARAR e decidir**: {fmt_list(e['conflitos'])}")
            if e.get("cmd_waiter"):
                L += ["", f"Waiter (tmux `{e.get('tmux_waiter')}`; não precisa publicar nada):", "```bash", e["cmd_waiter"], "```"]
            if e.get("cmd_waiter_OPCIONAL_com_OK"):
                L += ["", "**Só com OK do coordenador** (réplica seed43, reserva F4; começa do zero em state/seed43, "
                          "dentro da base larvae que já existe):", "```bash", e["cmd_waiter_OPCIONAL_com_OK"], "```"]
            L.append("")
            continue
        L += [f"## {e['id']} — {e.get('algoritmo')} — **{e.get('migration_status')}**", ""]
        L.append(f"- jobs {e.get('n_jobs')} {e.get('status_jobs')}, brackets concluídos {e.get('brackets_concluidos')}, "
                 f"done={e.get('done')}, GPU-h {e.get('gpu_hours')}; fid máx {e.get('fidelidade_max')} "
                 f"({e.get('n_ckpt_fid_max')} ckpts); ckpts carregados em CPU {e.get('checkpoints_carregados_cpu')}, "
                 f"ausentes {e.get('checkpoints_ausentes')}; staging {e.get('disk_staging_gb')} GB.")
        L.append(f"- `search_state.json` sha256 `{e.get('state_sha256')}` (gravado {e.get('state_updated')}); "
                 f"jobs failed: {e.get('jobs_failed') or 'nenhum'}.")
        L.append(f"- resume: {e.get('resume_status')}; próximo job {e.get('proximo_job')}.")
        L.append(f"- controller Jaci `{e.get('controller_id_jaci')}` ({e.get('fit_sig')}); recursos {e.get('recursos')}.")
        L.append(f"- config: `{e.get('config')}`")
        if e.get("errors"):
            L.append(f"- **ERROS**: {fmt_list(e['errors'])}")
        if e.get("obs"):
            L.append(f"- obs: {e['obs']}")
        if e.get("marcador_texto"):
            L.append(f"- marcador: \"{e['marcador_texto'].strip()}\"")
        if e.get("warnings"):
            L.append(f"- avisos: {fmt_list(e['warnings'])}")
        if e.get("conflitos"):
            L.append(f"- **CONFLITO — PARAR e decidir**: {fmt_list(e['conflitos'])}")
        if e.get("cmd_publicar"):
            L += ["", "Publicação (atômica, só se o destino não existir):", "```bash", e["cmd_publicar"], "```"]
        if e.get("cmd_publicar_ALTERNATIVO_so_seed"):
            L += ["", "**NÃO executar sem OK do coordenador** — alternativa: publicar só o subdiretório "
                      f"state/seed{e.get('seed')} dentro do destino existente:", "```bash",
                  e["cmd_publicar_ALTERNATIVO_so_seed"], "```"]
        if e.get("cmd_waiter"):
            L += ["", f"Waiter (tmux `{e.get('tmux_waiter')}`; rodar DEPOIS da publicação):", "```bash", e["cmd_waiter"], "```"]
        L.append("")
    tmp = AG5 / ".resume_validation_report.md.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, AG5 / "resume_validation_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="segundos entre passadas (0 = uma passada)")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--exit-on-new", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    while True:
        new, entries = one_pass(force=a.force)
        a.force = False
        print(f"{time.strftime('%H:%M:%S')} passada: {len(entries)} entradas; novas validadas: {new}", flush=True)
        if not a.loop or (a.exit_on_new and new) or time.time() - t0 > a.max_hours * 3600:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
