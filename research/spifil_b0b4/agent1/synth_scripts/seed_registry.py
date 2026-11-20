#!/usr/bin/env python
"""seed_registry.py v1 — gerador ÚNICO do agent1/seed_registry.csv (Agente 1, 17/11/2026).

Corrige as condições C1/C2/C3 de research/spifil_b0b4/check/gate_A1_audit.md:
  C1  população completa, lida dos search_state.json atuais (não só jobs com teste);
  C2  entradas congeladas num snapshot (cópia .gz + md5); o registro é gerado SÓ do snapshot;
      uma linha por sub-régua ORIG (nunca mistura sub-réguas no argmax); κ ORIG degenerado
      (≤ 0,05) é inutilizável; caminhos de checkpoint vêm do próprio search_state;
  C3  desempate do evaluation_contract §2.3: |Δκ| < MDE é empate prático → menos FLOPs,
      menos params, menor fidelidade numérica, menor cand_id.
Nenhum número de TESTE é lido.

Uso:
  # 1) congelar entradas (cria synth_scripts/snapshot_<tag>/ + manifest.json com md5)
  python seed_registry.py --make-snapshot --tag 20261117_A1v1
  # 2) gerar o registro só a partir do snapshot (confere md5 de tudo antes)
  python seed_registry.py --tag 20261117_A1v1 --out ../seed_registry.csv
"""
import argparse, csv, glob, gzip, hashlib, io, json, os, shutil, sys, time
import yaml

VERSION = "v1.1"  # v1.1: corrige caminhos de checkpoint de diretórios de busca renomeados
ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = "/dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena"

BANDS = [("B0", 0, 300e3), ("B1", 300e3, 800e3), ("B2", 800e3, 1.5e6), ("B3", 1.5e6, 3e6), ("B4", 3e6, 5.5e6)]
CAP = 5.5e6
DS = ["remote_sensing", "ham10000", "flowers", "eggs", "larvae", "protozoan", "eggs_raw", "larvae_raw",
      "protozoan_raw", "coconut_trees", "aid"]
PRI = {"remote_sensing": 1, "ham10000": 1, "flowers": 1, "eggs": 2, "larvae": 2, "protozoan": 2, "eggs_raw": 2,
       "larvae_raw": 2, "protozoan_raw": 2, "coconut_trees": 3, "aid": 3}
DOM = {"eggs": "pre", "larvae": "pre", "protozoan": "pre", "eggs_raw": "raw", "larvae_raw": "raw",
       "protozoan_raw": "raw"}
# tamanho do train/val de cada split usado na busca (A1c_generalizacao.md:57-67)
N_TRAIN = {"aid": 6996, "coconut_trees": 9511, "flowers": 1922, "remote_sensing": 682, "ham10000": 4129,
           "eggs": 8885, "eggs_raw": 8885, "larvae": 1118, "larvae_raw": 1118, "protozoan": 26159,
           "protozoan_raw": 26159}
N_VAL = {"aid": 1502, "coconut_trees": 2038, "flowers": 412, "remote_sensing": 159, "ham10000": 878,
         "eggs": 1903, "eggs_raw": 1903, "larvae": 240, "larvae_raw": 240, "protozoan": 5607,
         "protozoan_raw": 5607}
# MDE PROVISÓRIO do contrato §5.3 até o X01 medir: 0,02; 0,03 com n_val ≤ 240
MDE = {d: (0.03 if N_VAL[d] <= 240 else 0.02) for d in DS}
DEGENERATE_KAPPA = 0.05

# -------- entradas congeladas
STATE_GLOBS = [
    "artifacts/modehb_hybrid/*/state/seed42/search_state.json",
    "artifacts/modehb_lemonade/*/state/seed42/search_state.json",
    # legado (régua head sem scaler = ORIG, pipeline antiga): só para a régua ORIG_legado
    "artifacts/modehb_spifil/*/state/seed42/search_state.json",
    "artifacts/modehb_strat_*/*/state/seed42/search_state.json",
    "artifacts/modehb_v3/*/state/seed42/search_state.json",
    "artifacts/modehb_v3_old/*/state/seed42/search_state.json",
]
EXTRA = {
    "bundle_pnas_cs_search_state": BUNDLE + "/pnas_cs/search_state.json",
    "pnas_cs_yaml": ROOT + "/ga_search/config/modehb/pnas_cs.yaml",
    "agent6_E0_jsonl": ROOT + "/research/spifil_b0b4/agent6/results/E0_val_jaci_cpu.jsonl",
    "agent6_ckpt_csv": ROOT + "/research/spifil_b0b4/agent6/checkpoints_E0_E8_congelado_20261117_0159.csv",
    "A1b_embed_diag_csv": ROOT + "/research/spifil_b0b4/agent1/parts/A1b_embed_diag.csv",
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def make_snapshot(tag):
    snap = os.path.join(HERE, "snapshot_" + tag)
    if os.path.exists(snap):
        sys.exit(f"snapshot {snap} já existe; não sobrescrevo (escolha outra --tag)")
    os.makedirs(snap)
    files = {}
    srcs = {}
    for g in STATE_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            rel = os.path.relpath(p, ROOT)
            srcs[rel] = p
            cfg = os.path.join(os.path.dirname(p), "config_resolved.yaml")
            if os.path.exists(cfg):
                srcs[os.path.relpath(cfg, ROOT)] = cfg
    for k, p in EXTRA.items():
        srcs["extra/" + k + os.path.splitext(p)[1]] = p
    for rel, p in srcs.items():
        dst = os.path.join(snap, rel + ".gz")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        m = md5_file(p)
        with open(p, "rb") as fi, gzip.GzipFile(dst, "wb", compresslevel=6, mtime=0) as fo:
            shutil.copyfileobj(fi, fo)
        files[rel] = {"source": p, "md5_source": m, "bytes": os.path.getsize(p), "mtime": os.path.getmtime(p),
                      "gz": os.path.relpath(dst, snap), "md5_gz": md5_file(dst)}
    man = {"tag": tag, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": os.uname().nodename,
           "script_version": VERSION, "files": files}
    json.dump(man, open(os.path.join(snap, "manifest.json"), "w"), indent=1)
    print(f"snapshot {snap}: {len(files)} arquivos")


class Snap:
    def __init__(self, tag):
        self.dir = os.path.join(HERE, "snapshot_" + tag)
        self.man = json.load(open(os.path.join(self.dir, "manifest.json")))
        bad = []
        for rel, f in self.man["files"].items():
            raw = gzip.open(os.path.join(self.dir, f["gz"]), "rb").read()
            if hashlib.md5(raw).hexdigest() != f["md5_source"]:
                bad.append(rel)
        if bad:
            sys.exit(f"md5 divergente no snapshot: {bad[:5]}")

    def read(self, rel):
        return gzip.open(os.path.join(self.dir, self.man["files"][rel]["gz"]), "rb").read()

    def rels(self):
        return list(self.man["files"].keys())


def signature(sf):
    readout = sf.get("readout") or "head"
    C, sc, kw, mt = float(sf.get("C")), str(sf.get("scaler")), float(sf.get("kappa_weight", 1)), sf.get("max_train")
    if readout == "cov_head" and C == 100 and sc == "standard" and kw == 1.0:
        return "COV", f"C100_kw1_cov_head_standard_mt{mt}"
    if readout == "head" and C == 100 and sc in ("none", "None", "False"):
        return "ORIG_legado_gpuamp", f"C100_kw{kw:g}_head_none_mt{mt}"
    return None, f"C{C:g}_kw{kw:g}_{readout}_{sc}_mt{mt}"


def load_jobs(snap):
    """Todos os jobs (ok e não ok) de todos os search_state do snapshot, com assinatura de fitness."""
    jobs = []
    for rel in snap.rels():
        if not rel.endswith("search_state.json") and rel != "extra/bundle_pnas_cs_search_state.json":
            continue
        s = json.loads(snap.read(rel))
        if rel.startswith("extra/"):
            sf = yaml.safe_load(snap.read("extra/pnas_cs_yaml.yaml"))["svm_fitness"]
            run, ds_fixed, host = "artifacts/pnas_cs", "remote_sensing", "athena"
        else:
            cfg = yaml.safe_load(snap.read(rel.replace("search_state.json", "config_resolved.yaml")))
            sf = cfg.get("svm_fitness") or (cfg.get("eval") or {}).get("svm_fitness")
            run, ds_fixed, host = "/".join(rel.split("/")[:2]), rel.split("/")[2], "jaci"
        regua, sig = signature(sf)
        for j in s["jobs"]:
            r = j.get("result") or {}
            g = j.get("genome") or {}
            ds = ds_fixed
            ck, ck_fix = r.get("checkpoint") or "", ""
            # C2: o diretório da busca foi renomeado depois do job (ex. modehb_strat -> modehb_strat_prefix_2130);
            # o search_state guarda o caminho antigo. Reescreve para o diretório real do search_state.
            if ck and host == "jaci" and "/state/" in ck and not os.path.isfile(os.path.join(ROOT, ck)):
                alt = rel.split("/state/")[0] + "/state/" + ck.split("/state/", 1)[1]
                if os.path.isfile(os.path.join(ROOT, alt)):
                    ck, ck_fix = alt, f"caminho corrigido (search_state guarda {r.get('checkpoint')})"
            jobs.append(dict(
                state_rel=rel, state_md5=snap.man["files"][rel]["md5_source"], run=run, kind=run.split("/")[-1].replace("modehb_", ""),
                host=host, dataset=ds, regua_busca=regua, sig=sig, role=j.get("role"), status=j.get("status"),
                rstatus=r.get("status"), error=str(j.get("error", ""))[:120], cand=j.get("cand_id"),
                fid=float(j.get("fidelity")) if j.get("fidelity") is not None else None,
                kappa=r.get("fitness_kappa"), params=r.get("params"), flops=r.get("flops"),
                checkpoint=ck, ck_fix=ck_fix, wise_alpha=r.get("wise_alpha"),
                depth=g.get("depth"), embed_dim=g.get("embed_dim"), pooling=g.get("pooling"), protect=g.get("protect"),
                n_train_used=r.get("fitness_n_train_used")))
    return jobs


def pick(cands, mde):
    """Desempate do contrato §2.3. cands: lista de dicts com kappa, flops, params, fid, cand."""
    kmax = max(c["kappa"] for c in cands)
    strict = sorted(cands, key=lambda c: (-c["kappa"], c["fid"] or 0, c["flops"] or 0, c["cand"]))[0]
    tie = [c for c in cands if c["kappa"] > kmax - mde]
    chosen = sorted(tie, key=lambda c: (c["flops"] if c["flops"] is not None else float("inf"),
                                        c["params"] or 0, c["fid"] or 0, c["cand"]))[0]
    return chosen, strict, len(tie)


def build(tag, out):
    snap = Snap(tag)
    jobs = load_jobs(snap)
    ag6 = list(csv.DictReader(io.StringIO(snap.read("extra/agent6_ckpt_csv.csv").decode())))
    md5map = {r["checkpoint"]: r["md5"] for r in ag6}
    by_ckpt = {}
    for j in jobs:
        if j["status"] == "ok" and j["checkpoint"]:
            by_ckpt.setdefault(j["checkpoint"], j)

    def ck_md5(path, host):
        if path in md5map:
            return md5map[path], "agent6/checkpoints_E0_E8_congelado_20261117_0159.csv"
        f = os.path.join(ROOT, path)
        if host == "jaci" and os.path.isfile(f):
            return md5_file(f), "calculado na geração (Jaci)"
        return "", ("ausente_na_jaci (só athena)" if host == "athena" else "ausente_na_jaci")

    rows = []

    def emit(ds, B, regua, det, chosen, strict, n_band, n_tie, obs, lacuna=False, utilizavel="sim", origem=""):
        c = chosen or {}
        ck = c.get("checkpoint", "")
        m, ms = ck_md5(ck, c.get("host", "jaci")) if ck else ("", "")
        fl = c.get("flops")
        rows.append(dict(
            dataset=ds, prioridade=PRI[ds], dominio=DOM.get(ds, "natural"), faixa=B, regua=regua,
            regua_detalhe=det, lacuna="sim" if lacuna else "nao", utilizavel=utilizavel,
            host=c.get("host", ""), run=c.get("run", ""), busca=c.get("kind", ""), cand=c.get("cand", ""),
            fid=("f%d" % c["fid"]) if c.get("fid") is not None else "",
            kappa_val=round(c["kappa"], 4) if c.get("kappa") is not None else "",
            mde_provisorio=MDE[ds], n_cand_na_faixa=n_band, n_empate_pratico=n_tie,
            argmax_estrito=(f"{strict['cand']} f{int(strict['fid'])} {strict['kappa']:.4f}" if strict else ""),
            params=c.get("params", ""), flops_MAC=fl if fl is not None else "",
            flops_2MAC=2 * fl if fl is not None else "", depth=c.get("depth", ""), embed_dim=c.get("embed_dim", ""),
            pooling=c.get("pooling", ""), protect=c.get("protect", ""), wise_alpha=c.get("wise_alpha", ""),
            status_search_state=c.get("status", ""), checkpoint=ck,
            checkpoint_existe_jaci=("sim" if ck and os.path.isfile(os.path.join(ROOT, ck)) and c.get("host") == "jaci" else ("nao" if ck else "")),
            md5=m, md5_fonte=ms, fonte_kappa=c.get("fonte", ""), search_state=c.get("state_rel", ""),
            search_state_md5=c.get("state_md5", ""), fitness_signature=c.get("sig", ""), origem=origem,
            obs="; ".join(x for x in (obs, c.get("ck_fix", "")) if x)))

    # ---------------- régua COV (fitness atual)
    cov = [j for j in jobs if j["regua_busca"] == "COV" and j["status"] == "ok" and j["rstatus"] in ("ok", None)
           and j["role"] == "search" and j["kappa"] is not None and j["params"] is not None and j["params"] <= CAP]
    for j in cov:
        j["fonte"] = "search_state result.fitness_kappa (COV-val)"
    excl = [j for j in jobs if j["regua_busca"] == "COV" and j["status"] != "ok" and j["params"] and j["params"] > 0]
    for ds in DS:
        for B, lo, hi in BANDS:
            band = [j for j in cov if j["dataset"] == ds and lo < j["params"] <= hi]
            fails = [f"{j['kind']} {j['cand']} f{int(j['fid'])} ({j['status']}: {j['error'][:50]})" for j in excl
                     if j["dataset"] == ds and lo < (j["params"] or 0) <= hi]
            ob = ("excluídos por status≠ok no search_state: " + "; ".join(fails[:5])) if fails else ""
            if band:
                ch, st, nt = pick(band, MDE[ds])
                if ch is not st:
                    ob = (ob + "; " if ob else "") + "desempate do contrato §2.3 (empate prático < MDE → menos FLOPs) escolheu ≠ argmax estrito"
                emit(ds, B, "COV", "cov_head+StandardScaler, C=100, mt4000, α do WiSE (fitness atual)", ch, st,
                     len(band), nt, ob, origem="busca GA atual (C100_kw1_cov)")
            else:
                emit(ds, B, "COV", "cov_head+StandardScaler, C=100, mt4000", None, None, 0, 0,
                     (ob + "; " if ob else "") + "nenhum candidato ok na faixa", lacuna=True, utilizavel="")

    # ---------------- régua ORIG, uma linha por sub-régua
    orig = []
    for j in jobs:  # legado: head sem scaler na própria busca (GPU-AMP)
        if j["regua_busca"] == "ORIG_legado_gpuamp" and j["status"] == "ok" and j["kappa"] is not None \
                and j["params"] is not None and j["params"] <= CAP:
            o = dict(j)
            o["sub"] = "ORIG_legado_gpuamp"
            o["fonte"] = f"search_state legado result.fitness_kappa ({j['sig']}; pipeline antiga sem AE/protect atuais)"
            orig.append(o)
    ed = list(csv.DictReader(io.StringIO(snap.read("extra/A1b_embed_diag_csv.csv").decode())))
    for e in ed:
        if e["variant"] != "saved":
            continue
        base = dict(by_ckpt.get(e["checkpoint"], {}))
        if not base:
            continue
        full = int(e["n_train"]) == N_TRAIN[e["dataset"]] and int(e["n_val"]) == N_VAL[e["dataset"]]
        base.update(kappa=float(e["k_orig_noscaler"]),
                    sub="ORIG_full_cpu32" if full else "ORIG_sub2000_cpu32",
                    fonte=f"A1b_embed_diag.csv variant=saved k_orig_noscaler (n_train={e['n_train']}, n_val={e['n_val']})")
        orig.append(base)
    for line in snap.read("extra/agent6_E0_jsonl.jsonl").decode().splitlines():
        jj = json.loads(line)
        if jj.get("exp") != "E0" or "cost" not in jj or jj.get("tta") != "none" or jj["genes"]["attention_type"] != "none":
            continue
        base = dict(by_ckpt.get(jj["checkpoint"], {}))
        if not base:
            continue
        full = int(jj["n_train"]) == N_TRAIN[jj["dataset"]]
        base.update(kappa=float(jj["kappa"]), sub="ORIG_full_cpu32" if full else "ORIG_sub4000_cpu32",
                    fonte=f"agent6 E0_val_jaci_cpu.jsonl (n_train={jj['n_train']}, embed_dim={jj['embed_dim']}, {jj['device']} {jj['precision']})")
        orig.append(base)
    # dedup: mesmo checkpoint + mesma sub-régua → primeira medição
    seen, od = set(), []
    for o in orig:
        k = (o["checkpoint"], o["sub"])
        if k not in seen:
            seen.add(k)
            od.append(o)
    SUBDESC = {
        "ORIG_full_cpu32": "head≥1280, fit_svm C=100 sem scaler, train inteiro → val inteira, CPU FP32",
        "ORIG_sub4000_cpu32": "idem, SVM em subconjunto estratificado semeado ≤4000 (train > 4000), CPU FP32",
        "ORIG_sub2000_cpu32": "idem, treino ≤2000 (e val 1500 em protozoan), CPU FP32; só comparação interna",
        "ORIG_legado_gpuamp": "head≥1280 sem scaler C=100 mt4000/10000 (= train inteiro em RS/flowers), GPU-AMP, pipeline antiga: OUTRA POPULAÇÃO",
    }
    for ds in DS:
        for B, lo, hi in BANDS:
            any_row = False
            for sub in SUBDESC:
                band = [o for o in od if o["dataset"] == ds and o["sub"] == sub and o["params"] and lo < o["params"] <= hi]
                if not band:
                    continue
                any_row = True
                deg = [o for o in band if o["kappa"] <= DEGENERATE_KAPPA]
                ok = [o for o in band if o["kappa"] > DEGENERATE_KAPPA]
                ob = ""
                if deg:
                    ob = "INUTILIZÁVEL (κ ≤ 0,05, embedding degenerado): " + "; ".join(
                        f"{o['cand']} f{int(o['fid'])} κ={o['kappa']:.3f}" for o in deg)
                if ok:
                    ch, st, nt = pick(ok, MDE[ds])
                    if ch is not st:
                        ob = (ob + "; " if ob else "") + "desempate do contrato §2.3 escolheu ≠ argmax estrito"
                    if len(ok) < 3:
                        ob = (ob + "; " if ob else "") + f"só {len(ok)} medido(s) na faixa: re-pontuar a faixa inteira no X00 antes de congelar"
                    emit(ds, B, "ORIG", SUBDESC[sub] + f" [{sub}]", ch, st, len(band), nt, ob,
                         origem=("busca legada" if sub == "ORIG_legado_gpuamp" else "busca GA atual, re-pontuada"))
                else:
                    emit(ds, B, "ORIG", SUBDESC[sub] + f" [{sub}]", deg[0], None, len(band), 0, ob,
                         utilizavel="nao", origem="busca GA atual, re-pontuada")
            if not any_row:
                emit(ds, B, "ORIG", "nenhuma sub-régua medida", None, None, 0, 0,
                     "LACUNA: re-pontuar os candidatos COV da faixa em ORIG_full-val (X00); nunca usar teste",
                     lacuna=True, utilizavel="")

    cols = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log = {"script": os.path.abspath(__file__), "script_md5": md5_file(os.path.abspath(__file__)), "version": VERSION,
           "snapshot": snap.dir, "snapshot_manifest_md5": md5_file(os.path.join(snap.dir, "manifest.json")),
           "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "out": os.path.abspath(out), "out_md5": md5_file(out),
           "n_rows": len(rows), "n_jobs_lidos": len(jobs), "mde_provisorio": MDE}
    json.dump(log, open(os.path.join(HERE, f"seed_registry_{tag}.log.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "mde_provisorio"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--make-snapshot", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "seed_registry.csv"))
    a = ap.parse_args()
    if a.make_snapshot:
        make_snapshot(a.tag)
    else:
        build(a.tag, a.out)
