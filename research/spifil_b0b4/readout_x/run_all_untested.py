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
"""run_all_untested.py (athena) -- testa (SVM original: linear, C=100, ovo, max_iter=-1, sem scaler)
TODOS os modelos das 5 bases sem teste ainda. Copia adaptada do run_all_untested.py da Jaci.

Fontes: ga = artifacts/modehb_*; pnas = artifacts/pnas_cs* (sem smoke); growth = nos ok das cadeias
hmodehb_b0b4/{growth,growth_v3,growth_v3_fixed} sem linha no test_orig.jsonl. Um checkpoint por modelo:
a MAIOR fidelidade. Ordem: passe 1 = top-N por (base, fonte) por kappa de VAL, gap1280 + so2_4096_pca
+ TTA8; passe 2 = o resto, gap1280 + so2_4096_pca. Bases e fontes intercaladas.
GPU pela memoria REAL: a cada 10 s le o nvidia-smi e lanca numa GPU se free - (pico ainda nao
alocado dos nossos) >= pico + margem. GPUs 0/1 so sem processo de outro usuario. Teste nunca alimenta
selecao. Saida: results/all_untested_test.jsonl (idempotente pelo backbone_id); uma parte por modelo
em results/all_untested_parts/, um log por modelo em results/all_untested_logs/."""
import argparse, csv, glob, hashlib, json, math, os, re, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
PY = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python"
LIB = "/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib"
B_NAT = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
B_PAR = ["eggs", "larvae", "protozoan", "eggs_raw", "larvae_raw", "protozoan_raw"]   # so TESTE (18/11), depois das naturais
SRC = ["growth", "pnas", "ga"]                          # passe 1 (top-N) so destas
XSRC = ["a2", "ssl", "fm"]                               # (e) outras: so passe 2 (fm: so nos passes 5/6)
FM_EMB = Path("/exps/moliveira/benchmark/emb")           # embeddings dos FMs (benchmark da discovery)
FMS = ["dinov2-giant", "ijepa_vitg16_22k", "aimv2-3B-patch14-448", "convnextv2-pico-1k-224", "resnet50.a1_in1k"]
MIRROR = Path("/dados/home/moliveira/exps/jaci_results_mirror/readout_x_results")   # resultados da Jaci (15 min)
GROWTH = Path("/dados/home/moliveira/exps/growth/code/artifacts/hmodehb_b0b4")
CODE = {"growth": "/dados/home/moliveira/exps/growth/code", "growth_v3": "/dados/home/moliveira/exps/growth_v3/code",
        "a2": "/dados/home/moliveira/exps/a2/code", "oct": "/dados/home/moliveira/exps/oct_hybrid/code"}
COPIES = ["/dados/home/moliveira/exps/attn6/ckpts", "/dados/home/moliveira/exps/growth/code/artifacts"]
ATTN6_CSV = "/dados/home/moliveira/exps/attn6/checkpoints_E0_E8_congelado_20261117_0159.csv"
HEADS = ["gap1280", "so2_4096_pca"]
ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=40); ap.add_argument("--gpus", default="2,3")   # 0/1 livres (ordem 18/11)
ap.add_argument("--peak-gb", type=float, default=7.0); ap.add_argument("--margin-gb", type=float, default=2.0)
ap.add_argument("--jobs", type=int, default=28); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--parasitas", action="store_true")
ap.add_argument("--variant", choices=["spifil_init"], default=None)
ap.add_argument("--train-pct", action="store_true")   # passes 5/6: SVM em 1..100% do treino, TODOS os modelos   # mesma arquitetura, so init SPiFiL (spifil_init.py)   # teste de parasitas: so com ordem direta do usuario
ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--out", default=None); a = ap.parse_args()
a.out = a.out or str(HERE / "results" / "all_untested_test.jsonl")
B = B_NAT + (B_PAR if a.parasitas else [])
OUT = Path(a.out); PARTS = HERE / "results" / "all_untested_parts"; LOGS = HERE / "results" / "all_untested_logs"
for d in (PARTS, LOGS): d.mkdir(parents=True, exist_ok=True)

# ── ja testados FORA deste runner (ranking estavel: o que este runner fez continua no ranking)
tested, tested_ck, tested_md5 = set(), set(), set()
for l in open(REPO / "ga_search/modehb/reports/test_svm_C100.jsonl"):
    try: r = json.loads(l); tested.add((r["run"].rstrip("/").split("/")[-1], r["dataset"], r["cand_id"]))
    except Exception: pass
for f in glob.glob(str(HERE / "results" / "*.jsonl")):
    if Path(f).name.startswith("all_untested"): continue      # saidas deste runner (base e variantes)
    for l in open(f):
        try: r = json.loads(l); tested.add(("id", r["backbone_id"])); tested_ck.add(r["ckpt"]); tested_md5.add(r.get("md5"))
        except Exception: pass
for f in Path("/dados/home/moliveira/exps/jaci_results_mirror/readout_x_results").glob("*.jsonl"):   # Jaci: so com tta8
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("head") == "so2_4096_pca_tta8" and not re.search(r"_p\d+$", r["backbone_id"]):
            tested.add(("id", r["backbone_id"])); tested_ck.add(os.path.realpath(r["ckpt"])); tested_md5.add(r.get("md5"))

# ── candidatos: (base, fonte) -> {chave: (fid, kval, ckpt, id, kind, code_root)}
def scan(excl=True):     # excl=False: TODOS os modelos (passes 5/6, % de treino), sem tirar os ja testados
    tested, tested_ck, tested_md5 = (globals()["tested"], globals()["tested_ck"], set(globals()["tested_md5"])) if excl else (set(), set(), set())
    per = {(b, s): {} for b in B for s in SRC + XSRC}
    kv_copy = {(r["kind"], r["dataset"], r["cand"], r["fid"]): float(r["kappa_val_search"]) for r in csv.DictReader(open(ATTN6_CSV))}
    for root in [str(REPO / "artifacts"), *COPIES]:        # copias (attn6, growth/code) so entram se o original sumiu
        for f in sorted(glob.glob(f"{root}/*/*/state/seed42/candidates/c*/f*/encoder.pt")):
            ck = Path(f); run, ds, c, fd = ck.parts[-8], ck.parts[-7], ck.parts[-3], ck.parts[-2]
            if not fd[1:].isdigit(): continue
            fid = int(fd[1:])
            src = "ga" if run.startswith("modehb_") else "pnas" if run.startswith("pnas_cs") and "smoke" not in run else None
            if src is None or ds not in B or (run, ds, c) in tested or os.path.realpath(ck) in tested_ck: continue
            if (ck.parent / "result.json").is_file():
                try: r = json.loads((ck.parent / "result.json").read_text())
                except Exception: continue
                if r.get("status") != "ok": continue
                kv = r.get("fitness_kappa") or 0
            elif root == str(REPO / "artifacts"): continue
            else: kv = kv_copy.get((run, ds, c, fd), 0)
            i = f"{ds}_{run.replace('modehb_', '')}_{c}_f{fid}"
            if ("id", i) in tested: continue
            k = (run, c); cur = per[(ds, src)].get(k)
            if cur is None or fid > cur[0]:
                per[(ds, src)][k] = (fid, kv, str(ck), i, "modehb", str(REPO))
    for f in glob.glob(CODE["a2"] + "/artifacts/a2_ablation/*/candidates/*/f*/result.json"):   # ablacao de init (a2)
        p = Path(f); ds, name, fd = p.parts[-5], p.parts[-3], p.parts[-2]; ck = p.parent / "encoder.pt"
        if ds not in B or not fd[1:].isdigit() or not ck.is_file() or os.path.realpath(ck) in tested_ck: continue
        try: r = json.loads(p.read_text())
        except Exception: continue
        fid = int(fd[1:]); i = f"{ds}_a2_{name}_f{fid}"; cur = per[(ds, "a2")].get(name)
        if r.get("status") == "ok" and ("id", i) not in tested and (cur is None or fid > cur[0]):
            per[(ds, "a2")][name] = (fid, r.get("fitness_kappa") or r.get("quality") or 0, str(ck), i, "modehb", CODE["a2"])
    old = set()                                               # ssl_pequeno com teste no protocolo antigo
    for f in glob.glob("/exps/moliveira/results/svm_ssl_pequeno_*_test_*.csv") if excl else []:
        old |= {r.get("ckpt", "").split("spifil_growth/")[-1] for r in csv.DictReader(open(f))}
    for f in sorted(glob.glob("/dados/home/moliveira/artifacts/external_ssl_pequeno/*_split1_pct100/*/checkpoints/best_kappa.ckpt")) + \
             sorted(glob.glob("/exps/moliveira/spifil_growth/ssl_pequeno_*/*_split1_pct100/L*/checkpoints/best_kappa.ckpt")):
        ck = Path(f); ds = ck.parts[-4].rsplit("_split", 1)[0]
        if ds not in B_NAT or f.split("spifil_growth/")[-1] in old or os.path.realpath(ck) in tested_ck: continue
        with open(ck, "rb") as fh: md5 = hashlib.file_digest(fh, "md5").hexdigest()
        if md5 in tested_md5: continue                        # copia external <- spifil_growth: uma vez so
        tested_md5.add(md5)
        mp = ck.parents[1] / "run_metadata.json"; m = json.loads(mp.read_text()) if mp.is_file() else {}
        kv = m.get("best_val_svm_kappa") if "kappa" in str(m.get("ckpt_monitor")) else 0
        i = f"{ds}_extsslp_{ck.parts[-3]}" if "external_ssl_pequeno" in f else f"{ds}_{ck.parts[-5]}_{ck.parts[-3]}"
        if ("id", i) not in tested: per[(ds, "ssl")][i] = (0, kv or 0, str(ck), i, "hybrid", CODE["oct"])
    for st in glob.glob(str(GROWTH / "growth*/*/*/seed42/growth_state.json")):
        sd = Path(st).parent; chain = sd.relative_to(GROWTH).parts[:-1]
        if "smoke" in str(sd): continue
        ds = json.loads(Path(st).read_text())["config_full"]["run"]["data"]["dataset"]
        if ds not in B: continue
        to = sd / "test_orig.jsonl"
        shas = {json.loads(l).get("checkpoint_sha256") for l in open(to) if l.strip()} if excl and to.is_file() else set()
        code = CODE["growth_v3" if chain[0].startswith("growth_v3") else "growth"]
        for nd in sorted((sd / "nodes").iterdir()):
            res = sorted((q for q in nd.glob("f*/result.json") if q.parent.name[1:].isdigit()),
                         key=lambda q: -int(q.parent.name[1:])) or [nd / "result.json"]
            for p in res:                                     # f mais alto com status ok
                try: r = json.loads(p.read_text())
                except Exception: continue
                ck = p.parent / "backbone.pt"
                if r.get("status") == "ok" and ck.is_file(): break
            else: continue
            fid = int(r.get("fid") or 0); i = f"{ds}_{'-'.join(chain)}_{nd.name}_f{fid}"
            if r.get("checkpoint_sha256") in shas or os.path.realpath(ck) in tested_ck or ("id", i) in tested: continue
            per[(ds, "growth")][(str(sd), nd.name)] = (fid, r.get("kappa_val_orig") or 0, str(ck), i, "growth", code)
    return per

per = scan()

queues = {k: sorted(v.values(), key=lambda x: -x[1]) for k, v in per.items()}
def interleave(qs, bases):
    qs = {k: list(v) for k, v in qs.items() if k[0] in bases}; o = []
    while any(qs.values()):
        for b in bases:
            for s in SRC + XSRC:
                if qs.get((b, s)): o.append((b, s, *qs[(b, s)].pop(0)))
    return o
FLOOR = 100_000     # 19/11 usuario: modelozinhos (< 100 mil params do encoder) ficam fora da prioridade
_np = {}
def npar(t):        # t = (fid, kval, ckpt, id, kind, code)
    if t[2] not in _np:
        r = Path(t[2]).parent / "result.json"
        try: n = json.loads(r.read_text()).get("params") if r.is_file() else None
        except Exception: n = None
        if t[4] == "fm": n = float("inf")
        elif n is None and t[4] == "hybrid":
            import torch
            sd = torch.load(t[2], map_location="cpu", weights_only=False, mmap=True).get("state_dict", {})
            n = sum(v.numel() for v in {v.untyped_storage().data_ptr(): v for k, v in sd.items() if k.startswith("model.encoder.")}.values())
        _np[t[2]] = float("inf") if n is None else n     # sem contagem: nao exclui
    return _np[t[2]]
# 19/11 usuario: so os 30% MELHORES por kappa de VAL de cada base (todas as fontes juntas, inclusive os ja testados),
# tirando os < FLOOR params (nao entram mesmo no top-30%); FMs sempre (no %). O resto NAO e testado.
ALL = scan(excl=False)
TOP, NTOP = set(), {}
for b in B:
    q = sorted((t for s in SRC + XSRC if s != "fm" for t in ALL.get((b, s), {}).values()), key=lambda t: -t[1])
    tb = {os.path.realpath(t[2]) for t in q[:math.ceil(0.3 * len(q))] if npar(t) >= FLOOR}
    TOP |= tb; NTOP[b] = (len(tb), len(q))
print("[RUNNER] top-30% por val (>= 100k params) / modelos da base: " + str(NTOP), flush=True)
intop = lambda t: t[4] == "fm" or os.path.realpath(t[2]) in TOP
def split_top(q):   # (no top-30%, fora)
    return [t for t in q if intop(t)], [t for t in q if not intop(t)]
todo = [(ps, t) for ps in (1, 2) for bases in (B_NAT, B_PAR)          # passe 1 (naturais, parasitas), passe 2 idem
        for t in interleave({k: split_top(q)[ps != 1] for k, q in queues.items()}, [b for b in bases if b in B])]
if a.variant:   # passes 3/4 = variante (top-N com TTA8, depois o resto), DEPOIS dos passes 1/2; so naturais;
    # L0 do ssl ja e SPiFiL puro; rand_* nao tem versao SPiFiL
    todo += [(ps + 2, (*t[:5], f"{t[5]}__{a.variant}", *t[6:])) for ps, t in todo if t[0] in B_NAT
            and not (t[1] == "ssl" and ("/L0/" in t[4] or "rand_" in t[4]))]
PCTS = [1, 5, 10, 25, 50, 75, 100]
if a.train_pct:  # passes 5/6 = % de treino (encoder congelado; so a SVM em p% do treino), TODOS os modelos
    qa = {k: sorted(v.values(), key=lambda x: -x[1]) for k, v in ALL.items()}
    for b in B_NAT:                                      # FMs: embedding ja extraido, passe 5 (referencia)
        qa[(b, "fm")] = [(0, 1.0, str(FM_EMB / m / f"{b}.npz"), f"FM_{m}_{b}", "fm", str(REPO))
                         for m in FMS if (FM_EMB / m / f"{b}.npz").is_file()]
    # parasitas: so larvae_raw tem splits_incremental = split da busca (eggs/protozoan e larvae pre: outro split)
    todo += [(ps, (*t[:5], f"{t[5]}__pct", *t[6:])) for ps in (5, 6) for bases in (B_NAT, ["larvae_raw"])
             for t in interleave({k: split_top(q)[ps != 5] for k, q in qa.items()},
                                 [b for b in bases if b in B])]
META = {t[5]: {"src": t[1], "fid": t[2], "kval": t[3], "pass": ps,
               "variant": {3: a.variant, 4: a.variant, 5: "train_pct", 6: "train_pct"}.get(ps)} for ps, t in todo}
if a.only is not None: todo = [t for t in todo if t[1][5] in a.only]

# ── saida: partes por modelo -> jsonl principal (um so escritor: este runner)
PCT_OUT = OUT.with_name("train_pct_test.jsonl")         # linhas por % de treino (passes 5/6)
key = lambda r: (r["backbone_id"], r["head"], r["pct"]) if "pct" in r else (r["backbone_id"], r["head"])
have = set()
for f in (OUT, PCT_OUT):
    if f.is_file():
        for l in open(f):
            try: have.add(key(json.loads(l)))
            except Exception: pass
def merge(part):
    if not part.is_file(): return                       # toda parte entra (base/variante -> OUT; % -> PCT_OUT)
    for l in open(part):
        try: r = json.loads(l)
        except Exception: continue
        if key(r) in have: continue
        r.setdefault("item", META.get(r["backbone_id"]))   # o eval do oct_hybrid nao grava o item
        with open(PCT_OUT if "pct" in r else OUT, "a") as fh: fh.write(json.dumps(r, default=str) + "\n")
        have.add(key(r))
for part in sorted(PARTS.glob("*.jsonl")): merge(part)
# TTA8 em TODO teste (ordem 19/11): passes 1-4 com gap + so2 + so2_tta8; passe 7 = re-teste so do TTA8 de
# quem ja tem linha no OUT sem so2_4096_pca_tta8 (melhor kval primeiro, bases intercaladas, naturais antes)
TTA = "so2_4096_pca_tta8"
rows = {}
for l in (open(OUT) if OUT.is_file() else []):
    try: r = json.loads(l)
    except Exception: continue
    if "pct" not in r: rows.setdefault(r["backbone_id"], r)
need = {i for i, r in rows.items() if (i, "so2_4096_pca") in have and (i, TTA) not in have and r["dataset"] in B}
code_of = lambda r: ((CODE["a2"] if "/exps/a2/code/" in r["ckpt"] else str(REPO)) if r["kind"] == "modehb" else
                     CODE["growth_v3" if "growth_v3" in r["ckpt"] else "growth"] if r["kind"] == "growth" else CODE["oct"])
rq = {}
for i in need:
    r = rows[i]; it = r.get("item") or {}
    rq.setdefault(r["dataset"], []).append((int(it.get("fid") or 0), float(it.get("kval") or r.get("kappa_val") or 0),
                                            r["ckpt"], i, r["kind"], code_of(r)))
def jaci_rows():      # mirror da Jaci + copias em results/jaci_*.jsonl
    for f in [*MIRROR.glob("*.jsonl"), *(HERE / "results").glob("jaci_*.jsonl")]:
        try:
            for l in open(f):
                try: yield json.loads(l)
                except ValueError: pass
        except OSError: pass
# mirror da Jaci: so conta como testado com so2_tta8; com so gap/so2 e o MESMO md5 aqui -> passe 7 (so tta8)
jh, jr = {}, {}
for r in jaci_rows():
    if "pct" in r or r.get("dataset") not in B or re.search(r"_p\d+$", r["backbone_id"]): continue   # % da Jaci
    jh.setdefault(r.get("md5"), set()).add(r["head"]); jr.setdefault(r.get("md5"), r)
out_md5 = {r.get("md5") for r in rows.values()}
for m, hs in jh.items():
    r = jr[m]; ck = Path(r["ckpt"])
    if TTA in hs or "so2_4096_pca" not in hs or m in out_md5 or r["backbone_id"] in rows or not ck.is_file(): continue
    with open(ck, "rb") as fh:
        if hashlib.file_digest(fh, "md5").hexdigest() != m: continue
    res = ck.parent / "result.json"
    kv = (json.loads(res.read_text()).get("fitness_kappa") if res.is_file() else None) or r.get("kappa_val") or 0
    fid = int(ck.parent.name[1:]) if ck.parent.name[1:].isdigit() else 0
    rq.setdefault(r["dataset"], []).append((fid, float(kv), str(ck), r["backbone_id"], r["kind"], code_of(r)))
    need.add(r["backbone_id"])
retest = []
for bases in (B_NAT, B_PAR):
    qs = {b: sorted(rq.get(b, []), key=lambda x: -x[1]) for b in bases}
    while any(qs.values()):
        for b in bases:
            if qs[b]: retest.append((7, (b, "retest", *qs[b].pop(0))))
META.update({t[5]: {"src": "retest", "fid": t[2], "kval": t[3], "pass": 7,
                    "variant": "spifil_init" if t[5].endswith("__spifil_init") else None} for _, t in retest})
todo = [t for t in todo if t[1][5] not in need or t[0] > 4]
k = next((j for j, t in enumerate(todo) if t[0] > 4), len(todo))
todo[k:k] = retest                                       # depois dos passes 1-4, antes do % (5/6)
want = lambda ps: [TTA] if ps == 7 else HEADS + ([TTA] if ps <= 5 else [])   # % : TTA8 so no top-20 (passe 5)
done = lambda ps, i: all(((i.removesuffix("__pct") + "_p100", h, 100) if ps > 4 else (i, h)) in have
                         for h in (["fm_emb"] if ps > 4 and i.startswith("FM_") else want(ps)))
todo = [t for t in todo if not done(t[0], t[1][5])]
# ORDEM (19/11, usuario): uma fila so de PRIORIDADE com os melhores por kappa de VAL (nunca teste): passes 1, 3, 5 e o
# passe 7 de quem esta no top-N por val da sua base; depois o resto: 7, 2, 4, 6. Em cada fila: melhor kval primeiro,
# bases intercaladas, naturais antes dos parasitas.
top_ck = {os.path.realpath(t[2]) for q in (qa if a.train_pct else queues).values() for t in split_top(q)[0]}
prio = lambda t: t[0] in (1, 3, 5) or (t[0] == 7 and os.path.realpath(t[1][4]) in top_ck)
def rr(items):
    o = []
    for bases in (B_NAT, B_PAR):
        qs = {b: sorted((t for t in items if t[1][0] == b), key=lambda t: -t[1][3]) for b in bases}
        while any(qs.values()):
            for b in bases:
                if qs[b]: o.append(qs[b].pop(0))
    return o + [t for t in items if t[1][0] not in B_NAT + B_PAR]
(HERE / "results" / f"fila_excluidos_top30_{time.strftime('%Y%m%d')}.txt").write_text(   # arquivo: o que saiu da fila
    "".join(f"{t[0]}\t{t[1][0]}\t{t[1][1]}\t{t[1][3]:.4f}\t{t[1][5]}\t{t[1][4]}\n" for t in todo if not prio(t)))
todo = rr([t for t in todo if prio(t)])                 # so o top-30% (os passes 2/4/6 e o 7 de fora nao rodam)
NPRIO = sum(1 for t in todo if prio(t))
(HERE / "results" / "fila_prioridade.txt").write_text("".join(f"{t[0]}\t{t[1][0]}\t{t[1][1]}\t{t[1][3]:.4f}\t{t[1][5]}\n" for t in todo[:NPRIO]))
print(f"[RUNNER] fila de PRIORIDADE {NPRIO}: " + str({b: n for b in B if (n := sum(1 for t in todo if prio(t) and t[1][0] == b))}), flush=True)
cnt = lambda ps: {f"{b}/{s}": n for b in B for s in SRC + XSRC + ["retest"]
                  if (n := sum(1 for t in todo if t[0] == ps and t[1][:2] == (b, s)))}
print(f"[RUNNER] {time.strftime('%d/%m %H:%M')} {len(todo)} modelos | "
      + " ".join(f"passe{k} {sum(1 for t in todo if t[0] == k)}" for k in (1, 2, 3, 4, 7, 5, 6))
      + "".join(f"\n  passe{k} {cnt(k)}" for k in (1, 2, 3, 4, 7, 5, 6) if cnt(k)), flush=True)
if a.dry_run:
    first = {}
    for ps, t in todo: first.setdefault((ps, t[0], t[1]), t)       # 1o de cada (passe, base, fonte)
    for (ps, *_), t in first.items(): print(f"  [{ps}] {t[0]:<15} {t[1]:<6} kval={t[3]:.4f} {t[5]} {t[4]}")
    raise SystemExit

# ── GPU pela memoria real
GPUS = [int(g) for g in a.gpus.split(",")]
def smi(q):
    return [l.split(", ") for l in subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines() if l.strip()]
def owner(pid):
    try: return os.stat(f"/proc/{pid}").st_uid
    except OSError: return None                            # invisivel = alheio
def gpu_state():
    g = {u: (int(i), int(fr)) for i, u, fr in smi("gpu=index,uuid,memory.free")}
    used, alien, ours = {}, set(), set()
    for u, pid, mem in smi("compute-apps=gpu_uuid,pid,used_memory"):
        if u not in g: continue
        used[int(pid)] = int(mem) / 1024
        if owner(int(pid)) != os.getuid(): alien.add(g[u][0])
        else:
            try:
                if b"readout_x/eval_backbone_head.py" in Path(f"/proc/{pid}/cmdline").read_bytes(): ours.add(int(pid))
            except OSError: pass
    return {i: fr / 1024 for i, fr in g.values()}, used, alien, ours   # ours: testes de QUALQUER runner

env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
           PYTHONWARNINGS="ignore", LD_LIBRARY_PATH=LIB, CUDA_DEVICE_ORDER="PCI_BUS_ID",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
class _Adopted:   # teste de um runner anterior ainda vivo (start_new_session): nao relanca, so espera e mescla
    def __init__(self, pid): self.pid, self.returncode = pid, None
    def poll(self):
        try:
            if b"eval_backbone_head.py" in Path(f"/proc/{self.pid}/cmdline").read_bytes(): return None
        except OSError: pass
        self.returncode = 0
        return 0
def _cmd(pid):
    try: return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError: return []
run, tries = [], {}
for d in Path("/proc").iterdir():                       # adota os testes vivos deste PARTS (relancamento)
    c = _cmd(d.name) if d.name.isdigit() and owner(d.name) == os.getuid() else []
    if b"--out" not in c or not any(b"eval_backbone_head.py" in x for x in c): continue
    ppid = Path(f"/proc/{d.name}/stat").read_text().rsplit(")", 1)[1].split()[1]
    if any(b"eval_backbone_head.py" in x for x in _cmd(ppid)): continue   # worker do DataLoader
    out = Path(c[c.index(b"--out") + 1].decode())
    if out.parent == PARTS and out.stem in META:
        todo = [t for t in todo if t[1][5] != out.stem]
        run.append({"p": _Adopted(int(d.name)), "i": out.stem, "fh": open(os.devnull, "w"), "t": time.time(),
                    "g": -1, "pk": 0.0, "job": None})
print(f"[RUNNER] {len(run)} testes vivos adotados (nao relancados)", flush=True)
_jc = {}
def jaci_md5(kind):  # o que a Jaci ja mediu (mirror relido a cada 10 min): "tta" = so2_tta8 no 100%;
    c = _jc.setdefault(kind, {"t": 0.0, "s": set()})   # "pct5"/"pct6" = % completo na Jaci (ids _p1.._p75; p5: com tta8)
    if time.time() - c["t"] > 600:
        pp = {}
        for r in jaci_rows():
            m = re.search(r"_p(\d+)$", r.get("backbone_id", ""))
            if kind == "tta" and not m and r.get("head") == TTA: pp[r.get("md5")] = None
            elif kind != "tta" and m and (kind == "pct6" or r.get("head") == TTA):
                pp.setdefault(r.get("md5"), set()).add(int(m.group(1)))
        c["t"], c["s"] = time.time(), {k for k, v in pp.items() if v is None or {1, 5, 10, 25, 50, 75} <= v}
    return c["s"]
# pico reservado por fonte (19/11): medido no log (pico_smi p95: ga/pnas/a2 <= 4,8 GB; growth 10,6; spifil_init ate 10);
# OOM refila ate 3 vezes, entao subestimar e recuperavel. --peak-gb fica para o que nao tem medida (hybrid/ssl).
need = lambda job: (0.0 if job[1][6] == "fm" else 10.0 if job[0] in (3, 4) else 11.0 if job[1][6] == "growth"
                    else 5.0 if job[1][6] == "modehb" else a.peak_gb)
while todo or run:
    free, used, alien, ours = gpu_state()
    ok = [g for g in GPUS if g in free and not (g in (0, 1) and g in alien)]
    pend = {g: sum(max(0.0, t.get("need", a.peak_gb) - used.get(t["p"].pid, 0.0)) for t in run         # reserva o pico so nos
                   if t["g"] == g and time.time() - t["t"] < 180) for g in ok}                # 3 min iniciais
    for g in ok * 99:                                     # um por GPU por volta (espalha a carga)
        # o 1o da fila que CABE nesta GPU (entre os 30 primeiros): um crescimento de 11 GB nao trava os de 5 GB atras dele
        j = next((j for j, t in enumerate(todo[:30]) if free[g] - pend[g] >= need(t) + a.margin_gb), None)
        if j is not None and len(ours | {t["p"].pid for t in run}) < a.jobs \
                and os.getloadavg()[0] < 0.72 * os.cpu_count():   # 19/11: SVM na CPU -> ~28% de folga p/ os outros usuarios
            ps, (b, s, fid, kv, ck, i, kind, code) = todo.pop(j)
            if ps in (1, 2, 5, 6, 7) and kind != "fm":   # 1/2: a Jaci pode ter terminado com tta8 depois do inicio
                with open(ck, "rb") as fh: md5 = hashlib.file_digest(fh, "md5").hexdigest()
                if md5 in jaci_md5({1: "tta", 2: "tta", 7: "tta", 5: "pct5", 6: "pct6"}[ps]):
                    print(f"[SKIP-JACI] {i} (md5 {md5[:8]} ja medido na Jaci: {'%' if ps in (5, 6) else 'tta8'})", flush=True)
                    continue
            part = PARTS / f"{i}.jsonl"
            ev = [str(HERE / "run_ssl_pequeno_wrapper.py")] * (kind == "hybrid") + [str(HERE / "eval_backbone_head.py")]
            cmd = ["nice", "-n", "15", PY, *ev, "--code-root", code,
                   "--item", f"id={i},kind={kind},ckpt={ck},src={s},fid={fid},kval={kv:.6f},pass={ps}"
                   + (",init=spifil,variant=spifil_init" if ps in (3, 4) or i.endswith("__spifil_init") else ""),
                   "--heads", *(["so2_4096_pca"] if ps == 7 else HEADS), *(["--tta8"] if ps != 6 else []),
                   "--device", "cpu" if kind == "fm" else "cuda",
                   *(["--train-pcts", *map(str, PCTS)] if ps in (5, 6) else []),
                   "--batch-size", "32", "--num-workers", "4", "--data-cache", str(REPO / "artifacts/readout_x_cache"),
                   "--out", str(part), "--table-md", os.devnull]
            fh = open(LOGS / f"{i}.log", "a")
            p = subprocess.Popen(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)       # fechar o tmux nao derruba teste em andamento
            run.append({"p": p, "i": i, "fh": fh, "t": time.time(), "g": g, "pk": 0.0, "job": (ps, (b, s, fid, kv, ck, i, kind, code)),
                        "need": need((ps, (b, s, fid, kv, ck, i, kind, code)))})
            pend[g] += run[-1]["need"]
            print(f"[START] {time.strftime('%d/%m %H:%M:%S')} gpu{g} p{ps} {i} (free {free[g]:.1f} GB, pend {pend[g]:.1f})", flush=True)
    time.sleep(10)
    _, used, _, _ = gpu_state()
    for t in list(run):
        t["pk"] = max(t["pk"], used.get(t["p"].pid, 0.0))
        if t["p"].poll() is None: continue
        t["fh"].close(); run.remove(t); merge(PARTS / f"{t['i']}.jsonl")
        rc = t["p"].returncode
        oom = rc != 0 and "out of memory" in (LOGS / f"{t['i']}.log").read_text()[-4000:].lower()
        oom = oom and tries.get(t["i"], 0) < 3
        if oom: tries[t["i"]] = tries.get(t["i"], 0) + 1; todo.insert(0, t["job"])
        print(f"[END] {time.strftime('%d/%m %H:%M')} {t['i']} rc={rc} gpu{t['g']} {(time.time() - t['t']) / 60:.1f} min "
              f"pico_smi={t['pk']:.2f} GB{' OOM->refila' if oom else ''}", flush=True)
print("[RUNNER] fim", flush=True)
