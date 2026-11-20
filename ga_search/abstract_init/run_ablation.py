"""Driver de baseline/ablação do Agente 2: treina e avalia GENOMAS FIXOS com o pipeline real do MO-DEHB.

    python -m ga_search.abstract_init.run_ablation --dataset remote_sensing --genome rs_b1 \
        --inits spifil,random --ae noae,joint,decoder_first --seeds 42,1042,2042 --fidelity 45 \
        [--out-root artifacts/a2_ablation] [--device cuda:0] [--cov-eigh-workers 1] [--dry-run]

Pipeline (o MESMO objeto da produção, ``ga_search.modehb.run_search.Evaluator``, sem nenhuma mudança):
init (por camada) → AE local (Fase B) → LeJEPA + SIGReg (Fase C) → WiSE-FT α ∈ {0, .25, .5, .75, 1} → SVM.

Braços
* ``--inits``: ``spifil`` (SPiFiL-0 de produção), ``random`` (gêmeo He com a topologia EFETIVA do SPiFiL do mesmo
  genoma/seed/AE — ``--random-mode twin``, padrão; ``requested`` = caminho aleatório da produção nos canais
  pedidos) e qualquer inicializador registrado em ``ga_search.abstract_init`` (``spifil0``, ``s0plus``, ``s0data``,
  ``spifil1`` …), com opções ``nome:chave=valor,chave=valor`` (ex.: ``s0plus:knn_smooth=5``).
* ``--ae``: ``noae`` (Fase B desligada, mesmas épocas totais no SSL = ``ae_mode off`` do PNAS-CS), ``joint``,
  ``decoder_first`` (``genome.ae_schedule``).
* ``--seeds``: seed do genoma (SPiFiL, AE, cabeça, SSL) e do embaralhamento do treino. O split, os doadores
  (pool) e o subconjunto de treino do SVM ficam FIXOS (seed do config = 42): é a régua.
* ``--fidelity``: épocas totais (45 = 15 AE/camada + 30 SSL; noae = 45 SSL).

Métricas (só VALIDAÇÃO; o teste nunca é carregado — ``progressive_data.load_data`` só devolve treino/val):
* ``kappa_val_ORIG``: embedding ``head(encoder(x))`` (dim ≥ 1280), ``core.svm.fit_svm`` linear C=100 SEM scaler,
  treino → validação (``supervised_fitness.svm_fitness``, kappa_weight 1).
* ``kappa_val_COV``: leitura ``cov_head`` (CovReadout k=64 + head) com StandardScaler, C=100 (a fitness atual).
* ``--wise-ruler orig`` (padrão, a fitness nova da missão) escolhe o α do WiSE pela ORIG; a COV é medida no α
  escolhido (e em todos os α com ``--other-ruler-all-alphas``). ``--wise-ruler cov`` inverte.

Saída: ``<out-root>/<dataset>/results.jsonl`` (1 linha por run) + ``candidates/<run>/f<F>/{encoder.pt,
result.json, ablation.json, val_embedding.pt}``. Retomada idempotente: um run com ``ablation.json`` é pulado; um
diretório de run sem ``ablation.json`` (run morto) é renomeado para ``*.interrompido_<ts>`` (nada é apagado) e o
run é refeito do zero com as mesmas seeds.

Threads: OPENBLAS_NUM_THREADS=2, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8 (definidas antes de importar numpy/torch;
``--keep-thread-env`` respeita as do ambiente). ``--cov-eigh-workers 1`` = eigh do cov_head numa stream só
(chamada histórica; para o torch 2.2 do athena).
"""
from __future__ import annotations

import os
import sys

if "--keep-thread-env" not in sys.argv:
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VERSION = "1.0.0"
DEFAULT_CONFIG = "ga_search/config/modehb/search_v3.yaml"
AE_ARMS = ("noae", "joint", "decoder_first")
RULERS = {
    "ORIG": "head(encoder(x)) dim>=1280; core.svm.fit_svm linear C=100 ovo max_iter=-1; scaler none; tta 1; "
            "train(stratified<=max_train)->val; kappa",
    "COV": "cov_head (CovReadout k=64 per block + head); StandardScaler; core.svm.fit_svm linear C=100; "
           "train(stratified<=max_train)->val; kappa",
}


# ─── plano ───────────────────────────────────────────────────────────────────

def parse_init(spec: str) -> tuple[str, dict]:
    name, _, opts = spec.partition(":")
    kw = {}
    for item in filter(None, opts.split(",")):
        k, _, v = item.partition("=")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return name.strip(), kw


def init_tag(spec: str) -> str:
    name, kw = parse_init(spec)
    return name + "".join(f"-{k}{str(v).replace('.', 'p')}" for k, v in sorted(kw.items()))


def run_name(genome: str, init: str, ae: str, seed: int, protect: str | None = None) -> str:
    """``protect`` só entra no nome quando sobrescreve o do genoma (nomes antigos continuam válidos)."""
    return f"{Path(genome).stem}__{init_tag(init)}__{ae}__s{int(seed)}" + (f"__p{protect}" if protect else "")


def build_plan(genomes, inits, aes, seeds, protect=None):
    """Ordem: por genoma, seed, AE — ``spifil`` antes de ``random`` (o gêmeo precisa do SPiFiL)."""
    order = sorted(inits, key=lambda s: (parse_init(s)[0] != "spifil", parse_init(s)[0] == "random"))
    return [dict(genome=g, init=i, ae=a, seed=int(s), protect=protect, name=run_name(g, i, a, s, protect))
            for g in genomes for s in seeds for a in aes for i in order]


# ─── utilidades ──────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str, sort_keys=True) + "\n"
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str, sort_keys=True))
    os.replace(tmp, path)


def _file_md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_generator(loader):
    dl = getattr(loader, "loader", loader)
    return getattr(dl, "generator", None)


# ─── réguas ──────────────────────────────────────────────────────────────────

def _ruler_cfg(ruler: str, base: dict) -> dict:
    sv = dict(base)
    if ruler == "orig":
        sv.update(readout="head", scaler="none", tta=1, whiten="none")
    elif ruler == "cov":
        sv.update(readout="cov_head", scaler="standard", tta=1, whiten="none")
    else:
        raise ValueError(ruler)
    sv.update(C=100, kappa_weight=1.0, solver="libsvm")
    return sv


def measure(ev, encoder, head, ruler: str) -> dict:
    """κ de validação numa régua, no encoder/head dados (mesmas funções do Evaluator; SVM = core.svm.fit_svm)."""
    from ga_search.modehb.run_search import _blas_threads
    from ga_search.modehb.supervised_fitness import svm_fitness
    sv = ev.cfg["eval"]["svm_fitness"]
    encoder.to(ev.device), head.to(ev.device)
    t0 = time.time()
    if ruler == "orig":
        tr = ev._embed(encoder, head, ev.svm_train_x, tta=1, readout="head").numpy()
        va = ev._embed(encoder, head, ev.val_x, tta=1, readout="head").numpy()
        scaler = "none"
    else:
        tr, va = ev._embed_cov(encoder, head)
        scaler = "standard"
    with _blas_threads(ev._fitness_speed()["svm_blas_threads"]):
        r = svm_fitness(tr, ev.svm_train_y.numpy(), va, ev.val_y.numpy(), C=100.0, scaler=scaler,
                        max_train=sv.get("max_train", 10000), kappa_weight=1.0, seed=ev.seed)
    r["dim"] = int(np.prod(tr.shape[1:]))
    r["seconds"] = time.time() - t0
    return r


def _blend(encoder, head, anchor, trained, a):
    for mod, key in ((encoder, "encoder"), (head, "head")):
        sd = {}
        for n, t in trained[key].items():
            a0 = anchor[key].get(n)
            sd[n] = ((1 - a) * a0.to(t.device) + a * t) if (a0 is not None and t.is_floating_point()
                                                               and a0.shape == t.shape) else t
        mod.load_state_dict(sd)


# convs suportados por métodos que não declaram ``supported_convs`` (o SPiFiL-3 v0.1 lança InfeasibleCandidate)
KNOWN_SUPPORTED = {"spifil3": ("conv", "pointwise"), "spifil3_rand": ("conv", "pointwise")}


def applicability(initializer, genome: dict) -> list[str]:
    """Camadas que o inicializador NÃO sabe inicializar (lista vazia = aplicável)."""
    from ga_search.modehb.spifil_ops import ops_repair
    conv_ok = getattr(initializer, "supported_convs", None) or KNOWN_SUPPORTED.get(initializer.name)
    bad = []
    for i, L in enumerate(genome["layers"][:int(genome["depth"])]):
        kind = ops_repair({k: v for k, v in L.items() if k != "active"}).get("conv", "conv")
        if conv_ok is not None and kind not in conv_ok:
            bad.append(f"layer{i}:{kind}")
    return bad


# ─── despachante dos inicializadores abstratos ───────────────────────────────

class _Dispatch:
    """Substitui ``ga_search.modehb.init.init_block`` SÓ durante um run de inicializador registrado.

    ``Evaluator._grow`` chama ``I.init_block(genome, i, prefix, donors, spifil_cfg, prefix_id)``; aqui isso vira
    ``initializer.init_block(genome, i, prefix, TrainData, InitContext)``. Fora do ``with`` a produção é a original.
    """

    def __init__(self, ev, initializer, budget=None):
        self.ev, self.init = ev, initializer
        self.budget = budget
        self.calls = 0

    def train_data(self, donors):
        from ga_search.abstract_init.base import TrainData
        ev = self.ev
        X, Y = ev.svm_train_x, ev.svm_train_y
        bs = int(ev.cfg["data"].get("batch_size", 64))
        lf = self.init.config.get("label_policy", self.init.label_policy) == "label_free"

        def loader_factory(seed: int):
            g = torch.Generator().manual_seed(int(seed))
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                yield X[j], (torch.full((len(j),), -1, dtype=torch.long) if lf else Y[j])
        fp = hashlib.sha256(f"{ev.split_hash}:svm_train_subset:{len(X)}:{int(ev.seed)}".encode()).hexdigest()
        return TrainData(donors=list(donors), donor_pool=ev.spifil_pool, loader_factory=loader_factory,
                         split_key="train", split_fingerprint=fp, n_classes=len(ev.spifil_pool),
                         image_size=int(ev.cfg["data"].get("image_size", 224)), n_images=int(len(X)),
                         label_policy=self.init.config.get("label_policy", self.init.label_policy),
                         mask_policy="no_mask" if lf else "mask")

    def __call__(self, genome, layer_idx, prefix, samples, config, prefix_id=None):
        from ga_search.abstract_init.base import InitBudget, InitContext
        ev = self.ev
        self.calls += 1
        ctx = InitContext(seed=int(genome.get("seed", 0)) + int(layer_idx), device=ev.device,
                          layer_idx=int(layer_idx), prefix_id=str(prefix_id), config=dict(self.init.config),
                          spifil_config=dict(config or {}), genome=genome,
                          diag_batch=ev.svm_train_x[-16:], amp_dtype=getattr(ev, "amp_dtype", None),
                          budget=self.budget or InitBudget())
        return self.init.init_block(genome, layer_idx, prefix, self.train_data(samples), ctx)

    @contextlib.contextmanager
    def active(self):
        import ga_search.modehb.init as I
        old = I.init_block
        I.init_block = self
        try:
            yield self
        finally:
            I.init_block = old


# ─── runner ──────────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out_root) / args.dataset
        self.jsonl = self.out / "results.jsonl"
        self.ev = None
        self.cfg = None

    # -- config / evaluator
    def build_cfg(self):
        from ga_search.modehb.run_search import load_config
        a = self.args
        cfg = load_config(a.config, a.dataset, None, a.set)
        cfg["output_dir"] = str(self.out)
        if a.device:
            cfg["device"] = a.device
        cfg["wandb"] = {**cfg.get("wandb", {}), "enabled": bool(a.wandb)}
        if a.gpu_cap_gb is not None:
            cfg["gpu_process_cap_gb"] = float(a.gpu_cap_gb)
        ev_cfg = cfg["eval"]
        if ev_cfg.get("fitness_mode") != "svm_kappa":
            raise ValueError("o runner exige eval.fitness_mode = svm_kappa (search_v3.yaml)")
        sv = _ruler_cfg(a.wise_ruler, ev_cfg.get("svm_fitness") or {})
        if a.cov_eigh_workers is not None:
            sv["cov_eigh_workers"] = int(a.cov_eigh_workers)
        ev_cfg["svm_fitness"] = sv
        # pool de doadores grande o bastante para o S0-data (aninhado: os k primeiros não mudam)
        n_cls = int(cfg["search"].get("n_classes") or 0)
        need = [1]
        for spec in a.inits:
            name, kw = parse_init(spec)
            if name == "s0data":
                from ga_search.abstract_init.spifil0 import Spifil0Data
                need.append(Spifil0Data(**kw).ipc_for(n_cls))
        ch = sorted(set(int(c) for c in cfg["search"].get("spifil_images_choices", [1, 2, 4, 8, 16, 32])) | set(need))
        cfg["search"]["spifil_images_choices"] = ch
        self.cfg = cfg
        return cfg

    def evaluator(self):
        if self.ev is not None:
            return self.ev
        from ga_search.modehb.run_search import Evaluator, _stratified_indices
        from torch.utils.data import DataLoader, TensorDataset
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config_resolved.yaml").write_text(__import__("yaml").safe_dump(self.cfg, sort_keys=False))
        _seed_everything(int(self.cfg["seed"]))
        ev = Evaluator(self.cfg, self.out)
        a = self.args
        if a.debug_train:          # SÓ TESTE: subconjunto estratificado do treino (SVM + AE/SSL)
            idx = torch.as_tensor(_stratified_indices(ev.svm_train_y.numpy(), int(a.debug_train), ev.seed))
            ev.svm_train_x, ev.svm_train_y = ev.svm_train_x[idx], ev.svm_train_y[idx]
            bs = int(self.cfg["data"].get("batch_size", 64))
            ds = TensorDataset(ev.svm_train_x)
            ev.train_loader = DataLoader(ds, batch_size=bs, shuffle=True,
                                         generator=torch.Generator().manual_seed(ev.seed),
                                         drop_last=len(ds) > bs)
        if a.debug_val:
            idx = torch.as_tensor(_stratified_indices(ev.val_y.numpy(), int(a.debug_val), ev.seed))
            ev.val_x, ev.val_y = ev.val_x[idx], ev.val_y[idx]
        self.ev = ev
        return ev

    # -- estado
    def cand_dir(self, name):
        return self.out / "candidates" / name / f"f{int(self.args.fidelity)}"

    def done(self, name) -> dict | None:
        p = self.cand_dir(name) / "ablation.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _quarantine_partial(self, name):
        d = self.cand_dir(name)
        if d.exists() and not (d / "ablation.json").exists():
            dst = d.with_name(d.name + f".interrompido_{time.strftime('%Y%m%d_%H%M%S')}")
            os.replace(d, dst)
            print(f"[a2] run interrompido anterior movido para {dst}", flush=True)

    # -- um run
    def genome_for(self, item):
        from ga_search.abstract_init.genomes import get_genome
        ev = self.evaluator()
        info = get_genome(item["genome"], self.args.dataset, n_classes=len(ev.spifil_pool),
                          choices=self.cfg["search"]["spifil_images_choices"],
                          pool_sizes=[len(v) for _, v in sorted(ev.spifil_pool.items())])
        g = copy.deepcopy(info["genome"])
        g.pop("external", None)
        g["seed"] = int(item["seed"])
        if item["ae"] != "noae":
            g["ae_schedule"] = item["ae"]
        if item.get("protect"):                 # braço registrado: sobrescreve o protect do genoma
            g["protect"] = item["protect"]
        return g, info

    def run_one(self, item):
        a, ev = self.args, self.evaluator()
        name = item["name"]
        genome, info = self.genome_for(item)
        iname, ikw = parse_init(item["init"])
        ctx = {"cand_id": name, "generation": 0, "role": "ablation", "parent_id": None, "rung": 0,
               "twin_of": None}
        if item["ae"] == "noae":
            ctx["ae_mode"] = "off"
        dispatch = None
        twin = None
        if iname == "spifil":
            genome["init"] = "spifil"
        elif iname == "random":
            genome["init"] = "random"
            if a.random_mode == "twin":
                src_name = run_name(item["genome"], "spifil", item["ae"], item["seed"], item.get("protect"))
                src = self.done(src_name)
                if not src or src.get("status") != "ok":
                    st = "skipped_no_source" if (src and src.get("status") in ("infeasible", "not_applicable")) \
                        else "skipped"         # "skipped" (fonte ainda não rodou) não é definitivo
                    return {"status": st, "reason": f"gêmeo sem fonte SPiFiL ok ({src_name})"}
                res = json.loads((self.cand_dir(src_name) / "result.json").read_text())
                ctx["twin_source"] = {"result": res}
                ctx["twin_of"] = src_name
                twin = src_name
        else:
            from ga_search.abstract_init.base import get_initializer
            initializer = get_initializer(iname, **ikw)
            genome["init"] = "spifil"
            if hasattr(initializer, "genome_overrides"):
                genome.update(initializer.genome_overrides(genome, len(ev.spifil_pool), ev.spifil_k_max))
            bad = applicability(initializer, genome)
            if bad:
                return {"status": "not_applicable", "reason": f"{iname} não inicializa {bad}"}
            from ga_search.abstract_init.base import InitBudget
            dispatch = _Dispatch(ev, initializer, InitBudget(max_seconds=float(a.init_max_seconds)))
        seed = int(item["seed"])
        _seed_everything(seed)
        gen = _loader_generator(ev.train_loader)
        if gen is not None:
            gen.manual_seed(seed)
        t0 = time.time()
        with (dispatch.active() if dispatch else contextlib.nullcontext()):
            result = ev(genome, float(a.fidelity), ctx)
        if dispatch is not None and dispatch.calls == 0:
            raise RuntimeError("inicializador abstrato nunca foi chamado (roteamento do _grow mudou?)")
        # ---- réguas
        sel = a.wise_ruler
        other = "cov" if sel == "orig" else "orig"
        ck = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        enc, head = ck["encoder"], ck["head"]
        m_other = measure(ev, enc, head, other)
        per_alpha_other = {}
        if a.other_ruler_all_alphas and ck.get("ssl_state") is not None and ck.get("anchor") is not None:
            for al in (ev.cfg["eval"]["svm_fitness"].get("wise_alphas") or []):
                e2, h2 = copy.deepcopy(enc), copy.deepcopy(head)
                _blend(e2, h2, ck["anchor"], ck["ssl_state"], float(al))
                per_alpha_other[str(float(al))] = measure(ev, e2, h2, other)["kappa"]
        with torch.no_grad():                        # dimensão real do embedding ORIG (R11: >= 1280)
            orig_dim = int(head.to(ev.device).eval()(enc.to(ev.device).eval()(ev.val_x[:1].to(ev.device))).shape[-1])
        K = {sel.upper(): {"kappa": result.get("fitness_kappa"), "macro_f1": result.get("fitness_macro_f1"),
                           "balanced_acc": result.get("fitness_balanced_acc"),
                           "per_alpha": result.get("wise_kappas"), "selects_alpha": True},
             other.upper(): {"kappa": m_other["kappa"], "macro_f1": m_other["macro_f1"],
                             "balanced_acc": m_other["balanced_acc"], "dim": m_other["dim"],
                             "per_alpha": per_alpha_other or None, "selects_alpha": False,
                             "svm_seconds": m_other["seconds"]}}
        eff = result.get("effective_specs") or []
        rec = {
            "status": "ok", "run": name, "dataset": a.dataset, "genome": item["genome"],
            "genome_source": info.get("source"), "init": item["init"], "init_name": iname, "init_kwargs": ikw,
            "ae": item["ae"], "seed": seed, "fidelity": float(a.fidelity),
            "kappa_val_ORIG": K["ORIG"]["kappa"], "kappa_val_COV": K["COV"]["kappa"],
            "f1_val_ORIG": K["ORIG"]["macro_f1"], "f1_val_COV": K["COV"]["macro_f1"],
            "wise_ruler": sel.upper(), "wise_alpha": result.get("wise_alpha"),
            "kappa_val_ORIG_alpha0": (K["ORIG"]["per_alpha"] or {}).get("0.0") if sel == "orig"
            else per_alpha_other.get("0.0"),
            "rulers": K, "ruler_defs": RULERS, "test_read": False,
            "twin_of": twin, "random_mode": a.random_mode if iname == "random" else None,
            "params": result.get("params"), "flops": result.get("flops"),
            "embed_dim": int(genome.get("embed_dim", 1280)), "embed_dim_orig_measured": orig_dim,
            "widths_effective": [int(s.get("channels", 0)) for s in eff],
            "widths_requested": [int(L["channels"]) for L in genome["layers"][:genome["depth"]]],
            "spifil_images_per_class": result.get("spifil_images_per_class"),
            "spifil_n_images": result.get("spifil_n_images"), "spifil_fit_seconds": result.get("spifil_fit_seconds"),
            "ae_epochs_per_layer": result.get("ae_epochs_per_layer"), "ssl_epochs": result.get("ssl_epochs"),
            "ssl_final_loss": result.get("ssl_final_loss"), "protect": genome.get("protect"),
            "protect_override": item.get("protect"),
            "wall_hours": result.get("wall_hours"), "run_seconds": time.time() - t0,
            "peak_gpu_mem_gb": result.get("peak_gpu_mem_gb"), "peak_gpu_reserved_gb": result.get("peak_gpu_reserved_gb"),
            "checkpoint": result.get("checkpoint"), "data_sha256": result.get("data_sha256"),
            "n_train_svm": int(len(ev.svm_train_y)), "n_val": int(len(ev.val_y)),
            "debug_limits": {"train": a.debug_train, "val": a.debug_val} if (a.debug_train or a.debug_val) else None,
            "device": str(ev.device), "torch": torch.__version__, "runner_version": VERSION,
            "runner_md5": _file_md5(__file__), "config": a.config, "overrides": list(a.set),
            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "cov_eigh_workers": ev.cfg["eval"]["svm_fitness"].get("cov_eigh_workers"),
            "init_provenance": [{k: lg.get("prov", {}).get(k) for k in ("origin", "method", "abstract_init",
                                                                        "labels_used", "requested_channels",
                                                                        "actual_channels")}
                                for lg in result.get("layers", [])],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return rec

    def execute(self, item):
        from flim.progressive import InfeasibleCandidate
        from ga_search.modehb.modehb import _is_oom
        a = self.args
        name = item["name"]
        for attempt in range(int(a.oom_retries) + 1):
            self._quarantine_partial(name)
            try:
                rec = self.run_one(item)
                break
            except InfeasibleCandidate as exc:
                st = "not_applicable" if "unsupported" in str(exc) else "infeasible"
                rec = {"status": st, "error": f"{type(exc).__name__}: {exc}",
                       "details": getattr(exc, "details", None)}
                break
            except Exception as exc:  # noqa: BLE001
                if _is_oom(exc) and attempt < int(a.oom_retries):
                    print(f"[a2] OOM em {name} ({exc}); retry {attempt + 1}/{a.oom_retries} em {a.oom_sleep}s",
                          flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(float(a.oom_sleep))
                    continue
                rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
                break
        rec = {"run": name, "dataset": a.dataset, "genome": item["genome"], "init": item["init"],
               "ae": item["ae"], "seed": item["seed"], "fidelity": float(a.fidelity), **rec}
        d = self.cand_dir(name)
        if rec["status"] in ("ok", "infeasible", "not_applicable", "skipped_no_source"):   # definitivos
            d.mkdir(parents=True, exist_ok=True)
            _atomic_json(d / "ablation.json", rec)
        _append_jsonl(self.jsonl, rec)
        print(f"[a2] {name}: {rec['status']} ORIG={rec.get('kappa_val_ORIG')} COV={rec.get('kappa_val_COV')} "
              f"alpha={rec.get('wise_alpha')}", flush=True)
        return rec

    def main(self):
        a = self.args
        plan = build_plan(a.genomes, a.inits, a.ae, a.seeds, a.protect)
        self.build_cfg()
        todo = [it for it in plan if not self.done(it["name"])]
        print(f"[a2] {a.dataset}: {len(plan)} runs no plano, {len(plan) - len(todo)} feitos, {len(todo)} a fazer "
              f"(fidelidade {a.fidelity}, régua do WiSE {a.wise_ruler.upper()}, saída {self.out})", flush=True)
        if a.dry_run:
            for it in plan:
                print(("  feito  " if self.done(it["name"]) else "  a fazer ") + it["name"])
            return 0
        n_fail = 0
        for it in todo:
            if self.done(it["name"]):
                continue
            rec = self.execute(it)
            n_fail += rec["status"] == "failed"
        summarize(self.jsonl)
        return 1 if n_fail else 0


def summarize(jsonl: Path):
    """Média ± desvio por braço (último registro de cada run com status ok)."""
    if not jsonl.exists():
        return
    last = {}
    for line in jsonl.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        last[r["run"]] = r
    arms = {}
    for r in last.values():
        if r.get("status") != "ok":
            continue
        arms.setdefault((r["genome"], r["init"], r["ae"], r["fidelity"]), []).append(r)
    if not arms:
        return
    print(f"{'genoma':<12} {'init':<16} {'ae':<14} {'F':>4} {'n':>2}  {'κ ORIG':>15}  {'κ COV':>15}")
    for k in sorted(arms):
        rs = arms[k]
        o = np.array([r["kappa_val_ORIG"] for r in rs], float)
        c = np.array([r["kappa_val_COV"] for r in rs], float)
        print(f"{k[0]:<12} {k[1]:<16} {k[2]:<14} {k[3]:>4g} {len(rs):>2}  {o.mean():.4f}±{o.std(ddof=0):.4f}  "
              f"{c.mean():.4f}±{c.std(ddof=0):.4f}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--genome", dest="genomes", default="rs_b1",
                    help="lista separada por vírgula: rs_b1, rs_b1_band, flowers_b1, seed_small ou caminho .json")
    ap.add_argument("--inits", default="spifil,random")
    ap.add_argument("--ae", default="noae,joint,decoder_first")
    ap.add_argument("--seeds", default="42,1042,2042")
    ap.add_argument("--fidelity", type=float, default=45)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--set", action="append", default=[], metavar="a.b=VALUE",
                    help="override pontuado aplicado ao config (como run_search)")
    ap.add_argument("--out-root", default="artifacts/a2_ablation")
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (padrão: o do config/auto)")
    ap.add_argument("--gpu-cap-gb", type=float, default=None)
    ap.add_argument("--wise-ruler", choices=("orig", "cov"), default="orig")
    ap.add_argument("--other-ruler-all-alphas", action="store_true",
                    help="mede a outra régua em todos os α do WiSE (custo extra; padrão só no α escolhido)")
    ap.add_argument("--protect", choices=("full", "lowlr", "freeze"), default=None,
                    help="sobrescreve genome.protect (escala do lr das convs no AE e no SSL); padrão = o do genoma. "
                         "Com 'freeze' (rs_b1/flowers_b1) as convs SPiFiL não mudam no AE nem no SSL.")
    ap.add_argument("--init-max-seconds", type=float, default=1800.0,
                    help="InitBudget.max_seconds por camada dos inicializadores abstratos (padrão da interface: 600)")
    ap.add_argument("--random-mode", choices=("twin", "requested"), default="twin")
    ap.add_argument("--cov-eigh-workers", type=int, default=None, help="1 = chamada histórica (torch 2.2/athena)")
    ap.add_argument("--oom-retries", type=int, default=10)
    ap.add_argument("--oom-sleep", type=float, default=300)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--keep-thread-env", action="store_true")
    ap.add_argument("--debug-train", type=int, default=None, help="SÓ TESTE: nº de imagens de treino (estratificado)")
    ap.add_argument("--debug-val", type=int, default=None, help="SÓ TESTE: nº de imagens de validação")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true", help="só imprime o resumo do results.jsonl")
    args = ap.parse_args(argv)
    args.genomes = [g for g in args.genomes.split(",") if g]
    args.inits = [i for i in _split_inits(args.inits) if i]
    args.ae = [x for x in args.ae.split(",") if x]
    bad = [x for x in args.ae if x not in AE_ARMS]
    if bad:
        ap.error(f"--ae inválido: {bad} (use {AE_ARMS})")
    args.seeds = [int(s) for s in args.seeds.split(",") if s]
    return args


def _split_inits(s: str):
    """Vírgula separa inits, exceto dentro das opções ``nome:k=v,k=v`` (uma opção tem '=' e nenhum ':')."""
    out = []
    for tok in s.split(","):
        if out and "=" in tok and ":" not in tok and ":" in out[-1]:
            out[-1] += "," + tok
        else:
            out.append(tok.strip())
    return out


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    r = Runner(args)
    if args.summary:
        summarize(r.jsonl)
        return 0
    return r.main()


if __name__ == "__main__":
    sys.exit(main())
