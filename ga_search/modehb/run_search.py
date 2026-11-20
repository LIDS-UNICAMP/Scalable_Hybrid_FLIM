"""Entry point of the progressive MO-DEHB NAS.

    python -m ga_search.modehb.run_search --config ga_search/config/modehb/<name>.yaml [--resume]

The YAML (owned by I11 under ``ga_search/config/modehb/``) is the only input. Sections
(all optional, defaults in ``DEFAULTS``):

* top level: ``output_dir``, ``seed``, ``device`` (auto|cpu|cuda[:i]), ``dataset_name``.
* ``data``: forwarded to ``ga_search.progressive_data.load_data`` (dataset, split,
  percentage, image_size, batch_size, num_workers, synthetic, n_images, one_per_class...)
  plus ``ram_cache`` (preload train/val tensors once), ``val_max_samples``, ``drop_last``.
* ``search``: ``modehb.DEFAULTS`` (fidelities, eta, epsilon, spifil_fraction, ...).
* ``train``: ``ae_ssl_split`` [1,2], ``lr_ae``, ``lr_old``, ``lr_new``, ``sigreg_lambda``,
  ``max_steps_per_epoch``, ``softhebb`` {epochs, lr, temperature}, ``zca_fit_batches``.
* ``spifil``: forwarded to ``init.init_block`` / ``flim.progressive.fit_new_block``.
  SPiFiL donors = first ``genome["spifil_images_per_class"]`` (default 1) training images
  per class of a fixed pool (``build_spifil_pool``, max of ``search.spifil_images_choices``).
* ``eval``: ``n_views`` (TTA views, 1|4|8), ``geometry``, ``max_samples``,
  ``collapse_total_multiplier`` (0.0), ``collapse_dimensional_multiplier`` (0.5),
  ``embedding_eval`` (EmbeddingEvalConfig fields).
* ``cost``: ``latency_batch_size``, ``latency_warmup``, ``latency_repeats``.
* ``wandb``: ``enabled``, ``project``, ``entity``, ``mode``.
* ``gpu_process_cap_gb``: CUDA allocator cap for this process (shared GPU); a candidate
  that needs more fails with OOM and is recorded as infeasible.

The I11 schema (``search.yaml`` with ``dataset_config_dir``) is translated by
``from_search_yaml`` and needs ``--dataset``; ``--seed`` and ``--set a.b=VALUE`` override.

Fidelity semantics: a fidelity of F epochs = ``ae = round(F*a/(a+b))`` local-AE epochs
per layer (Fase B) + ``ssl = F - ae`` global SSL epochs (Fase C), ``[a,b] = ae_ssl_split``.
A promotion (same genome, same seed, lower-fidelity checkpoint exists) resumes that
checkpoint and spends the remaining ``F_hi - F_lo`` epochs on global SSL (layers are not
re-grown). Twins replay the SPiFiL candidate's path, so budgets match exactly.
Labels are never read: Fase D uses only the fixed, unlabeled validation images; the only
labelled data used is SPiFiL's donor set (training split), which SPiFiL requires.
Exception (opt-in, ``eval.fitness_mode: svm_kappa`` / search.yaml ``fitness_mode``): the
quality is the supervised validation score of a linear SVM fit on train embeddings
(``supervised_fitness.svm_fitness``, quality = w*kappa + (1-w)*macro-F1); the label-free
EmbeddingEvaluator metrics are still computed and logged as diagnostics. Test split untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import random
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ga_search.modehb import modehb as M

DEFAULTS = dict(
    output_dir="artifacts/modehb/run", seed=0, device="auto", dataset_name=None,
    data=dict(dataset="helminth-eggs", split=1, percentage=100, image_size=200, batch_size=32,
              num_workers=4, ram_cache=True, val_max_samples=512, drop_last=True),
    search={},
    train=dict(ae_ssl_split=[1, 2], lr_ae=1e-3, lr_old=1e-4, lr_new=1e-3, sigreg_lambda=0.05,
               max_steps_per_epoch=None, zca_fit_batches=8,
               softhebb=dict(epochs=1, lr=0.01, temperature=1.0)),
    spifil={},
    eval=dict(n_views=4, geometry="rot_flip", collapse_total_multiplier=0.0,
              collapse_dimensional_multiplier=0.5, embedding_eval={}),
    cost=dict(latency_batch_size=1, latency_warmup=2, latency_repeats=5),
    wandb=dict(enabled=False, project=None, entity=None, mode=None),
)


SPIFIL_IMAGES_CHOICES = [1, 2, 4, 8, 16]


def build_spifil_pool(selected, train, k_max, seed):
    """Per-class SPiFiL donor pool (training split only), ``{label: [Sample, ...]}``.

    Each class starts with the sample(s) ``progressive_data`` already selected for it (so
    k=1 reproduces the previous single-donor protocol), followed by the remaining training
    images of that class in a permutation seeded by ``(seed, label)``; truncated to
    ``k_max``. Taking the first k per class gives nested subsets (k=2 within k=4 ...).
    Only classes present in ``selected`` are used (respects ``paper_classes``).
    """
    train = list(train)
    train_paths = {s.image_path.resolve() for s in train}
    if not {s.image_path.resolve() for s in selected} <= train_paths:
        raise ValueError("SPiFiL donors must be contained exclusively in training")
    pool = {}
    for label in sorted({int(s.label) for s in selected}):
        first = [s for s in selected if int(s.label) == label]
        names = {s.name for s in first}
        rest = sorted((s for s in train if int(s.label) == label and s.name not in names),
                      key=lambda s: s.name)
        random.Random(f"spifil_pool:{int(seed)}:{label}").shuffle(rest)
        pool[label] = (first + rest)[: max(int(k_max), 1)]
    return pool


def spifil_subset(pool, k):
    """First ``k`` donors of every class (fewer when a class has fewer training images)."""
    return [s for label in sorted(pool) for s in pool[label][: max(int(k), 1)]]


def _measured(v, default):
    """search.yaml placeholders ("MEASURED_IN_Q1") fall back to ``default``."""
    return default if v is None or isinstance(v, str) else v


def from_search_yaml(raw: dict, dataset: str, seed=None) -> dict:
    """Translate the I11 schema (``search.yaml`` + ``datasets/<name>.yaml``) into this
    module's schema. Selected by the presence of ``dataset_config_dir`` in the YAML."""
    if dataset is None:
        raise ValueError("search.yaml schema needs --dataset (one of %s)" % raw.get("datasets"))
    seed = int(raw.get("seed", 42) if seed is None else seed)
    ds_cfg = yaml.safe_load((Path(raw["dataset_config_dir"]) / f"{dataset}.yaml").read_text())
    tr, pop, res = raw.get("training", {}), raw.get("population", {}), raw.get("resources", {})
    size = int(raw.get("image_size", ds_cfg.get("image_size", 224)))
    n_train = int(ds_cfg.get("expected_counts", {}).get("train", 0))
    cache_gb = _measured(res.get("ram_cache_gb"), 0)
    est_gb = n_train * 3 * size * size * 4 / 1e9          # float32 IFT-LAB tensors
    col = raw.get("collapse", {})
    split = raw.get("fidelity_split", {"ae": 1, "ssl": 2})
    out = Path(raw["paths"]["state_dir"].format(dataset=dataset)) / f"seed{seed}"
    return {
        "output_dir": str(out), "seed": seed, "dataset_name": dataset,
        "data": {"dataset": dataset, "dataset_config": ds_cfg, "split": int(raw.get("split", 1)),
                 "percentage": int((ds_cfg.get("pcts") or [100])[0]), "image_size": size,
                 "batch_size": int(tr.get("batch_size", 64)),
                 "num_workers": int(_measured(res.get("num_workers"), 4)),
                 "ram_cache": bool(cache_gb) and est_gb <= float(cache_gb),
                 # whole validation split: val is name-sorted (= class-sorted), so the 512 default
                 # keeps only the first classes (coconut_trees: class 1 only -> kappa 0 for all)
                 "val_max_samples": raw.get("val_max_samples")},
        "search": {"fidelities": raw["fidelities"], "eta": raw.get("eta", 3),
                   "n_classes": int(ds_cfg.get("n_classes") or 0),
                   "max_brackets": int(raw.get("max_brackets", 6)),
                   "epsilon": pop.get("exploration_epsilon", 0.1),
                   "spifil_fraction": pop.get("spifil_offspring_share", 0.75),
                   "mutation_factor": pop.get("de", {}).get("mutation_factor", 0.5),
                   "crossover_prob": pop.get("de", {}).get("crossover_prob", 0.5),
                   "initial_channels": pop.get("initial_grid", {}).get("filters", [16, 32, 64, 128]),
                   "initial_depths": pop.get("initial_grid", {}).get("depth", [2, 3]),
                   "initial_pyramid_growth": float(pop.get("initial_grid", {}).get("pyramid_growth", 1.0)),
                   "initial_ssl": tr.get("ssl", "lejepa"),
                   "spifil_images_choices": [int(k) for k in
                                             raw.get("spifil_images_choices", SPIFIL_IMAGES_CHOICES)],
                   **({"gpu_hours_budget": raw["gpu_hours_budget"]} if raw.get("gpu_hours_budget") else {}),
                   # opt-in integer channel space (search_v3.yaml); absent -> discrete grid
                   **({"channel_space": raw["channel_space"]} if raw.get("channel_space") else {}),
                   **_search_v3_keys(raw, pop),
                   # opt-in PNAS-CS (pnas_cs.yaml): compound-scaled families + growth by morphisms
                   **({"pnas_cs": {**raw["pnas_cs"], "dataset": dataset}}
                      if (raw.get("pnas_cs") or {}).get("enabled") else {}),
                   # opt-in external pretrained seed (external_seed.py); per-dataset paths resolved here
                   **({"external_seed": _external_seed_cfg(raw, dataset, ds_cfg)}
                      if (raw.get("external_seed") or {}).get("enabled") else {})},
        "train": {"ae_ssl_split": [split.get("ae", 1), split.get("ssl", 2)],
                  **({"max_images_per_epoch": int(tr["max_images_per_epoch"])} if tr.get("max_images_per_epoch") else {}),
                  # opt-in mixed precision of the training phases (pnas_cs.yaml: bf16); absent -> fp32
                  **({"amp_dtype": str(tr["amp_dtype"])} if tr.get("amp_dtype") else {}),
                  "lr_old": tr.get("lr_old", 1e-4), "lr_new": tr.get("lr_new", 1e-3)},
        # the fitness (kNN view consistency, clustering stability) compares 2 views, so the
        # evaluation needs >= 2 views (tta_transforms accepts 1|4|8 -> 4) although TTA view
        # averaging stays at tta.search=1; the quality reads views 0 and 1 only.
        "eval": {"n_views": 4 if int(raw.get("tta", {}).get("search", 1)) < 4 else int(raw["tta"]["search"]),
                 "embedding_eval": {"knn_k": raw.get("fitness", {}).get("knn_k", 10),
                                    "tau_total": float(col.get("min_embedding_std", 1e-3)) ** 2,
                                    "tau_dimensional": col.get("min_effective_rank_frac", 0.05)}},
        "wandb": {"enabled": True, "project": raw.get("project"), "entity": raw.get("entity")},
        "gpu_process_cap_gb": _measured(res.get("gpu_process_cap_gb"), None),
        # opt-in supervised fitness (search_v3.yaml); absent -> label-free quality
        **({"eval_fitness": {"fitness_mode": raw["fitness_mode"],
                             "svm_fitness": dict(raw.get("svm_fitness") or {})}}
           if raw.get("fitness_mode") else {}),
        # opt-in extended SPiFiL operators (init.init_block -> spifil_ops); key absent when off
        **({"spifil": {"spifil_ops": True}} if raw.get("spifil_ops", False) else {}),
    }


def _external_seed_cfg(raw: dict, dataset: str, ds_cfg: dict) -> dict:
    from ga_search.modehb.external_seed import resolve_config
    es = dict(raw["external_seed"])
    d = ((raw.get("pnas_cs") or {}).get("external_seed_dirs") or {}).get(dataset)
    if d and (raw.get("pnas_cs") or {}).get("enabled"):   # PNAS-CS per-dataset seed directory (ham10000: train-only)
        es.update(arch=f"{d}/L0/architecture.json", checkpoint=f"{d}/lejepa/checkpoints/best_kappa.ckpt")
    return resolve_config(es, dataset, int(raw.get("split", 1)), int((ds_cfg.get("pcts") or [100])[0]))


def _search_v3_keys(raw: dict, pop: dict) -> dict:
    """Opt-in search_v3 switches; every key is absent unless set in the YAML (defaults kept).

    * ``population.random_baseline: false`` -> SPiFiL only (no twins / random offspring /
      random seeds); ``population.initial_size`` -> number of SPiFiL seeds.
    * ``population.initial_operators: sampled`` -> seeds' per-layer operators sampled.
    * ``operators_from_start: true`` (or ``stage1_brackets: N``) -> stage lock off from bracket N.
    * ``search_space_from_yaml: true`` -> per-gene vocabularies from ``stages.stage2.space``.
    * ``lemonade.enabled: true`` -> LEMONADE warm start + morphisms + inheritance (lemonade.py)."""
    out = {}
    if "random_baseline" in pop or "random_baseline" in raw:
        out["random_baseline"] = bool(pop.get("random_baseline", raw.get("random_baseline")))
        if not out["random_baseline"]:
            out["initial_size"] = int(pop.get("initial_size", 16))
    if pop.get("initial_operators"):
        out["initial_operators"] = str(pop["initial_operators"])
    if raw.get("operators_from_start"):
        out["stage1_brackets"] = 0
    elif raw.get("stage1_brackets") is not None:
        out["stage1_brackets"] = int(raw["stage1_brackets"])
    if (raw.get("stratified") or {}).get("enabled"):   # parameter-stratified MO-DEHB (strata.py)
        out["stratified"] = dict(raw["stratified"])
    if (raw.get("lemonade") or {}).get("enabled"):    # LEMONADE network morphisms (lemonade.py)
        out["lemonade"] = dict(raw["lemonade"])
    if raw.get("search_space_from_yaml"):
        space = (raw.get("stages", {}).get("stage2", {}) or {}).get("space")
        if space:
            out["search_space"] = dict(space)
    return out


def _set_dotted(cfg, item):
    key, _, val = item.partition("=")
    d = cfg
    *parents, last = key.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(val)


def load_config(path, dataset=None, seed=None, overrides=()) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("MO-DEHB config must be a YAML mapping")
    if "dataset_config_dir" in raw:
        raw = from_search_yaml(raw, dataset, seed)
        raw["eval"] = {**raw.get("eval", {}), **raw.pop("eval_fitness", {})}
    elif seed is not None:
        raw["seed"] = int(seed)
    for item in overrides or ():
        _set_dotted(raw, item)
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in raw.items():
        cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["search"] = {**cfg["search"], "seed": int(cfg["seed"]),
                     "image_size": int(cfg["data"].get("image_size", 200))}
    return cfg


def split_epochs(fidelity, split=(1, 2)):
    a, b = (float(x) for x in split)
    f = int(round(float(fidelity)))
    ae = max(1, int(round(f * a / (a + b))))
    return ae, max(1, f - ae)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class _NullRun:
    id = None

    def log(self, *a, **k):
        pass

    def alert(self, *a, **k):
        pass

    def log_checkpoint(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


class _MonotonicRun:
    """Proxy for a CandidateRun: Fase A/B (per layer) and Fase C each restart their own
    ``step`` at 0, which W&B would drop (steps must increase). Re-bases every restart onto
    the run's last step; same-step groupings are preserved."""

    def __init__(self, run):
        self._run, self._offset, self._last_in, self._last_out = run, 0, -1, -1

    def __getattr__(self, name):
        return getattr(self._run, name)

    def log(self, data, step=None, **kw):
        if step is None:
            return self._run.log(data, **kw)
        step = int(step)
        if step < self._last_in:                       # a new phase restarted its counter
            self._offset = self._last_out + 1 - step
        self._last_in = step
        out = max(step + self._offset, self._last_out)
        self._last_out = out
        return self._run.log(data, step=out, **kw)

    def log_now(self, data):
        """Log at the current last step + 1 (used for per-candidate summaries)."""
        return self.log(data, step=self._last_in + 1 if self._last_in >= 0 else 0)


class _Inference(nn.Module):
    """Adapter for ``progressive_cost.profile_model`` (expects ``inference_model()``)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def inference_model(self):
        return nn.Sequential(copy.deepcopy(self.encoder), copy.deepcopy(self.head))


def profile(encoder, head, cost_cfg, image_size):
    """profile_model (Conv2d/Linear MACs, latency) for 1280-d; same methodology otherwise."""
    from ga_search.progressive_cost import profile_model
    cfg = {**cost_cfg, "image_size": image_size, "in_channels": 3}
    wrapper = _Inference(encoder, head)
    if getattr(head, "nominal_dim", 1280) == 1280:
        return profile_model(wrapper, cfg)
    # profile_model hard-checks a 1280-d embedding; mirror its MAC/latency methodology.
    model = wrapper.inference_model().eval()
    dev = next(model.parameters()).device
    bs = int(cfg.get("latency_batch_size", 1))
    x = torch.zeros(bs, 3, image_size, image_size, device=dev)
    macs = 0

    def count(mod, inp, out):
        nonlocal macs
        if isinstance(mod, nn.Conv2d):
            macs += out.numel() * mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
        elif isinstance(mod, nn.Linear):
            macs += out.numel() * mod.in_features
    hs = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    times = []
    with torch.no_grad():
        for _ in range(int(cfg.get("latency_warmup", 2))):
            model(x)
        for _ in range(int(cfg.get("latency_repeats", 5))):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t) * 1000)
    params = sum(p.numel() for p in model.parameters())
    return {"inference_macs": int(macs // bs), "inference_parameters": params,
            "latency_ms_mean": float(np.mean(times)), "latency_ms_std": float(np.std(times)),
            "mac_methodology": "Conv2d/Linear kernel terms (profile_model methodology)"}


class _CappedLoader:
    """Iterates at most ``n_steps`` batches of a shuffled DataLoader per pass (one "epoch")."""

    def __init__(self, loader, n_steps):
        self.loader, self.n_steps = loader, int(n_steps)
        self.dataset = loader.dataset

    def __len__(self):
        return min(self.n_steps, len(self.loader))

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n_steps:
                break
            yield b


def _amp_dtype(name, device):
    """``train.amp_dtype`` -> torch dtype of the training autocast (None = fp32, the default).
    Only bf16 (no GradScaler needed); CPU runs and GPUs without bf16 stay fp32."""
    if not name or str(name).lower() in ("none", "fp32", "float32", "off"):
        return None
    if str(name).lower() not in ("bf16", "bfloat16"):
        raise ValueError(f"train.amp_dtype must be 'bf16' or 'none', got {name!r}")
    if device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        print("[run_search] WARNING: bf16 not supported on this GPU -> fp32", flush=True)
        return None
    return torch.bfloat16


def _amp_kw(ev):
    dt = getattr(ev, "amp_dtype", None)
    return {"amp_dtype": dt} if dt is not None else {}


class Evaluator:
    """``evaluate(genome, fidelity, ctx)`` for MODEHB: Fases A-D + cost + W&B."""

    def __init__(self, cfg, out_dir):
        self.cfg, self.out = cfg, Path(out_dir)
        self.device = torch.device(cfg["device"])
        self.seed = int(cfg["seed"])
        self.dataset = cfg.get("dataset_name") or str(cfg["data"].get("dataset", "data"))
        self._load_data()
        self.git = _git_commit()
        self.amp_dtype = _amp_dtype((self.cfg.get("train") or {}).get("amp_dtype"), self.device)
        cap = self.cfg.get("gpu_process_cap_gb")
        if cap and self.device.type == "cuda":     # shared GPU: OOM -> candidate infeasible, host safe
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) / total), idx)

    # ---- data
    def _load_data(self):
        from ga_search.progressive_data import load_data
        d = dict(self.cfg["data"])
        d.setdefault("seed", self.seed)
        train_loader, val_loader, selected, identity = load_data(
            {k: v for k, v in d.items() if k not in ("ram_cache", "val_max_samples", "drop_last")},
            self.out / "data")
        self.samples, self.identity = selected, identity
        choices = self.cfg.get("search", {}).get("spifil_images_choices") or SPIFIL_IMAGES_CHOICES
        self.spifil_k_max = max(int(k) for k in choices)
        self.spifil_pool = build_spifil_pool(selected, train_loader.dataset.samples,
                                             self.spifil_k_max, int(d["seed"]))
        self.split_hash = identity["sha256"]
        bs, nw = int(d.get("batch_size", 32)), int(d.get("num_workers", 0))
        vmax = d.get("val_max_samples")
        val, val_y = [], []
        for x, y, *_ in val_loader:                   # labels only used by fitness_mode=svm_kappa
            val.append(x), val_y.append(torch.as_tensor(y))
            if vmax and sum(v.shape[0] for v in val) >= int(vmax):
                break
        self.val_x = torch.cat(val)[: int(vmax) if vmax else None]
        self.val_y = torch.cat(val_y)[: int(vmax) if vmax else None]
        supervised = self.cfg.get("eval", {}).get("fitness_mode") == "svm_kappa"
        if d.get("ram_cache", True):
            xs, ys = [], []
            for x, y, *_ in train_loader:
                xs.append(x), ys.append(torch.as_tensor(y))
            xs, ys = torch.cat(xs), torch.cat(ys)
            ds, nw = TensorDataset(xs), 0
        else:
            ds, ys = _ImagesOnly(train_loader.dataset), None
        if supervised:
            self._cache_svm_train(train_loader, xs if ys is not None else None, ys)
        g = torch.Generator().manual_seed(self.seed)
        self.train_loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=nw,
                                       drop_last=bool(d.get("drop_last", True)) and len(ds) > bs,
                                       pin_memory=self.device.type == "cuda",
                                       persistent_workers=nw > 0)
        cap = (self.cfg.get("train") or {}).get("max_images_per_epoch")
        if cap and len(ds) > int(cap):   # "epoch" = at most ``cap`` random train images, so a
            self.train_loader = _CappedLoader(self.train_loader, max(1, int(cap) // bs))   # fidelity
            # costs the same on 682 or 26k-image datasets (AE, SoftHebb and SSL all see it)

    def _cache_svm_train(self, train_loader, xs=None, ys=None):
        """Stratified (per-class proportional, seeded) subset of <= max_train TRAIN images +
        labels for the supervised SVM fitness; cached once in RAM for all candidates."""
        mx = int((self.cfg["eval"].get("svm_fitness") or {}).get("max_train", 10000))
        if ys is None:
            ys = torch.as_tensor([int(s.label) for s in train_loader.dataset.samples])
        idx = _stratified_indices(ys.numpy(), mx, self.seed)
        if xs is None:
            ds = train_loader.dataset
            xs = torch.stack([ds[int(i)][0] for i in idx])
        else:
            xs = xs[torch.as_tensor(idx)]
        self.svm_train_x, self.svm_train_y = xs, ys[torch.as_tensor(idx)]

    @torch.no_grad()
    def _embed(self, encoder, head, x, tta: int = 1, readout: str = "head"):
        """Eval-mode embeddings of a RAM tensor, AMP on CUDA. ``tta`` > 1: mean over the
        rot/flip views of ``pooling_head.encode_tta`` (8 = eval/svm.py ``gem_tta``)."""
        from ga_search.modehb.pooling_head import encode_tta
        encoder.eval(), head.eval()
        bs = int(self.cfg["data"].get("batch_size", 32))
        # growth_fix4: ``embed_amp=False`` (atributo da instância) = leitura fp32; padrão histórico inalterado
        amp = self.device.type == "cuda" and bool(getattr(self, "embed_amp", True))
        out = []
        with torch.no_grad():
            for b in x.split(bs):
                with torch.autocast(self.device.type, enabled=amp, **_amp_kw(self)):
                    out.append(encode_tta(encoder, head, b.to(self.device, non_blocking=True), n=int(tta),
                                          readout=readout).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def _embed_cov(self, encoder, head, extra=None):
        """CovReadout fitted on train pixels (label-free) -> (train, val[, extra]) features."""
        from ga_search.modehb.pooling_head import CovReadout
        encoder.eval(), head.eval()
        k = int((self.cfg["eval"].get("svm_fitness") or {}).get("cov_k", 64))
        sp = self._fitness_speed()
        cr = CovReadout(k=k, seed=self.seed, eigh_device=sp["cov_eigh"],
                        eigh_workers=sp["cov_eigh_workers"]).fit(encoder, self.svm_train_x, self.device)
        bs = int(self.cfg["data"].get("batch_size", 32))
        emb = lambda X: cr.embed(encoder, head, X, bs, self.device).numpy()   # == cat of cr(batch), bitwise
        out = (emb(self.svm_train_x), emb(self.val_x))
        return out + (emb(extra),) if extra is not None else out

    def _fitness_speed(self, n_alphas: int = 1) -> dict:
        """Speed knobs of the SVM-fitness phase (``eval.svm_fitness``), none of which touches the SVM
        (core.svm.fit_svm: linear SVC, C, scaler, libsvm) or the features beyond fp rounding:

        * ``cov_eigh``: ``device`` (default) = per-image fp64 eigh of cov_head on the GPU, chunks run
          concurrently on ``cov_eigh_workers`` CUDA streams (``pooling_head._eigh_many``: the same
          cusolver call per matrix, bitwise identical to the historical single batched call, ~3.5x
          faster); ``cpu`` = opt-in CPU LAPACK (``_eigh_host``): faster still but only equal up to
          rounding, which the cov_head readout amplifies on rank-deficient blocks (kappa can move).
        * ``cov_eigh_workers``: threads/streams of that eigh (default 8; 1 = the historical call).
        * ``svm_workers``: WiSE alphas whose libsvm fits run concurrently (threads; libsvm releases the
          GIL) while the GPU extracts the next alpha's features (default min(#alphas, torch threads);
          1 = sequential). Each fit is the same deterministic computation, so the result is identical.
          ``svm_ram_gb`` (default 8) caps it further by RAM: ~3 float64 copies of n_train x D per fit
          (``_svm_workers_for``), so a 4000 x 18k cov_head never holds 5 x 1.7 GB at once.
        * ``svm_blas_threads``: BLAS threads during the fitness phase (default 1; null = leave the
          process cap). libsvm's kernel is a BLAS ddot that OpenBLAS splits over threads for D > 10000:
          one thread is faster (no spin/sync per kernel product) and makes the result independent of
          the host's thread count (the split changes the rounding).
        * ``gram_reduce`` (opt-in, default false; read by ``_svm_score``): same linear SVM fitted on
          exact train-span coordinates (``supervised_fitness.span_coordinates``) when D > n_train."""
        kw = (self.cfg["eval"].get("svm_fitness") or {})
        nthr = max(1, torch.get_num_threads())
        dev = str(kw.get("cov_eigh", "device")).lower()
        if dev not in ("cpu", "device", "gpu", "cuda"):
            raise ValueError(f"svm_fitness.cov_eigh must be cpu|device, got {dev!r}")
        bt = kw.get("svm_blas_threads", 1)
        return {"cov_eigh": "cpu" if dev == "cpu" else None,
                "cov_eigh_workers": max(1, int(kw.get("cov_eigh_workers") or 8)),
                "svm_workers": max(1, min(int(n_alphas), int(kw.get("svm_workers") or min(int(n_alphas), nthr)))),
                "svm_blas_threads": None if bt in (None, 0, "none", "null") else max(1, int(bt))}

    def _svm_workers_for(self, workers: int, tr) -> int:
        """``svm_workers`` capped so the concurrent fits fit in ``svm_ram_gb`` (default 8 GB)."""
        if workers <= 1:
            return 1
        ram = float((self.cfg["eval"].get("svm_fitness") or {}).get("svm_ram_gb") or 8.0)
        per_fit = 3 * 8 * int(tr.shape[0]) * int(np.prod(tr.shape[1:])) / 1e9 + 0.05
        return max(1, min(int(workers), int(ram // per_fit)))

    def _svm_features(self, encoder, head):
        """(train, val) features read by the SVM fitness (GPU part of ``_svm_quality``)."""
        kw = dict(self.cfg["eval"].get("svm_fitness") or {})
        tta = int(kw.get("tta", 1))          # readout of the hybrid SPiFiL recipe: tta 8 + whiten pw
        readout = str(kw.get("readout", "head"))
        if readout == "cov_head":            # second-order readout (pooling_head.CovReadout)
            return self._embed_cov(encoder, head)
        tr = self._embed(encoder, head, self.svm_train_x, tta=tta, readout=readout).numpy()
        va = self._embed(encoder, head, self.val_x, tta=tta, readout=readout).numpy()
        return tr, va

    def _svm_score(self, tr, va):
        """CPU part of ``_svm_quality``: the repo SVM (``supervised_fitness.svm_fitness``) on features."""
        from ga_search.modehb.supervised_fitness import svm_fitness
        kw = dict(self.cfg["eval"].get("svm_fitness") or {})
        kw = {k: kw[k] for k in ("C", "scaler", "max_train", "kappa_weight", "whiten", "gram_reduce") if k in kw}
        if "C" in kw:
            kw["C"] = float(kw["C"])
        if not kw.get("gram_reduce"):        # opt-in; absent -> the historical call, kwargs unchanged
            kw.pop("gram_reduce", None)
        return svm_fitness(tr, self.svm_train_y.numpy(), va, self.val_y.numpy(), seed=self.seed, **kw)

    def _svm_quality(self, encoder, head):
        with _blas_threads(self._fitness_speed()["svm_blas_threads"]):
            return self._svm_score(*self._svm_features(encoder, head))

    # ---- one evaluation
    def __call__(self, genome, fidelity, ctx):
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()                  # reserved memory must not pile up across candidates
            torch.cuda.reset_peak_memory_stats(self.device)
        self._peak = {"alloc": 0.0, "reserved": 0.0}
        cand_dir = self.out / "candidates" / ctx["cand_id"] / f"f{int(fidelity)}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        run = self._start_run(genome, fidelity, ctx)
        try:
            result = self._evaluate(genome, fidelity, ctx, cand_dir, run)
        except Exception as exc:
            import traceback
            (cand_dir / "error.txt").write_text(traceback.format_exc())   # full traceback per failure
            try:
                run.alert("evaluation failed", f"{type(exc).__name__}: {exc}", "ERROR")
            finally:
                run.finish(exit_code=1)
                if self.device.type == "cuda":     # e.g. OOM under the process cap -> infeasible
                    torch.cuda.empty_cache()
            raise
        hours = (time.time() - t0) / 3600.0
        result["wall_hours"] = hours
        if self.device.type == "cuda":
            self._sample_peak()
            result["peak_gpu_mem_gb"] = self._peak["alloc"]
            result["peak_gpu_reserved_gb"] = self._peak["reserved"]
        result["gpu_hours"] = hours if self.device.type == "cuda" else 0.0
        self._gpu_hours_cum = getattr(self, "_gpu_hours_cum", 0.0) + result["gpu_hours"]
        result["front_rank"] = ctx["front_rank"](result) if callable(ctx.get("front_rank")) else None
        summary = {"fitness/quality": result["quality"], "fitness/quality_raw": result["quality_raw"],
                   "fitness/collapse_multiplier": result["collapse_multiplier"],
                   **{f"fitness/{k}": result.get(f"fitness_{k}")
                      for k in ("kappa", "macro_f1", "balanced_acc", "svm_fit_s")},
                   "cost/flops": result["flops"], "cost/params": result["params"],
                   "cost/latency_ms": result["latency_ms"], "cost/gpu_hours": result["gpu_hours"],
                   "cost/gpu_hours_cum": self._gpu_hours_cum,
                   "cost/peak_gpu_mem_gb": result.get("peak_gpu_mem_gb"),
                   "cost/peak_gpu_reserved_gb": result.get("peak_gpu_reserved_gb"), "front_rank": result["front_rank"]}
        summary = {k: v for k, v in summary.items() if v is not None}
        (run.log_now if hasattr(run, "log_now") else run.log)(summary)
        if hasattr(run, "summary"):
            for k, v in summary.items():
                run.summary[k] = v
        meta = {"run_id": getattr(run, "id", None), "generation": ctx["generation"], "fidelity": fidelity,
                "front_rank": result["front_rank"], "cand_id": ctx["cand_id"], "role": ctx["role"],
                "twin_of": ctx.get("twin_of"), "arch_hash": M.arch_hash(genome), "init": genome["init"]}
        try:
            run.log_checkpoint(result["checkpoint"], meta)
        except Exception as exc:
            print(f"[run_search] artifact upload failed: {exc}")
        result["wandb_run_id"] = getattr(run, "id", None)
        (cand_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
        run.finish()
        return result

    def _sample_peak(self):
        """Accumulate the CUDA peak across the resets done by fit_new_block / profile_model."""
        if self.device.type != "cuda":
            return
        pk = getattr(self, "_peak", {"alloc": 0.0, "reserved": 0.0})
        pk["alloc"] = max(pk["alloc"], torch.cuda.max_memory_allocated(self.device) / 1e9)
        pk["reserved"] = max(pk["reserved"], torch.cuda.max_memory_reserved(self.device) / 1e9)
        self._peak = pk

    def _start_run(self, genome, fidelity, ctx):
        w = self.cfg["wandb"]
        if not w.get("enabled"):
            return _NullRun()
        from ga_search.modehb import wandb_logger as WL
        kw = {k: w[k] for k in ("project", "entity", "mode") if w.get(k)}
        return _MonotonicRun(WL.start_candidate(
            genome, self.dataset, fidelity, ctx["parent_id"], ctx["generation"], self.seed,
            self.split_hash, M.arch_hash(genome), self.git, cand_id=ctx["cand_id"], role=ctx["role"],
            twin_of=ctx.get("twin_of"), rung=ctx["rung"],
            spifil_ops=bool(self.cfg.get("spifil", {}).get("spifil_ops", False)), **kw))

    def spifil_samples(self, genome):
        """Donors for this genome: first ``spifil_images_per_class`` of each class in the pool."""
        k = min(int(genome.get("spifil_images_per_class", 1)), self.spifil_k_max)
        return k, spifil_subset(self.spifil_pool, k)

    def _evaluate(self, genome, fidelity, ctx, cand_dir, run):
        tr, ev = self.cfg["train"], self.cfg["eval"]
        k_img, donors = self.spifil_samples(genome)
        self._spifil_fit = {"seconds": 0.0, "peak_gb": 0.0}
        (run.log_now if hasattr(run, "log_now") else run.log)(
            {"spifil/images_per_class": k_img, "spifil/n_images": len(donors),
             "spifil/n_classes": len(self.spifil_pool)})
        if hasattr(run, "summary"):
            run.summary["spifil/images_per_class"] = k_img
            run.summary["spifil/n_images"] = len(donors)
        ae_ep, ssl_ep = split_epochs(fidelity, tr["ae_ssl_split"])
        ae_off = ctx.get("ae_mode") == "off"           # PNAS-CS Phase 0: no local AE, same total epochs
        if ae_off:
            ae_ep, ssl_ep = 0, max(1, int(round(float(fidelity))))
        seed = int(genome["seed"])
        resume = ctx.get("resume")
        layer_logs, specs = [], []
        self._fp_info = None                          # PNAS-CS report: step-0 preservation of _grow_fp
        self._fp_masks = None                         # function-preserving growth: inherited elements
        if resume:                                    # promotion: continue the lower-fidelity model
            ck = torch.load(resume["result"]["checkpoint"], map_location=self.device, weights_only=False)
            encoder, head = ck["encoder"], ck["head"]
            self._fp_masks = ck.get("protect_masks")
            anchor = ck.get("anchor")                 # WiSE-FT: SPiFiL(+AE) weights of this lineage
            if ck.get("ssl_state") is not None:       # continue the SSL trajectory, not the blend
                encoder.load_state_dict(ck["ssl_state"]["encoder"])
                head.load_state_dict(ck["ssl_state"]["head"])
            specs = resume["result"].get("effective_specs", [])
            ssl_epochs = max(1, int(round(float(fidelity) - float(resume["fidelity"]))))
            step_offset = int(resume["result"].get("ssl_steps", 0))
            new_names = []
        else:
            inh = ctx.get("inherit")
            if genome.get("external") and not inh:   # external pretrained seed (+ its growth grid)
                from ga_search.modehb.external_seed import external_inherit
                inh = external_inherit(self.cfg["search"]["external_seed"], genome, self.out)
            if inh and inh.get("layer_map") is not None:   # function-preserving (Net2Net) growth
                encoder, specs, layer_logs = self._grow_fp(genome, ctx, ae_ep, seed, run, inh)
            elif ctx.get("inherit"):                  # LEMONADE: parent's trained leading blocks
                encoder, specs, layer_logs = self._grow(genome, ctx, ae_ep, seed, run, inherit=ctx["inherit"])
            else:
                encoder, specs, layer_logs = self._grow(genome, ctx, ae_ep, seed, run,
                                                        **({"ae_off": True} if ae_off else {}))
            c_out = self._out_channels(encoder)
            from ga_search.modehb.pooling_head import EmbeddingHead
            torch.manual_seed(seed)
            head = EmbeddingHead(c_out, int(genome["embed_dim"]), genome["pooling"]).to(self.device)
            ssl_epochs, step_offset = ssl_ep, 0
            new_names = [f"{genome['depth'] - 1}."]   # last block + head get lr_new
            anchor = {"encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                      "head": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        from ga_search.modehb.ssl_trainers import train_global_ssl
        from ga_search.modehb.external_seed import ElementProtect
        masks = self._fp_masks                        # None: protect scales every conv (SPiFiL lineage)
        p_scale = M.PROTECT_LR_SCALE[genome.get("protect", "full")]
        with ElementProtect(encoder, masks, p_scale):   # fp growth: protect only the inherited elements
            ssl_log = train_global_ssl(encoder, head, self.train_loader, genome["ssl"], genome["reg"],
                                       ssl_epochs, float(tr["lr_old"]), float(tr["lr_new"]), new_names,
                                       self.device, run, seed, float(tr["sigreg_lambda"]),
                                       step_offset=step_offset, max_steps_per_epoch=tr.get("max_steps_per_epoch"),
                                       conv_lr_scale=1.0 if masks is not None else p_scale,
                                       projector=bool(genome.get("projector", False)), **_amp_kw(self))
        # ---- Hybrid SPiFiL / WiSE-FT: blend (1-a)*SPiFiL(+AE) + a*SSL weights, a chosen by the
        # fitness on validation. a=0 is the SPiFiL model itself, so SSL can only ADD to SPiFiL.
        wise = {}
        alphas = (ev.get("svm_fitness") or {}).get("wise_alphas")
        ssl_state = {"encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                     "head": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if alphas and anchor is not None and ev.get("fitness_mode") == "svm_kappa":
            wise = self._wise_select(encoder, head, anchor, ssl_state, [float(a) for a in alphas], run)
        # ---- Fase D: label-free embedding evaluation on the fixed validation images
        views = self._views(encoder, head)
        prev = self._load_emb(resume["result"].get("embedding_path")) if resume else None
        src = ctx.get("twin_source")
        other = self._load_emb(src["result"].get("embedding_path")) if src else None
        from ga_search.modehb.embedding_eval import EmbeddingEvaluator
        evaluator = EmbeddingEvaluator(ev.get("embedding_eval") or None)
        metrics = evaluator.evaluate(views, prev, other,
                                     structural_rank=min(head.out_dim, head.n_blocks * head.in_ch))
        q_raw = float(evaluator.quality(metrics))
        col = evaluator.collapse(metrics)
        fit_extra = {}
        if ev.get("fitness_mode") == "svm_kappa":     # supervised quality; label-free = diagnostics
            sv = wise.get("sv") or self._svm_quality(encoder, head)
            fit_extra = {"fitness_mode": "svm_kappa", "quality_label_free": q_raw,
                         "fitness_kappa": float(sv["kappa"]), "fitness_macro_f1": float(sv["macro_f1"]),
                         "fitness_balanced_acc": float(sv["balanced_acc"]),
                         "fitness_svm_fit_s": float(sv["fit_s"]),
                         "fitness_n_train_used": int(sv.get("n_train_used", 0))}
            q_raw = float(sv["quality"])
            (run.log_now if hasattr(run, "log_now") else run.log)(
                {"fitness/mode": "svm_kappa", "fitness/quality_label_free": fit_extra["quality_label_free"]})
            if hasattr(run, "summary"):
                run.summary["fitness/mode"] = "svm_kappa"
        mult = 1.0
        # svm_kappa: the label-free rank test is a diagnostic only, never halves the kappa
        # fitness (opt back in with svm_fitness.dimensional_collapse_penalty: true)
        penalize_dim = (ev.get("fitness_mode") != "svm_kappa"
                        or bool((ev.get("svm_fitness") or {}).get("dimensional_collapse_penalty", False)))
        if col.get("dimensional") and penalize_dim:
            mult = float(ev["collapse_dimensional_multiplier"])
        if col.get("total"):
            mult = float(ev["collapse_total_multiplier"])
        emb_log = {f"emb/{k}": v for k, v in metrics.items()
                   if isinstance(v, (int, float, bool)) and not isinstance(v, str)}
        emb_log["emb/spectrum_topk"] = metrics.get("spectrum_topk")
        (run.log_now if hasattr(run, "log_now") else run.log)(
            {k: (float(v) if isinstance(v, bool) else v) for k, v in emb_log.items() if v is not None})
        hist = {"loss": [h.get("ssl/loss_total", float("nan")) for h in ssl_log.get("history", [])][:-1]}
        try:
            evaluator.alerts({**metrics, "loss": ssl_log.get("final_loss_total")}, run, history=hist)
        except Exception:
            pass
        emb_path = cand_dir / "val_embedding.pt"
        torch.save(views[0].cpu(), emb_path)
        # ---- cost
        self._sample_peak()                           # profile_model resets the CUDA peak
        prof = profile(encoder, head, self.cfg["cost"], int(self.cfg["data"].get("image_size", 200)))
        ckpt = cand_dir / "encoder.pt"
        cpu = lambda d: {k: {n: t.detach().cpu() for n, t in v.items()} for k, v in d.items()} if d else None
        torch.save({"encoder": encoder.cpu(), "head": head.cpu(), "genome": genome, "fidelity": fidelity,
                    "effective_specs": specs, "cand_id": ctx["cand_id"],
                    "anchor": cpu(anchor), "ssl_state": cpu(ssl_state) if wise else None,
                    "wise_alpha": wise.get("alpha"),
                    **({"protect_masks": {n: m.cpu() for n, m in masks.items()}} if masks is not None else {})},
                   ckpt)
        encoder.to(self.device), head.to(self.device)
        return {"status": "ok", "quality": q_raw * mult, "quality_raw": q_raw, "collapse": col,
                "collapse_multiplier": mult, "flops": prof["inference_macs"],
                "params": prof.get("inference_parameters"), "latency_ms": prof["latency_ms_mean"],
                "checkpoint": str(ckpt), "embedding_path": str(emb_path), "effective_specs": specs,
                **fit_extra,
                **({"wise_alpha": wise["alpha"], "wise_kappas": wise["kappas"]} if wise else {}),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))},
                "layers": layer_logs, "ae_epochs_per_layer": ae_ep if not resume else 0,
                "ssl_epochs": ssl_epochs, "ssl_steps": step_offset + int(ssl_log.get("steps", 0)),
                "ssl_final_loss": ssl_log.get("final_loss_total"), "data_sha256": self.split_hash,
                "spifil_images_per_class": k_img, "spifil_n_images": len(donors),
                "spifil_sample_ids": [s.name for s in donors],
                "spifil_fit_seconds": self._spifil_fit["seconds"],
                "spifil_fit_peak_gb": self._spifil_fit["peak_gb"],
                # opt-in channel_space only: SPiFiL may build fewer channels than the genome asks
                **({"effective_monotonic": M.widths_monotonic(s.get("channels", 0) for s in specs)}
                   if self.cfg["search"].get("channel_space") else {}),
                # opt-in PNAS-CS only: encoder-only params, Phase-0 AE mode, bf16, fp-growth preservation
                **({"params_encoder": int(sum(p.numel() for p in encoder.parameters())),
                    "amp_dtype": str(getattr(self, "amp_dtype", None)), "ae_mode": "off" if ae_off else "on",
                    "pnas": _jsonable({"fp": self._fp_info} if self._fp_info else {})}
                   if self.cfg["search"].get("pnas_cs") else {})}

    def _wise_select(self, encoder, head, anchor, trained, alphas, run):
        """Evaluate the SVM fitness at each blend alpha; leave encoder/head at the best one."""
        def load(a):
            for mod, key in ((encoder, "encoder"), (head, "head")):
                sd = {}
                for n, t in trained[key].items():
                    a0 = anchor[key].get(n)
                    sd[n] = ((1 - a) * a0.to(t.device) + a * t) if (a0 is not None and t.is_floating_point()
                                                                       and a0.shape == t.shape) else t
                mod.load_state_dict(sd)
        sp = self._fitness_speed(len(alphas))
        workers, svs = sp["svm_workers"], []
        with _blas_threads(sp["svm_blas_threads"]):
            if workers <= 1:
                for a in alphas:
                    load(a)
                    svs.append(self._svm_score(*self._svm_features(encoder, head)))
            else:                            # GPU features of alpha i+1 overlap the libsvm fits of <= i
                from concurrent.futures import ThreadPoolExecutor
                futs = []
                with ThreadPoolExecutor(workers, thread_name_prefix="wise_svm") as ex:
                    try:
                        for a in alphas:
                            load(a)
                            tr, va = self._svm_features(encoder, head)
                            workers = self._svm_workers_for(workers, tr)   # RAM cap (fixed after alpha 0)
                            while sum(not f.done() for f in futs) >= workers:   # <= workers fits in flight
                                next(f for f in futs if not f.done()).result()
                            futs.append(ex.submit(self._svm_score, tr, va))
                            del tr, va
                        svs = [f.result() for f in futs]
                    except BaseException:
                        for f in futs:
                            f.cancel()
                        raise
        best, kappas = None, {}
        for a, sv in zip(alphas, svs):       # same order and tie rule as the sequential loop
            kappas[str(a)] = float(sv["kappa"])
            if best is None or sv["quality"] > best[1]["quality"]:
                best = (a, sv)
        load(best[0])
        (run.log_now if hasattr(run, "log_now") else run.log)(
            {"wise/alpha": best[0], **{f"wise/kappa_a{k}": v for k, v in kappas.items()}})
        return {"alpha": best[0], "kappas": kappas, "sv": best[1]}

    # ---- Fases A + (SoftHebb) + B, layer by layer
    def _grow(self, genome, ctx, ae_ep, seed, run, inherit=None, ae_off=False):
        """``inherit`` (opt-in, LEMONADE ``ctx["inherit"]``): the first ``inherit["n_blocks"]``
        blocks are the parent's TRAINED blocks (``lemonade.load_inherited_blocks``), not re-fit;
        only the following layers get SPiFiL init + local AE. None = original behaviour."""
        from ga_search.modehb import init as I
        from ga_search.modehb.local_ae import train_local_ae
        tr = self.cfg["train"]
        blocks, specs, logs = [], [], []
        cin = 3
        src = ctx.get("twin_source")
        inh_blocks, inh_specs = [], []
        if inherit and inherit.get("prebuilt") is not None:   # _grow_fp: function-preserving prefix
            inh_blocks, inh_specs = inherit["prebuilt"]
        elif inherit:
            from ga_search.modehb.lemonade import load_inherited_blocks
            inh_blocks, inh_specs = load_inherited_blocks(inherit, self.device)
            (run.log_now if hasattr(run, "log_now") else run.log)(
                {"lemonade/n_inherited_blocks": len(inh_blocks),
                 "lemonade/parent_fidelity": float(inherit.get("parent_fidelity", 0.0))})
        for i in range(genome["depth"]):
            L = {k: v for k, v in genome["layers"][i].items() if k != "active"}
            if i < len(inh_blocks):                   # inherited trained block: no SPiFiL, no AE
                block = inh_blocks[i]
                for p in block.parameters():
                    p.grad = None
                blocks.append(block)
                cin = self._out_channels(nn.Sequential(*blocks))
                specs.append(_jsonable(inh_specs[i]))
                logs.append({"layer": i, "prov": {"origin": "inherited",
                                                  "parent_cand_id": inherit.get("parent_cand_id"),
                                                  "parent_fidelity": inherit.get("parent_fidelity")}, "ae": {}})
                continue
            prefix = nn.Sequential(*blocks).to(self.device).eval()
            prefix.out_channels = cin
            prefix_id = M.arch_hash({**genome, "depth": i}) if i else "input"
            if inh_blocks:                            # provenance: prefix holds inherited weights
                prefix_id = f"{prefix_id}+inherit:{inherit.get('parent_cand_id')}@{inherit.get('parent_fidelity')}"
            spifil_cfg = {**self.cfg["spifil"], "device": str(self.device), "seed": seed,
                          "image_size": int(self.cfg["data"].get("image_size", 200))}
            spifil_cfg.pop("in_channels", None)
            if src is not None:                       # random twin of a SPiFiL candidate
                eff = src["result"]["effective_specs"][i]
                block = I.random_twin_block(eff, seed + i, in_channels=cin, device=self.device)
                prov = {"origin": "random_twin", "effective_spec": eff}
                post_norm = True
            elif genome["init"] == "spifil" or I.spifil_supported(L):
                self._sample_peak()                   # fit_new_block resets the CUDA peak
                k_img, donors = self.spifil_samples(genome)
                t_fit = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                block, prov = I.init_block(genome, i, prefix, donors, spifil_cfg, prefix_id)
                fit_s = time.perf_counter() - t_fit
                fit_gb = (torch.cuda.max_memory_allocated(self.device) / 1e9
                          if self.device.type == "cuda" else 0.0)
                prov = {**prov, "images_per_class": k_img, "n_images": len(donors)}
                fit = getattr(self, "_spifil_fit", {"seconds": 0.0, "peak_gb": 0.0})
                fit["seconds"] += fit_s
                fit["peak_gb"] = max(fit["peak_gb"], fit_gb)
                self._spifil_fit = fit
                (run.log_now if hasattr(run, "log_now") else run.log)(
                    {f"spifil/fit_seconds_layer{i}": fit_s, f"spifil/fit_peak_gb_layer{i}": fit_gb,
                     "spifil/fit_seconds_total": fit["seconds"], "spifil/fit_peak_gb": fit["peak_gb"]})
                eff = prov.get("effective_spec", I.to_block_spec(L))
                post_norm = True
            else:                                     # random lineage, operator outside SPiFiL
                from ga_search.modehb.ops_norm_act import ConvBlock
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + i)
                    block = ConvBlock({**L, "in_channels": cin})
                    for m in block.modules():
                        if isinstance(m, nn.Conv2d) and L["activation"] != "selu":
                            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                eff = {**L, "in_channels": cin, "builder": "ConvBlock"}
                prov = {"origin": "random_convblock"}
                post_norm = False
            block = block.to(self.device)
            if (post_norm and L["norm"] not in ("none", "batch")   # init.py: only BN is native
                    and not (isinstance(eff, dict) and eff.get("norm_in_block"))):  # spifil_ops
                from ga_search.modehb.ops_norm_act import make_norm
                c = self._out_channels(nn.Sequential(prefix, block), cin_check=False)
                block = nn.Sequential(block, make_norm(L["norm"], c)).to(self.device)
            if L.get("softhebb"):
                from ga_search.modehb.softhebb import fit_softhebb
                conv = next(m for m in block.modules() if isinstance(m, nn.Conv2d))
                sh = tr["softhebb"]
                fit_softhebb(conv, self.train_loader, int(sh["epochs"]), float(sh["lr"]),
                             float(sh["temperature"]), self.device, run, input_transform=prefix)
            self._fit_zca(prefix, block)
            self._log_init_diag(i, prefix, block, prov, run)
            if ae_off:                                 # PNAS-CS Phase 0 ablation arm "no AE"
                ae = {"skipped": "phase0_ae_off"}
            else:
                ae = train_local_ae(prefix, block, self.train_loader, genome["ae_target"], ae_ep,
                                    float(tr["lr_ae"]), self.device, run, seed + i,
                                    conv_lr_scale=M.PROTECT_LR_SCALE[genome.get("protect", "full")]
                                    if genome["init"] == "spifil" and not (inherit or {}).get("free_new") else 1.0,
                                    decoder_warmup_frac=M.AE_WARMUP_FRAC[genome.get("ae_schedule", "joint")],
                                    **_amp_kw(self))
            for p in block.parameters():               # becomes the frozen prefix of layer i+1
                p.grad = None
            blocks.append(block)
            cin = self._out_channels(nn.Sequential(*blocks))
            specs.append(_jsonable(eff))
            logs.append({"layer": i, "prov": _jsonable({k: v for k, v in prov.items()
                                                         if isinstance(v, (int, float, str, bool, list, dict))}),
                         "ae": _jsonable({k: v for k, v in ae.items() if not isinstance(v, torch.Tensor)})})
        encoder = nn.Sequential(*blocks).to(self.device)
        for p in encoder.parameters():
            p.requires_grad_(True)
        return encoder, specs, logs

    def _grow_fp(self, genome, ctx, ae_ep, seed, run, inherit):
        """Function-preserving growth from a parent checkpoint (external seed / LEMONADE with
        ``function_preserving``), external_seed.grow_blocks: (1) build every mappable layer (kept
        filters copied, NEW filters SPiFiL-fitted with the exact-width allocator, identity / zero
        residual layers) and log the step-0 preservation error; (2) local AE only on blocks that have
        new elements, the inherited ones moving at ``protect`` x lr (ElementProtect), new ones free;
        (3) the remaining layers (first unmappable one onwards) through ``_grow`` (SPiFiL + AE, free).
        Sets ``self._fp_masks`` (inherited elements, encoder parameter names) for SSL / checkpoint."""
        from ga_search.modehb import external_seed as ES
        from ga_search.modehb.local_ae import train_local_ae
        tr = self.cfg["train"]
        parents, _ = ES.load_parent_blocks(inherit["checkpoint"], self.device)
        with torch.no_grad():                         # ACTUAL widths (SPiFiL may have built fewer)
            h, pch = self.val_x[:2].to(self.device), []
            for b in parents:
                h = b.eval()(h)
                pch.append(int(h.shape[1]))
        layers = [{k: v for k, v in L.items() if k != "active"} for L in genome["layers"][:genome["depth"]]]
        k_img, donors = self.spifil_samples(genome)
        fit_s = [0.0]

        def new_filters(i, prefix_blocks, L, cin, n_new, kernel):
            prefix = nn.Sequential(*prefix_blocks).to(self.device).eval()
            prefix.out_channels = cin
            cfg = {**self.cfg["spifil"], "device": str(self.device), "seed": seed + i,
                   "image_size": int(self.cfg["data"].get("image_size", 200))}
            cfg.pop("in_channels", None)
            t = time.perf_counter()
            out = ES.fit_new_filters(prefix, donors, L, cin, n_new, kernel, cfg)
            fit_s[0] += time.perf_counter() - t
            return out
        g = ES.grow_blocks(parents, inherit["parent_layers"], layers, inherit["layer_map"],
                           new_filters=new_filters, device=self.device, parent_channels=pch)
        err = ES.preservation_error(parents, g, self.val_x[:8].to(self.device)) if g.blocks else None
        if err is not None:                           # relative to the parent's feature scale (PNAS-CS report)
            with torch.no_grad():
                ref = nn.Sequential(*parents[:g.last_src + 1]).eval()(self.val_x[:8].to(self.device))
            rel = float(err) / max(float(ref.abs().max()), 1e-12)
        else:
            rel = None
        self._fp_info = {"n_fp_layers": len(g.blocks), "depth": int(genome["depth"]), "cut": bool(g.cut),
                         "max_abs_diff": err, "rel_err": rel, "parent_cand_id": inherit.get("parent_cand_id"),
                         "parent_fidelity": inherit.get("parent_fidelity"),
                         "origins": [lg["prov"]["origin"] for lg in g.logs]}
        self._spifil_fit = {"seconds": fit_s[0], "peak_gb": 0.0}
        (run.log_now if hasattr(run, "log_now") else run.log)(
            {"growth/n_fp_layers": len(g.blocks), "growth/cut": float(g.cut),
             **({"growth/fp_max_abs_diff": err} if err is not None else {}),
             "growth/spifil_new_filters_s": fit_s[0]})
        scale = M.PROTECT_LR_SCALE[genome.get("protect", "full")]
        for i, (blk, mk) in enumerate(zip(g.blocks, g.masks)):
            n_new = g.logs[i]["prov"]["n_new_elements"]
            g.logs[i]["prov"].update(parent_cand_id=inherit.get("parent_cand_id"),
                                     parent_fidelity=inherit.get("parent_fidelity"),
                                     **({"fp_max_abs_diff": err} if err is not None else {}))
            if not n_new:                             # fully inherited: never retrained locally
                continue
            prefix = nn.Sequential(*g.blocks[:i]).to(self.device).eval()
            with ES.ElementProtect(blk, mk, scale):
                ae = train_local_ae(prefix, blk, self.train_loader, genome["ae_target"], ae_ep,
                                    float(tr["lr_ae"]), self.device, run, seed + i, conv_lr_scale=1.0,
                                    decoder_warmup_frac=M.AE_WARMUP_FRAC[genome.get("ae_schedule", "joint")],
                                    **_amp_kw(self))
            for p in blk.parameters():
                p.grad = None
            g.logs[i]["ae"] = _jsonable({k: v for k, v in ae.items() if not isinstance(v, torch.Tensor)})
        n = len(g.blocks)
        if n < int(genome["depth"]):
            encoder, specs, logs = self._grow(genome, ctx, ae_ep, seed, run,
                                              inherit={**inherit, "prebuilt": (g.blocks, g.specs), "n_blocks": n,
                                                       "free_new": True})
            logs[:n] = g.logs
        else:
            encoder, specs, logs = nn.Sequential(*g.blocks).to(self.device), list(g.specs), list(g.logs)
            for p in encoder.parameters():
                p.requires_grad_(True)
        self._fp_masks = {f"{i}.{name}": m for i, mk in enumerate(g.masks) for name, m in mk.items()}
        return encoder, specs, logs

    @torch.no_grad()
    def _log_init_diag(self, i, prefix, block, prov, run):
        """Kernel/activation diagnostics of the freshly initialised layer (before Fase B)."""
        from ga_search.modehb.init import kernel_diagnostics
        conv = next((m for m in block.modules() if isinstance(m, nn.Conv2d)), None)
        if conv is None:
            return
        block.eval()
        feats = block(prefix(self.val_x[:16].to(self.device)))
        block.train()
        tag = "spifil" if prov.get("origin") == "spifil" else "init_random"
        d = kernel_diagnostics(conv.weight, feats)
        rec = {f"{tag}/layer{i}/{k}": v for k, v in d.items()}
        for k in ("requested_channels", "actual_channels"):
            if isinstance(prov.get(k), (int, float)):
                rec[f"{tag}/layer{i}/{k}"] = prov[k]
        rec[f"{tag}/layer{i}/origin"] = str(prov.get("origin"))
        (run.log_now if hasattr(run, "log_now") else run.log)(rec)

    def _fit_zca(self, prefix, block):
        from ga_search.modehb.ops_norm_act import ZCAWhitening
        zcas = [m for m in block.modules() if isinstance(m, ZCAWhitening)]
        if not zcas:
            return
        acc = {id(z): None for z in zcas}       # streamed moments: never stores the feature maps

        def _hook(mod, inp):
            n, sm, xtx = ZCAWhitening.moments(inp[0])
            a = acc[id(mod)]
            acc[id(mod)] = (n, sm, xtx) if a is None else (a[0] + n, a[1] + sm, a[2] + xtx)
        hooks = [z.register_forward_pre_hook(_hook) for z in zcas]
        block.eval()
        with torch.no_grad():
            for n, (x, *_) in enumerate(self.train_loader):
                block(prefix(x.to(self.device)))
                if n + 1 >= int(self.cfg["train"]["zca_fit_batches"]):
                    break
        for h in hooks:
            h.remove()
        for z in zcas:
            z.fit_moments(*acc[id(z)])
        block.train()

    @torch.no_grad()
    def _out_channels(self, model, cin_check=True):
        model.eval()
        x = self.val_x[:2].to(self.device)
        return int(model(x).shape[1])

    @torch.no_grad()
    def _views(self, encoder, head):
        from ga_search.modehb.pooling_head import tta_transforms
        ev = self.cfg["eval"]
        encoder.eval(), head.eval()
        out = []
        for t in tta_transforms(int(ev["n_views"]), ev.get("geometry", "rot_flip")):
            zs = [head(encoder(t(x.to(self.device)))).float().cpu()
                  for x in self.val_x.split(int(self.cfg["data"].get("batch_size", 32)))]
            out.append(torch.cat(zs))
        return torch.stack(out)                       # [V, N, D]

    @staticmethod
    def _load_emb(path):
        return torch.load(path, map_location="cpu") if path and Path(path).exists() else None


def _stratified_indices(y, max_n, seed):
    """Sorted indices of a seeded per-class-proportional subset of size <= max_n (all if fewer)."""
    y = np.asarray(y)
    if max_n is None or len(y) <= int(max_n):
        return np.arange(len(y))
    rng = np.random.default_rng(int(seed))
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts * int(max_n) / len(y)).astype(int))
    idx = [rng.permutation(np.flatnonzero(y == c))[:q] for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(idx))[: int(max_n)]


class _ImagesOnly(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return (self.ds[i][0],)


def _blas_threads(n):
    """Context limiting the BLAS pools (numpy's and scipy's OpenBLAS) to ``n`` threads; None = no-op."""
    import contextlib
    if not n:
        return contextlib.nullcontext()
    try:
        import scipy.linalg  # noqa: F401  (load scipy's OpenBLAS so the limit reaches libsvm's ddot)
        import sklearn.svm  # noqa: F401
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=int(n), user_api="blas")
    except Exception as exc:              # pragma: no cover - threadpoolctl/scipy always installed
        print(f"[run_search] BLAS thread limit unavailable: {exc}")
        return contextlib.nullcontext()


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _controller(cfg):
    w = cfg["wandb"]
    if not w.get("enabled"):
        return None
    from ga_search.modehb import wandb_logger as WL
    s = {**M.DEFAULTS, **cfg["search"]}
    n_obj = 3 if s["use_latency"] else 2
    kw = {k: w[k] for k in ("project", "entity", "mode") if w.get(k)}
    import hashlib
    # one controller run per search directory: a restart RESUMES it (same panels, no frozen copies)
    # The SVM fitness is part of the controller identity: same dir re-run with another C/kappa_weight
    # gets a NEW controller (no C=0.1 and C=100 curves mixed). Legacy fitness (C=0.1, kw=0.75) keeps the old id.
    sv = (cfg.get("eval") or {}).get("svm_fitness") or {}
    fit_sig = f"C{float(sv.get('C', 100.0)):g}_kw{float(sv.get('kappa_weight', 0.75)):g}"
    key = str(Path(cfg["output_dir"]).resolve()) + ("" if fit_sig == "C0.1_kw0.75" else "|" + fit_sig)
    rid = "ctrl-" + hashlib.sha1(key.encode()).hexdigest()[:16]
    tag = "" if fit_sig == "C0.1_kw0.75" else "-" + fit_sig
    return WL.ControllerRun(ref_point=[float(s["ref_point"])] * n_obj, max_fidelity=float(s["fidelities"][-1]),
                            name=f"{'pnas_cs' if s.get('pnas_cs') else 'modehb'}-controller-"
                                 f"{cfg.get('dataset_name', '')}-{Path(cfg['output_dir']).name}{tag}",
                            group=cfg.get("dataset_name"), config=cfg, id=rid, resume="allow", **kw)


def _cap_cpu_threads():
    """Cap intra-op CPU threads (torch + BLAS/OpenMP via threadpoolctl). Many searches share a host;
    each one defaulting to all cores oversubscribed the CPU (athena: load 220 on 96 cores) and left the
    GPUs at 0% util. MODEHB_THREADS overrides; default = cpu_count // 12 (>= 4).

    threadpoolctl only reaches libraries ALREADY loaded: scipy's own OpenBLAS (libscipy_openblas, the
    ddot of sklearn's libsvm) and sklearn's libgomp load lazily with the first SVM fit and kept their
    64/128-thread defaults -> every libsvm kernel product with D > 10000 (cov_head with >= 5 blocks) ran
    on 64 spinning threads (jaci: 2900% CPU per search, load 330). So the env vars are exported first
    (read when a BLAS/OpenMP runtime loads, also by child processes) and scipy.linalg / sklearn.svm are
    imported BEFORE the limit. Same SVM (core.svm.fit_svm untouched); only the BLAS thread count."""
    import os
    n = int(os.environ.get("MODEHB_THREADS") or max(4, (os.cpu_count() or 48) // 12))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    try:                                  # load the lazy BLAS/OpenMP runtimes now, so the limit covers them
        import scipy.linalg  # noqa: F401
        import sklearn.svm  # noqa: F401
    except Exception as exc:              # pragma: no cover - always installed in the search env
        print(f"[run_search] scipy/sklearn preload failed: {exc}")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception as exc:              # pragma: no cover - optional dependency
        print(f"[run_search] threadpoolctl unavailable: {exc}")
    print(f"[run_search] CPU threads capped at {n}", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="replay search_state.json and continue")
    ap.add_argument("--dataset", help="dataset name (required with the search.yaml schema)")
    ap.add_argument("--seed", type=int, help="search seed (search.yaml: one of `seeds`)")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=VALUE",
                    help="dotted override applied after translation, VALUE parsed as YAML")
    args = ap.parse_args(argv)
    # big datasets (protozoan: ~100k imgs) exhaust file descriptors with the default
    # fd-sharing strategy when DataLoader workers feed the RAM cache
    torch.multiprocessing.set_sharing_strategy("file_system")
    _cap_cpu_threads()
    cfg = load_config(args.config, args.dataset, args.seed, args.set)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "config_resolved.yaml"
    if not resolved.exists():
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    evaluator = Evaluator(cfg, out)
    if cfg["search"].get("external_seed"):              # opt-in: convert once; missing file -> warning, no seed
        from ga_search.modehb.external_seed import prepare
        cfg["search"]["external_seed"] = prepare(cfg["search"]["external_seed"], out)
    controller = _controller(cfg)
    spifil_repair = None
    if cfg.get("spifil", {}).get("spifil_ops", False):      # opt-in: keep extended operators
        import functools
        from ga_search.modehb.init import repair_for_spifil
        spifil_repair = functools.partial(repair_for_spifil, spifil_ops=True)
    # real donors per class (a class with < k training images gives fewer SPiFiL seeds)
    cfg["search"]["spifil_pool_sizes"] = [len(v) for _, v in sorted(evaluator.spifil_pool.items())]
    search = M.MODEHB(cfg["search"], evaluator, out, logger=controller, spifil_repair=spifil_repair)
    resume = args.resume or (out / "search_state.json").exists()
    state = search.run(resume=resume)
    top = float(search.fidelities[-1])
    summary = {"n_jobs": len(state["jobs"]), "gpu_hours": state["gpu_hours"],
               "pareto_max_fidelity": [{"cand_id": j["cand_id"], "init": j["genome"]["init"], "F": j["F"],
                                        "checkpoint": j["result"].get("checkpoint")}
                                       for j in search.front(top)],
               "brackets": state["brackets_log"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if controller is not None:
        controller.finish()
    print(json.dumps({k: summary[k] for k in ("n_jobs", "gpu_hours")}))


if __name__ == "__main__":
    main()
