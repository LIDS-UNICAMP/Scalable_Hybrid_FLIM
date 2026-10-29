"""Opt-in progressive experiment driven by the existing growth and GA entries.

The SPiFiL adapter, decoder, transforms and split readers are shared with the
repository. Legacy growth/SVM experiments keep their existing selection rules.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from flim.progressive import InfeasibleCandidate, fit_new_block
from ga_search.progressive_genetics import (EvaluationCache, GeneticSearch, atomic_json,
                                          digest, validate_genome)
from ga_search.progressive_metrics import evaluate_embeddings, extract_embeddings
from ga_search.progressive_logging import ExperimentLogger
from methods.autoencoder.progressive import EvolutionaryCNN, ProgressiveTrainer


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    result = {"python": platform.python_version()}
    for name in ("torch", "numpy", "scikit-learn", "hdbscan", "spifil", "pyift", "wandb"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    root = Path(__file__).resolve().parents[1]
    # Include dirty implementation content, not only HEAD, in cache identity.
    files = [Path(__file__), root / "flim/progressive.py", root / "flim/spifil.py",
             root / "ga_search/progressive_metrics.py", root / "ga_search/progressive_data.py",
             root / "ga_search/progressive_cost.py", root / "ga_search/progressive_genetics.py",
             root / "ga_search/progressive_logging.py",
             root / "core/data/transforms.py", root / "core/data/loaders.py", root / "core/blocks/init.py",
             root / "core/svm.py", root / "core/metrics.py",
             root / "methods/autoencoder/resnet_decoder.py", root / "methods/autoencoder/residual_up_block.py",
             root / "methods/autoencoder/progressive.py"]
    result["implementation"] = {str(p.relative_to(root)): file_digest(p) for p in files if p.exists()}
    import spifil
    package = Path(spifil.__file__).parent
    result["spifil_source"] = digest({str(p.relative_to(package)): file_digest(p)
                                     for p in sorted(package.rglob("*.py"))})
    return result


DEFAULTS = dict(seed=42, device="auto", image_size=200, in_channels=3, embedding_dim=1280,
                max_depth=10, batch_size=2, num_workers=0, epochs_a=1, epochs_b=1,
                training_patience=None, training_min_delta=0.0,
                lr=0.001, refine_lr=0.0001, auxiliary_init="he", n_superpixels=100,
                imagenet_norm=False, spifil_in_image=True, spifil_in_feature=False,
                superpixels="slic", budget_check=True, impurities=False,
                one_per_class=True, paper_classes=False, n_images=20,
                max_parameters=2_000_000, max_macs=500_000_000,
                max_latency_ms=None, ssim_weight=0.0,
                svm_scaler=False, svm_C=100.0, svm_max_iter=-1,
                wandb=dict(enabled=False, project="progressive_nas_ga", mode="online",
                           upload_artifacts=True, init_timeout=60),
                clustering=dict(sparsity_lambda=0.004, min_coverage=0.8, penalty=1e12,
                                min_cluster_size=5, min_samples=3, standardize=False),
                genetic=dict(seed=42, min_depth=1, max_depth=10, population_size=4,
                             generations=2, elitism=1, mutation_rate=0.8, patience=2))


def read_config(path, overrides=None):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("progressive configuration must be a YAML mapping")
    allowed = set(DEFAULTS) | {"mode", "output_dir", "synthetic", "synthetic_train_samples",
                              "synthetic_val_samples", "dataset", "split", "percentage",
                              "dataset_config", "dataset_cfg", "genome", "complementary_svm",
                              "weight_inheritance", "weight_decay", "latency_warmup", "latency_repeats",
                              "latency_batch_size"}
    if set(raw) - allowed or set(overrides or {}) - allowed:
        raise ValueError(f"unknown progressive configuration keys: {(set(raw) | set(overrides or {})) - allowed}")
    config = {**copy.deepcopy(DEFAULTS), **raw, **(overrides or {})}
    for key in ("clustering", "genetic", "wandb"):
        config[key] = {**DEFAULTS[key], **raw.get(key, {}), **(overrides or {}).get(key, {})}
    if not isinstance(config["wandb"].get("enabled"), bool):
        raise ValueError("wandb.enabled must be a boolean")
    if config["wandb"].get("mode") not in ("online", "offline"):
        raise ValueError("wandb.mode must be online or offline")
    for key in ("seed", "image_size", "in_channels", "embedding_dim", "max_depth", "batch_size",
                "num_workers", "epochs_a", "epochs_b", "n_superpixels", "n_images"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    patience = config["training_patience"]
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("training_patience must be None or a positive integer")
    for key in ("lr", "refine_lr", "max_parameters", "max_macs", "ssim_weight", "training_min_delta"):
        config[key] = float(config[key])
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["lr"] <= 0 or config["refine_lr"] <= 0 or config["refine_lr"] > config["lr"]:
        raise ValueError("require 0 < refine_lr <= lr")
    if config["auxiliary_init"] not in ("he", "xavier"):
        raise ValueError("auxiliary_init must be he or xavier for convolutions")
    if config.get("weight_inheritance", False):
        raise ValueError("cross-individual weight inheritance is not validated; disable it")
    if config["embedding_dim"] != 1280 or not 1 <= config["max_depth"] <= 10:
        raise ValueError("embedding_dim=1280 and max_depth in 1..10 required")
    if config["in_channels"] != 3:
        raise ValueError("SPiFiL LAB input pipeline currently validates exactly 3 input channels")
    if any(int(config[key]) < 1 for key in ("epochs_a", "epochs_b", "batch_size", "image_size")):
        raise ValueError("positive epochs_a, epochs_b, batch_size and image_size required")
    if config["spifil_in_image"] == config["spifil_in_feature"]:
        raise ValueError("choose exactly one SPiFiL superpixel mode")
    if config.get("mode", "search") not in ("train", "search"):
        raise ValueError("mode must be train or search")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["genetic"]["max_depth"] = min(config["genetic"]["max_depth"], config["max_depth"])
    config["output_dir"] = str(Path(config.get("output_dir", "artifacts/progressive_nas")).resolve())
    return config


@torch.no_grad()
def reconstruction_loss(model, loader, device):
    modes = [(m, m.training) for m in model.modules()]
    total, count = 0.0, 0
    try:
        model.eval()
        for batch in loader:
            x = batch[0].to(device)
            loss = F.l1_loss(model(x), x)
            if not torch.isfinite(loss):
                raise InfeasibleCandidate("nonfinite validation reconstruction")
            total += float(loss) * len(x)
            count += len(x)
    finally:
        for module, mode in modes:
            module.training = mode
    if not count:
        raise ValueError("empty validation loader")
    return total / count


def complementary_probe(model, train_loader, val_loader, device, config=None):
    """Optional fixed SVM/kappa probe; labels never influence J."""
    from core.metrics import compute_metrics
    from core.svm import fit_svm
    config = config or {}
    def collect(loader):
        vectors, _ = extract_embeddings(model, loader, device=device)
        labels = np.concatenate([np.asarray(batch[1]) for batch in loader])
        return vectors, labels
    x, y = collect(train_loader)
    z, labels = collect(val_loader)
    classes = sorted(set(y.tolist()) | set(labels.tolist()))
    mapping = {label: index for index, label in enumerate(classes)}
    y = np.asarray([mapping[label] for label in y])
    labels = np.asarray([mapping[label] for label in labels])
    probe = fit_svm(x, y, scaler=bool(config.get("svm_scaler", False)),
                    C=float(config.get("svm_C", 100.0)), max_iter=int(config.get("svm_max_iter", -1)),
                    tag="NAS_COMPLEMENTARY")
    prediction = probe.predict(z)
    return {**{f"svm_{name}": value for name, value in compute_metrics(labels, prediction, len(classes)).items()},
            **probe.fit_diagnostics_}


class ExperimentLoggingError(Exception):
    """Infrastructure failure must not become an infeasible architecture."""


class ProgressiveExperiment:
    def __init__(self, config, resume=False):
        from ga_search.progressive_data import load_data
        self.config, self.resume = config, resume
        self.root = Path(config["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root / "experiment.json"
        self.version_id = versions()
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if previous["config"] != config or previous["versions"] != self.version_id:
                raise ValueError("experiment config/data/code changed; use a new output_dir")
            if not resume:
                raise FileExistsError("experiment already exists; use --resume")
        self.train, self.validation, self.samples, self.data_id = load_data(config, self.root)
        self.identity = dict(data=self.data_id, config=config, versions=self.version_id)
        self.identity_hash = digest(self.identity)
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if previous["identity_hash"] != self.identity_hash:
                raise ValueError("experiment config/data/code changed; use a new output_dir")
            if not resume:
                raise FileExistsError("experiment already exists; use --resume")
        else:
            atomic_json(metadata, dict(identity_hash=self.identity_hash, **self.identity,
                                       nondeterminism="CUDA algorithms may be nondeterministic; CPU tested"))
        atomic_json(self.root / "data_identity.json", self.data_id)
        self.cache = EvaluationCache(self.root / "cache")
        self.logger = ExperimentLogger(config, self.root)

    def _emit(self, event, **payload):
        try:
            return self.logger.emit(event, **payload)
        except Exception as exc:
            raise ExperimentLoggingError(f"failed to log {event}: {exc}") from exc

    def evaluate(self, genome):
        from ga_search.progressive_cost import profile_model
        cfg = self.config
        penalty = float(cfg["clustering"]["penalty"])
        key = digest(dict(requested_genome=genome, experiment=self.identity_hash))
        directory = self.root / "individuals" / key
        directory.mkdir(parents=True, exist_ok=True)
        result_path, last = directory / "result.json", directory / "last.ckpt"
        if result_path.exists():
            # Only the same exact experiment+requested genome can reach this path.
            result = json.loads(result_path.read_text())
            self._emit("individual_cache_hit", individual_id=key, requested_genome=genome,
                       result=result, cache=True, cache_source="completed_individual")
            return result
        records, init_records = [], []
        metadata = dict(depth_complete=True, records=records, initializations=init_records)
        trainer = None
        try:
            validate_genome(genome, cfg["max_depth"])
            random.seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            trainer = ProgressiveTrainer(EvolutionaryCNN(cfg), cfg)
            if last.exists():
                metadata = trainer.load_checkpoint(last)
                if metadata.get("individual_id") != key:
                    raise ValueError("candidate checkpoint identity mismatch")
                records, init_records = metadata["records"], metadata["initializations"]
            first = trainer.model.depth + (1 if metadata["depth_complete"] else 0)
            first = max(1, first)
            for depth in range(first, len(genome["blocks"]) + 1):
                stage = directory / f"depth_{depth}"
                stage.mkdir(exist_ok=True)
                selected = stage / "best_J.ckpt"
                if trainer.model.depth < depth:
                    prefix_id = file_digest(last) if last.exists() else "empty_prefix"
                    block, provenance = fit_new_block(trainer.model.encoder, self.samples,
                                                     genome["blocks"][depth - 1], cfg,
                                                     prefix_id=prefix_id)
                    trainer.model.add_block(block, provenance["effective_spec"])
                    trainer.model.provenance[f"encoder{depth}"] = provenance
                    init_records.append(provenance)
                    self._emit("initialization", individual_id=key, depth=depth,
                               requested_genome=genome, effective_genome={"blocks": trainer.model.effective_specs},
                               result=provenance)
                    trainer.sync_optimizer()
                    # Reject shape/spatial incompatibilities before gradient training.
                    trainer.model.eval()
                    with torch.no_grad():
                        probe = torch.zeros(1, cfg["in_channels"], cfg["image_size"], cfg["image_size"],
                                            device=trainer.device)
                        reconstruction = trainer.model(probe)
                        if reconstruction.shape != probe.shape:
                            raise InfeasibleCandidate("decoder output shape differs from input")
                    metadata = dict(depth_complete=False, phase="A", epoch_done=0,
                                    best_J=None, records=records, initializations=init_records,
                                    phase_best_J=None, phase_stale=0, phase_early_stopped=False,
                                    individual_id=key)
                    trainer.save_checkpoint(last, **metadata)
                cost = profile_model(trainer.model, cfg)
                self._emit("model_cost", individual_id=key, depth=depth, result=dict(cost=cost))
                if cost["inference_parameters"] > cfg["max_parameters"] or cost["inference_macs"] > cfg["max_macs"]:
                    raise InfeasibleCandidate("inference parameter/MAC constraint exceeded", details={"cost": cost})
                if cfg["max_latency_ms"] is not None and cost["latency_ms_mean"] > cfg["max_latency_ms"]:
                    raise InfeasibleCandidate("inference latency constraint exceeded", details={"cost": cost})
                context = dict(effective_genome={"blocks": trainer.model.effective_specs},
                               data=self.data_id, config=cfg, seeds=dict(seed=cfg["seed"]),
                               sample_ids=[s.name for s in self.samples], versions=self.version_id,
                               prefix_id=init_records[-1]["prefix_id"])
                cache_hit = self.cache.get(context)
                cache_usable = bool(cache_hit and Path(cache_hit["checkpoint"]).exists()
                                    and file_digest(cache_hit["checkpoint"]) == cache_hit["checkpoint_id"])
                if cache_usable:
                    self._emit("depth_cache_hit", individual_id=key, depth=depth,
                               result=cache_hit, cache=True, cache_source=cache_hit["checkpoint_id"])
                    # A hit is only legal with the full effective/prefix identity.
                    cached_metadata = trainer.load_checkpoint(cache_hit["checkpoint"])
                    for record in cached_metadata["records"]:
                        if record["depth"] == depth:
                            records.append({**record, "individual_id": key, "requested_genome": genome,
                                            "cache_source_checkpoint": cache_hit["checkpoint_id"]})
                    metadata = {**metadata, "best_J": cached_metadata["best_J"]}
                    selected = Path(cache_hit["checkpoint"])
                resume_phase, resume_epoch = metadata.get("phase", "A"), metadata.get("epoch_done", 0)
                best_j = metadata.get("best_J")
                training_patience = cfg.get("training_patience")
                training_min_delta = float(cfg.get("training_min_delta", 0.0))
                for phase in (() if cache_usable else ("A", "B")):
                    if phase == "A" and resume_phase == "B":
                        continue
                    resuming_phase = phase == resume_phase
                    phase_best_j = metadata.get("phase_best_J") if resuming_phase else None
                    phase_stale = int(metadata.get("phase_stale", 0)) if resuming_phase else 0
                    phase_early_stopped = bool(metadata.get("phase_early_stopped", False)) if resuming_phase else False
                    if training_patience is not None and phase_early_stopped:
                        continue
                    trainer.set_phase(phase)
                    epochs = cfg["epochs_a"] if phase == "A" else cfg["epochs_b"]
                    start = resume_epoch if phase == resume_phase else 0
                    for epoch in range(start, epochs):
                        training = trainer.train_epoch(self.train)
                        embeddings, ids = extract_embeddings(trainer.model, self.validation, device=trainer.device)
                        metrics = evaluate_embeddings(embeddings, cfg["clustering"])
                        val_loss = reconstruction_loss(trainer.model, self.validation, trainer.device)
                        row = dict(individual_id=key, depth=depth, phase_epoch=epoch + 1,
                                   seed=cfg["seed"], requested_genome=genome,
                                   effective_genome={"blocks": copy.deepcopy(trainer.model.effective_specs)},
                                   validation_reconstruction_loss=val_loss, **training, **metrics)
                        records.append(row)
                        better = best_j is None or row["J"] < best_j
                        if better:
                            best_j = row["J"]
                        valid_j = row.get("status") == "ok" and math.isfinite(float(row["J"]))
                        if training_patience is not None:
                            if valid_j and (phase_best_j is None or row["J"] < phase_best_j - training_min_delta):
                                phase_best_j = row["J"]
                                phase_stale = 0
                            else:
                                phase_stale += 1
                            phase_early_stopped = phase_stale >= training_patience
                        metadata = dict(individual_id=key, depth_complete=False, phase=phase,
                                        epoch_done=epoch + 1, best_J=best_j,
                                        phase_best_J=phase_best_j, phase_stale=phase_stale,
                                        phase_early_stopped=phase_early_stopped,
                                        records=records, initializations=init_records)
                        if better:
                            trainer.save_checkpoint(selected, **metadata)
                        trainer.save_checkpoint(last, **metadata)
                        atomic_json(directory / "history.json", records)
                        self._emit("epoch_end", individual_id=key, depth=depth, phase=phase,
                                   epoch=epoch + 1, requested_genome=genome,
                                   effective_genome=row["effective_genome"], result=row)
                        print(f"NAS {key[:8]} depth={depth} phase={phase} epoch={epoch+1} "
                              f"L1={training['reconstruction_loss']:.6g} J={metrics['J']:.6g} "
                              f"status={metrics['status']}", flush=True)
                        if phase_early_stopped:
                            reason = "no_valid_J" if phase_best_j is None else "no_valid_J_improvement"
                            self._emit("training_early_stop", individual_id=key, depth=depth,
                                       phase=phase, epoch=epoch + 1, reason=reason,
                                       phase_stale=phase_stale, phase_best_J=phase_best_j,
                                       patience=training_patience, min_delta=training_min_delta,
                                       result=dict(J=row["J"], status=row.get("status")))
                            print(f"NAS {key[:8]} depth={depth} phase={phase} early_stop "
                                  f"epoch={epoch+1} stale={phase_stale} reason={reason}", flush=True)
                            break
                # Save the exact J-selected model/optimizer; keep the complete history.
                trainer.load_checkpoint(selected)
                for i, provenance in enumerate(init_records, 1):
                    trainer.model.provenance[f"encoder{i}"] = provenance
                trainer.model.encoder.eval().requires_grad_(False)
                checkpoint = stage / "selected.ckpt"
                metadata = dict(individual_id=key, depth_complete=True, records=records,
                                initializations=init_records, best_J=best_j,
                                phase=metadata.get("phase"), epoch_done=metadata.get("epoch_done", 0),
                                phase_best_J=metadata.get("phase_best_J"),
                                phase_stale=metadata.get("phase_stale", 0),
                                phase_early_stopped=metadata.get("phase_early_stopped", False))
                trainer.save_checkpoint(checkpoint, **metadata)
                checkpoint_id = file_digest(checkpoint)
                embeddings, ids = extract_embeddings(trainer.model, self.validation,
                                                       stage / "embeddings.npz", checkpoint_id,
                                                       trainer.device)
                final_metrics = evaluate_embeddings(embeddings, cfg["clustering"])
                cost = profile_model(trainer.model, cfg, checkpoint)
                stage_result = dict(**final_metrics, cost=cost, checkpoint=str(checkpoint),
                                    checkpoint_id=checkpoint_id, depth=depth,
                                    effective_genome={"blocks": copy.deepcopy(trainer.model.effective_specs)},
                                    embedding_path=str(stage / "embeddings.npz"))
                selected_row = min((r for r in records if r["depth"] == depth), key=lambda r: r["J"])
                stage_result.update(validation_reconstruction_loss=selected_row["validation_reconstruction_loss"],
                                    training_reconstruction_loss=selected_row["reconstruction_loss"],
                                    selected_phase=selected_row["phase"], selected_epoch=selected_row["phase_epoch"])
                if cfg.get("complementary_svm", False):
                    self._emit("svm_start", individual_id=key, depth=depth,
                               scaler=cfg["svm_scaler"], max_iter=cfg["svm_max_iter"], C=cfg["svm_C"])
                    import time
                    svm_started = time.perf_counter()
                    stage_result.update(complementary_probe(trainer.model, self.train, self.validation,
                                                            trainer.device, cfg))
                    stage_result["svm_elapsed_seconds"] = time.perf_counter() - svm_started
                    self._emit("svm_end", individual_id=key, depth=depth, result=stage_result)
                atomic_json(stage / "result.json", stage_result)
                atomic_json(directory / "history.json", records)
                self.cache.put(context, stage_result)
                trainer.save_checkpoint(last, **metadata)
                self._emit("depth_end", individual_id=key, depth=depth,
                           requested_genome=genome, effective_genome=stage_result["effective_genome"],
                           result=stage_result)
            final = json.loads((directory / f"depth_{len(genome['blocks'])}" / "result.json").read_text())
            inference_path = directory / "encoder_embedding.pt"
            torch.save(trainer.model.inference_model(), inference_path)
            final.update(individual_id=key, requested_genome=genome, seed=cfg["seed"],
                         inference_path=str(inference_path), initialization_records=init_records,
                         history_path=str(directory / "history.json"))
        except (InfeasibleCandidate, ValueError, RuntimeError, FloatingPointError) as exc:
            final = dict(status="infeasible", reason=str(exc), J=penalty,
                         individual_id=key, requested_genome=genome, seed=cfg["seed"],
                         initialization_records=init_records,
                         effective_genome={"blocks": trainer.model.effective_specs} if trainer else None,
                         details=getattr(exc, "details", {}))
            print(f"NAS {key[:8]} infeasible: {exc}", flush=True)
        atomic_json(result_path, final)
        self._emit("individual_end", individual_id=key, requested_genome=genome,
                   effective_genome=final.get("effective_genome"), result=final)
        return final

    def run(self):
        cfg = self.config
        status = "failed"
        try:
            self._emit("experiment_start", config=cfg, data=self.data_id,
                       identity_hash=self.identity_hash, resumed=self.resume,
                       wandb_url=self.logger.url, wandb_run_id=self.logger.run_id)
            if cfg.get("mode", "search") == "train":
                if "genome" not in cfg:
                    raise ValueError("mode train requires genome.blocks in config")
                result = self.evaluate(cfg["genome"])
            else:
                search = GeneticSearch(cfg["genetic"], self.evaluate, self.root / "search",
                                       context=self.identity, logger=self.logger)
                state = search.run(resume=self.resume)
                result = dict(status="finished", best=state["best"], generations=state["generation"],
                              history_path=str(self.root / "search/search_state.json"))
            atomic_json(self.root / "result.json", result)
            self._emit("experiment_end", result=result)
            status = "finished"
            return result
        except KeyboardInterrupt:
            status = "interrupted"
            self._emit("experiment_interrupted", status=status)
            raise
        except Exception as exc:
            self._emit("experiment_failed", error=type(exc).__name__, reason=str(exc))
            raise
        finally:
            self.logger.finish(status=status)


def run_config(path, *, resume=False, overrides=None):
    return ProgressiveExperiment(read_config(path, overrides), resume=resume).run()


def run_ray_arm(exp, point, gpu_id, cpus, dry_run=False):
    """Called by the existing growth runner, with its reserved GPU and subprocess."""
    from experiments.ray.runners.train import _run
    from experiments.constants import DATASET_LONG_TO_SHORT
    config_path = exp.runner_args["progressive_nas"]
    tag = f"{DATASET_LONG_TO_SHORT[point['dataset']]}_split{point['split']}_pct{point['percentage']}"
    output = Path(exp.output.work_dir) / tag
    cmd = [sys.executable, "-m", "ga_search.progressive_runtime", "--progressive-nas", str(config_path),
           "--resume", "--datasets", point["dataset"], "--splits", str(point["split"]),
           "--pcts", str(point["percentage"]), "--nas-output", str(output), "--nas-device", "cuda:0"]
    if dry_run:
        return dict(tag=tag, gpu_id=gpu_id, plan=[cmd], rows=[], failed=False, reason="progressive NAS")
    # Each arm has a stable, distinct display name and resumable config snapshot.
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["wandb"] = {**raw.get("wandb", {}), "name": f"{exp.name}_{tag}"}
    output.mkdir(parents=True, exist_ok=True)
    arm_config = output / "arm_config.yaml"
    encoded = yaml.safe_dump(raw, sort_keys=False)
    if arm_config.exists() and arm_config.read_text() != encoded:
        raise ValueError(f"refusing to change existing arm configuration: {arm_config}")
    arm_config.write_text(encoded)
    cmd[cmd.index("--progressive-nas") + 1] = str(arm_config)
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "NAS_PHYSICAL_GPU": str(gpu_id), "NAS_CPU_THREADS": str(cpus),
           "NAS_PRELOAD_WORKERS": str(cpus), "NAS_DBCV_WORKERS": str(cpus),
           "OMP_NUM_THREADS": str(cpus), "OPENBLAS_NUM_THREADS": str(cpus),
           "MKL_NUM_THREADS": str(cpus)}
    completed = _run(cmd, env=env, name=tag, gpu_id=gpu_id)
    return dict(tag=tag, gpu_id=gpu_id, plan=[cmd], rows=[],
                failed=completed["returncode"] != 0, reason=f"progressive NAS exit {completed['returncode']}")
