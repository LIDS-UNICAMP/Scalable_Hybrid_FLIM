"""Resource-only runtime overlay for an existing progressive experiment.

Scientific configuration and checkpoints retain their identity. This entry
preloads the exact deterministic inputs, sets measured CPU limits, and installs
the tested equivalent DBCV implementation. Its own hashes/resources/timings are
recorded separately in the output directory. Run from a protected main entry.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    output_arg = sys.argv.index("--nas-output") + 1
    root = Path(sys.argv[output_arg]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("NAS_CPU_THREADS", "16"))
    workers = int(os.environ.get("NAS_PRELOAD_WORKERS", "32"))
    dbcv_workers = int(os.environ.get("NAS_DBCV_WORKERS", "32"))
    if min(threads, workers, dbcv_workers) < 1:
        raise ValueError("runtime thread/worker counts must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)

    # Legacy CLI sets numerical defaults during import. Reset the loaded pools
    # explicitly afterwards; changing a parent shell alone would have no effect.
    from ga_search import run_ga
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits, threadpool_info
    from ga_search import progressive, progressive_data
    original_read_config = progressive.read_config

    def canonical_config(*args, **kwargs):
        # JSON checkpoints store class-map keys as strings. YAML loads them as
        # integers; canonicalize before strict resume comparison (same values).
        return json.loads(json.dumps(original_read_config(*args, **kwargs)))

    progressive.read_config = canonical_config
    from ga_search.progressive_cpu import cache_loader
    from ga_search.progressive_dbcv import parallel_validity_index
    from ga_search.progressive_spifil_memory import StreamingFisherScorer
    from methods.autoencoder.progressive import ProgressiveTrainer
    import hdbscan.validity
    import spifil.learner
    # Runtime-only equivalent scorer: retain all donors/patches, stream rows.
    spifil.learner.FisherScorer = StreamingFisherScorer
    import faulthandler
    import signal

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(4, threads))
    limits = threadpool_limits(limits=threads)
    stack_log = (root / "runtime_stacks.log").open("a")
    faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=True)
    context = {}
    active_logger = {}
    io_lock = threading.Lock()
    stop = threading.Event()

    def emit(path, record):
        with io_lock, (root / path).open("a") as stream:
            stream.write(json.dumps({"time": time.time(), **record}, allow_nan=False) + "\n")
        logger = active_logger.get("logger")
        if logger is not None and not getattr(logger, "_finished", False):
            event = "runtime_resources" if path == "runtime_resources.jsonl" else "runtime_stage"
            if event == "runtime_resources" or record.get("event") in ("end", "error"):
                try:
                    logger.emit(event, **dict(context), result=record)
                except Exception as exc:
                    with io_lock, (root / "runtime_logging_errors.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(time=time.time(), error=str(exc))) + "\n")
                    print(f"RUNTIME W&B logging error: {exc}", flush=True)

    def timed(name, function, describe=None):
        @functools.wraps(function)
        def call(*args, **kwargs):
            info = {**context, **(describe(*args, **kwargs) if describe else {})}
            start = time.perf_counter()
            emit("runtime_timings.jsonl", dict(event="start", stage=name, **info))
            print(f"RUNTIME start {name} {info}", flush=True)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                emit("runtime_timings.jsonl", dict(event="error", stage=name, seconds=time.perf_counter()-start,
                                                   error=type(exc).__name__, **info))
                raise
            elapsed = time.perf_counter() - start
            detail = {}
            if isinstance(result, dict):
                for key in ("J", "status", "n_clusters", "coverage", "reconstruction_loss", "epoch", "phase"):
                    if key in result:
                        detail[key] = result[key]
            emit("runtime_timings.jsonl", dict(event="end", stage=name, seconds=elapsed, result=detail, **info))
            print(f"RUNTIME end {name} seconds={elapsed:.3f} {detail}", flush=True)
            return result
        return call

    original_data = progressive_data.load_data

    def cached_data(config, output_dir):
        train, validation, samples, identity = original_data(config, output_dir)
        statistics = {}
        for name, loader in (("train", train), ("validation", validation)):
            cached, stats = cache_loader(loader, workers=workers, device=config["device"])
            for index in (0, len(loader.dataset) - 1):
                expected, actual = loader.dataset[index], cached.dataset[index]
                if not torch.equal(expected[0], actual[0]) or expected[1:] != actual[1:]:
                    raise ValueError("preprocessing cache altered data")
            statistics[name] = stats
            print(f"RUNTIME cached {name} {stats}", flush=True)
            if name == "train":
                train = cached
            else:
                validation = cached
        (root / "runtime_cache.json").write_text(json.dumps(statistics, indent=2))
        return train, validation, samples, identity

    progressive_data.load_data = timed("preload", cached_data)
    original_init = progressive.ProgressiveExperiment.__init__

    @functools.wraps(original_init)
    def initialize(experiment, *args, **kwargs):
        original_init(experiment, *args, **kwargs)
        active_logger["logger"] = experiment.logger
        experiment._emit("runtime_profile", result=profile)
        cache_path = root / "runtime_cache.json"
        if cache_path.exists():
            experiment._emit("runtime_cache", result=json.loads(cache_path.read_text()))

    progressive.ProgressiveExperiment.__init__ = initialize
    original_evaluate = progressive.ProgressiveExperiment.evaluate

    @functools.wraps(original_evaluate)
    def evaluate(experiment, genome):
        context.clear()
        context.update(requested_depth=len(genome["blocks"]), genome_id=progressive.digest(genome))
        return timed("individual", original_evaluate)(experiment, genome)

    progressive.ProgressiveExperiment.evaluate = evaluate
    ProgressiveTrainer.train_epoch = timed("train", ProgressiveTrainer.train_epoch,
        lambda trainer, loader: dict(depth=trainer.model.depth, phase=trainer.phase, epoch=trainer.epoch+1))
    original_fit = timed("spifil", progressive.fit_new_block)

    def memory_bounded_fit(*args, **kwargs):
        try:
            return original_fit(*args, **kwargs)
        finally:
            # Fitting leaves large dead distance buffers in CUDA's allocator.
            # Release only unused blocks; live encoder weights stay allocated.
            import gc
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    progressive.fit_new_block = memory_bounded_fit
    progressive.extract_embeddings = timed("embedding", progressive.extract_embeddings,
        lambda model, loader, *args, **kwargs: dict(depth=model.depth, samples=len(loader.dataset)))
    progressive.reconstruction_loss = timed("reconstruction", progressive.reconstruction_loss,
        lambda model, loader, *args, **kwargs: dict(depth=model.depth, samples=len(loader.dataset)))
    progressive.evaluate_embeddings = timed("clustering", progressive.evaluate_embeddings,
        lambda embeddings, config: dict(samples=len(embeddings)))
    hdbscan.validity.validity_index = timed("dbcv", functools.partial(parallel_validity_index,
                                                                      workers=dbcv_workers, progress=True))

    def monitor():
        import subprocess
        processes = {}
        parent = psutil.Process()
        while not stop.wait(5):
            try:
                family = [parent, *parent.children(recursive=True)]
                active, cpu, rss = [], 0.0, 0
                for process in family:
                    try:
                        sample = processes.setdefault(process.pid, process)
                        cpu += sample.cpu_percent()
                        rss += sample.memory_info().rss
                        active.append(sample.pid)
                    except psutil.Error:
                        pass
                processes = {pid: processes[pid] for pid in active}
                gpu = subprocess.check_output(["nvidia-smi", "-i", os.environ.get("NAS_PHYSICAL_GPU", "2"),
                    "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=3)
                gpu_memory, gpu_util = map(float, gpu.strip().split(","))
                emit("runtime_resources.jsonl", dict(pid=parent.pid, total_cpu_percent=cpu,
                    parent_threads=parent.num_threads(), processes=len(family), rss_sum_gib=rss/2**30,
                    gpu_memory_mib=gpu_memory, gpu_util_percent=gpu_util,
                    cuda_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,
                    cuda_reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0))
            except (psutil.Error, OSError, ValueError, subprocess.SubprocessError) as exc:
                emit("runtime_resources.jsonl", dict(error=str(exc)))

    sources = [Path(__file__), Path(cache_loader.__code__.co_filename),
               Path(parallel_validity_index.__code__.co_filename),
               Path(__import__(StreamingFisherScorer.__module__, fromlist=["__file__"]).__file__)]
    profile = dict(cpu_threads=threads, preload_workers=workers, dbcv_workers=dbcv_workers,
                   cpu_count=os.cpu_count(), pid=os.getpid(), threadpools=threadpool_info(),
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   scientific_configuration="same configuration and donor IDs", input_equivalence="exact cached tensors",
                   spifil_memory_policy="streamed Fisher rows; same formula, float32 GPU reductions may differ in rounding",
                   dbcv_equivalence="tested upstream-identical scalar and MST ties")
    (root / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        run_ga.main()
    finally:
        stop.set()
        watcher.join(timeout=6)
        limits.restore_original_limits()


if __name__ == "__main__":
    main()
