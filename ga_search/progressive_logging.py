"""Thread-safe local NAS events and optional W&B telemetry, without model uploads."""
from __future__ import annotations

import datetime as dt
import importlib
import json
import math
import numbers
from pathlib import Path
import re
import threading
import uuid


_PRIVATE = re.compile(r"(^|[_\-])(api[_\-]?key|token|password|secret|credential|authorization|netrc)(s|$|[_\-])", re.I)


def _safe(value, key=""):
    """JSON-safe, finite values; credential-bearing keys are always redacted."""
    if _PRIVATE.search(str(key)):
        return "[REDACTED]"
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    return str(value)


def _atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _scalars(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_scalars(item, name))
        elif isinstance(item, (int, float, bool)) and math.isfinite(float(item)):
            result[name] = item
    return result


def _valid_result(result):
    return (isinstance(result, dict) and result.get("status") in ("ok", "valid", "feasible")
            and isinstance(result.get("J"), (float, int)) and math.isfinite(float(result["J"])))


class ExperimentLogger:
    """Local JSONL always; import/init W&B only when config.wandb.enabled=true.

    ``emit`` writes local events first and uses one monotonic SDK step per event.
    Run identity and sequence survive reconstruction from the same directory.
    Online init/upload errors propagate rather than switching modes silently.
    """

    def __init__(self, config, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._config = _safe(config)
        self.options = dict(config.get("wandb") or {})
        self.enabled = bool(self.options.get("enabled", False))
        self.path = self.directory / "events.jsonl"
        self.state_path = self.directory / "logger_state.json"
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.run_id = state.get("run_id") or uuid.uuid4().hex[:12]
        self._sequence = int(state.get("sequence", -1))
        # Recover the append-before-state crash window without reusing a step.
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    self._sequence = max(self._sequence, int(json.loads(line)["sequence"]))
                except (ValueError, KeyError, TypeError):
                    raise ValueError(f"invalid event history in {self.path}") from None
        self.url = state.get("url")
        self.run = None
        self._sdk = None
        self._finished = False
        self._individuals = {}
        self._best_valid_J = state.get("best_valid_J")
        _atomic(self.directory / "logger_config.json", self._config)
        if self.enabled:
            mode = self.options.get("mode", "online")
            if mode not in ("online", "offline"):
                raise ValueError("wandb.mode must be online or offline when enabled")
            self._sdk = importlib.import_module("wandb")
            entity = self.options.get("entity")
            if entity is None:
                from core.wandb import ENTITY
                entity = ENTITY
            self.run = self._sdk.init(
                project=self.options.get("project", "progressive_nas_ga"), entity=entity,
                name=self.options.get("name"), group=self.options.get("group"),
                tags=self.options.get("tags"), mode=mode, id=self.run_id,
                resume="allow", config=self._config, dir=str(self.directory),
                settings=self._sdk.Settings(init_timeout=self.options.get("init_timeout", 60)),
            )
            if self.run is None or (mode == "online" and getattr(self.run.settings, "mode", mode) != "online"):
                raise RuntimeError("W&B did not initialize the requested mode")
            self.run_id = self.run.id
            self.url = self.run.url if mode == "online" else None
            self.run.define_metric("event_sequence")
            self.run.define_metric("*", step_metric="event_sequence")
            self.run.summary["experiment/dataset"] = config.get("dataset", "synthetic")
            self.run.summary["experiment/percentage"] = config.get("percentage")
            self.run.summary["experiment/split"] = config.get("split")
        self._save_state()

    def _save_state(self):
        _atomic(self.state_path, {"run_id": self.run_id, "url": self.url,
                                 "sequence": self._sequence, "enabled": self.enabled,
                                 "best_valid_J": self._best_valid_J})

    def _generation_summary(self, payload):
        population = payload.get("population", payload.get("results", []))
        rows = [entry.get("result", entry) for entry in population]
        if not rows:
            generation = payload.get("generation")
            rows = [row["result"] for row in self._individuals.values()
                    if row.get("generation") == generation]
        valid = [float(row["J"]) for row in rows if _valid_result(row)]
        summary = {"ga/valid_count": len(valid), "ga/invalid_count": len(rows) - len(valid),
                   "ga/J_min": min(valid) if valid else None,
                   "ga/J_mean": sum(valid) / len(valid) if valid else None,
                   "ga/J_max": max(valid) if valid else None}
        if valid and (self._best_valid_J is None or min(valid) < self._best_valid_J):
            self._best_valid_J = min(valid)
        summary["ga/best_valid_J"] = self._best_valid_J
        return summary

    def emit(self, event, **payload):
        with self._lock:
            if self._finished:
                raise RuntimeError("cannot emit after logger.finish")
            clean = _safe(payload)
            self._sequence += 1
            row = {**clean, "event": str(event), "sequence": self._sequence,
                   "time": dt.datetime.now(dt.timezone.utc).isoformat()}
            if event == "individual_result":
                identity = (clean.get("generation"), clean.get("individual_index", clean.get("individual_id", self._sequence)))
                self._individuals[identity] = clean
            if event == "generation_end":
                row["generation_summary"] = self._generation_summary(clean)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
            self._save_state()
            generation = clean.get("generation")
            display_generation = generation + 1 if isinstance(generation, (int, float)) else "?"
            if event == "generation_start":
                print(f"GA generation={display_generation} population={clean.get('population_size')}", flush=True)
            elif event == "individual_start":
                print(f"GA evaluate generation={display_generation} individual={clean.get('individual_index', 0) + 1} "
                      f"genome={str(clean.get('genome_id', ''))[:12]} cache={clean.get('cache', False)}", flush=True)
            elif event == "individual_result":
                result = clean.get("result", {})
                print(f"GA result generation={display_generation} individual={clean.get('individual_index', 0) + 1} "
                      f"status={result.get('status')} J={result.get('J')} reason={result.get('reason', '')}", flush=True)
            elif event == "generation_end":
                print(f"GA generation_end={display_generation} {row['generation_summary']}", flush=True)
            if self.run is not None:
                metrics = {"event": str(event), "event_sequence": self._sequence, **_scalars(clean)}
                if generation is not None:
                    metrics["ga/generation"] = generation + 1
                if clean.get("individual_index") is not None:
                    metrics["ga/individual_index"] = clean["individual_index"] + 1
                metrics.update(self._named_metrics(event, clean))
                if "generation_summary" in row:
                    metrics.update(row["generation_summary"])
                    self.run.summary.update(row["generation_summary"])
                if event in ("individual_result", "individual_end"):
                    columns = ["generation", "individual_index", "individual_id", "status", "reason", "requested_genome", "effective_genome", "J"]
                    result = clean.get("result", {})
                    values = [clean.get("generation"), clean.get("individual_index"),
                              clean.get("individual_id", clean.get("genome_id")),
                              result.get("status", clean.get("status")), result.get("reason", clean.get("reason")),
                              json.dumps(clean.get("requested_genome", clean.get("genome")), sort_keys=True),
                              json.dumps(clean.get("effective_genome", result.get("effective_genome")), sort_keys=True), result.get("J")]
                    metrics["ga/individual"] = self._sdk.Table(columns=columns, data=[values])
                self.run.log(metrics, step=self._sequence)
                if event == "generation_end" and self.options.get("upload_artifacts", True):
                    self._upload_snapshot()
            return row

    def _named_metrics(self, event, payload):
        result = payload.get("result", {})
        if not isinstance(result, dict):
            return {}
        names = {"reconstruction_loss": "train/L1", "validation_reconstruction_loss": "validation/L1",
                 "J": "validation/J", "dbi": "clustering/DBI", "dbi_norm": "clustering/DBI_norm",
                 "hoyer_mean": "clustering/Hoyer", "hoyer_std": "clustering/Hoyer_std",
                 "coverage": "clustering/coverage", "dbcv": "clustering/DBCV",
                 "n_clusters": "clustering/cluster_count", "svm_kappa": "svm/kappa",
                 "svm_acc": "svm/accuracy_macro", "svm_f1": "svm/F1_macro",
                 "svm_fit_status": "svm/fit_status", "svm_elapsed_seconds": "timing/svm_seconds"}
        metrics = {destination: result[source] for source, destination in names.items() if source in result}
        if "J" in result:
            metrics["validation/J_valid"] = result["J"] if _valid_result(result) else None
            metrics["validation/valid"] = int(_valid_result(result))
        if "status" in result:
            metrics["validation/status"] = result["status"]
        if "dbcv_status" in result:
            metrics["clustering/DBCV_status"] = result["dbcv_status"]
        weight = float(self._config.get("clustering", {}).get("sparsity_lambda", 0.004))
        if isinstance(result.get("dbi_norm"), (int, float)):
            metrics["fitness/clustering_term"] = (1 - weight) * result["dbi_norm"]
        if isinstance(result.get("hoyer_mean"), (int, float)):
            metrics["fitness/sparsity_term"] = weight * (1 - result["hoyer_mean"])
        if isinstance(result.get("cost"), dict):
            metrics.update(_scalars(result["cost"], "model"))
        if event == "runtime_resources":
            metrics.update(_scalars(result, "resources"))
        if event == "runtime_stage" and "stage" in result and "seconds" in result:
            metrics[f"timing/{result['stage']}_seconds"] = result["seconds"]
        if event == "initialization":
            metrics.update(_scalars(result, "spifil"))
        return metrics

    def _upload_snapshot(self):
        # Allowlist structured logs only. Never traverse image or checkpoint dirs.
        staging = self.directory / "artifact_snapshots" / f"step_{self._sequence:08d}"
        staging.mkdir(parents=True, exist_ok=True)
        sources = [self.path, self.state_path, self.directory / "logger_config.json"]
        for pattern in ("config*.json", "config*.yaml", "config*.yml", "run.yaml", "experiment.json", "data_identity.json",
                        "runtime*.json", "runtime*.jsonl", "search_state.json", "search/search_state.json",
                        "generations.jsonl", "generation*.json"):
            sources.extend(self.directory.glob(pattern))
        artifact = self._sdk.Artifact(f"progressive-logs-{self.run_id}", type="nas-logs",
                                      metadata={"sequence": self._sequence})
        for source in sorted(set(sources)):
            if not source.is_file():
                continue
            if source.suffix == ".jsonl":
                content = "".join(json.dumps(_safe(json.loads(line)), allow_nan=False) + "\n"
                                  for line in source.read_text().splitlines() if line)
            elif source.suffix == ".json":
                content = json.dumps(_safe(json.loads(source.read_text())), indent=2, allow_nan=False)
            else:
                import yaml
                content = json.dumps(_safe(yaml.safe_load(source.read_text())), indent=2, allow_nan=False)
            relative = source.relative_to(self.directory)
            if source.suffix in (".yaml", ".yml"):
                relative = relative.with_suffix(relative.suffix + ".json")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
            artifact.add_file(str(destination), name=str(relative))
        self.run.log_artifact(artifact, aliases=["latest", f"step-{self._sequence}"])

    def finish(self, status="finished"):
        with self._lock:
            if self._finished:
                return
            self.emit("logger_finish", status=status)
            if self.run is not None:
                self.run.summary["status"] = str(status)
                self.run.summary["ga/best_valid_J"] = self._best_valid_J
                if self.options.get("upload_artifacts", True):
                    self._upload_snapshot()
                self.run.finish(exit_code=0 if status == "finished" else 1)
            self._finished = True
