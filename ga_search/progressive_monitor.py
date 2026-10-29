"""Read-only experiment/W&B monitor; only its report and status file are written.

The CLI polls every 60s until both jobs end. It never modifies experiments,
connects to model processes, changes selection, or reads credentials explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit


DEFAULT_ROOTS = ("results/nas_protozoan_50_wandb", "results/nas_eggs_50_wandb")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    rows, malformed = [], 0
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows, malformed
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed += 1  # An in-flight partial final line is safe to retry.
    return rows, malformed


def timestamp(value):
    try:
        return float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def valid(row):
    return row.get("status") in ("ok", "valid", "feasible") and finite(row.get("J"))


def identity(event):
    genome = event.get("requested_genome")
    if genome is not None:
        return hashlib.sha256(json.dumps(genome, sort_keys=True).encode()).hexdigest()
    return event.get("genome_id", event.get("individual_id"))


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


class Monitor:
    def __init__(self, roots=DEFAULT_ROOTS, *, report="reports/ga_monitor_latest.md",
                 status="results/ga_monitor/status.json", api=None, remote=True,
                 stale_seconds=180, now=time.time):
        self.roots = [Path(root) for root in roots]
        self.report, self.status = Path(report), Path(status)
        self.api, self.remote = api, remote
        self.stale_seconds, self.now = stale_seconds, now
        self.remote_errors = {}

    def _online(self, root, logger, now):
        previous = self.remote_errors.get(str(root), 0)
        result = dict(enabled=bool(logger.get("enabled")), verified=False,
                      error_polls=previous, url=None)
        url = logger.get("url")
        if not result["enabled"]:
            result["status"] = "disabled_or_not_initialized"
            return result
        # Only expose a canonical run URL, never URL query parameters or auth.
        parts = urlsplit(url or "")
        path = parts.path.strip("/").split("/")
        if parts.hostname != "wandb.ai" or len(path) != 4 or path[2] != "runs":
            self.remote_errors[str(root)] = previous + 1
            result.update(status="online_url_unavailable", error_polls=previous + 1)
            return result
        result["url"] = f"https://wandb.ai/{'/'.join(path)}"
        if not self.remote:
            result["status"] = "verification_disabled"
            return result
        try:
            if self.api is None:
                import wandb
                self.api = wandb.Api(timeout=15)
            if hasattr(self.api, "flush"):
                self.api.flush()
            run = self.api.run(f"{path[0]}/{path[1]}/{path[3]}")
            summary = dict(run.summary)
            # Allowlist only metrics; never expose API exception messages/config.
            result.update(status="verified", verified=True, state=run.state,
                          remote_step=summary.get("event_sequence", summary.get("_step")),
                          remote_timestamp=summary.get("_timestamp"),
                          best_valid_J=summary.get("ga/best_valid_J"), error_polls=0)
            self.remote_errors[str(root)] = 0
            remote_time = timestamp(result["remote_timestamp"])
            result["remote_age_seconds"] = max(0, now - remote_time) if remote_time is not None else None
            if hasattr(run, "logged_artifacts"):
                artifacts = list(run.logged_artifacts())
                result["nas_log_artifacts"] = sum(getattr(artifact, "type", None) == "nas-logs" for artifact in artifacts)
        except Exception as exc:
            self.remote_errors[str(root)] = previous + 1
            result.update(status="verification_error", verified=False, error_polls=previous + 1,
                          error_type=type(exc).__name__)
        return result

    def inspect(self, root):
        now = self.now()
        events, malformed = read_jsonl(root / "events.jsonl")
        logger = read_json(root / "logger_state.json", {})
        search = read_json(root / "search/search_state.json", {})
        final = read_json(root / "result.json", {})
        exit_path = root / "exit_code.txt"
        terminal = exit_path.exists() or bool(final and search.get("finished"))
        result = dict(root=str(root), exists=root.exists(), terminal=terminal,
                      result_status=final.get("status"),
                      generation_completed=search.get("generation", 0),
                      search_finished=search.get("finished", False),
                      event_count=len(events), event_counts=dict(Counter(row.get("event") for row in events)),
                      malformed_event_lines=malformed, local_sequence=logger.get("sequence"), alerts=[])
        if exit_path.exists():
            result["exit_code"] = exit_path.read_text().strip()
        event_time = timestamp(events[-1].get("time")) if events else None
        age = max(0, now - event_time) if event_time is not None else None
        result.update(last_event=events[-1].get("event") if events else None,
                      last_event_age_seconds=age)
        if not terminal and age is not None and age > self.stale_seconds:
            result["alerts"].append(f"Sem evento GA há {age:.0f}s; verificar etapa ativa, sem assumir processo morto.")
        if not events and root.exists():
            result["alerts"].append("Ainda não há eventos locais verificáveis.")
        unique = {}
        for event in events:
            if event.get("event") == "individual_result":
                key = identity(event)
                if key is not None:
                    unique[key] = event.get("result", {})
        # Files supply completed results before/without the corresponding event.
        for path in (root / "individuals").glob("*/result.json"):
            row = read_json(path, {})
            key = identity(row)
            if key is not None:
                unique.setdefault(key, row)
        rows = list(unique.values())
        good = [row for row in rows if valid(row)]
        bad = [row for row in rows if not valid(row)]
        result.update(unique_architectures=len(rows), valid_architectures=len(good),
                      invalid_architectures=len(bad), invalid_rate=len(bad) / len(rows) if rows else None,
                      invalid_reasons=dict(Counter(row.get("reason", "unspecified") for row in bad)),
                      cached_or_retained_events=sum(bool(row.get("cache")) for row in events
                                                    if row.get("event") == "individual_result"))
        if rows and len(bad) / len(rows) >= .5:
            result["alerts"].append("Pelo menos 50% das arquiteturas únicas são inválidas/inviáveis; revisar motivos e limites.")
        best = min(good, key=lambda row: row["J"], default=None)
        result["best_valid"] = ({key: best.get(key) for key in
                                ("J", "dbi", "dbi_norm", "hoyer_mean", "hoyer_std", "coverage", "dbcv", "depth", "cost")}
                               if best else None)
        epochs = [row for row in events if row.get("event") == "epoch_end"]
        if epochs:
            latest = epochs[-1]
            result["latest_epoch"] = {"depth": latest.get("depth"), "phase": latest.get("phase"),
                                      "epoch": latest.get("epoch"), "metrics": latest.get("result", {})}
            same = [row.get("result", {}) for row in epochs
                    if row.get("individual_id") == latest.get("individual_id") and row.get("depth") == latest.get("depth")]
            if len(same) >= 2:
                a, b = same[-2:]
                if all(finite(row.get(key)) for row in (a, b) for key in ("reconstruction_loss", "J")):
                    if b["reconstruction_loss"] < a["reconstruction_loss"] and b["J"] >= a["J"]:
                        result["alerts"].append("L1 caiu sem melhora de J nesta profundidade: reconstrução não comprova melhora dos clusters.")
        for name in ("runtime_resources.jsonl", "runtime_timings.jsonl"):
            runtime, _ = read_jsonl(root / name)
            if runtime:
                last = runtime[-1]
                result[name.removesuffix(".jsonl")] = {key: last.get(key) for key in
                    ("time", "event", "stage", "seconds", "depth", "total_cpu_percent", "gpu_util_percent", "gpu_memory_mib", "rss_sum_gib")}
        profile = read_json(root / "runtime_profile.json", {})
        result["runtime_profile"] = {key: profile.get(key) for key in
                                     ("cpu_threads", "preload_workers", "dbcv_workers", "cpu_count", "pid")}
        result["remote"] = self._online(root, logger, now)
        online = result["remote"]
        if online["error_polls"] >= 3:
            result["alerts"].append("W&B não verificado em três consultas consecutivas; não declarar sincronização online.")
        local_step, remote_step = logger.get("sequence"), online.get("remote_step")
        if online.get("verified") and isinstance(local_step, int):
            lag = local_step - remote_step if isinstance(remote_step, (int, float)) else None
            result["remote"]["sequence_lag"] = lag
            remote_age = online.get("remote_age_seconds")
            first_event_time = timestamp(events[0].get("time")) if events else None
            unconfirmed_for = (remote_age if remote_age is not None else
                               now - first_event_time if first_event_time is not None else 0)
            if (lag is None or lag > 0) and unconfirmed_for > self.stale_seconds:
                result["alerts"].append("W&B sem sequência atual confirmada após tolerância de 180s; verificar upload.")
            if terminal and online.get("nas_log_artifacts") == 0:
                result["alerts"].append("Nenhum artefato nas-logs confirmado no W&B para execução encerrada.")
        return result

    def poll(self):
        data = dict(time=datetime.fromtimestamp(self.now(), timezone.utc).isoformat(),
                    roots=[self.inspect(root) for root in self.roots])
        atomic(self.status, json.dumps(data, indent=2, allow_nan=False))
        lines = [f"# Monitor GA — {data['time']}", "", "J menor é melhor; L1 é complementar. Contagens usam arquiteturas únicas, sem duplicar elites/cache.", ""]
        for row in data["roots"]:
            lines += [f"## {Path(row['root']).name}", "", f"Gerações completas: {row['generation_completed']}; encerrado: {row['terminal']}. "
                      f"Arquiteturas únicas: {row['unique_architectures']}; válidas: {row['valid_architectures']}; inválidas/inviáveis: {row['invalid_architectures']}."]
            best = row["best_valid"]
            if best:
                lines += [f"Melhor válido: J={best['J']:.8g}, DBI={best['dbi']}, Hoyer={best['hoyer_mean']}, cobertura={best['coverage']}, profundidade={best['depth']}."]
            remote = row["remote"]
            lines += [f"W&B: {remote['status']}; sequência local={row['local_sequence']}; remota={remote.get('remote_step')}."]
            if remote.get("url"):
                lines += [f"[Abrir run]({remote['url']})"]
            lines += ["", *[f"- {alert}" for alert in row["alerts"]], ""]
        atomic(self.report, "\n".join(lines))
        return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--report", default="reports/ga_monitor_latest.md")
    parser.add_argument("--status", default="results/ga_monitor/status.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    monitor = Monitor(args.roots, report=args.report, status=args.status, remote=not args.no_remote)
    while True:
        state = monitor.poll()
        print(json.dumps({"time": state["time"], "runs": [{"root": row["root"], "generation": row["generation_completed"], "alerts": row["alerts"]} for row in state["roots"]]}), flush=True)
        terminal = all(row["terminal"] for row in state["roots"])
        remote_done = all(not row["remote"]["enabled"] or row["remote"].get("state") in ("finished", "failed", "crashed", "killed") or row["remote"]["error_polls"] >= 3 for row in state["roots"])
        if args.once or (terminal and (remote_done or args.no_remote)):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
