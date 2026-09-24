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

"""run_experiments.py — Grid runner for all percentage/** experiments.

Discovers every YAML under configs/data/percentage/ and runs it against
every model variant listed in `model_variants`.

Log file
--------
logs/run_experiments_run_ids.log  (append-only, one structured line per run)

Line format (tab-separated):
    <ISO-timestamp> TAB <experiment_name> TAB <run_id> TAB <yaml_path> TAB <model_path> TAB <status> [TAB <reason>]
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# FLIM weight resolution
# ---------------------------------------------------------------------------

_FLIM_BASE = "data/to_mateus/model/ch24_32_48_a0.5_f5"

# Maps dataset parasite prefix → FLIM subfolder name
_PARASITE_KEY_MAP: dict[str, str] = {
    "helminth-eggs":   "eggs",
    "helminth-larvae": "larvae",
    "protozoan-cysts": "protozoan",
}

# Exact per-parasite-split FLIM model YAMLs: (parasite_key, split_id) → yaml.
# Each YAML already contains the correct arch_json and flim_weights_path —
# no CLI overrides are needed.
_FLIM_YAMLS: dict[tuple[str, str], str] = {
    ("eggs",      "1"): "configs/model/lejepa_line_flim_eggs_train1.yaml",
    ("eggs",      "2"): "configs/model/lejepa_line_flim_eggs_train2.yaml",
    ("eggs",      "3"): "configs/model/lejepa_line_flim_eggs_train3.yaml",
    ("larvae",    "1"): "configs/model/lejepa_line_flim_larvae_train1.yaml",
    ("larvae",    "2"): "configs/model/lejepa_line_flim_larvae_train2.yaml",
    ("larvae",    "3"): "configs/model/lejepa_line_flim_larvae_train3.yaml",
    ("protozoan", "1"): "configs/model/lejepa_line_flim_protozoan_train1.yaml",
    ("protozoan", "2"): "configs/model/lejepa_line_flim_protozoan_train2.yaml",
    ("protozoan", "3"): "configs/model/lejepa_line_flim_protozoan_train3.yaml",
}


def _resolve_flim(components: dict[str, str]) -> str:
    """Return the exact FLIM model YAML for the given experiment components.

    The selected YAML already contains the correct arch_json and
    flim_weights_path for this (parasite, split) — no CLI overrides needed.

    Raises ValueError if the parasite or split is unknown, or
    FileNotFoundError if the YAML or its referenced paths are missing.
    """
    parasite = components["parasite"]
    split_id = components["split_id"]

    parasite_key = _PARASITE_KEY_MAP.get(parasite)
    if parasite_key is None:
        raise ValueError(
            f"Cannot resolve FLIM weights: unknown parasite '{parasite}'. "
            f"Known: {list(_PARASITE_KEY_MAP)}"
        )

    key = (parasite_key, split_id)
    model_yaml = _FLIM_YAMLS.get(key)
    if model_yaml is None:
        raise ValueError(
            f"No FLIM YAML registered for (parasite_key='{parasite_key}', "
            f"split_id='{split_id}'). Known: {list(_FLIM_YAMLS)}"
        )

    if not os.path.isfile(model_yaml):
        raise FileNotFoundError(
            f"FLIM model YAML not found: '{model_yaml}' "
            f"(dataset='{parasite}_split_{split_id}')"
        )
    train_key = f"train{split_id}"
    base = os.path.join(_FLIM_BASE, parasite_key, train_key)
    arch_json = os.path.join(base, "architecture.json")
    flim_weights = os.path.join(base, "models")
    if not os.path.isfile(arch_json):
        raise FileNotFoundError(
            f"FLIM architecture.json not found: '{arch_json}' "
            f"(dataset='{parasite}_split_{split_id}')"
        )
    if not os.path.isdir(flim_weights):
        raise FileNotFoundError(
            f"FLIM models/ directory not found: '{flim_weights}' "
            f"(dataset='{parasite}_split_{split_id}')"
        )

    return model_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_stream_and_capture(
    cmd: List[str],
    env: dict[str, str],
) -> tuple[int, str]:
    """Stream subprocess output to terminal AND capture it.

    Returns (return_code, combined_output).
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def _tail_single_line(text: str, n_chars: int = 300) -> str:
    """Return the last *n_chars* characters of *text*, collapsed to one line."""
    tail = text[-n_chars:] if len(text) > n_chars else text
    return " ".join(tail.split())


def _extract_wandb_run_id(output: str) -> Optional[str]:
    """Try to extract a W&B run ID from subprocess output.

    Handles three forms:
      - Online URL:  .../runs/<run_id>
      - Online dir:  wandb/run-YYYYMMDD_HHMMSS-<run_id>
      - Offline dir: wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
    """
    # Pattern 1: URL
    m = re.search(r"wandb\.ai/[^/]+/[^/]+/runs/([a-zA-Z0-9]+)", output)
    if m:
        return m.group(1)
    # Pattern 2: online or offline local dir
    m = re.search(r"(?:offline-)?run-\d{8}_\d{6}-([a-zA-Z0-9]+)", output)
    if m:
        return m.group(1)
    return None


def _parse_yaml_path(yaml_path: str) -> dict[str, str]:
    """Extract experiment components from a percentage/** YAML path.

    Expected structure:
        configs/data/percentage/<parasite>_split_<split_id>/<pct>/<file>.yaml

    Returns a dict with keys: parasite, split_id, pct, yaml_stem.
    """
    parts = yaml_path.replace("\\", "/").split("/")
    # Find the index of the <parasite>_split_N folder
    # (the first folder whose name matches *_split_*)
    parasite_split_folder = ""
    pct_folder = ""
    yaml_file = os.path.basename(yaml_path)
    for i, part in enumerate(parts):
        if re.search(r"_split_\d+$", part):
            parasite_split_folder = part
            if i + 1 < len(parts):
                pct_folder = parts[i + 1]
            break

    if not parasite_split_folder:
        # Fallback for unexpected layouts
        return {
            "parasite": "unknown",
            "split_id": "0",
            "pct": pct_folder or "0",
            "yaml_stem": os.path.splitext(yaml_file)[0],
        }

    m = re.match(r"^(.+)_split_(\d+)$", parasite_split_folder)
    parasite = m.group(1) if m else parasite_split_folder
    split_id = m.group(2) if m else "0"

    return {
        "parasite": parasite,
        "split_id": split_id,
        "pct": pct_folder,
        "yaml_stem": os.path.splitext(yaml_file)[0],
    }


def _experiment_name(components: dict[str, str], model_tag: str) -> str:
    """Build a unique, filesystem-safe experiment name."""
    return (
        f"{components['yaml_stem']}"
        f"_{components['parasite']}"
        f"_split_{components['split_id']}"
        f"_pct_{components['pct']}"
        f"_model_{model_tag}"
    )


def _log_run(
    log_path: str,
    *,
    experiment_name: str,
    run_id: Optional[str],
    yaml_path: str,
    model_path: str,
    status: str,
    reason: str = "",
) -> None:
    """Append a single structured line to the run-ID log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    rid = run_id or "n/a"
    parts = [ts, experiment_name, rid, yaml_path, model_path, status]
    if reason:
        parts.append(reason)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\t".join(parts) + "\n")


def _load_done_runs(log_path: str) -> set[str]:
    """Return the set of experiment names that completed with status OK.

    Only OK runs are skipped on resume; FAILED runs are retried.
    """
    done: set[str] = set()
    if not os.path.isfile(log_path):
        return done
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            # format: timestamp, exp_name, run_id, yaml, model, status [, reason]
            if len(parts) >= 6 and parts[5] == "OK":
                done.add(parts[1])
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Model variants (source of truth for model configs) ──────────────────
    # "flim" model_cfg is a placeholder replaced per-experiment by _resolve_flim(),
    # which selects the exact per-parasite-split YAML (no CLI overrides needed).
    model_variants: List[Tuple[str, str]] = [
        ("xavier", "configs/model/lejepa_line_xavier.yaml"),
        ("random", "configs/model/lejepa_line_random.yaml"),
        ("he",     "configs/model/lejepa_line_he.yaml"),
        ("flim",   ""),  # resolved dynamically via _resolve_flim()
    ]

    # ── Discover all YAMLs under configs/data/percentage/ ───────────────────
    yaml_pattern = "configs/data/percentage/**/*.yaml"
    all_yamls: List[str] = sorted(glob.glob(yaml_pattern, recursive=True))

    if not all_yamls:
        print(f"No YAMLs found matching: {yaml_pattern}")
        return

    total_runs = len(all_yamls) * len(model_variants)

    # ── Structured log file ─────────────────────────────────────────────────
    log_path = "logs/run_experiments_run_ids.log"
    os.makedirs("logs", exist_ok=True)

    # ── Per-run W&B dir ─────────────────────────────────────────────────────
    base_wandb_dir = "wandb_runs"
    os.makedirs(base_wandb_dir, exist_ok=True)

    # ── Resume: skip experiments already logged as OK ────────────────────────
    done_runs = _load_done_runs(log_path)

    succeeded = len(done_runs)
    failed = 0
    failures: List[dict] = []

    print(f"\nDiscovered {len(all_yamls)} YAML(s) × {len(model_variants)} model(s) = {total_runs} total runs")
    if done_runs:
        print(f"Resuming: {len(done_runs)} already-OK run(s) will be skipped.")
    print(f"Run-ID log: {log_path}\n")

    for yaml_path in all_yamls:
        components = _parse_yaml_path(yaml_path)

        for model_tag, model_cfg in model_variants:
            exp_name = _experiment_name(components, model_tag)

            if exp_name in done_runs:
                print(f"[SKIP] {exp_name}")
                continue

            wandb_dir = os.path.join(base_wandb_dir, exp_name)
            os.makedirs(wandb_dir, exist_ok=True)

            # ── Resolve exact FLIM YAML (fail fast if missing) ──────────────
            if model_tag == "flim":
                try:
                    model_cfg = _resolve_flim(components)
                except (ValueError, FileNotFoundError) as exc:
                    failed += 1
                    reason = str(exc)
                    failures.append({"name": exp_name, "rc": -1, "reason": reason})
                    _log_run(
                        log_path,
                        experiment_name=exp_name,
                        run_id=None,
                        yaml_path=yaml_path,
                        model_path="flim/unresolved",
                        status="FAILED",
                        reason=reason,
                    )
                    print(f"[FAILED] {exp_name}")
                    print(f"  Reason: {reason}")
                    continue

            group = f"{components['parasite']}_split_{components['split_id']}"
            tags = [
                f"p{components['pct']}",
                model_tag,
                "line",
                components['parasite'],
                f"split{components['split_id']}",
            ]
            tags_str = "[" + ",".join(f'"{t}"' for t in tags) + "]"

            env = os.environ.copy()
            env["WANDB_DIR"] = wandb_dir

            cmd = [
                sys.executable, "-m", "src.main",
                "fit",
                "--config", "configs/default.yaml",
                "--config", yaml_path,
                "--config", model_cfg,
                "--use_wandb", "true",
                "--trainer.accelerator=gpu",
                f"--trainer.logger.init_args.group={group}",
                f"--trainer.logger.init_args.tags={tags_str}",
                f"--trainer.logger.init_args.name={exp_name}",
            ]

            print(f"\n{'='*80}")
            print(f"RUN  : {exp_name}")
            print(f"YAML : {yaml_path}")
            print(f"MODEL: {model_cfg}")
            print(f"{'='*80}")

            return_code, output = run_stream_and_capture(cmd, env=env)
            run_id = _extract_wandb_run_id(output)

            if return_code == 0:
                succeeded += 1
                _log_run(
                    log_path,
                    experiment_name=exp_name,
                    run_id=run_id,
                    yaml_path=yaml_path,
                    model_path=model_cfg,
                    status="OK",
                )
                print(f"[OK] {exp_name}  run_id={run_id}")
            else:
                failed += 1
                reason = _tail_single_line(output)
                failures.append({"name": exp_name, "rc": return_code, "reason": reason})
                _log_run(
                    log_path,
                    experiment_name=exp_name,
                    run_id=run_id,
                    yaml_path=yaml_path,
                    model_path=model_cfg,
                    status="FAILED",
                    reason=reason,
                )
                print(f"[FAILED rc={return_code}] {exp_name}  run_id={run_id}")
                print(f"  Reason: {reason[:200]}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'#'*80}")
    print(f"SUMMARY")
    print(f"  Total planned : {total_runs}")
    print(f"  Succeeded     : {succeeded}")
    print(f"  Failed        : {failed}")
    if failures:
        print("\nFailed runs:")
        for i, f in enumerate(failures, 1):
            print(f"  {i:03d}) {f['name']}  rc={f['rc']}  {f['reason'][:120]}")
    print(f"\nRun-ID log: {log_path}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
