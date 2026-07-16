# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""wandb_cache.py — Local cache for W&B run metadata.

Stores a snapshot of W&B project runs in::

    configs/wandb_update/ids_wandb.json

Scripts use this cache by default to avoid repeated W&B API calls.
Pass ``update=True`` (or ``--wandb-update`` on the CLI) to refresh from W&B.

Usage::

    # Refresh cache from W&B
    python -m src.utils.wandb_cache --update

    # Show cache summary
    python -m src.utils.wandb_cache
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(_ROOT, "configs", "wandb_update")
CACHE_FILE = os.path.join(CACHE_DIR, "ids_wandb.json")


# ─── Internal W&B fetch ───────────────────────────────────────────────────────


def _to_plain(obj):
    """Recursively convert W&B types (SummarySubDict, etc.) to plain Python."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    # W&B SummarySubDict and similar objects expose .items() like a dict
    if hasattr(obj, "items"):
        try:
            return {k: _to_plain(v) for k, v in obj.items()}
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    # Fallback: cast to string to avoid TypeError
    return str(obj)


def _fetch_from_wandb(entity: str, project: str) -> dict[str, dict]:
    """Fetch all runs from W&B and return ``{run_id: metadata_dict}``.

    Only ``name``, ``created_at``, and ``state`` are fetched — ``config`` and
    ``summary`` are skipped to avoid per-run lazy API calls that dominate
    latency on large projects (1 000+ runs).  ``per_page=1000`` minimises the
    number of HTTP round-trips.
    """
    import wandb  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    api  = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=1000)
    result: dict[str, dict] = {}
    for run in tqdm(runs, desc=f"Fetching {entity}/{project}", unit="run"):
        result[run.id] = {
            "name":       run.name,
            "created_at": run.created_at or "",
            "state":      getattr(run, "state", ""),
            "config":     {},
            "summary":    {},
        }
    return result


# ─── Cache I/O ────────────────────────────────────────────────────────────────


def save_cache(
    entity: str,
    project: str,
    cache_path: str = CACHE_FILE,
) -> dict[str, dict]:
    """Fetch from W&B, persist to *cache_path*, and return the raw runs dict."""
    print(f"[W&B cache] Refreshing from {entity}/{project} …")
    runs = _fetch_from_wandb(entity, project)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "entity": entity,
        "project": project,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "runs": runs,
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(
        f"[W&B cache] Saved {len(runs)} runs → "
        f"{os.path.relpath(cache_path, _ROOT)}"
    )
    return runs


def load_cache(cache_path: str = CACHE_FILE) -> dict | None:
    """Load cache from disk.  Returns ``None`` if the file does not exist."""
    if not os.path.isfile(cache_path):
        return None
    with open(cache_path, encoding="utf-8") as fh:
        return json.load(fh)


# ─── Public API ───────────────────────────────────────────────────────────────


def get_runs_dict_cached(
    entity: str,
    project: str,
    deduplicate: bool = False,
    update: bool = False,
    cache_path: str = CACHE_FILE,
) -> dict[str, str]:
    """Return ``{run_id: run_name}`` using the local cache.

    Behaviour:

    * ``update=False`` (default): load from *cache_path*; auto-fetch from W&B
      on first run (cache absent) and save a fresh copy.
    * ``update=True``: always fetch from W&B and overwrite *cache_path*.

    Args:
        entity:      W&B entity (user / organisation).
        project:     W&B project name.
        deduplicate: When ``True``, keep only the most-recently-created run
                     per unique run name (same semantics as the live
                     ``get_runs_dict``).
        update:      When ``True``, refresh cache from W&B before returning.
        cache_path:  Absolute path to the JSON cache file.

    Returns:
        ``{run_id: run_name}`` mapping.
    """
    if update:
        runs = save_cache(entity, project, cache_path)
    else:
        cached = load_cache(cache_path)
        if cached is None:
            print(
                f"[W&B cache] No cache at "
                f"{os.path.relpath(cache_path, _ROOT)} — fetching from W&B …"
            )
            runs = save_cache(entity, project, cache_path)
        else:
            age = cached.get("updated_at", "unknown")
            runs_count = len(cached.get("runs", {}))
            print(
                f"[W&B cache] Using local cache "
                f"({runs_count} runs, updated {age})"
            )
            runs = cached.get("runs", {})

    if not deduplicate:
        return {rid: info["name"] for rid, info in runs.items()}

    # Keep the newest run (by created_at) for each unique name.
    # ISO-8601 strings compare lexicographically.
    best: dict[str, tuple[str, str]] = {}  # name → (run_id, created_at)
    for rid, info in runs.items():
        name = info["name"]
        created_at = info.get("created_at", "")
        if name not in best or created_at > best[name][1]:
            best[name] = (rid, created_at)

    return {rid: name for name, (rid, _) in best.items()}


# ─── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Manage the local W&B metadata cache (ids_wandb.json)."
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Fetch fresh metadata from W&B and overwrite the cache.",
    )
    parser.add_argument(
        "--cache-path", default=CACHE_FILE,
        help=f"Cache file path (default: {os.path.relpath(CACHE_FILE, _ROOT)})",
    )
    args = parser.parse_args()

    if args.update:
        runs = save_cache(ENTITY, PROJECT, args.cache_path)
        print(f"Cache updated with {len(runs)} runs.")
    else:
        cached = load_cache(args.cache_path)
        if cached is None:
            print(f"No cache found at {args.cache_path}. Run with --update to create it.")
        else:
            print(f"Cache at: {args.cache_path}")
            print(f"  Entity  : {cached.get('entity')}")
            print(f"  Project : {cached.get('project')}")
            print(f"  Updated : {cached.get('updated_at')}")
            print(f"  Runs    : {len(cached.get('runs', {}))}")
