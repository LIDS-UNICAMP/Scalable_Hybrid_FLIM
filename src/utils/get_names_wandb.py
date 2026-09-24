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

from __future__ import annotations

import re
from collections import Counter

import wandb

ENTITY = "ophira-ai"     # sua organização / usuário
PROJECT = "flim-ssl"     # nome do projeto

# ─── W&B run fetching ─────────────────────────────────────────────────────────

def get_runs_dict(
    entity: str,
    project: str,
    deduplicate: bool = False,
) -> dict[str, str]:
    """Return ``{run_id: run_name}`` for all runs in the W&B project.

    Args:
        deduplicate: When True, keep only the most-recently-created run per
                     unique run name.  Use this when the same experiment was
                     restarted and you want to evaluate only the latest run.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    if not deduplicate:
        return {run.id: run.name for run in runs}

    # Keep the newest run (by created_at) for each unique name.
    # run.created_at is an ISO-8601 string; lexicographic comparison works.
    best: dict[str, tuple[str, str]] = {}  # name -> (run_id, created_at)
    for run in runs:
        name = run.name
        created_at = run.created_at or ""
        if name not in best or created_at > best[name][1]:
            best[name] = (run.id, created_at)

    return {run_id: name for name, (run_id, _) in best.items()}


# ─── Fine-tuning W&B name builder ────────────────────────────────────────────

_CANONICAL_RE = re.compile(
    r"^lejepa_line_([a-z\-]+)_split_(\d+)_pct_\d+_model_(xavier|random|he|flim|trunc_normal)$"
)


def _base_finetune_name(run_name: str) -> str | None:
    """Convert a canonical experiment name to a base fine-tuning W&B run name.

    Strips the ``pct_*`` segment, yielding:
        ``finetune_lejepa_line_<dataset>_split_<N>_model_<init>``

    Returns ``None`` for non-canonical names (e.g. legacy ``line_p*`` names).
    """
    m = _CANONICAL_RE.match(run_name)
    if not m:
        return None
    return f"finetune_lejepa_line_{m.group(1)}_split_{m.group(2)}_model_{m.group(3)}"


def build_finetune_name_dict(
    experiments: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return ``{run_id: finetune_wandb_name}`` with collision-safe naming.

    Names follow the pattern::

        finetune_lejepa_line_<dataset>_split_<N>_model_<init>

    When two run-ids resolve to the same base name (e.g. different ``pct``
    values), the ``run_id`` is appended to disambiguate::

        finetune_lejepa_line_helminth-eggs_split_1_model_xavier_7rkcbbnk

    Runs whose names cannot be parsed (legacy format) are silently excluded.

    Args:
        experiments: ``{run_id: run_name}`` dict.  Defaults to all runs
                     fetched live from W&B.
    """
    if experiments is None:
        experiments = get_runs_dict(ENTITY, PROJECT)

    # First pass — compute base names
    base_names: dict[str, str] = {}
    for run_id, run_name in experiments.items():
        base = _base_finetune_name(run_name)
        if base is not None:
            base_names[run_id] = base

    # Detect collisions
    counts = Counter(base_names.values())

    # Second pass — apply run_id suffix only where a collision exists
    result: dict[str, str] = {}
    for run_id, base in base_names.items():
        result[run_id] = f"{base}_{run_id}" if counts[base] > 1 else base

    return result


if __name__ == "__main__":
    # ── Live W&B run names ─────────────────────────────────────────────────
    runs_dict = get_runs_dict(ENTITY, PROJECT)
    print("=== Live W&B run dictionary ===")
    for k, v in runs_dict.items():
        print(f"  {k}: {v}")

    # ── Collision-safe finetune names ──────────────────────────────────────
    finetune_names = build_finetune_name_dict()
    print(f"\n=== Fine-tuning W&B names ({len(finetune_names)} entries) ===")
    for run_id, name in finetune_names.items():
        suffix = "  ← collision-resolved" if name.endswith(f"_{run_id}") else ""
        print(f"  {run_id}: {name}{suffix}")