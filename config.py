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

"""
Dataset path registry and split-file helpers used by DatasetParasite.

Layout convention
-----------------
Each dataset is registered in DATASETS below with two keys:
  - "images" : directory that holds the image files.
  - "splits"  : directory that holds the split JSON files.

Split JSON files are named following the pattern:
  split_{split}_p{percentage}.json

where `split` is an integer fold index and `percentage` is the
fraction of labelled training data (e.g. 100 means full set).

Split JSON structure:
  {
    "training":   ["<label>_<img_id>.png", ...],
    "validation": ["<label>_<img_id>.png", ...],
    "test":       ["<label>_<img_id>.png", ...]
  }

Filenames encode the class label as the first field (1-indexed),
e.g. "000002_00000042.png" → class index 1 (after subtracting 1).
"""

import os
import re

# ---------------------------------------------------------------------------
# Root of the project (resolved relative to this file so it works regardless
# of the working directory from which scripts are launched).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs(rel_path: str) -> str:
    """Convert a path relative to the project root to an absolute path."""
    return os.path.join(_PROJECT_ROOT, rel_path)


# ---------------------------------------------------------------------------
# Dataset registry
# Add a new entry here whenever you introduce a new dataset.
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict[str, str]] = {
    "to_modules": {
        "images": _abs("data/to_modules/images"),
        "splits": _abs("data/to_modules/splits"),
    },
    "parasito": {
        "classes": [
            {
                "name": "helminth-eggs_split_2",
                "images": _abs("data/to_modules/new_split_parasito/helminth-eggs/images"),
                "masks": _abs("data/to_modules/new_split_parasito/helminth-eggs/masks"),
                "splits": _abs("data/to_modules/new_split_parasito/helminth-eggs/splits"),
                "splits_incremental": _abs("data/to_modules/new_split_parasito/helminth-eggs/splits_incremental"),
            },
            {
                "name": "helminth-larvae_split_2",
                "images": _abs("data/to_modules/new_split_parasito/helminth-larvae/images"),
                "masks": _abs("data/to_modules/new_split_parasito/helminth-larvae/masks"),
                "splits": _abs("data/to_modules/new_split_parasito/helminth-larvae/splits"),
                "splits_incremental": _abs("data/to_modules/new_split_parasito/helminth-larvae/splits_incremental"),
            },
            {
                "name": "protozoan-cysts_split_2",
                "images": _abs("data/to_modules/new_split_parasito/protozoan-cysts/images"),
                "masks": _abs("data/to_modules/new_split_parasito/protozoan-cysts/masks"),
                "splits": _abs("data/to_modules/new_split_parasito/protozoan-cysts/splits"),
                "splits_incremental": _abs("data/to_modules/new_split_parasito/protozoan-cysts/splits_incremental"),
            },
        ],
    },
}


def get_dataset_paths(dataset_name: str) -> dict[str, str]:
    """Return the path dict for *dataset_name*.

    Returns a dict with at least the keys ``"images"`` and ``"splits"``.

    Raises
    ------
    KeyError
        If *dataset_name* is not registered in :data:`DATASETS`.
    """
    if dataset_name not in DATASETS:
        available = list(DATASETS.keys())
        raise KeyError(
            f"Dataset '{dataset_name}' is not registered. "
            f"Available datasets: {available}. "
            f"Add it to DATASETS in config.py."
        )
    return DATASETS[dataset_name]


def get_parasito_split_paths(split: int | str, percentage: int | float, name: str="parasito") -> list[dict]:
    """Return per-class split info for the parasito dataset.

    Each element is a dict with keys ``"images_dir"`` and ``"split_json"``.

    For the full split (percentage is None) the file is::
        <splits_dir>/split{split}.json

    For incremental splits the file is::
        <splits_incremental_dir>/split{split}/data_descriptor_perc{pct}.json
    """
    classes = DATASETS[name]["classes"]
    result = []
    for cls in classes:
        if percentage is None:
            json_path = os.path.join(cls["splits"], f"split{split}.json")
        else:
            pct = int(percentage) if float(percentage) == int(percentage) else percentage
            json_path = os.path.join(
                cls["splits_incremental"],
                f"split{split}",
                f"data_descriptor_perc{pct}.json",
            )
        result.append({
            "name": cls["name"],
            "images_dir": cls["images"],
            "masks_dir": cls["masks"],
            "split_json": json_path,
        })
    return result


def get_single_parasite_paths(
    parasite_name: str, split: int | str, percentage: int | float
) -> list[dict]:
    """Return split info for a **single** parasite (single-parasite mode).

    Identical in structure to :func:`get_parasito_split_paths` but restricted
    to the one parasite identified by *parasite_name*.

    Parameters
    ----------
    parasite_name:
        One of the registered parasite names, e.g. ``"helminth-eggs_split_2"``.
    split:
        Fold index (integer) or a custom string identifier.
    percentage:
        Percentage of labelled training data.  ``None`` selects the full
        (non-incremental) split JSON.

    Raises
    ------
    KeyError
        If *parasite_name* is not found in the ``"parasito"`` registry.
    """
    # Support ``"helminth-eggs_split_2"``-style names.
    # Lookup strategy:
    #   1. Exact match on registered class name.
    #   2. Strip ``_split_N`` suffix from both the lookup key and each
    #      registered name and match on the resulting base names.
    # This allows splits 1/2/3 to all resolve to the same image/split paths
    # regardless of which variant is registered in DATASETS.
    base_name = re.sub(r"_split_\d+$", "", parasite_name)
    classes = DATASETS["parasito"]["classes"]
    cls = next((c for c in classes if c["name"] == parasite_name), None)
    if cls is None:
        cls = next(
            (c for c in classes if re.sub(r"_split_\d+$", "", c["name"]) == base_name),
            None,
        )
    if cls is None:
        available = [c["name"] for c in classes]
        raise KeyError(
            f"Parasite '{parasite_name}' (base='{base_name}') not registered. "
            f"Available: {available}"
        )
    if percentage is None:
        json_path = os.path.join(cls["splits"], f"split{split}.json")
    else:
        pct = int(percentage) if float(percentage) == int(percentage) else percentage
        json_path = os.path.join(
            cls["splits_incremental"],
            f"split{split}",
            f"data_descriptor_perc{pct}.json",
        )
    return [{
        "name": cls["name"],
        "images_dir": cls["images"],
        "masks_dir": cls["masks"],
        "split_json": json_path,
    }]


def get_split_path_incremental(
    dataset_name: str, split: int | str, percentage: int | float
) -> str:
    """Return the absolute path to the split JSON file.

    The file is expected at::

        <splits_dir>/split_{split}_p{percentage}.json

    Parameters
    ----------
    dataset_name:
        Key registered in :data:`DATASETS`.
    split:
        Fold index (integer) or a custom string identifier.
    percentage:
        Percentage of labelled training data (e.g. 100 for the full set).
        Stored as an integer in the filename (100.0 → "100").

    Returns
    -------
    str
        Absolute path to the JSON file (may or may not exist yet).
    """
    paths = get_dataset_paths(dataset_name)
    splits_dir = paths["splits"]
    pct = int(percentage) if float(percentage) == int(percentage) else percentage
    filename = f"split_{split}_p{pct}.json"
    return os.path.join(splits_dir, filename)
