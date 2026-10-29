"""Prepare reproducible Eggs/Larvae NAS configurations using train-only holdouts.

Original descriptors are read only. Only their 100% ``train`` list is consumed;
native validation/test entries are never used to construct the new protocol.
Run from the repository root: python -m ga_search.prepare_multires_grid.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random

import yaml

from config import get_single_parasite_paths


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "ga_search/config/progressive_multires"
PERCENTAGES = (1, 5, 25, 50, 75, 100)
SPLITS = (1, 2, 3)
RESOLUTIONS = (128, 160, 200, 256)
DATASETS = ("helminth-eggs", "helminth-larvae")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_exact(path: Path, text: str) -> str:
    content = text.encode("utf-8")
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite a different prepared file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return _sha256(content)


def _json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def prepare(destination: Path = DESTINATION, *, datasets=DATASETS, manifest_name="manifest.json") -> dict:
    destination = Path(destination).resolve()
    base = yaml.safe_load((ROOT / "ga_search/config/progressive_nas_eggs_50.yaml").read_text())
    manifest = {
        "protocol": "train_pool_stratified_fixed_validation_v1",
        "source": "Only native percentage100 train lists; no native validation/test samples consumed",
        "seed": "42 + split",
        "validation_fraction": 0.2,
        "percentage_definition": "Per-class percentage of remaining80% training pool, floor, at least1 perclass",
        "validation_fixed_across_percentages_and_resolutions": True,
        "spifil_donor_protocol": {
            "helminth-eggs": {"one_per_class": True, "n_images": 20},
            "helminth-larvae": {"one_per_class": True, "n_images": 2,
                                "selection": "one training donor per class; reject insufficient patch budgets"},
        },
        "datasets": list(datasets), "splits": list(SPLITS),
        "percentages": list(PERCENTAGES), "image_sizes": list(RESOLUTIONS),
        "total_searches": len(datasets) * len(SPLITS) * len(PERCENTAGES) * len(RESOLUTIONS),
        "descriptor_records": [], "configurations": [],
    }
    for dataset in datasets:
        short = "protozoan" if dataset == "protozoan-cysts" else dataset.removeprefix("helminth-")
        paths = get_single_parasite_paths(dataset, 1, 100)[0]
        metadata = yaml.safe_load((ROOT / f"ga_search/config/dataset_old_{dataset.replace('-', '_')}.yaml").read_text())
        for split in SPLITS:
            source = Path(get_single_parasite_paths(dataset, split, 100)[0]["split_json"])
            # Do not consult any other descriptor key: the native100 validation
            # duplicates train, and native test is explicitly outside this task.
            names = json.loads(source.read_text())["train"]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate IDs in native training pool: {source}")
            groups = {}
            for name in sorted(names):
                groups.setdefault(int(Path(name).stem.split("_", 1)[0]), []).append(name)
            rng = random.Random(42 + split)
            train_groups, validation = {}, []
            for label, group in sorted(groups.items()):
                if len(group) < 2:
                    raise ValueError(f"Insufficient train-only samples for stratified holdout: {dataset}/{split}/{label}")
                rng.shuffle(group)
                count = max(1, int(len(group) * 0.2))
                validation.extend(group[:count])
                train_groups[label] = group[count:]
            validation = sorted(validation)
            previous = set()
            for percentage in PERCENTAGES:
                train = sorted(name for group in train_groups.values()
                               for name in group[:max(1, int(len(group) * percentage / 100))])
                current = set(train)
                assert not current.intersection(validation)
                assert previous <= current
                assert current.union(validation) <= set(names)
                assert {int(n.split("_", 1)[0]) for n in train} == set(groups)
                previous = current
                descriptor = destination / "descriptors" / dataset / f"split{split}" / f"data_descriptor_perc{percentage}.json"
                sha = _write_exact(descriptor, _json({"train": train, "validation": validation}))
                manifest["descriptor_records"].append({
                    "dataset": dataset, "split": split, "percentage": percentage,
                    "source": str(source), "source_train_sha256": _sha256(_json(sorted(names)).encode()),
                    "descriptor": str(descriptor), "sha256": sha,
                    "native_train_pool_count": len(names),
                    "remaining_training_pool_count": sum(map(len, train_groups.values())),
                    "train_count": len(train), "validation_count": len(validation),
                    "class_train_counts": {str(label): sum(int(n.split("_", 1)[0]) == label for n in train)
                                           for label in groups},
                })
        dataset_config = {
            "name": f"{short}_trainonly_fixed_holdout", "root": str(Path(paths["images_dir"]).parent),
            "images_dir": paths["images_dir"], "masks_dir": paths["masks_dir"],
            "split_json": str(destination / "descriptors" / dataset / "split{split}" / "data_descriptor_perc{pct}.json"),
            "has_mask": True, "train_key": "train", "eval": {"source": "validation"},
            "classes": metadata["classes"], "n_classes": metadata["n_classes"],
            "paper_classes": metadata["paper_classes"],
        }
        for size in RESOLUTIONS:
            config = copy.deepcopy(base)
            config.update(dataset=dataset, image_size=size, batch_size=16,
                          dataset_config=dataset_config,
                          output_dir=f"results/nas_multires/{dataset}/r{size}")
            if short == "larvae":
                config.update(one_per_class=True, n_images=2)
            config["genetic"]["space"]["channels"] = ({"eggs": [9, 18, 27, 36],
                "larvae": [4, 8, 12, 16], "protozoan": [7, 14, 21, 28]})[short]
            config["genetic"]["space"]["dropout"] = [0.0, 0.05, 0.1, 0.2, 0.3]
            for block in config["genome"]["blocks"]:
                block["channels"] = {"eggs": 18, "larvae": 8, "protozoan": 14}[short]
            config["wandb"].update(
                name=f"ga_{dataset}_r{size}", group=f"gpt_sol_G_A_multires_{dataset}_r{size}",
                tags=["gpt_sol_G_A", "multires", "train-only-fixed-holdout", "spifil", "genetic-search", f"r{size}"])
            progressive_path = destination / f"{short}_r{size}.yaml"
            config_sha = _write_exact(progressive_path, yaml.safe_dump(config, sort_keys=False))
            ray = {
                "name": f"progressive_nas_{short}_r{size}", "method": "autoencoder", "runner": "growth",
                "grid": {"init": ["flim"], "datasets": [dataset], "splits": list(SPLITS), "percentages": list(PERCENTAGES)},
                "resources": {"gpu_ids": [3 if short == "larvae" else 2], "max_per_gpu": 1,
                              "cpus_per_experiment": 16, "num_workers": 0},
                "runner_args": {"progressive_nas": str(progressive_path)},
                "output": {"work_dir": f"results/nas_multires/{dataset}/r{size}"},
                "wandb": {"enabled": False}, "evaluate": {"SVM": False}, "skip": "state",
            }
            ray_path = destination / f"ray_{short}_r{size}.yaml"
            ray_sha = _write_exact(ray_path, yaml.safe_dump(ray, sort_keys=False))
            manifest["configurations"].append({"dataset": dataset, "image_size": size,
                "progressive_config": str(progressive_path), "sha256": config_sha,
                "ray_config": str(ray_path), "ray_sha256": ray_sha, "searches": 18})
    if "protozoan-cysts" in datasets:
        manifest["spifil_donor_protocol"]["protozoan-cysts"] = {"one_per_class": True, "n_images": 20}
    _write_exact(destination / manifest_name, _json(manifest))
    return manifest


if __name__ == "__main__":
    result = prepare()
    print(f"Prepared {result['total_searches']} searches, {len(result['descriptor_records'])} train/validation descriptors, "
          f"{len(result['configurations'])} progressive and Ray configuration pairs at {DESTINATION}")
