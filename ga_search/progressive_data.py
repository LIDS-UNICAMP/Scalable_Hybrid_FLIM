"""Training/validation data for progressive NAS, using repository preprocessing.

Only RGB files and the repository's three-channel IFT LAB input are validated.
``in_channels != 3`` is rejected. The test split is never loaded. Synthetic
images are annotated fixtures, not evidence of performance on real data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from config import get_single_parasite_paths
from core.constants import PARASITE_NAME
from core.data.loaders import ift_lab_loader
from core.data.transforms import build_test
from flim.spifil import train_samples as hybrid_train_samples
from ga_search import data as existing_data
from spifil.data import Sample


class ReconstructionDataset(Dataset):
    def __init__(self, samples, image_size, imagenet_norm=False):
        self.samples = list(samples)
        self.transform = build_test(image_size, imagenet_norm=imagenet_norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        lab = np.ascontiguousarray(ift_lab_loader(sample.image_path))
        image = self.transform(torch.from_numpy(lab).permute(2, 0, 1).float())
        return image, sample.label, sample.name


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_samples(config, output_dir):
    root = Path(output_dir) / "synthetic_data"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    size = int(config.get("image_size", 200))
    splits = {}
    for split, default in (("train", 4), ("validation", 16)):
        count = int(config.get(f"synthetic_{'val' if split == 'validation' else split}_samples", default))
        if count < 4:
            raise ValueError(f"synthetic {split} requires >=4 samples across two classes")
        images, masks = root / split / "images", root / split / "masks"
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        samples = []
        for i in range(count):
            label = i % 2
            # Two distinct colour populations with reproducible spatial texture;
            # full masks keep enough superpixel donors for a real SPiFiL fit.
            noise = rng.integers(0, 45, (size, size, 3), dtype=np.uint8)
            base = np.array([155, 35, 45] if label == 0 else [35, 155, 100], dtype=np.uint8)
            pixels = noise + base
            name = f"{label}_{split}_{i:04d}.png"
            path, mask_path = images / name, masks / name
            if not path.exists():
                Image.fromarray(pixels).save(path)
            if not mask_path.exists():
                Image.fromarray(np.full((size, size), 255, dtype=np.uint8)).save(mask_path)
            samples.append(Sample(Path(name).stem, path, mask_path, label))
        splits[split] = samples
    return splits["train"], splits["validation"]


def _real_samples(config):
    name = config.get("dataset", "protozoan-cysts")
    split, percentage = int(config.get("split", 1)), int(config.get("percentage", 5))
    ds_cfg = config.get("dataset_config", config.get("dataset_cfg"))
    short = next((key for key, value in PARASITE_NAME.items()
                  if name == key or name == value or name == value.rsplit("_split_", 1)[0]), None)
    use_hybrid = ds_cfg is None and short is not None
    if use_hybrid:
        paths = get_single_parasite_paths(PARASITE_NAME[short], split, percentage)[0]
        ds_cfg = {**paths, "has_mask": True, "train_key": "train", "eval": {"source": "validation"}}
    elif ds_cfg is None:
        ds_cfg = existing_data.load_dataset_cfg(name.replace("-", "_"))
    elif not isinstance(ds_cfg, dict):
        ds_cfg = yaml.safe_load(Path(ds_cfg).read_text())
    ds_cfg = dict(ds_cfg)
    for key in ("images_dir", "masks_dir", "split_json"):
        if ds_cfg.get(key):
            ds_cfg[key] = ds_cfg[key].replace("{root}", str(ds_cfg.get("root", "")))
    if ds_cfg.get("train_key", "train") not in ("train", "training"):
        raise ValueError("SPiFiL and reconstruction must use a training split")
    if ds_cfg.get("eval", {}).get("source", "validation") != "validation":
        raise ValueError("progressive NAS must evaluate validation; test selection is forbidden")
    train = existing_data._samples(ds_cfg, percentage, split, ds_cfg.get("train_key", "train"))
    validation = existing_data._samples(ds_cfg, percentage, split, "validation")
    selection = {"one_per_class": bool(config.get("one_per_class", True)),
                 "n_images": int(config.get("n_images", 200)),
                 "paper_classes": bool(config.get("paper_classes", False)),
                 "seed": int(config.get("seed", 42))}
    if use_hybrid:
        selected = hybrid_train_samples(short, split, percentage, **selection)
    else:
        # Explicit alternative datasets use their existing class/sampling schema.
        selected = existing_data.train_samples(ds_cfg, percentage, split, **selection)
    return train, validation, selected


def load_data(config, output_dir):
    """Return train loader, validation loader, fixed SPiFiL samples, identity.

    Config is flat. Standard defaults: dataset protozoan-cysts, split 1,
    percentage 5, image_size 200, batch_size 2, num_workers 0, seed 42.
    Set synthetic=True for deterministic annotated fixtures. Optional
    dataset_config accepts an existing ga_search dataset YAML path or dict.
    Both loaders yield ``(images, labels, sample_ids)``; ordering is fixed so
    checkpoint RNG state suffices for exact continuation without sampler state.
    """
    config = dict(config)
    if int(config.get("in_channels", 3)) != 3:
        raise ValueError("progressive data loader validates in_channels=3 only (IFT LAB)")
    if config.get("synthetic", False):
        train, validation = _synthetic_samples(config, output_dir)
        selected = list(train)
    else:
        train, validation, selected = _real_samples(config)
    if not train or not validation or not selected:
        raise ValueError("training, validation and SPiFiL donor sets must be nonempty")
    train_names, val_names = {s.name for s in train}, {s.name for s in validation}
    train_paths = {s.image_path.resolve() for s in train}
    val_paths = {s.image_path.resolve() for s in validation}
    train_masks = {s.mask_path.resolve() for s in train if s.mask_path}
    val_masks = {s.mask_path.resolve() for s in validation if s.mask_path}
    if len(train_names) != len(train) or len(val_names) != len(validation):
        raise ValueError("duplicate sample IDs within a split")
    if train_names & val_names or train_paths & val_paths or train_masks & val_masks:
        raise ValueError("train/validation overlap in sample IDs or resolved image/mask paths")
    if not {s.image_path.resolve() for s in selected} <= train_paths:
        raise ValueError("SPiFiL donors must be contained exclusively in training")
    records = {}
    for split, samples in (("train", train), ("validation", validation)):
        records[split] = [{"id": s.name, "label": int(s.label),
                           "image_sha256": _file_hash(s.image_path),
                           "mask_sha256": _file_hash(s.mask_path) if s.mask_path else None}
                          for s in samples]
    identity = {"dataset": "synthetic" if config.get("synthetic", False) else config.get("dataset", "protozoan-cysts"),
                "split": int(config.get("split", 1)), "percentage": int(config.get("percentage", 5)),
                "seed": int(config.get("seed", 42)), "image_size": int(config.get("image_size", 200)),
                "imagenet_norm": bool(config.get("imagenet_norm", False)),
                "selected_sample_ids": [s.name for s in selected], "samples": records}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    loader_kwargs = {"batch_size": int(config.get("batch_size", 2)),
                     "num_workers": int(config.get("num_workers", 0)), "shuffle": False}
    train_dataset = ReconstructionDataset(train, identity["image_size"], identity["imagenet_norm"])
    val_dataset = ReconstructionDataset(validation, identity["image_size"], identity["imagenet_norm"])
    return DataLoader(train_dataset, **loader_kwargs), DataLoader(val_dataset, **loader_kwargs), selected, identity
