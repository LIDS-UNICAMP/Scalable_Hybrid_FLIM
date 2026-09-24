# -*- coding: utf-8 -*-

"""
Creates stratified train/test splits and incremental training subsets.

Output structure:
  <output_dir>/
    splits/
      split1.json        # keys: train, validation (==train), test  — 50/50 stratified
      split2.json
      ...
    splits_incremental/
      split1/
        data_descriptor_perc1.json    # 1% of the 50% train pool
        data_descriptor_perc5.json
        data_descriptor_perc25.json
        data_descriptor_perc50.json
        data_descriptor_perc75.json
        data_descriptor_perc100.json  # 100% of the 50% train pool (== splits/splitN train)
        ...
      split2/

Filename convention assumed:  <class-id>_<image-id>.ext
  The class label is extracted as everything before the first underscore.

Usage:
  python create_splits.py \\
    -i /path/to/images_folder \\
    -o /path/to/output_dir \\
    -n 5 \\
    [--seed 42] \\
    [--ext png]
"""

import argparse
import json
import os
import random


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate stratified 50/50 splits and incremental training subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        required=True,
        help="Folder containing images. Sub-folders are searched recursively.",
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        required=True,
        help="Root output folder. 'splits/' and 'splits_incremental/' will be created here.",
    )
    parser.add_argument(
        "-n", "--n_splits",
        type=int,
        default=5,
        help="Number of independent splits to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each split uses seed + split_index for reproducibility).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Image file extension to look for (without dot).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image discovery & class grouping
# ---------------------------------------------------------------------------

def collect_images(folder, ext):
    """Return sorted list of image filenames (basename only) found under folder."""
    images = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith("." + ext):
                images.append(fname)
    images = sorted(images)
    if not images:
        raise FileNotFoundError(
            "No *.{} files found under '{}'. "
            "Check --input_folder and --ext.".format(ext, folder)
        )
    return images


def group_by_class(images):
    """Group filenames by the class prefix (text before the first '_')."""
    groups = {}
    for img in images:
        cls = img.split("_")[0]
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(img)
    return groups


# ---------------------------------------------------------------------------
# 50 / 50 stratified split
# ---------------------------------------------------------------------------

def stratified_half_split(groups):
    """
    For each class, shuffle and split 50% -> train, 50% -> test.
    Returns (train_images, test_images) as flat lists.
    """
    train, test = [], []
    for cls in sorted(groups):
        imgs = groups[cls][:]   # copy so original is untouched
        random.shuffle(imgs)
        mid = len(imgs) // 2
        # If only one image exists for a class, put it in train only.
        train.extend(imgs[:mid] if mid > 0 else imgs)
        test.extend(imgs[mid:])
    return train, test


# ---------------------------------------------------------------------------
# Incremental training subsets
# ---------------------------------------------------------------------------

INCREMENTAL_PERCS = [1, 5, 25, 50, 75, 100]


def incremental_subsets(train_images):
    """
    Build cumulative stratified subsets of train_images at each percentage
    in INCREMENTAL_PERCS.

    Strategy:
      - Maintain a running 'selected' set that only grows.
      - At each step, randomly pick enough additional images from the
        remaining pool to reach the target count per class.
      - validation = full train pool MINUS selected (at each step).
      - perc1 is treated specially: sampled from the perc5 selected pool,
        so it is always a strict subset of perc5.

    Returns:
      { perc: (train_list, validation_list) }
    """
    groups = group_by_class(train_images)
    classes = sorted(groups.keys())

    # Running state per class
    selected = {cls: [] for cls in classes}
    results = {}

    for perc in [5, 25, 50, 75, 100]:  # build cumulatively; 1% handled after 5%
        for cls in classes:
            pool = groups[cls]
            n_total = len(pool)
            n_target = max(1, int(n_total * perc / 100))
            n_already = len(selected[cls])
            n_needed = n_target - n_already

            if n_needed > 0:
                selected_set = set(selected[cls])
                remaining = [img for img in pool if img not in selected_set]
                n_needed = min(n_needed, len(remaining))
                newly_picked = random.sample(remaining, n_needed)
                selected[cls].extend(newly_picked)

        # Flatten
        train_list = [img for cls in classes for img in selected[cls]]
        val_list = [
            img
            for cls in classes
            for img in groups[cls]
            if img not in set(selected[cls])
        ]
        # If validation is empty (100% case), mirror train
        if not val_list:
            val_list = train_list[:]

        results[perc] = (train_list, val_list)

        # After building perc5, derive perc1 as a stratified subset of perc5
        if perc == 5:
            perc5_selected = {cls: selected[cls][:] for cls in classes}
            perc1_train = []
            perc1_val = []
            for cls in classes:
                n_total = len(groups[cls])
                n_imgs = max(1, int(n_total * 1 / 100))
                chosen = random.sample(
                    perc5_selected[cls], min(n_imgs, len(perc5_selected[cls]))
                )
                perc1_train.extend(chosen)
                chosen_set = set(chosen)
                perc1_val.extend(img for img in groups[cls] if img not in chosen_set)
            results[1] = (perc1_train, perc1_val)

    return results


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def make_split_json(train, test):
    """50/50 split descriptor. validation == train."""
    return {
        "train": train,
        "validation": train,    # intentionally the same list
        "test": test,
    }


def make_incremental_json(train, val, test):
    return {
        "training": train,
        "validation": val,
        "test": test,
    }


def write_json(path, data):
    dirpath = os.path.dirname(path)

    if dirpath and not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except OSError:
            if not os.path.isdir(dirpath):
                raise

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    print("[INFO] Collecting *.{} images from '{}' ...".format(
        args.ext, args.input_folder
    ))
    images = collect_images(args.input_folder, args.ext)
    groups = group_by_class(images)

    print("[INFO] Found {} images across {} classes:".format(len(images), len(groups)))
    for cls in sorted(groups):
        print("       Class {}: {} images".format(cls, len(groups[cls])))

    splits_dir = os.path.join(args.output_folder, "splits")
    incremental_dir = os.path.join(args.output_folder, "splits_incremental")
    for d in [splits_dir, incremental_dir]:
        if not os.path.exists(d):
           try:
              os.makedirs(d)
           except OSError:
              if not os.path.isdir(d):
                raise

    for split_idx in range(1, args.n_splits + 1):
        # Each split gets its own seed for full reproducibility
        random.seed(args.seed + split_idx)

        print("\n[INFO] -- Split {} / {} --".format(split_idx, args.n_splits))

        # -- 50 / 50 base split -----------------------------------------------
        train, test = stratified_half_split(groups)
        print("       train={}, test={}".format(len(train), len(test)))

        split_path = os.path.join(splits_dir, "split{}.json".format(split_idx))
        write_json(split_path, make_split_json(train, test))
        print("       Written: {}".format(split_path))

        # -- Incremental subsets ----------------------------------------------
        subsets = incremental_subsets(train)
        split_inc_dir = os.path.join(incremental_dir, "split{}".format(split_idx))

        for perc in INCREMENTAL_PERCS:
            inc_train, inc_val = subsets[perc]
            out_path = os.path.join(
                split_inc_dir, "data_descriptor_perc{}.json".format(perc)
            )
            write_json(out_path, make_incremental_json(inc_train, inc_val, test))
            print("       perc{:>3}%  train={:>5}, val={:>5}  -> {}".format(
                perc, len(inc_train), len(inc_val),
                os.path.basename(out_path)
            ))

    print("\n[INFO] Done.")
    print("       Base splits : {}".format(splits_dir))
    print("       Incremental : {}".format(incremental_dir))


if __name__ == "__main__":
    main()
