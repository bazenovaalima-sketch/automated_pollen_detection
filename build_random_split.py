#!/usr/bin/env python3
# ============================================================================
# build_random_split.py — make a RANDOM-split copy of the unified dataset for
# the leakage ablation. Pools all images from the depth-based train/valid/test
# and re-partitions them randomly into the SAME sizes (2523 / 120 / 358).
#
# A random split mixes images from the same slide across train and test, so the
# test set leaks slide-specific appearance → inflated metrics. Comparing a model
# trained on this split against the depth split quantifies that leakage.
#
# Usage:
#   python build_random_split.py
# ============================================================================

import random
import shutil
from pathlib import Path

import yaml

HERE   = Path(__file__).parent
SRC    = HERE / "unified_dataset"
DST    = HERE / "unified_dataset_random"
SEED   = 42                       # reproducible split
SPLITS = ["train", "valid", "test"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    # 1) pool every (image, label) pair from all depth-based splits
    pairs = []
    for sp in SPLITS:
        img_dir = SRC / sp / "images"
        lbl_dir = SRC / sp / "labels"
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, lbl_dir / (img.stem + ".txt")))

    sizes = {sp: len(list((SRC / sp / "images").iterdir())) for sp in SPLITS}
    print(f"Pooled {len(pairs)} images. Target sizes (same as depth split): {sizes}")

    # 2) shuffle and re-partition into identical sizes
    random.seed(SEED)
    random.shuffle(pairs)
    n_train, n_valid = sizes["train"], sizes["valid"]
    parts = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test":  pairs[n_train + n_valid:],
    }

    # 3) write out
    if DST.exists():
        shutil.rmtree(DST)
    for sp, items in parts.items():
        (DST / sp / "images").mkdir(parents=True, exist_ok=True)
        (DST / sp / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy(img, DST / sp / "images" / img.name)
            if lbl.exists():
                shutil.copy(lbl, DST / sp / "labels" / lbl.name)
        print(f"  {sp}: {len(items)} images")

    # 4) copy data.yaml (keep class names + group tags, relative paths)
    cfg = yaml.safe_load(open(SRC / "data.yaml"))
    cfg["path"] = "."
    yaml.safe_dump(cfg, open(DST / "data.yaml", "w"),
                   sort_keys=False, allow_unicode=True)

    print(f"\n✅ Random-split dataset at: {DST}")
    print("   Same sizes as depth split, but slides are mixed across train/test.")


if __name__ == "__main__":
    main()
