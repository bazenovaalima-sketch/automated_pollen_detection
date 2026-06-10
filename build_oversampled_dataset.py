#!/usr/bin/env python3
# ============================================================================
# build_oversampled_dataset.py — image-level oversampling of rare pollen taxa.
#
# Rare-class mitigation ablation: pollen (Group A) classes with fewer than
# RARE_THRESHOLD training instances are "rare". Every TRAIN image containing at
# least one rare-class box is duplicated K times, increasing the model's
# exposure to rare taxa. Validation and test are left UNCHANGED (no leakage).
#
# Known caveat (stated in the paper): duplicating an image also duplicates any
# common-class boxes it contains; this is the standard limitation of image-level
# oversampling.
#
# Output: unified_dataset_oversampled/ (+ build a zip for Colab afterwards).
# Usage:
#   python build_oversampled_dataset.py
# ============================================================================

import shutil
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SRC  = HERE / "unified_dataset"
DST  = HERE / "unified_dataset_oversampled"
RARE_THRESHOLD = 100      # Group A pollen classes with < this many train boxes
K = 3                     # how many extra copies of each rare-containing image


def main():
    cfg = yaml.safe_load(open(SRC / "data.yaml"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    group_A = set(cfg["group_A_pollen"])

    # 1) count train instances, define rare pollen classes
    cnt = Counter()
    for lb in (SRC / "train" / "labels").glob("*.txt"):
        for line in lb.read_text().splitlines():
            if line.strip():
                cnt[int(float(line.split()[0]))] += 1
    rare = {c for c in group_A if cnt[c] < RARE_THRESHOLD}
    print("Rare pollen classes (<%d train instances):" % RARE_THRESHOLD)
    for c in sorted(rare):
        print(f"  {c:>2} {names[c]:<16} {cnt[c]} instances")

    # 2) fresh copy of the whole dataset
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    # patch data.yaml path -> relative
    dcfg = yaml.safe_load(open(DST / "data.yaml"))
    dcfg["path"] = "."
    yaml.safe_dump(dcfg, open(DST / "data.yaml", "w"),
                   sort_keys=False, allow_unicode=True)

    # 3) duplicate train images that contain a rare class
    img_dir = DST / "train" / "images"
    lbl_dir = DST / "train" / "labels"
    src_imgs = {p.stem: p for p in img_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    n_dup_imgs = 0
    for lb in list(lbl_dir.glob("*.txt")):
        classes = {int(float(l.split()[0]))
                   for l in lb.read_text().splitlines() if l.strip()}
        if classes & rare:                       # image has a rare taxon
            img = src_imgs.get(lb.stem)
            if img is None:
                continue
            for k in range(1, K + 1):
                new_stem = f"{lb.stem}_ovs{k}"
                shutil.copy(img, img_dir / f"{new_stem}{img.suffix}")
                shutil.copy(lb, lbl_dir / f"{new_stem}.txt")
            n_dup_imgs += 1

    n_train = len(list(img_dir.glob("*")))
    print(f"\nDuplicated {n_dup_imgs} rare-containing images ×{K}.")
    print(f"Train images: {len(src_imgs)} -> {n_train}")
    print(f"\n✅ Oversampled dataset at: {DST}")
    print("   val/ and test/ are unchanged (identical to the depth split).")


if __name__ == "__main__":
    main()
