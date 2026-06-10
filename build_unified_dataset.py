"""
Build one unified, training-ready YOLO dataset from the 6 separately-annotated
depth folders. Applies all class merges (typos, synonyms, Asteraceae subtypes,
rare-taxa -> Other_pollen / Other_NPP) and remaps every label file to a single
canonical class scheme.

Split is DEPTH-BASED (no leakage):
    Train : 1101, 1102, 1104, 1148
    Val   : 1012
    Test  : 1188   (held out — never seen in training)

Output:
    repo/unified_dataset/
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels
        data.yaml
        class_counts.csv      (final per-class counts per split)
"""

import csv
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
# "Annotated data/" = the 6 raw per-depth Roboflow export folders (one per
# slide: 1101, 1102, 1104, 1148, 1012, 1188), not redistributed in this repo.
SRC = Path(__file__).parent / "Annotated data"
OUT = Path(__file__).parent / "unified_dataset"

# ── Depth → split mapping ─────────────────────────────────────────────────────
SPLIT = {
    "1101 annotated": "train",
    "1102 annotated": "train",
    "1104 annotated": "train",
    "1148 annotated": "train",
    "1012 annotated": "valid",
    "1188 annotated": "test",
}

# ── Final canonical class list (order = class ID) ─────────────────────────────
FINAL_CLASSES = [
    # pollen (0-13)
    "Pine", "Artemisia", "Poaceae", "Betula pendula", "Chenopodiaceae",
    "Picea", "Alnus viridis", "Rumex", "Ulmus", "Asteraceae",
    "Cyperaceae", "Apiaceae", "Fagus", "Salix",
    # pollen catch-all (14)
    "Other_pollen",
    # non-pollen objects / NPP (15-20)
    "Type-128", "Lycopodium", "Charcoal",
    "Pediastrum integrum", "Pediastrum boryanum", "Pinus stomata",
    # NPP catch-all (21)
    "Other_NPP",
]
CLASS_ID = {name: i for i, name in enumerate(FINAL_CLASSES)}

# ── Raw label string → canonical class ────────────────────────────────────────
CANON = {
    # --- kept pollen (+ typo/synonym fixes) ---
    "Pine": "Pine",
    "Artemisia": "Artemisia",
    "Poaceae": "Poaceae", "Poacea": "Poaceae",
    "Betula pendula": "Betula pendula",
    "Chenopodiaceae": "Chenopodiaceae",
    "Picea": "Picea",
    "Alnus viridis": "Alnus viridis",
    "Rumex": "Rumex",
    "Ulmus": "Ulmus",
    "Cyperaceae": "Cyperaceae",
    "Apiaceae": "Apiaceae", "Peucedanum": "Apiaceae",
    "Fagus": "Fagus",
    "Salix": "Salix",
    # --- Asteraceae (all subtypes merged) ---
    "Asteraceae": "Asteraceae",
    "Asteraceae senecio": "Asteraceae",
    "Aster type": "Asteraceae",
    "Asteraceae liguliflorae": "Asteraceae",
    "Asteraceae cirsium": "Asteraceae",
    "Achillea": "Asteraceae",
    "Ambrosia": "Asteraceae",
    # --- pollen tail -> Other_pollen ---
    "Other_pollen": "Other_pollen",
    "Thalictrum": "Other_pollen",
    "Aconitum": "Other_pollen",
    "Quercus": "Other_pollen",
    "Polygonum": "Other_pollen", "Polygonum specie": "Other_pollen",
    "Juniperus": "Other_pollen", "Juiperus": "Other_pollen",
    "Corylus": "Other_pollen",
    "Galium": "Other_pollen",
    "Convolvulus": "Other_pollen",
    "Filipendula": "Other_pollen",
    "Ephedra": "Other_pollen",
    "Tilia": "Other_pollen",
    "Fraxinus": "Other_pollen",
    "Carpinus betulus": "Other_pollen",
    "Juglans": "Other_pollen",
    "Larix": "Other_pollen",
    "Brassicaceae": "Other_pollen", "Brassica": "Other_pollen", "Brassicacea": "Other_pollen",
    "Prunus": "Other_pollen",
    "Helianthemum": "Other_pollen", "Heliathemum": "Other_pollen",
    "Soldanella": "Other_pollen",
    "Lonicera": "Other_pollen",
    "Abies": "Other_pollen",
    "Plantago": "Other_pollen",
    # --- kept NPP / non-pollen objects ---
    "Type-128": "Type-128",
    "Lycopodium": "Lycopodium",
    "Charcoal": "Charcoal",
    "Pediastrum integrum": "Pediastrum integrum",
    "Pediastrum boryanum": "Pediastrum boryanum",
    "Pinus stomata": "Pinus stomata",
    # --- NPP tail -> Other_NPP ---
    "Filicales": "Other_NPP",
    "Equisetum": "Other_NPP",
    "Botryococcus": "Other_NPP",
    "Botrychium": "Other_NPP",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main():
    # Prepare output dirs
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))  # split -> class -> count
    unmapped = set()
    n_images = defaultdict(int)
    n_boxes_dropped = 0

    for folder_name, split in SPLIT.items():
        folder = SRC / folder_name
        names = yaml.safe_load(open(folder / "data.yaml"))["names"]
        depth = folder_name.split()[0]                       # e.g. "1101"
        img_dir = folder / "valid" / "images"
        lbl_dir = folder / "valid" / "labels"

        print(f"── {folder_name}  →  {split}  ({len(names)} source classes)")

        for img_path in img_dir.iterdir():
            if img_path.suffix not in IMAGE_EXTS:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            # unique output name (prefix with depth to avoid cross-folder clashes)
            out_stem = f"{depth}_{stem}"
            shutil.copy2(img_path, OUT / split / "images" / f"{out_stem}{img_path.suffix}")

            new_lines = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cid = int(parts[0])
                    if not (0 <= cid < len(names)):
                        n_boxes_dropped += 1
                        continue
                    raw = names[cid]
                    canon = CANON.get(raw)
                    if canon is None:
                        unmapped.add(raw)
                        continue
                    new_id = CLASS_ID[canon]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
                    counts[split][canon] += 1

            (OUT / split / "labels" / f"{out_stem}.txt").write_text("\n".join(new_lines))
            n_images[split] += 1

    # ── Safety check: nothing silently lost ───────────────────────────────────
    if unmapped:
        print("\n❌ ERROR — these raw labels were NOT in the mapping:")
        for u in sorted(unmapped):
            print(f"     {u}")
        print("   Add them to CANON and re-run. Aborting before writing data.yaml.")
        return

    # ── Write data.yaml ───────────────────────────────────────────────────────
    data_yaml = {
        "train": str(OUT / "train" / "images"),
        "val":   str(OUT / "valid" / "images"),
        "test":  str(OUT / "test" / "images"),
        "nc": len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Write class_counts.csv + print report ─────────────────────────────────
    with open(OUT / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "train", "valid", "test", "total"])
        for i, name in enumerate(FINAL_CLASSES):
            tr, va, te = counts["train"][name], counts["valid"][name], counts["test"][name]
            w.writerow([i, name, tr, va, te, tr + va + te])

    print("\n" + "=" * 60)
    print(f"{'ID':<3}{'CLASS':<22}{'TRAIN':>7}{'VAL':>6}{'TEST':>6}{'TOTAL':>7}")
    print("=" * 60)
    for i, name in enumerate(FINAL_CLASSES):
        tr, va, te = counts["train"][name], counts["valid"][name], counts["test"][name]
        flag = "  ← no train!" if tr == 0 else ("  ⚠️ weak" if tr < 50 else "")
        print(f"{i:<3}{name:<22}{tr:>7}{va:>6}{te:>6}{tr+va+te:>7}{flag}")
    print("=" * 60)
    print(f"Images   → train {n_images['train']} | valid {n_images['valid']} | test {n_images['test']}")
    print(f"Classes  → {len(FINAL_CLASSES)}")
    if n_boxes_dropped:
        print(f"⚠️  {n_boxes_dropped} boxes had out-of-range class IDs and were skipped")
    print(f"\n✅ Unified dataset → {OUT}")
    print(f"   data.yaml + class_counts.csv written")


if __name__ == "__main__":
    main()
