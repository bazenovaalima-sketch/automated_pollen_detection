# ============================================================================
# yolo_to_coco.py — convert the unified YOLO-format dataset to COCO format
# for RF-DETR training. Run ONCE (locally or in Colab).
#
# INPUT  (YOLO):                       OUTPUT (COCO, what RF-DETR expects):
#   unified_dataset/                     unified_dataset_coco/
#     train/images/*.jpg                   train/*.jpg
#     train/labels/*.txt                   train/_annotations.coco.json
#     valid/...                            valid/*.jpg + _annotations.coco.json
#     test/...                             test/*.jpg  + _annotations.coco.json
#
# Usage:
#   python yolo_to_coco.py --src unified_dataset --dst unified_dataset_coco
# ============================================================================

import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

# YOLO splits → RF-DETR/COCO split folder names (kept identical here)
SPLITS = {"train": "train", "valid": "valid", "test": "test"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_class_names(src: Path):
    """Read class names from data.yaml (id -> name)."""
    cfg = yaml.safe_load(open(src / "data.yaml"))
    names = cfg["names"]
    # names can be a dict {0: 'Pine', ...} or a list ['Pine', ...]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def yolo_box_to_coco(cx, cy, bw, bh, img_w, img_h):
    """Normalized YOLO (center x,y,w,h) -> absolute COCO (x_min,y_min,w,h)."""
    w = bw * img_w
    h = bh * img_h
    x_min = (cx * img_w) - w / 2.0
    y_min = (cy * img_h) - h / 2.0
    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]


def convert_split(src: Path, dst: Path, split_in: str, split_out: str,
                  class_names, ann_start_id=1):
    img_dir = src / split_in / "images"
    lbl_dir = src / split_in / "labels"
    out_dir = dst / split_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # COCO categories: 1-indexed (COCO convention); category_id = yolo_id + 1
    categories = [
        {"id": i + 1, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]

    images, annotations = [], []
    img_id = 1
    ann_id = ann_start_id

    img_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )

    for img_path in img_files:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # copy image into the split folder (flat, next to the json)
        shutil.copy(img_path, out_dir / img_path.name)

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                box = yolo_box_to_coco(cx, cy, bw, bh, img_w, img_h)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,          # 1-indexed
                    "bbox": box,
                    "area": round(box[2] * box[3], 2),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_json = out_dir / "_annotations.coco.json"
    json.dump(coco, open(out_json, "w"))

    print(f"  [{split_out}] {len(images)} images, {len(annotations)} annotations "
          f"-> {out_json}")
    return ann_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="unified_dataset",
                    help="source YOLO dataset dir (with data.yaml)")
    ap.add_argument("--dst", default="unified_dataset_coco",
                    help="output COCO dataset dir for RF-DETR")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert (src / "data.yaml").exists(), f"data.yaml not found in {src}"

    class_names = load_class_names(src)
    print(f"Classes ({len(class_names)}): {class_names}\n")
    print(f"Converting {src} -> {dst}")

    ann_id = 1
    for split_in, split_out in SPLITS.items():
        if (src / split_in / "images").exists():
            ann_id = convert_split(src, dst, split_in, split_out,
                                   class_names, ann_start_id=ann_id)
        else:
            print(f"  [skip] {split_in} (no images dir)")

    print(f"\n✅ Done. COCO dataset ready at: {dst}")
    print("   Structure: train/ valid/ test/  (each with _annotations.coco.json)")


if __name__ == "__main__":
    main()
