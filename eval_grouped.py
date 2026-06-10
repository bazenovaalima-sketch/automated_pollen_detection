"""
eval_grouped.py — evaluate a trained model and report metrics split into:
    Group A (pollen)  -> primary macro-mAP   (headline number)
    Group B (NPP)     -> reported separately  (van Geel, 2001)

Usage:
    python eval_grouped.py --weights path/to/best.pt --split test

Outputs a per-class AP table + the two group-level macro averages.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "unified_dataset" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(DATA_YAML))
    names = cfg["names"]
    group_A = set(cfg["group_A_pollen"])
    group_B = set(cfg["group_B_npp"])

    model = YOLO(args.weights)
    # validate on chosen split; ultralytics returns per-class metrics
    metrics = model.val(data=str(DATA_YAML), split=args.split, verbose=False)

    # per-class AP@0.5 (ap_class_index maps row -> class id)
    ap50 = metrics.box.ap50          # array, one per class present
    idxs = metrics.box.ap_class_index

    print(f"\n{'ID':<4}{'CLASS':<22}{'GROUP':<8}{'AP@0.5':>8}")
    print("-" * 44)
    a_vals, b_vals = [], []
    for row, cid in enumerate(idxs):
        cid = int(cid)
        grp = "Pollen" if cid in group_A else "NPP"
        val = float(ap50[row])
        (a_vals if cid in group_A else b_vals).append(val)
        print(f"{cid:<4}{names[cid]:<22}{grp:<8}{val:>8.3f}")
    print("-" * 44)

    def macro(v):
        return sum(v) / len(v) if v else float("nan")

    print(f"\nPRIMARY  — Group A (pollen) macro-mAP@0.5 : {macro(a_vals):.3f}  ({len(a_vals)} classes present)")
    print(f"SEPARATE — Group B (NPP)    macro-mAP@0.5 : {macro(b_vals):.3f}  ({len(b_vals)} classes present)")
    print(f"OVERALL  — all classes      mAP@0.5       : {float(metrics.box.map50):.3f}")
    print(f"           all classes      mAP@0.5:0.95  : {float(metrics.box.map):.3f}")
    print(f"\nNote: classes absent from the {args.split} split are not listed (reported as N/A in paper).")


if __name__ == "__main__":
    main()
