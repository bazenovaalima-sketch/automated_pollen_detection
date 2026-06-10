#!/usr/bin/env python3
# ============================================================================
# make_diagnostic_figures.py — confusion matrices (top-2 models) and
# qualitative detection panels on the test slide (1188).
#
# Run from repo/analysis/. Expects repo/unified_dataset/ (Zenodo dataset
# release) and repo/analysis/pollen_benchmark/<run>/weights/best.pt for each
# run in TOP2 (produced by the training scripts in repo/training/).
#
# Outputs into paper_assets/:
#   confusion_yolo12l.png, confusion_rtdetr-l.png
#   qualitative_detections.png   (GT vs prediction on sample test images)
# ============================================================================

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

HERE   = Path(__file__).parent
BENCH  = HERE / "pollen_benchmark"
REPO   = HERE.parent
DATA   = REPO / "unified_dataset" / "data.yaml"
TESTIM = REPO / "unified_dataset" / "test" / "images"
TESTLB = REPO / "unified_dataset" / "test" / "labels"
OUT    = HERE / "paper_assets"
OUT.mkdir(exist_ok=True)
DEVICE = "mps"

TOP2 = {"yolo12l": "yolo12l_seed1", "rtdetr-l": "rtdetr-l_seed0"}


def confusion_matrices():
    for name, run in TOP2.items():
        w = BENCH / run / "weights" / "best.pt"
        model = YOLO(str(w))
        model.val(data=str(DATA), split="test", imgsz=640, device=DEVICE,
                  plots=True, project=str(OUT / "_val"), name=name,
                  exist_ok=True, verbose=False)
        src = OUT / "_val" / name / "confusion_matrix_normalized.png"
        if src.exists():
            shutil.copy(src, OUT / f"confusion_{name}.png")
            print(f"  confusion_{name}.png saved")


QUAL_CONF = 0.40


def qualitative(names):
    """GT boxes vs model predictions on typical test images (yolo12l)."""
    import cv2
    model = YOLO(str(BENCH / TOP2["yolo12l"] / "weights" / "best.pt"))
    imgs = sorted(TESTIM.iterdir())

    def n_boxes(p):
        lb = TESTLB / (p.stem + ".txt")
        return len(lb.read_text().splitlines()) if lb.exists() else 0

    # pick images of typical (interquartile) annotation density, evenly
    # spaced by rank, so the figure shows representative rather than
    # worst-case (busiest) scenes
    counts = sorted(((n_boxes(p), p) for p in imgs if n_boxes(p) > 0))
    middle = counts[len(counts) // 4: 3 * len(counts) // 4]
    step = max(1, len(middle) // 4)
    picks = [p for _, p in middle[::step]][:4]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, img_path in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- top row: ground truth ---
        gt = img.copy()
        lb = TESTLB / (img_path.stem + ".txt")
        if lb.exists():
            for line in lb.read_text().splitlines():
                p = line.split()
                cid = int(float(p[0]))
                cx, cy, bw, bh = map(float, p[1:5])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(gt, (x1, y1), (x2, y2), (0, 200, 0), 2)
        axes[0, col].imshow(gt); axes[0, col].axis("off")

        # --- bottom row: predictions ---
        res = model.predict(str(img_path), conf=QUAL_CONF, imgsz=640,
                            device=DEVICE, verbose=False)[0]
        pred = res.plot()[:, :, ::-1]   # BGR->RGB
        axes[1, col].imshow(pred); axes[1, col].axis("off")

    axes[0, 0].set_title(
        f"Ground truth (top)  /  YOLOv12l prediction, conf$\\geq${QUAL_CONF:.2f} (bottom)",
        loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "qualitative_detections.png", dpi=150)
    plt.close(fig)
    print("  qualitative_detections.png saved")


def main():
    cfg = yaml.safe_load(open(DATA))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    print("Confusion matrices (top-2)...")
    confusion_matrices()
    print("Qualitative detections...")
    qualitative(names)
    # clean temp val dir
    shutil.rmtree(OUT / "_val", ignore_errors=True)
    print(f"\nSaved in {OUT}")


if __name__ == "__main__":
    main()
