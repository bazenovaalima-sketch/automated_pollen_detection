#!/usr/bin/env python3
# eval_robustness.py — Multi-severity robustness benchmark for YOLOv12l (seed 1).
#
# Applies three corruption types × 5 severity levels to test slide 1188,
# evaluates Group A (pollen) mAP@0.5 under each condition, and generates
# the robustness figure and CSV table for the paper.
#
# Run from repo/analysis/. Expects repo/unified_dataset/ (Zenodo dataset
# release) and repo/analysis/pollen_benchmark/yolo12l_seed1/weights/best.pt
# (produced by repo/training/train_yolo.py).
#
# Usage: python eval_robustness.py

import shutil
import statistics as st
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
HERE       = Path(__file__).parent
REPO_DIR   = HERE.parent
DATASET    = REPO_DIR / "unified_dataset"
DATA_YAML  = DATASET / "data.yaml"
WEIGHTS    = HERE / "pollen_benchmark" / "yolo12l_seed1" / "weights" / "best.pt"
OUT_DIR    = HERE / "robustness_results"
IMGSZ      = 640
DEVICE     = "mps"

OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Corruption functions — all operate on uint8 BGR numpy arrays
# ---------------------------------------------------------------------------
def blur(img, severity):
    sigmas = [1, 2, 3, 5, 8]
    s = sigmas[severity - 1]
    k = int(s * 6) | 1          # nearest odd kernel size
    return cv2.GaussianBlur(img, (k, k), s)

def noise(img, severity):
    stds = [5, 15, 25, 40, 60]
    rng  = np.random.default_rng(42)   # fixed seed for reproducibility
    n    = rng.normal(0, stds[severity - 1], img.shape)
    return np.clip(img.astype(np.int32) + n, 0, 255).astype(np.uint8)

def darkness(img, severity):
    factors = [0.80, 0.60, 0.40, 0.25, 0.10]
    return (img * factors[severity - 1]).astype(np.uint8)

CORRUPTIONS = {
    "blur":     blur,
    "noise":    noise,
    "darkness": darkness,
}

# ---------------------------------------------------------------------------
# Severity labels for figures / tables
# ---------------------------------------------------------------------------
SEVERITY_LABELS = {
    "blur":     ["σ=1", "σ=2", "σ=3", "σ=5", "σ=8"],
    "noise":    ["std=5", "std=15", "std=25", "std=40", "std=60"],
    "darkness": ["×0.80", "×0.60", "×0.40", "×0.25", "×0.10"],
}

# ---------------------------------------------------------------------------
# Group definitions (must match data.yaml)
# ---------------------------------------------------------------------------
def load_groups():
    cfg = yaml.safe_load(open(DATA_YAML))
    return set(cfg["group_A_pollen"]), set(cfg["group_B_npp"])


def group_mAP(metrics, group_A, group_B):
    ap50  = metrics.box.ap50
    idxs  = metrics.box.ap_class_index
    a_vals, b_vals = [], []
    for row, cid in enumerate(idxs):
        cid = int(cid)
        (a_vals if cid in group_A else b_vals).append(float(ap50[row]))
    macro = lambda v: sum(v) / len(v) if v else float("nan")
    return macro(a_vals), macro(b_vals)


# ---------------------------------------------------------------------------
# Evaluate one set of corrupted images
# ---------------------------------------------------------------------------
def evaluate_corrupted(model, corrupt_fn, severity, src_images, src_labels,
                       group_A, group_B):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_dir = tmp / "images"
        lbl_dir = tmp / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Write corrupted images; symlink labels (unchanged)
        for img_path in src_images:
            img = cv2.imread(str(img_path))
            corrupted = corrupt_fn(img, severity)
            cv2.imwrite(str(img_dir / img_path.name), corrupted)

        for lbl_path in src_labels:
            shutil.copy(lbl_path, lbl_dir / lbl_path.name)

        # Temporary data.yaml — keep real train/val, override only test
        orig_cfg = yaml.safe_load(open(DATA_YAML))
        tmp_cfg = dict(orig_cfg)
        tmp_cfg["path"]  = str(tmp)
        tmp_cfg["train"] = str((DATASET / "train" / "images").resolve())
        tmp_cfg["val"]   = str((DATASET / "valid" / "images").resolve())
        tmp_cfg["test"]  = str(img_dir.resolve())
        tmp_yaml = tmp / "data.yaml"
        yaml.safe_dump(tmp_cfg, open(tmp_yaml, "w"),
                       sort_keys=False, allow_unicode=True)

        m = model.val(data=str(tmp_yaml), split="test",
                      imgsz=IMGSZ, device=DEVICE, verbose=False, plots=False)
    return group_mAP(m, group_A, group_B)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading model: {WEIGHTS}")
    model   = YOLO(str(WEIGHTS))
    group_A, group_B = load_groups()

    src_images = sorted((DATASET / "test" / "images").glob("*.jpg"))
    src_labels = sorted((DATASET / "test" / "labels").glob("*.txt"))
    print(f"Test images: {len(src_images)}  labels: {len(src_labels)}\n")

    # ---- clean baseline first ----
    # patch data.yaml path for local use
    orig_cfg = yaml.safe_load(open(DATA_YAML))
    orig_cfg["path"] = str(DATASET.resolve())
    yaml.safe_dump(orig_cfg, open(DATA_YAML, "w"), sort_keys=False, allow_unicode=True)

    print("Clean baseline ...", end=" ", flush=True)
    m_clean = model.val(data=str(DATA_YAML), split="test",
                        imgsz=IMGSZ, device=DEVICE, verbose=False, plots=False)
    gA_clean, gB_clean = group_mAP(m_clean, group_A, group_B)
    print(f"Group A = {gA_clean:.4f}")

    # ---- corruption loop ----
    rows = []
    rows.append({"corruption": "clean", "severity": 0,
                 "label": "clean", "groupA": gA_clean, "groupB": gB_clean})

    for corr_name, corr_fn in CORRUPTIONS.items():
        for sev in range(1, 6):
            lbl = SEVERITY_LABELS[corr_name][sev - 1]
            print(f"{corr_name:10s} sev={sev} ({lbl}) ...", end=" ", flush=True)
            gA, gB = evaluate_corrupted(
                model, corr_fn, sev, src_images, src_labels, group_A, group_B
            )
            print(f"Group A = {gA:.4f}")
            rows.append({"corruption": corr_name, "severity": sev,
                         "label": lbl, "groupA": gA, "groupB": gB})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "robustness_results.csv", index=False)

    # ---- pivot table for paper ----
    pivot = df[df["corruption"] != "clean"].pivot(
        index="severity", columns="corruption", values="groupA"
    )[["blur", "noise", "darkness"]]
    pivot.index.name = "Severity"
    # prepend clean row
    clean_row = pd.DataFrame(
        [[gA_clean, gA_clean, gA_clean]],
        index=pd.Index([0], name="Severity"),
        columns=pivot.columns
    )
    pivot_full = pd.concat([clean_row, pivot])
    pivot_full.to_csv(OUT_DIR / "robustness_pivot.csv")
    print("\nPivot table (Group A mAP@0.5):")
    print(pivot_full.round(4).to_string())

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors  = {"blur": "#e41a1c", "noise": "#377eb8", "darkness": "#4daf4a"}
    markers = {"blur": "o",       "noise": "s",       "darkness": "^"}
    x = np.arange(1, 6)

    for corr in ["blur", "noise", "darkness"]:
        sub = df[df["corruption"] == corr].sort_values("severity")
        ax.plot(x, sub["groupA"].values,
                color=colors[corr], marker=markers[corr],
                linewidth=2, markersize=6, label=corr.capitalize())

    ax.axhline(gA_clean, color="black", linestyle="--", linewidth=1.2,
               label=f"Clean ({gA_clean:.3f})")
    ax.set_xlabel("Severity level", fontsize=11)
    ax.set_ylabel("Group A mAP@0.5", fontsize=11)
    ax.set_title("Robustness of YOLOv12l to image corruptions", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["1 (mild)", "2", "3", "4", "5 (severe)"])
    ax.legend(fontsize=9, frameon=True, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    ax.set_ylim(0, 0.62)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "paper_assets" / "fig_robustness.png", dpi=150)
    print(f"\nFigure saved → paper_assets/fig_robustness.png")

    # ---- summary stats ----
    print("\n--- Mean mAP drop from clean to severity 5 ---")
    for corr in ["blur", "noise", "darkness"]:
        sev5 = df[(df["corruption"] == corr) & (df["severity"] == 5)]["groupA"].values[0]
        drop = gA_clean - sev5
        print(f"  {corr:10s}: clean={gA_clean:.3f}  sev5={sev5:.3f}  drop={drop:.3f}")


if __name__ == "__main__":
    main()
