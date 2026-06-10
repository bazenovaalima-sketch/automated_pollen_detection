# ============================================================================
# train_yolo.py — paste cell-by-cell into Google Colab (Runtime -> GPU, T4)
#
# Trains one YOLO-family detector across 3 seeds on the unified 22-class
# fossil-pollen dataset (depth-based, leakage-free split). The same script,
# with only MODEL changed, was used for all six YOLO variants benchmarked in
# the paper: yolov8l.pt, yolov9c.pt, yolov10l.pt, yolo11l.pt, yolo12l.pt,
# yolo26l.pt. Hyperparameters and augmentation are identical across models
# (see experiment_config.yaml).
#
# Per-epoch checkpoints are backed up to Google Drive so a Colab disconnect
# never loses progress: re-running the script resumes from the last
# checkpoint (resume=True) and skips seeds already marked COMPLETED.
#
# Needs on Drive: fossil_pollen_dataset_v1.zip (produced by
# build_unified_dataset.py, zip the unified_dataset/ folder).
# ============================================================================

# %% [CELL 1] install + GPU check ------------------------------------------------
!pip -q install --upgrade ultralytics
import torch
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
# If GPU is NONE: Runtime -> Change runtime type -> GPU (T4)

# %% [CELL 2] mount Drive + unzip dataset ----------------------------------------
from google.colab import drive
from zipfile import ZipFile
from pathlib import Path

drive.mount("/content/drive")

DATA_ZIP   = "/content/drive/MyDrive/fossil_pollen_dataset_v1.zip"
EXTRACT_TO = Path("/content")

assert Path(DATA_ZIP).exists(), f"Zip not found: {DATA_ZIP}"
with ZipFile(DATA_ZIP) as z:
    z.extractall(EXTRACT_TO)

DATA_DIR = EXTRACT_TO / "unified_dataset"
print("Extracted to:", DATA_DIR, "| exists:", DATA_DIR.exists())

# %% [CELL 3] patch data.yaml so paths resolve on Colab --------------------------
import yaml
yaml_path = DATA_DIR / "data.yaml"
cfg = yaml.safe_load(open(yaml_path))
cfg["path"] = str(DATA_DIR)
yaml.safe_dump(cfg, open(yaml_path, "w"), sort_keys=False, allow_unicode=True)
print("Classes:", cfg["nc"])
print("path  =", cfg["path"])

# %% [CELL 4] choose model + diagnose what's on Drive before starting ------------
import os, shutil
from pathlib import Path

# One of: yolov8l.pt, yolov9c.pt, yolov10l.pt, yolo11l.pt, yolo12l.pt, yolo26l.pt
MODEL         = "yolo12l.pt"
SEEDS         = [0, 1, 2]
DRIVE_PROJECT = "/content/drive/MyDrive/pollen_benchmark"

# Batch size: 16 for all six architectures, except YOLOv12l, which needed a
# reduced batch on a 16 GB T4 (8 for seeds 0/2, 4 for seed 1). Ultralytics'
# "auto" optimizer (AdamW) rescales the learning rate accordingly.
BATCH = {s: (8 if MODEL == "yolo12l.pt" and s != 1 else
              4 if MODEL == "yolo12l.pt" else 16) for s in SEEDS}

print("\n-- STATUS CHECK -------------------------------------------")
for seed in SEEDS:
    stem      = Path(MODEL).stem
    drive_dir = os.path.join(DRIVE_PROJECT, f"{stem}_seed{seed}")
    done      = os.path.exists(os.path.join(drive_dir, "COMPLETED.txt"))
    has_last  = os.path.exists(os.path.join(drive_dir, "weights", "last.pt"))
    csv_path  = os.path.join(drive_dir, "results.csv")

    if done:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
            best_map = df["metrics/mAP50(B)"].max()
            print(f"  seed {seed}: COMPLETED  (best mAP50 = {best_map:.3f})")
        except Exception:
            print(f"  seed {seed}: COMPLETED")
    elif has_last:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
            last_epoch = int(df["epoch"].max()) + 1
            print(f"  seed {seed}: INTERRUPTED at epoch {last_epoch}/100 -> will RESUME")
        except Exception:
            print(f"  seed {seed}: INTERRUPTED -> will RESUME (last.pt found)")
    else:
        print(f"  seed {seed}: NOT STARTED -> will train fresh")
print("-------------------------------------------------------------\n")

# %% [CELL 5] train (or resume) each seed -----------------------------------------
from ultralytics import YOLO

LOCAL_PROJECT = "/content/runs/pollen"

BASE_ARGS = dict(
    data         = str(yaml_path),
    epochs       = 100,
    imgsz        = 640,
    patience     = 20,
    optimizer    = "auto",
    degrees      = 180.0,
    fliplr       = 0.5,
    flipud       = 0.5,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    translate    = 0.1,
    scale        = 0.5,
    mosaic       = 1.0,
    mixup        = 0.1,
    close_mosaic = 10,
    save         = True,
    plots        = True,
    verbose      = True,
)

def make_backup_cb(drive_dir):
    def cb(trainer):
        try:
            os.makedirs(os.path.join(drive_dir, "weights"), exist_ok=True)
            sd = str(trainer.save_dir)
            for rel in ["weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"]:
                src = os.path.join(sd, rel)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(drive_dir, rel))
        except Exception as e:
            print("  [backup warn]", e)
    return cb

for seed in SEEDS:
    stem      = Path(MODEL).stem
    name      = f"{stem}_seed{seed}"
    local_dir = os.path.join(LOCAL_PROJECT, name)
    drive_dir = os.path.join(DRIVE_PROJECT, name)
    done_flag = os.path.join(drive_dir, "COMPLETED.txt")

    if os.path.exists(done_flag):
        print(f"[skip] {name} already completed")
        continue

    print(f"\n{'='*60}\nTRAINING {name}\n{'='*60}")

    # restore last checkpoint from Drive and resume, if present
    resume = False
    if os.path.exists(os.path.join(drive_dir, "weights", "last.pt")):
        print(f"[resume] remounting Drive and copying last.pt -> local")
        from google.colab import drive as _drive
        _drive.mount("/content/drive", force_remount=True)
        os.makedirs(os.path.join(local_dir, "weights"), exist_ok=True)
        for rel in ["weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"]:
            src = os.path.join(drive_dir, rel)
            if os.path.exists(src):
                import time, shutil as _sh
                for attempt in range(3):
                    try:
                        _sh.copy(src, os.path.join(local_dir, rel))
                        break
                    except OSError:
                        print(f"  copy retry {attempt+1}/3 for {rel}")
                        time.sleep(3)
        resume = True

    # resume=True is required to continue from epoch N instead of restarting
    if resume:
        model = YOLO(os.path.join(local_dir, "weights", "last.pt"))
        model.add_callback("on_fit_epoch_end", make_backup_cb(drive_dir))
        model.train(resume=True)
    else:
        model = YOLO(MODEL)
        model.add_callback("on_fit_epoch_end", make_backup_cb(drive_dir))
        model.train(seed=seed, batch=BATCH[seed], project=LOCAL_PROJECT, name=name,
                    exist_ok=True, **BASE_ARGS)

    make_backup_cb(drive_dir)(model.trainer)
    with open(done_flag, "w") as f:
        f.write("done")

# %% [CELL 6] final 3-seed summary (validation set) -------------------------------
import statistics as st, pandas as pd

print(f"\n{'='*60}\n{Path(MODEL).stem} -- 3-seed summary (val set)\n{'='*60}")
rows = {}
for seed in SEEDS:
    csv = os.path.join(DRIVE_PROJECT, f"{Path(MODEL).stem}_seed{seed}", "results.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv); df.columns = df.columns.str.strip()
        best = df.loc[df["metrics/mAP50(B)"].idxmax()]
        rows[seed] = {
            "mAP50":     float(best["metrics/mAP50(B)"]),
            "mAP50_95":  float(best["metrics/mAP50-95(B)"]),
            "precision": float(best["metrics/precision(B)"]),
            "recall":    float(best["metrics/recall(B)"]),
        }

for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
    vals = [rows[s][metric] for s in sorted(rows)]
    if vals:
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {metric:<10} {st.mean(vals):.3f} +/- {sd:.3f}   {['%.3f'%v for v in vals]}")

# Headline test-set numbers (held-out slide 1188) come from eval_grouped.py /
# repo/analysis/make_paper_assets.py, not from this validation summary.
