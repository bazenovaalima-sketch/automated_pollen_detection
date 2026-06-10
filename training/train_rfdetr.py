# ============================================================================
# train_rfdetr.py — paste cell-by-cell into Google Colab (Runtime -> GPU)
# Trains RF-DETR-Large across 3 seeds on the fossil-pollen dataset (COCO format).
#
# RF-DETR (Roboflow, ICLR 2026) is a TRANSFORMER detector — a different package
# and data format from Ultralytics/YOLO. This is the transformer baseline that
# complements the YOLO family in the benchmark (FUTURE WORK — not yet run to
# completion at the time of writing).
#
# FAIRNESS NOTE: RF-DETR uses its own training recipe and a DINOv2-pretrained
# backbone, so it is trained under a *comparable budget* (same depth-based
# split, effective batch 16, COCO-pretrained init) rather than the identical
# YOLO hyperparameters. Report it as a separate transformer block.
#
# PREREQUISITE: run yolo_to_coco.py once to produce unified_dataset_coco/ and
# zip it to Drive as fossil_pollen_coco_v1.zip (see CELL 0 below).
# ============================================================================

# %% [CELL 0] (run ONCE, can be on your laptop) build the COCO dataset + zip ----
# In Colab you can instead upload yolo_to_coco.py and run:
#   !python yolo_to_coco.py --src unified_dataset --dst unified_dataset_coco
# then zip + upload to Drive. Easiest: do it locally and drag the zip to Drive.
#
#   python yolo_to_coco.py --src unified_dataset --dst unified_dataset_coco
#   (zip the folder)  ->  upload as  fossil_pollen_coco_v1.zip  to MyDrive

# %% [CELL 1] install + GPU check ------------------------------------------------
!pip -q install rfdetr
import torch
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
# If GPU is NONE: Runtime -> Change runtime type -> GPU (T4)

# %% [CELL 2] mount Drive + unzip COCO dataset -----------------------------------
from google.colab import drive
from zipfile import ZipFile
from pathlib import Path
import os, shutil

drive.mount("/content/drive")

DATA_ZIP   = "/content/drive/MyDrive/fossil_pollen_coco_v1.zip"
EXTRACT_TO = Path("/content")

assert Path(DATA_ZIP).exists(), f"COCO zip not found: {DATA_ZIP}\nRun yolo_to_coco.py and upload the zip first."
with ZipFile(DATA_ZIP) as z:
    z.extractall(EXTRACT_TO)

# locate the dataset dir (folder that contains train/ valid/ test/)
DATA_DIR = EXTRACT_TO / "unified_dataset_coco"
assert (DATA_DIR / "train" / "_annotations.coco.json").exists(), \
    f"COCO annotations not found under {DATA_DIR}/train/"
print("COCO dataset ready:", DATA_DIR)

# %% [CELL 3] diagnose what's on Drive before starting ---------------------------
MODEL_TAG     = "rfdetr_large"
SEEDS         = [0, 1, 2]
DRIVE_PROJECT = "/content/drive/MyDrive/pollen_benchmark"

print("\n-- STATUS CHECK --------------------------------------------")
for seed in SEEDS:
    drive_dir = os.path.join(DRIVE_PROJECT, f"{MODEL_TAG}_seed{seed}")
    done      = os.path.exists(os.path.join(drive_dir, "COMPLETED.txt"))
    # RF-DETR saves checkpoint.pth (latest) each epoch in output_dir
    has_ckpt  = os.path.exists(os.path.join(drive_dir, "checkpoint.pth"))
    if done:
        print(f"  seed {seed}: COMPLETED")
    elif has_ckpt:
        print(f"  seed {seed}: INTERRUPTED -> will RESUME from checkpoint.pth")
    else:
        print(f"  seed {seed}: NOT STARTED -> will train fresh")
print("--------------------------------------------------------------\n")

# %% [CELL 4] train RF-DETR-Large x 3 seeds  (Drive backup + auto-resume) --------
from rfdetr import RFDETRLarge

LOCAL_PROJECT = "/content/runs/rfdetr"
os.makedirs(LOCAL_PROJECT, exist_ok=True)

# Comparable budget to the YOLO runs:
#   effective batch = batch_size * grad_accum_steps = 4 * 4 = 16  (T4-safe)
#   RF-DETR converges faster than YOLO thanks to the DINOv2 pretrained backbone,
#   so 50 epochs is the standard fine-tuning budget (early stopping via patience).
TRAIN_ARGS = dict(
    epochs          = 50,
    batch_size      = 4,
    grad_accum_steps= 4,
    lr              = 1e-4,
    # resolution must be divisible by 56; 672 ~= YOLO's 640 (closest fair match)
    resolution      = 672,
    early_stopping  = True,
)

def backup_to_drive(local_dir, drive_dir):
    """Copy RF-DETR checkpoints + logs to Drive (best-effort, with retries)."""
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for fn in os.listdir(local_dir):
            if fn.endswith((".pth", ".json", ".txt", ".log", ".csv")):
                src = os.path.join(local_dir, fn)
                for attempt in range(3):
                    try:
                        shutil.copy(src, os.path.join(drive_dir, fn))
                        break
                    except OSError:
                        import time; time.sleep(3)
    except Exception as e:
        print("  [backup warn]", e)

for seed in SEEDS:
    name      = f"{MODEL_TAG}_seed{seed}"
    local_dir = os.path.join(LOCAL_PROJECT, name)
    drive_dir = os.path.join(DRIVE_PROJECT, name)
    done_flag = os.path.join(drive_dir, "COMPLETED.txt")

    if os.path.exists(done_flag):
        print(f"[skip] {name} already completed")
        continue

    print(f"\n{'='*60}\nTRAINING {name}\n{'='*60}")
    os.makedirs(local_dir, exist_ok=True)

    # restore checkpoint from Drive to resume
    resume_path = None
    drive_ckpt  = os.path.join(drive_dir, "checkpoint.pth")
    if os.path.exists(drive_ckpt):
        print(f"[resume] remounting Drive and restoring {name}")
        from google.colab import drive as _drive
        _drive.mount("/content/drive", force_remount=True)
        for fn in os.listdir(drive_dir):
            if fn.endswith((".pth", ".json", ".txt")):
                for attempt in range(3):
                    try:
                        shutil.copy(os.path.join(drive_dir, fn),
                                    os.path.join(local_dir, fn))
                        break
                    except OSError:
                        import time; time.sleep(3)
        resume_path = os.path.join(local_dir, "checkpoint.pth")

    model = RFDETRLarge()

    # per-epoch backup callback (best-effort; RF-DETR exposes a callbacks dict)
    try:
        model.callbacks["on_fit_epoch_end"].append(
            lambda data: backup_to_drive(local_dir, drive_dir)
        )
    except Exception as e:
        print("  [callback warn] per-epoch backup not registered:", e)

    train_kwargs = dict(
        dataset_dir = str(DATA_DIR),
        output_dir  = local_dir,
        seed        = seed,
        **TRAIN_ARGS,
    )
    if resume_path:
        train_kwargs["resume"] = resume_path

    model.train(**train_kwargs)

    # final backup + mark complete
    backup_to_drive(local_dir, drive_dir)
    with open(done_flag, "w") as f:
        f.write("done")

# %% [CELL 5] evaluate each seed on val + summarise ------------------------------
# RF-DETR evaluates during training; final COCO metrics are in the output logs.
# For a clean cross-seed summary, evaluate checkpoint_best_total.pth from each
# seed on the test split using eval_grouped.py-style Group A / B macro-mAP, the
# same way as the YOLO models.
print("\nTraining done. Best checkpoints per seed on Drive:")
for seed in SEEDS:
    print(f"  {DRIVE_PROJECT}/{MODEL_TAG}_seed{seed}/checkpoint_best_total.pth")
