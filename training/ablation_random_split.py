# ============================================================================
# ablation_random_split.py — LEAKAGE ABLATION (paste into Google Colab).
#
# Trains YOLOv12l (the best detector on the held-out test set) on a RANDOM
# split of the unified dataset (build_random_split.py), then evaluates on the
# random test split. Compared against the depth-split result for the SAME
# model (test Group A mAP50 = 0.496 +/- 0.035, 3 seeds), this quantifies how
# much a random split inflates metrics through slide-level leakage.
#
# Using the best model gives a conservative, hard-to-dispute leakage estimate.
# Expected: random-split test mAP >> depth-split test mAP.
#
# Needs on Drive:  fossil_pollen_random_v1.zip (zip of unified_dataset_random/,
# produced by build_random_split.py).
# ============================================================================

# %% [CELL 1] install + GPU check -----------------------------------------------
!pip -q install --upgrade ultralytics
import torch
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# %% [CELL 2] mount Drive + unzip random-split dataset --------------------------
from google.colab import drive
from zipfile import ZipFile
from pathlib import Path
import os, shutil

drive.mount("/content/drive")

DATA_ZIP   = "/content/drive/MyDrive/fossil_pollen_random_v1.zip"
EXTRACT_TO = Path("/content")
assert Path(DATA_ZIP).exists(), f"Zip not found: {DATA_ZIP}"
with ZipFile(DATA_ZIP) as z:
    z.extractall(EXTRACT_TO)

DATA_DIR = EXTRACT_TO / "unified_dataset_random"
print("Random-split dataset:", DATA_DIR.exists())

# %% [CELL 3] patch data.yaml ---------------------------------------------------
import yaml
yaml_path = DATA_DIR / "data.yaml"
cfg = yaml.safe_load(open(yaml_path))
cfg["path"] = str(DATA_DIR)
yaml.safe_dump(cfg, open(yaml_path, "w"), sort_keys=False, allow_unicode=True)
GROUP_A = set(cfg["group_A_pollen"])
GROUP_B = set(cfg["group_B_npp"])
names   = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Classes:", cfg["nc"])

# %% [CELL 4] train YOLOv12l on the random split (3 seeds) ----------------------
from ultralytics import YOLO

MODEL = "yolo12l.pt"        # best detector on the held-out (depth-split) test set
SEEDS = [0, 1, 2]            # match the 3-seed depth-split benchmark
DRIVE_PROJECT = "/content/drive/MyDrive/pollen_benchmark_random"
LOCAL_PROJECT = "/content/runs/pollen_random"

TRAIN_ARGS = dict(          # IDENTICAL hyperparameters to the main benchmark
    data=str(yaml_path), epochs=100, imgsz=640, batch=16, patience=20,
    optimizer="auto", degrees=180.0, fliplr=0.5, flipud=0.5, hsv_h=0.015,
    hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, mosaic=1.0, mixup=0.1,
    close_mosaic=10, save=True, plots=True, verbose=True,
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
    name      = f"{Path(MODEL).stem}_random_seed{seed}"
    local_dir = os.path.join(LOCAL_PROJECT, name)
    drive_dir = os.path.join(DRIVE_PROJECT, name)
    done_flag = os.path.join(drive_dir, "COMPLETED.txt")

    if os.path.exists(done_flag):
        print(f"[skip] {name} already completed"); continue
    print(f"\n{'='*60}\nTRAINING {name}\n{'='*60}")

    resume = False
    if os.path.exists(os.path.join(drive_dir, "weights", "last.pt")):
        print("[resume] restoring from Drive")
        from google.colab import drive as _drive
        _drive.mount("/content/drive", force_remount=True)
        os.makedirs(os.path.join(local_dir, "weights"), exist_ok=True)
        for rel in ["weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"]:
            src = os.path.join(drive_dir, rel)
            if os.path.exists(src):
                import time
                for _ in range(3):
                    try: shutil.copy(src, os.path.join(local_dir, rel)); break
                    except OSError: time.sleep(3)
        resume = True

    if resume:
        model = YOLO(os.path.join(local_dir, "weights", "last.pt"))
        model.add_callback("on_fit_epoch_end", make_backup_cb(drive_dir))
        model.train(resume=True)
    else:
        model = YOLO(MODEL)
        model.add_callback("on_fit_epoch_end", make_backup_cb(drive_dir))
        model.train(seed=seed, project=LOCAL_PROJECT, name=name,
                    exist_ok=True, **TRAIN_ARGS)
    make_backup_cb(drive_dir)(model.trainer)
    with open(done_flag, "w") as f: f.write("done")

# %% [CELL 5] evaluate on RANDOM test split + compare to depth split ------------
import statistics as st

# yolo12l depth-split test Group A mAP50, mean +/- SD over 3 seeds (this study,
# repo/analysis/test_results/test_summary.csv)
DEPTH_TEST_GROUPA_MEAN = 0.496
DEPTH_TEST_GROUPA_SD   = 0.035

print(f"\n{'='*64}\nLEAKAGE ABLATION — random split vs depth split (test set)\n{'='*64}")
random_scores = []
for seed in SEEDS:
    name = f"{Path(MODEL).stem}_random_seed{seed}"
    best = os.path.join(DRIVE_PROJECT, name, "weights", "best.pt")
    if not os.path.exists(best):
        print(f"  {name}: best.pt not found"); continue
    m = YOLO(best)
    r = m.val(data=str(yaml_path), split="test", imgsz=640, verbose=False, plots=False)
    ap50, idxs = r.box.ap50, r.box.ap_class_index
    a = [float(ap50[i]) for i, c in enumerate(idxs) if int(c) in GROUP_A]
    groupA = sum(a) / len(a) if a else float("nan")
    random_scores.append(groupA)
    print(f"  seed {seed}: random split test Group A mAP50 = {groupA:.3f}")

if random_scores:
    mean = st.mean(random_scores)
    sd   = st.pstdev(random_scores) if len(random_scores) > 1 else 0.0
    print(f"\n  RANDOM split test Group A mAP50 : {mean:.3f} +/- {sd:.3f}")
    print(f"  DEPTH  split test Group A mAP50 : {DEPTH_TEST_GROUPA_MEAN:.3f} +/- {DEPTH_TEST_GROUPA_SD:.3f}")
    print(f"  INFLATION (leakage)             : +{mean - DEPTH_TEST_GROUPA_MEAN:.3f}  "
          f"({100*(mean-DEPTH_TEST_GROUPA_MEAN)/DEPTH_TEST_GROUPA_MEAN:+.0f}%)")
print(f"\n{'='*64}")
print("If random >> depth, the depth split is doing its job: preventing")
print("slide-level leakage that would otherwise inflate reported accuracy.")
