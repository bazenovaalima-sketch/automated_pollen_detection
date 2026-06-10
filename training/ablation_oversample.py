# ============================================================================
# ablation_oversample.py — RARE-CLASS OVERSAMPLING ABLATION (Google Colab).
#
# Trains YOLOv12l (our best detector) on a training set where images containing
# rare pollen taxa (<100 train instances: Rumex, Ulmus, Asteraceae, Cyperaceae,
# Apiaceae, Fagus, Salix; class IDs 7-13) are duplicated x3
# (build_oversampled_dataset.py, K=3). Validation and test are UNCHANGED, so
# the comparison against the baseline is fair and leakage-free.
#
# Compares against the baseline (no oversampling) YOLOv12l on the SAME test
# slide (1188), 3-seed mean: Group A macro-mAP50 = 0.496, rare-class mean
# AP = 0.349 (repo/analysis/test_results/test_per_class.csv; Fagus has no test
# instances and is excluded from the rare-class mean).
#
# Needs on Drive:  fossil_pollen_oversampled_v1.zip (zip of
# unified_dataset_oversampled/, produced by build_oversampled_dataset.py).
# ============================================================================

# %% [CELL 1] install + GPU check -----------------------------------------------
!pip -q install --upgrade ultralytics
import torch
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# %% [CELL 2] mount Drive + unzip oversampled dataset ---------------------------
from google.colab import drive
from zipfile import ZipFile
from pathlib import Path
import os, shutil

drive.mount("/content/drive")
DATA_ZIP = "/content/drive/MyDrive/fossil_pollen_oversampled_v1.zip"
assert Path(DATA_ZIP).exists(), f"Zip not found: {DATA_ZIP}"
with ZipFile(DATA_ZIP) as z:
    z.extractall("/content")
DATA_DIR = Path("/content/unified_dataset_oversampled")
print("Oversampled dataset:", DATA_DIR.exists())

# %% [CELL 3] patch data.yaml ---------------------------------------------------
import yaml
yaml_path = DATA_DIR / "data.yaml"
cfg = yaml.safe_load(open(yaml_path))
cfg["path"] = str(DATA_DIR)
yaml.safe_dump(cfg, open(yaml_path, "w"), sort_keys=False, allow_unicode=True)
GROUP_A = set(cfg["group_A_pollen"])
names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Classes:", cfg["nc"])

# %% [CELL 4] train YOLOv12l on the oversampled set (3 seeds) -------------------
from ultralytics import YOLO

MODEL = "yolo12l.pt"
SEEDS = [0, 1, 2]            # match the 3-seed depth-split benchmark
DRIVE_PROJECT = "/content/drive/MyDrive/pollen_benchmark_oversample"
LOCAL_PROJECT = "/content/runs/pollen_ovs"

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
    name      = f"{Path(MODEL).stem}_ovs_seed{seed}"
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

# %% [CELL 5] evaluate on test (1188) + compare to baseline ---------------------
import statistics as st

RARE = {7: "Rumex", 8: "Ulmus", 9: "Asteraceae", 10: "Cyperaceae",
        11: "Apiaceae", 12: "Fagus", 13: "Salix"}
BASE_GROUPA   = 0.496     # baseline yolo12l Group A macro-mAP50, 3-seed mean (no oversampling)
BASE_RAREMEAN = 0.349     # baseline mean AP over rare classes w/ test instances (test 1188)

print(f"\n{'='*64}\nOVERSAMPLING ABLATION — test slide 1188\n{'='*64}")
groupA_scores, rare_scores = [], []
for seed in SEEDS:
    name = f"{Path(MODEL).stem}_ovs_seed{seed}"
    best = os.path.join(DRIVE_PROJECT, name, "weights", "best.pt")
    if not os.path.exists(best):
        print(f"  {name}: best.pt not found"); continue
    m = YOLO(best)
    r = m.val(data=str(yaml_path), split="test", imgsz=640, verbose=False, plots=False)
    ap50, idxs = r.box.ap50, r.box.ap_class_index
    per = {int(c): float(ap50[i]) for i, c in enumerate(idxs)}
    groupA = [per[c] for c in GROUP_A if c in per]
    rare   = [per[c] for c in RARE if c in per]
    gA = sum(groupA)/len(groupA) if groupA else float("nan")
    rM = sum(rare)/len(rare) if rare else float("nan")
    groupA_scores.append(gA)
    rare_scores.append(rM)

    print(f"\n  seed {seed}")
    print(f"  Group A macro-mAP50 : {gA:.3f}")
    print(f"  Rare-class mean AP  : {rM:.3f}")
    print("  per rare class:")
    for c, nm in RARE.items():
        v = per.get(c, float("nan"))
        print(f"    {nm:<12} {v:.3f}" if v == v else f"    {nm:<12} n/a (no test inst.)")

if groupA_scores:
    gA_mean, gA_sd = st.mean(groupA_scores), (st.pstdev(groupA_scores) if len(groupA_scores) > 1 else 0.0)
    rM_mean, rM_sd = st.mean(rare_scores), (st.pstdev(rare_scores) if len(rare_scores) > 1 else 0.0)
    print(f"\n{'='*64}\n3-seed summary\n{'='*64}")
    print(f"  Group A macro-mAP50 : {gA_mean:.3f} +/- {gA_sd:.3f}   "
          f"(baseline {BASE_GROUPA:.3f}, delta {gA_mean-BASE_GROUPA:+.3f})")
    print(f"  Rare-class mean AP  : {rM_mean:.3f} +/- {rM_sd:.3f}   "
          f"(baseline {BASE_RAREMEAN:.3f}, delta {rM_mean-BASE_RAREMEAN:+.3f})")
print(f"\n{'='*64}")
print("Report transparently: oversampling helps rare taxa, or it does not")
print("(then the bottleneck is data quantity, not sampling) — both are findings.")
