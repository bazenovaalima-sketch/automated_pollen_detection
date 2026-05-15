"""
train_rtdetr.py
===============
Full training script for RT-DETR-L on pollen dataset with smart class balancing.

This script:
  1. Loads the dataset from a zipped archive
  2. Applies smart oversampling to rare taxa using albumentations
  3. Configures the YOLO format data.yaml
  4. Trains RT-DETR-L for 100 epochs with class-weighted loss
  5. Backs up results to Google Drive (Colab) or local disk

Usage (Colab):
  1. Upload mergeee.zip to Google Drive
  2. Change DATA_ZIP to match your filename
  3. Run this script

Usage (local):
  1. Point DATA_ZIP to your local zip path
  2. Comment out drive.mount() and Colab-specific code
  3. python train_rtdetr.py
"""

import os
import yaml
import cv2
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

import albumentations as A
from ultralytics import RTDETR

# ============================================================================
# CONFIGURATION
# ============================================================================

# For Colab: set True; for local: set False
USE_COLAB = True

# Dataset archive filename
DATA_ZIP = "mergeee.zip"
DATA_ROOT = "/content/pollen_data" if USE_COLAB else "./pollen_data"

# Rare taxa class IDs that need oversampling
# (These should match the order in your data.yaml)
RARE_CLASS_IDS = [0, 8, 10, 11, 12, 17, 19, 23]

# Training hyperparameters
TRAIN_PARAMS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "patience": 20,  # early stopping
    "save": True,
    "project": "pollen_project",
    "name": "rtdetr_v2_balanced",
    # Augmentation
    "degrees": 180.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.05,
    "hsv_s": 0.8,
    "mixup": 0.1,
}


# ============================================================================
# SETUP (Colab only)
# ============================================================================

if USE_COLAB:
    print("🔧 Setting up Google Colab environment...")
    from google.colab import drive, files
    
    # Mount Google Drive
    drive.mount("/content/drive")
    print("✅ Google Drive mounted.")


# ============================================================================
# UNZIP DATASET
# ============================================================================

def unzip_dataset(zip_filename: str, data_root: str):
    """Download and extract the dataset."""
    if USE_COLAB:
        zip_path = f"/content/drive/MyDrive/{zip_filename}"
    else:
        zip_path = zip_filename
    
    if not os.path.exists(zip_path):
        print(f"❌ File {zip_path} not found!")
        print(f"   Make sure '{zip_filename}' is in your Google Drive root.")
        raise FileNotFoundError(zip_path)
    
    os.makedirs(data_root, exist_ok=True)
    print(f"📦 Extracting {zip_filename}...")
    os.system(f"unzip -q -o {zip_path} -d {data_root}")
    print("✅ Dataset extracted.")


# ============================================================================
# SMART OVERSAMPLING FOR RARE CLASSES
# ============================================================================

def augment_rare_classes(train_path: str, multiplier: int = 5):
    """
    Oversample images containing rare taxa using albumentations.
    
    Parameters
    ----------
    train_path : str
        Path to training folder (should contain 'images/' and 'labels/')
    multiplier : int
        How many augmented copies to create per rare image
    """
    img_dir = os.path.join(train_path, "images")
    lbl_dir = os.path.join(train_path, "labels")

    if not os.path.exists(lbl_dir):
        print(f"⚠️ Labels directory not found: {lbl_dir}")
        return

    # Define augmentation pipeline for rare classes
    aug = A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=40,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    print("🚀 Smart oversampling rare taxa...")
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

    for label_file in tqdm(label_files, desc="Augmenting rare classes"):
        label_path = os.path.join(lbl_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        # Check if this file contains any rare class
        classes_in_file = [int(line.split()[0]) for line in lines]
        if not any(cls in RARE_CLASS_IDS for cls in classes_in_file):
            continue

        # Find the image file (try multiple extensions)
        img_file = None
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            temp_path = os.path.join(img_dir, label_file.replace(".txt", ext))
            if os.path.exists(temp_path):
                img_file = label_file.replace(".txt", ext)
                img_path = temp_path
                break

        if img_file is None or img_path is None:
            continue

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse bounding boxes (YOLO format: x_center, y_center, width, height)
        bboxes = [list(map(float, line.split()[1:])) for line in lines]
        labels = classes_in_file

        # Create augmented copies
        for i in range(multiplier):
            try:
                transformed = aug(
                    image=image, bboxes=bboxes, class_labels=labels
                )
                new_img_name = f"aug_{i}_{img_file}"
                new_lbl_name = f"aug_{i}_{label_file}"

                # Save augmented image
                save_img = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(img_dir, new_img_name), save_img)

                # Save augmented labels
                with open(os.path.join(lbl_dir, new_lbl_name), "w") as f:
                    for lbl, box in zip(
                        transformed["class_labels"], transformed["bboxes"]
                    ):
                        f.write(f"{lbl} {' '.join(map(str, box))}\n")
            except Exception as e:
                # Skip if augmentation fails
                continue

    print("✅ Oversampling complete.")


# ============================================================================
# CONFIGURE DATA.YAML
# ============================================================================

def configure_yaml(data_root: str) -> str:
    """Update data.yaml with correct paths."""
    yaml_path = os.path.join(data_root, "data.yaml")

    if not os.path.exists(yaml_path):
        print(f"❌ data.yaml not found at {yaml_path}")
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Update paths to absolute
    config["train"] = os.path.join(data_root, "train/images")
    config["val"] = os.path.join(data_root, "valid/images")
    config["test"] = os.path.join(data_root, "test/images")

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    print(f"✅ data.yaml configured at {yaml_path}")
    return yaml_path


# ============================================================================
# BACKUP CALLBACK (for Colab)
# ============================================================================

def backup_to_drive(trainer):
    """Callback: backs up weights and metrics to Google Drive after each epoch."""
    if not USE_COLAB:
        return

    drive_backup_dir = "/content/drive/MyDrive/pollen_backups"
    os.makedirs(drive_backup_dir, exist_ok=True)

    try:
        weights_dir = os.path.join(trainer.save_dir, "weights")

        # Backup best.pt
        best_pt = os.path.join(weights_dir, "best.pt")
        if os.path.exists(best_pt):
            shutil.copy(best_pt, os.path.join(drive_backup_dir, "best.pt"))

        # Backup last.pt (for resuming training if interrupted)
        last_pt = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_pt):
            shutil.copy(last_pt, os.path.join(drive_backup_dir, "last.pt"))

        # Backup results.csv (training curves)
        results_csv = os.path.join(trainer.save_dir, "results.csv")
        if os.path.exists(results_csv):
            shutil.copy(results_csv, os.path.join(drive_backup_dir, "results.csv"))
    except Exception as e:
        # Silently fail if backup fails (don't interrupt training)
        pass


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train(data_yaml: str):
    """Load model and start training."""
    print("\n" + "=" * 70)
    print("🚀 STARTING RT-DETR-L TRAINING")
    print("=" * 70)

    model = RTDETR("rtdetr-l.pt")

    # Register backup callback
    if USE_COLAB:
        model.add_callback("on_train_epoch_end", backup_to_drive)

    results = model.train(
        data=data_yaml,
        **TRAIN_PARAMS,
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    return results


# ============================================================================
# EXPORT RESULTS
# ============================================================================

def export_results(project: str, name: str):
    """Archive training results to Google Drive (Colab) or local disk."""
    run_folder = os.path.join(project, name)

    if USE_COLAB:
        drive_final_dir = "/content/drive/MyDrive/pollen_results_final"
        os.makedirs(drive_final_dir, exist_ok=True)

        zip_path_drive = os.path.join(drive_final_dir, f"{name}_full_backup")
        print(f"📦 Archiving results to {drive_final_dir}...")
        shutil.make_archive(zip_path_drive, "zip", run_folder)
        print(f"✅ Results saved: {drive_final_dir}")

        # Try to download to browser
        try:
            files.download(f"{zip_path_drive}.zip")
        except Exception as e:
            print(f"⚠️ Browser download failed (expected on slow connections)")
    else:
        # Local: just copy to outputs
        output_dir = Path("./training_outputs")
        output_dir.mkdir(exist_ok=True)
        print(f"📦 Copying results to {output_dir}...")
        shutil.copytree(
            run_folder, output_dir / name, dirs_exist_ok=True
        )
        print(f"✅ Results saved: {output_dir / name}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RT-DETR on pollen dataset")
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip oversampling rare classes (for faster debugging)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_PARAMS["epochs"],
        help="Number of epochs",
    )
    args = parser.parse_args()

    # Override epochs if passed
    TRAIN_PARAMS["epochs"] = args.epochs

    try:
        # 1. Extract dataset
        unzip_dataset(DATA_ZIP, DATA_ROOT)

        # 2. Oversample rare classes
        if not args.skip_augment:
            augment_rare_classes(os.path.join(DATA_ROOT, "train"), multiplier=5)
        else:
            print("⏭️  Skipping augmentation (--skip-augment)")

        # 3. Configure data.yaml
        yaml_path = configure_yaml(DATA_ROOT)

        # 4. Train
        train(yaml_path)

        # 5. Export
        export_results(TRAIN_PARAMS["project"], TRAIN_PARAMS["name"])

        print("\n🎉 Pipeline complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
