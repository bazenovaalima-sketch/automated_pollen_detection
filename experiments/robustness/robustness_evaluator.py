import cv2
import numpy as np
import os
import shutil
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import yaml

# ==============================================================================
# 1. CONFIGURATION PATHS
# ==============================================================================
# Define absolute or relative paths to your dataset
DATASET_PATH = 'pollen_data' 
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, 'valid/images') 
VAL_LABELS_PATH = os.path.join(DATASET_PATH, 'valid/labels')
ORIGINAL_YAML_PATH = os.path.join(DATASET_PATH, 'data.yaml')

# Paths to the trained model weights
MODELS_TO_TEST = {
    "YOLOv8l": "models/weights/best_yolo8.pt",
    "YOLOv26l": "models/weights/best_yolo26.pt"
}

# Output directory for distorted datasets
ROBUST_ROOT = 'results/generated/robustness_test_data'
OUTPUT_REPORT_PATH = "results/generated/robustness_report.xlsx"

# ==============================================================================
# 2. DATA AUGMENTATION (DISTORTION) FUNCTIONS
# ==============================================================================
def apply_blur(img):
    """Applies Gaussian blur to simulate out-of-focus microscopy."""
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_noise(img):
    """Adds Gaussian noise to simulate sensor noise or dirty slides."""
    row, col, ch = img.shape
    gauss = np.random.normal(0, 30, (row, col, ch))
    noisy = np.clip(img + gauss, 0, 255)
    return noisy.astype(np.uint8)

def apply_darkness(img):
    """Reduces brightness to simulate poor illumination."""
    return cv2.convertScaleAbs(img, alpha=0.4, beta=0)

# ==============================================================================
# 3. DATASET PREPARATION
# ==============================================================================
def prepare_distorted_datasets():
    """Creates temporary datasets with specific visual distortions."""
    if os.path.exists(ROBUST_ROOT):
        shutil.rmtree(ROBUST_ROOT)
    os.makedirs(ROBUST_ROOT, exist_ok=True)

    with open(ORIGINAL_YAML_PATH, 'r') as f:
        original_yaml = yaml.safe_load(f)

    print(f"Loaded original dataset config. Classes detected: {original_yaml['nc']}")
    scenarios = ['Blur', 'Noise', 'Darkness']

    for sc in scenarios:
        img_dest = os.path.join(ROBUST_ROOT, sc, 'images/val')
        label_dest = os.path.join(ROBUST_ROOT, sc, 'labels/val')
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(label_dest, exist_ok=True)

        print(f"\n[INFO] Preparing scenario: {sc}...")
        
        # Copy original ground truth labels
        for label_file in os.listdir(VAL_LABELS_PATH):
            shutil.copy(os.path.join(VAL_LABELS_PATH, label_file), label_dest)

        # Apply distortions to images
        image_files = [f for f in os.listdir(VAL_IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
        for img_name in tqdm(image_files, desc=f"Generating {sc} images"):
            img_path = os.path.join(VAL_IMAGES_PATH, img_name)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            
            if sc == 'Blur': 
                distorted = apply_blur(img)
            elif sc == 'Noise': 
                distorted = apply_noise(img)
            elif sc == 'Darkness': 
                distorted = apply_darkness(img)
            
            cv2.imwrite(os.path.join(img_dest, img_name), distorted)
        
        # Generate new YAML config for the distorted dataset
        temp_yaml_path = os.path.join(ROBUST_ROOT, sc, 'data.yaml')
        temp_config = {
            'path': os.path.abspath(os.path.join(ROBUST_ROOT, sc)),
            'train': 'images/val',
            'val': 'images/val',
            'nc': original_yaml['nc'],           
            'names': original_yaml['names']      
        }
        
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_config, f)

    return scenarios

# ==============================================================================
# 4. EVALUATION PIPELINE
# ==============================================================================
def evaluate_models(scenarios):
    """Runs validation on both clean and distorted datasets for all models."""
    results = []

    for model_name, model_path in MODELS_TO_TEST.items():
        if not os.path.exists(model_path):
            print(f"\n[WARNING] Model weights not found: {model_path}. Skipping.")
            continue

        print(f"\n--- Evaluating Model: {model_name} ---")
        model = YOLO(model_path)
        
        # 1. Baseline evaluation (Clean Data)
        print("Testing on: Clean Baseline")
        res_clean = model.val(data=ORIGINAL_YAML_PATH, split='val', verbose=False)
        results.append({
            'Model': model_name, 
            'Scenario': 'Clean', 
            'mAP50': res_clean.box.map50
        })

        # 2. Robustness evaluation (Distorted Data)
        for sc in scenarios:
            print(f"Testing on: {sc}")
            scenario_yaml = os.path.join(ROBUST_ROOT, sc, 'data.yaml')
            
            res_robust = model.val(data=scenario_yaml, split='val', verbose=False)
            results.append({
                'Model': model_name, 
                'Scenario': sc, 
                'mAP50': res_robust.box.map50
            })

    return results

# ==============================================================================
# 5. MAIN EXECUTION & REPORTING
# ==============================================================================
if __name__ == "__main__":
    print("Starting Robustness Evaluation Pipeline...")
    
    # Step 1: Generate distorted validation sets
    active_scenarios = prepare_distorted_datasets()
    
    # Step 2: Evaluate models
    evaluation_results = evaluate_models(active_scenarios)
    
    if not evaluation_results:
        print("No results generated. Ensure model paths and dataset paths are correct.")
        exit()

    # Step 3: Format and save results
    df = pd.DataFrame(evaluation_results)
    df_pivot = df.pivot(index='Scenario', columns='Model', values='mAP50')

    # Calculate performance delta if both models exist
    if 'YOLOv8l' in df_pivot.columns and 'YOLOv26l' in df_pivot.columns:
        df_pivot['Delta (26l - 8l)'] = df_pivot['YOLOv26l'] - df_pivot['YOLOv8l']
        df_pivot['Delta (%)'] = (df_pivot['Delta (26l - 8l)'] / df_pivot['YOLOv8l'] * 100).round(2)

    print("\n=======================================================")
    print("FINAL ROBUSTNESS REPORT (mAP@0.5)")
    print("=======================================================")
    print(df_pivot.round(4))
    print("=======================================================\n")

    # Save to Excel
    df_pivot.to_excel(OUTPUT_REPORT_PATH)
    print(f"[SUCCESS] Report saved to: {OUTPUT_REPORT_PATH}")
