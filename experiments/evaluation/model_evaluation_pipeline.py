import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO, RTDETR

# ==============================================================================
# 1. Configuration and Paths
# ==============================================================================
DATASET_YAML = 'pollen_data/data.yaml'
SAVE_DIR = 'results/generated/model_evaluation'

# Dictionary of model weights. Ensure these files exist in the specified directory.
MODELS_DICT = {
    "YOLOv8l": "models/weights/best_yolo8.pt",
    "YOLOv9e": "models/weights/best_yolo9.pt",
    "YOLOv10l": "models/weights/best_yolo10.pt",
    "YOLOv11l": "models/weights/best_yolo11.pt",
    "YOLOv26l": "models/weights/best_yolo26.pt",
    "RT-DETR": "models/weights/best_rtdetr.pt"
}

TARGET_TTA_MODELS = ["YOLOv8l", "YOLOv26l"]

# ==============================================================================
# 2. Evaluation Pipeline
# ==============================================================================
def evaluate_all_models():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Initiating model evaluation pipeline...\n")

    results_summary = {}

    # --- Part 1: Standard Validation ---
    for model_name, weight_path in MODELS_DICT.items():
        if not os.path.exists(weight_path):
            print(f"[Warning] Skipping {model_name}: Weights not found at {weight_path}")
            continue

        print(f"-------------------------------------")
        print(f"Evaluating baseline: {model_name}...")

        # Initialize appropriate architecture
        if "RT-DETR" in model_name:
            model = RTDETR(weight_path)
        else:
            model = YOLO(weight_path)

        size_mb = os.path.getsize(weight_path) / (1024 * 1024)

        metrics = model.val(
            data=DATASET_YAML,
            split='val',
            project=SAVE_DIR,
            name=f"{model_name}_baseline",
            verbose=False
        )

        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        speed_ms = metrics.speed['inference']
        fps = 1000 / speed_ms

        results_summary[model_name] = {
            'mAP50': map50,
            'mAP50_95': map50_95,
            'FPS': fps,
            'Speed_ms': speed_ms,
            'Size_MB': size_mb,
            'path': weight_path,
            'Architecture': 'RT-DETR' if 'RT-DETR' in model_name else 'YOLO family'
        }

        print(f"mAP50: {map50:.4f} | FPS: {fps:.1f} | Size: {size_mb:.1f} MB")

    # --- Part 2: Test-Time Augmentation (TTA) ---
    print(f"-------------------------------------")
    print("Running Test-Time Augmentation (TTA) for selected models...")

    for tta_model_name in TARGET_TTA_MODELS:
        if tta_model_name in results_summary:
            print(f"Applying TTA for {tta_model_name}...")
            best_stats = results_summary[tta_model_name]

            if "RT-DETR" in tta_model_name:
                tta_model = RTDETR(best_stats['path'])
            else:
                tta_model = YOLO(best_stats['path'])

            tta_metrics = tta_model.val(
                data=DATASET_YAML,
                split='val',
                augment=True,
                project=SAVE_DIR,
                name=f"{tta_model_name}_TTA",
                verbose=False
            )

            tta_fps = 1000 / tta_metrics.speed['inference']
            
            # Save TTA results as a separate entry for visualization
            tta_key = f"{tta_model_name} + TTA"
            results_summary[tta_key] = {
                'mAP50': tta_metrics.box.map50,
                'mAP50_95': tta_metrics.box.map,
                'FPS': tta_fps,
                'Speed_ms': tta_metrics.speed['inference'],
                'Size_MB': best_stats['Size_MB'],
                'path': best_stats['path'],
                'Architecture': 'TTA Optimized'
            }

            print(f"TTA impact on {tta_model_name}:")
            print(f"mAP50: {best_stats['mAP50']:.4f} -> {tta_metrics.box.map50:.4f}")
            print(f"FPS:   {best_stats['FPS']:.1f} -> {tta_fps:.1f}")

    return results_summary

# ==============================================================================
# 3. Visualization and Reporting
# ==============================================================================
def generate_performance_plot(results_summary):
    if len(results_summary) < 2:
        print("\n[Info] Insufficient data to generate comparison plot.")
        return

    print(f"-------------------------------------")
    print("Generating performance visualization...")

    df = pd.DataFrame.from_dict(results_summary, orient='index').reset_index()
    df.rename(columns={'index': 'Model'}, inplace=True)

    # Save metrics to CSV
    csv_path = os.path.join(SAVE_DIR, 'evaluation_metrics.csv')
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 8), dpi=300)
    sns.set_theme(style="whitegrid")

    # Scatter plot
    scatter = sns.scatterplot(
        data=df,
        x='FPS',
        y='mAP50',
        hue='Architecture',
        size='Size_MB',
        sizes=(150, 600),
        palette='Set1',
        alpha=0.7,
        edgecolor="black"
    )

    # Annotations
    for i in range(df.shape[0]):
        plt.text(df['FPS'][i] + 0.5, df['mAP50'][i], df['Model'][i],
                 horizontalalignment='left', size='small', color='black', weight='medium')

    # Threshold line
    plt.axvline(x=30, color='red', linestyle='--', alpha=0.3, label='Real-time (30 FPS)')

    plt.title('Detection Accuracy vs. Inference Speed Performance', fontsize=16, pad=20)
    plt.xlabel('Inference Speed (FPS)', fontsize=12)
    plt.ylabel('Accuracy (mAP@0.5)', fontsize=12)

    # Legend cleanup
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Model Category")
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, 'accuracy_speed_tradeoff.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Accuracy vs speed plot saved to: {plot_path}")
    print(f"Metric summary saved to: {csv_path}")

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    final_results = evaluate_all_models()
    generate_performance_plot(final_results)
    print("\nModel evaluation pipeline completed successfully.")
