# Automated Pollen Identification and Microscope Scanning

This repository contains the code, hardware files, model checkpoints, and evaluation artifacts for a project on automated pollen identification. The project has two connected parts:

1. A computer-vision pipeline for detecting 24 pollen taxas and microscopy-related classes in microscope images.
2. A prototype automated microscope stage controlled by Arduino and stepper motors, with live object detection from a microscope camera.

The final research comparison focuses on **YOLOv8l** and **YOLOv26l**. YOLOv26l gives the strongest clean validation metrics for mAP50-95 and recall, while YOLOv8l is more robust under blur, darkness, and noise. For deployment in the live microscope scanner, YOLOv8l is used by default because microscope acquisition can easily suffer from focus, illumination, and sensor-noise variation.

## Project Highlights

- Custom pollen object-detection dataset with **2,007 images**, **2,778 annotations**, and **24 classes**.
- Evaluation of six detector checkpoints: YOLOv8l, YOLOv9e, YOLOv10l, YOLOv11l, YOLOv26l, and RT-DETR.
- Deeper comparison of YOLOv8l and YOLOv26l using standard validation, test-time augmentation, robustness tests, heatmaps, and validation-image visual inspection.
- Arduino MEGA based microscope-stage automation using two 28BYJ-48 stepper motors and ULN2003 driver boards.
- Live microscope camera loop that moves the stage, runs detection, displays annotated frames, saves positive detections, and logs results to CSV.

![Microscope automation setup](hardware/arduino/setup.png)

## Repository Structure

```text
repo/
├── README.md
├── requirements.txt
├── requirements-training.txt
├── data/
│   └── Dataset_info.txt
├── models/
│   └── weights/
│       ├── best_yolo8.pt
│       ├── best_yolo26.pt
│       ├── best_yolo9.pt
│       ├── best_yolo10.pt
│       ├── best_yolo11.pt
│       └── best_rtdetr.pt
├── src/
│   └── pollen_scanner/
│       ├── config.py
│       ├── motor_control.py
│       └── auto_scan.py
├── experiments/
│   ├── training/
│   ├── evaluation/
│   ├── robustness/
│   └── explainability/
├── results/
│   ├── model_comparison/
│   ├── model_evaluation/
│   ├── robustness/
│   └── heatmaps/
├── hardware/
    ├── arduino/
    └── stl/

```

## Dataset

Dataset summary from `data/Dataset_info.txt`:

| Property | Value |
|---|---:|
| Project type | Object Detection |
| Total images | 2,007 |
| Total annotations | 2,778 |
| Total classes | 24 |
| Annotation platform | Roboflow |

The dataset is strongly class-imbalanced. Pine, Lycopodium, Artemisia, Poaceae, and Charcoal dominate the annotation distribution, while classes such as Galium, Picea, Aconitum, Convolvulus, and Fagus have very few samples. To reduce class imbalance, rare pollen taxa were oversampled during training using Albumentations. Images containing underrepresented classes were augmented with color shifts, blur, noise, rotations, and horizontal/vertical flips while preserving YOLO-format bounding boxes. This helped increase the representation of rare taxa without changing the original validation/test sets.

## Model Comparison

The full comparison table is stored in `results/model_comparison/pollen_models_comparison.txt`.

| Model | mAP50 | mAP50-95 | Precision | Recall | Inference (ms) | FPS | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8l | 0.8039 | 0.6064 | 0.8055 | 0.7319 | 30.67 | 32.6 | 333.7 |
| YOLOv9e | 0.7797 | 0.4555 | 0.6589 | 0.7727 | 59.76 | 16.7 | 445.5 |
| YOLOv10l | 0.7892 | 0.6128 | 0.7271 | 0.7138 | 31.11 | 32.1 | 198.0 |
| YOLOv11l | 0.7912 | 0.6235 | 0.6811 | 0.7475 | 27.41 | 36.5 | 194.3 |
| YOLOv26l | 0.8127 | 0.6272 | 0.6876 | 0.7886 | 28.80 | 34.7 | 201.1 |
| RT-DETR | 0.7763 | 0.6058 | 0.7627 | 0.7749 | 40.75 | 24.5 | 251.8 |
| YOLOv8l + TTA | 0.8158 | 0.6141 | 0.8069 | 0.7296 | 73.58 | 13.6 | 333.7 |
| YOLOv26l + TTA | 0.8127 | 0.6272 | 0.6876 | 0.7886 | 27.74 | 36.0 | 201.1 |

Main observations:

- Highest mAP50: **YOLOv8l + TTA** at 0.8158.
- Highest mAP50-95: **YOLOv26l** at 0.6272.
- Highest precision: **YOLOv8l + TTA** at 0.8069.
- Highest recall: **YOLOv26l** at 0.7886.
- Best practical robustness: **YOLOv8l**, especially under blur, darkness, and noise.

## Robustness Results

YOLOv8l and YOLOv26l were tested on distorted validation data to simulate common microscopy problems: out-of-focus images, sensor noise, and poor illumination.

| Scenario | YOLOv26l mAP50 | YOLOv8l mAP50 | Delta (26l - 8l) |
|---|---:|---:|---:|
| Clean | 0.8127 | 0.8039 | +0.0088 |
| Blur | 0.7903 | 0.8122 | -0.0220 |
| Darkness | 0.7952 | 0.8074 | -0.0122 |
| Noise | 0.5940 | 0.6518 | -0.0578 |

Interpretation: YOLOv26l is slightly better on clean validation images, but YOLOv8l is more stable under degraded microscope conditions. This supports YOLOv8l as the default deployment model for the live scanning prototype.

## Explainability and Visual Comparison

The project includes heatmap and visual comparison artifacts:

- `results/heatmaps/Heatmap-based using EigenCAM.png`
- `results/model_comparison/Visual comparison of YOLOv8l vs. YOLOv26l.png`
- `results/model_comparison/Detection accuracy (mAP@0.5) and inference speed (FPS).png`

These figures are useful for the thesis and publication discussion because they show not only numerical performance, but also whether the models attend to biologically meaningful pollen structures.

## Hardware Prototype

The microscope automation part is stored in `hardware/`.

Main components:

| Component | Purpose |
|---|---|
| Arduino MEGA 2560 | Controls the two stepper motors through Firmata |
| 28BYJ-48 stepper motors | Move the microscope stage on X and Y axes |
| ULN2003 drivers | Stepper motor driver boards |
| Microscope camera | Provides live frames for detection |
| 3D-printed parts | Mechanical coupling between motors and microscope stage |

Arduino wiring and setup images:

- `hardware/arduino/setup.png`
- `hardware/arduino/breadboard.png`
- `hardware/arduino/schematic.png`

3D-printable files:

- `hardware/stl/base_stand.stl`
- `hardware/stl/x-axis_drive_coupler.stl`
- `hardware/stl/y-axis_drive_gear.stl`

## Installation

From the `repo/` directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For training and evaluation scripts:

```bash
pip install -r requirements-training.txt
```

Flash **StandardFirmata** to the Arduino MEGA using the Arduino IDE:

```text
File -> Examples -> Firmata -> StandardFirmata
```

## Live Microscope Detection

Edit `src/pollen_scanner/config.py` for your hardware:

```python
SERIAL_PORT = "/dev/cu.usbmodem14101"
CAMERA_INDEX = 0
MODEL_TYPE = "yolo"
MODEL_PATH = "models/weights/best_yolo8.pt"
CONF_THRESHOLD = 0.30
```

Run from the `repo/` directory:

```bash
PYTHONPATH=src python3 -m pollen_scanner.auto_scan
```

The application:

1. Connects to Arduino.
2. Opens the microscope camera.
3. Loads the selected detector.
4. Moves the microscope stage.
5. Runs live detection.
6. Saves annotated images to `captures/`.
7. Logs detections to `auto_scan_log.csv`.

Press `q` to stop.

## Experiments

Training:

```bash
python3 experiments/training/train_rtdetr.py
```

Model evaluation:

```bash
python3 experiments/evaluation/model_evaluation_pipeline.py
```

Robustness evaluation:

```bash
python3 experiments/robustness/robustness_evaluator.py
```

Heatmap formatting:

```bash
python3 experiments/explainability/pollen_heatmap_visualizer.py
```

The experiment scripts assume the full dataset exists locally as `pollen_data/` in YOLO/Ultralytics format with a `data.yaml` file.

## Publication Direction

A strong thesis/publication framing is:

> This work presents an end-to-end automated pollen identification prototype combining a custom annotated microscopy dataset, comparative detector evaluation, robustness analysis under microscopy-like degradation, heatmap-based model inspection, and a low-cost Arduino-driven automated microscope stage.

Suggested discussion focus:

- Dataset imbalance and its effect on rare pollen taxa.
- Accuracy-speed tradeoff across YOLO and RT-DETR detectors.
- Why clean validation metrics alone are not enough for microscope deployment.
- Robustness advantage of YOLOv8l under blur, darkness, and noise.
- Integration of computer vision with low-cost microscope automation.

## Notes on Large Files

The model checkpoints in `models/weights/` are large. 

## License

Software code is released under the MIT License. Hardware files should be cited and licensed separately if they are published as open hardware.
