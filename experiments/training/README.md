# Training Notes

This folder contains the training script used during the model-development stage of the project.

The project originally started with RT-DETR experiments, then expanded into a broader detector comparison that included YOLOv8l, YOLOv9e, YOLOv10l, YOLOv11l, YOLOv26l, and RT-DETR.

## Dataset

The scripts assume a YOLO/Ultralytics dataset layout:

```text
pollen_data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The dataset summary is stored in `../../data/Dataset_info.txt`.

## Current Script

`training.py` trains RT-DETR-L and includes rare-class oversampling logic. It is kept because RT-DETR was part of the experimental comparison and thesis development history.

For the final discussion, use the consolidated results in:

- `../../results/model_comparison/pollen_models_comparison.txt`
- `../../results/model_evaluation/`
- `../../results/robustness/pollen_robustness_results.txt`

## Environment

Install the extended experiment dependencies from the repository root:

```bash
pip install -r requirements-training.txt
```

The original training was designed for Google Colab with GPU acceleration. Local use may require editing dataset paths and disabling Colab-specific Drive mounting.
