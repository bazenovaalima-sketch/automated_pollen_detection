# Project Structure for Thesis and Publication

The repository is organized around the scientific workflow:

| Folder | Purpose |
|---|---|
| `data/` | Dataset description and class statistics |
| `models/weights/` | Trained detector checkpoints |
| `src/pollen_scanner/` | Live microscope automation and detection code |
| `experiments/training/` | Training scripts and notes |
| `experiments/evaluation/` | Standard validation and model-comparison scripts |
| `experiments/robustness/` | Blur, noise, and darkness robustness testing |
| `experiments/explainability/` | Heatmap/visual explanation utilities |
| `results/` | Final figures, metrics, validation outputs, and thesis artifacts |
| `hardware/` | Arduino setup images, wiring notes, and STL files |
| `other/` | Preserved legacy files, duplicate docs, system files, and old empty layout |

Recommended thesis flow:

1. Dataset creation and annotation protocol.
2. Detector training and baseline comparison.
3. Selection of YOLOv8l and YOLOv26l for deeper analysis.
4. TTA, robustness, heatmap, and visual comparison.
5. Microscope-stage automation and live inference integration.
6. Limitations and future work.
