# Leakage-Free Benchmarking of CNN and Transformer Detectors for Automated Fossil Pollen Identification

Code and reproducibility materials for the paper *"Leakage-Free Benchmarking of
CNN and Transformer Detectors for Automated Fossil Pollen Identification"*
(Bazenova, Pál & Magyari), submitted to *Journal of Imaging*.

The project benchmarks **seven object detectors** (six YOLO variants + RT-DETR)
on a custom microscopy dataset of fossil pollen and non-pollen palynomorphs
(NPPs) from the LIL-DEEP sediment core, Lake Latoriței, Southern Carpathians,
Romania, using a **depth-based, leakage-free train/val/test split**. It also
includes two ablations (a random-split leakage estimate and a rare-class
oversampling experiment), a multi-corruption robustness benchmark, and a
comparison against expert pollen counts on independent slides.

A related, earlier-stage component — an Arduino-driven automated microscope
stage with live detection — is kept in this repo for reference; see
[Related Prototype](#related-prototype-automated-microscope-stage) below.

## Project Highlights

- **3,001 microscope images / 8,460 annotations / 22 classes** (15 pollen taxa
  + 7 non-pollen palynomorphs), see [DATASET_README.md](DATASET_README.md).
  Dataset archived on Zenodo: [DOI 10.5281/zenodo.20549576](https://doi.org/10.5281/zenodo.20549576).
- **Depth-based split** (train: depths 1101/1102/1104/1148; valid: 1012; test:
  1188, fully held out) — no slide appears in more than one split.
- **Seven detectors benchmarked** under an identical training protocol:
  YOLOv8l, YOLOv9c, YOLOv10l, YOLOv11l, YOLOv12l, YOLOv26l, RT-DETR-l.
- **Leakage ablation**: a random (non-depth-aware) split inflates Group A
  mAP@0.5 by **+0.253 (+51%)** for the same model and hyperparameters.
- **Oversampling ablation**: 3x duplication of rare-class images leaves Group A
  mAP@0.5 essentially unchanged (0.494 vs. 0.496) — the bottleneck is data
  quantity, not sampling.
- **Robustness benchmark**: YOLOv12l evaluated under Gaussian blur, additive
  Gaussian noise, and reduced brightness at 5 severity levels.
- **Expert comparison**: model-derived relative pollen abundances on 3
  independent slides correlate with expert counts (Pearson r=0.86, Spearman
  rho=0.80).

## Repository Structure

```text
repo/
├── README.md
├── DATASET_README.md            # dataset description, classes, split, citation
├── experiment_config.yaml       # single source of truth: hyperparameters, split, software versions
├── build_unified_dataset.py     # raw per-depth Roboflow exports -> unified_dataset/ (depth split)
├── build_random_split.py        # unified_dataset/ -> unified_dataset_random/ (leakage ablation)
├── build_oversampled_dataset.py # unified_dataset/ -> unified_dataset_oversampled/ (oversampling ablation)
├── yolo_to_coco.py               # unified_dataset/ -> unified_dataset_coco/ (for RF-DETR)
├── eval_grouped.py                # Group A / Group B macro-mAP evaluation for one checkpoint
├── training/                      # Colab/Kaggle "paste cell-by-cell" training scripts
│   ├── train_yolo.py              # all 6 YOLO variants, 3 seeds each
│   ├── train_rtdetr.py            # RT-DETR-l, 3 seeds
│   ├── ablation_random_split.py   # leakage ablation (YOLOv12l on the random split)
│   ├── ablation_oversample.py     # oversampling ablation (YOLOv12l)
│   └── train_rfdetr.py            # RF-DETR-Large (transformer baseline, future work)
├── analysis/                       # evaluation outputs + figure/table generation
│   ├── make_paper_assets.py       # master results table + main figures
│   ├── make_diagnostic_figures.py # confusion matrices + qualitative detections
│   ├── eval_robustness.py         # multi-severity corruption benchmark
│   ├── test_results/              # master 7-model benchmark (test slide 1188)
│   ├── test_results_random/       # leakage-ablation results
│   ├── test_results_oversample/   # oversampling-ablation results
│   ├── speed_results/             # inference speed / model size
│   ├── robustness_results/        # corruption-sweep results
│   ├── expert_comparison/         # model vs. expert relative abundances
│   └── paper_assets/              # generated master table + figures
├── requirements.txt
├── requirements-training.txt
├── data/Dataset_info.txt          # legacy 24-class dataset info (earlier prototype)
├── hardware/                       # Arduino microscope-scanner prototype
├── scanner/                         # live detection + stage control
├── models/weights/                  # legacy 24-class checkpoints (Git LFS)
├── experiments/, results/             # legacy 24-class thesis-stage experiments
└── LICENSE
```

## Dataset

Full description, class list, and the depth-based split table are in
[DATASET_README.md](DATASET_README.md). The dataset itself (images + YOLO
labels + `data.yaml`) is **not** stored in this repository — download it from
Zenodo ([DOI 10.5281/zenodo.20549576](https://doi.org/10.5281/zenodo.20549576))
and place it at `repo/unified_dataset/`.

To rebuild the unified dataset from the raw per-depth Roboflow exports
yourself (not redistributed here), run `build_unified_dataset.py` from
`repo/` with an `Annotated data/` folder containing the six per-depth export
folders (1101, 1102, 1104, 1148, 1012, 1188).

## Reproducing the Benchmark

1. **Get the dataset.** Download from Zenodo and unzip to
   `repo/unified_dataset/` (or rebuild it with `build_unified_dataset.py`).
2. **(Ablations only)** Build the auxiliary datasets:
   - `python build_random_split.py` -> `unified_dataset_random/`
   - `python build_oversampled_dataset.py` -> `unified_dataset_oversampled/`
   - `python yolo_to_coco.py --src unified_dataset --dst unified_dataset_coco`
     (for RF-DETR)
3. **Train.** Zip the relevant dataset folder, upload it to Google Drive, and
   run the corresponding script in `training/` cell-by-cell in Google Colab
   (T4 GPU). Each script trains 3 seeds with identical hyperparameters (see
   `experiment_config.yaml`), backing up checkpoints to Drive every epoch and
   resuming automatically after disconnects. Each run produces
   `<run_name>/weights/best.pt`.
4. **Collect runs.** Copy/symlink the completed `<run_name>/` directories into
   `repo/analysis/pollen_benchmark/` (main benchmark),
   `repo/analysis/pollen_benchmark_random/` (leakage ablation), and
   `repo/analysis/pollen_benchmark_oversample/` (oversampling ablation).
5. **Evaluate.** From `repo/analysis/`:
   - `python ../eval_grouped.py --weights pollen_benchmark/<run>/weights/best.pt --split test`
     for Group A / Group B macro-mAP of a single checkpoint.
   - `python eval_robustness.py` for the corruption-severity sweep
     (YOLOv12l seed 1).
   - `python make_diagnostic_figures.py` for confusion matrices and the
     qualitative detection panel (top-2 models).
   - `python make_paper_assets.py` for the master results table and the
     remaining paper figures.

The CSVs already included under `analysis/*_results/` and
`analysis/paper_assets/master_results.csv` are the final results reported in
the paper, so the figures and table can be regenerated without re-running any
training.

## Results

### Master Benchmark (held-out test slide 1188)

Mean +/- SD over 3 seeds, except RT-DETR-l (n=2; one seed's run was lost).
FPS measured on an Apple M1 Pro, batch size 1, imgsz 640.

| Model     | Group A mAP@0.5   | Group B mAP@0.5 | mAP@0.5:0.95 | Params (M) | Size (MB) | FPS |
|-----------|------------------:|----------------:|-------------:|-----------:|----------:|----:|
| YOLOv12l  | **0.496 +/- 0.035** | 0.499          | 0.398        | 26.41      | 53.6      | 7.9 |
| RT-DETR-l | 0.490 +/- 0.001   | 0.482          | 0.364        | 32.85      | 66.3      | 7.9 |
| YOLOv8l   | 0.484 +/- 0.004   | 0.444          | 0.378        | 43.65      | 87.7      | 9.9 |
| YOLOv9c   | 0.474 +/- 0.027   | 0.494          | 0.380        | 25.55      | 51.6      | 10.7 |
| YOLOv26l  | 0.459 +/- 0.043   | **0.504**      | 0.371        | 26.21      | 53.0      | 9.1 |
| YOLOv10l  | 0.456 +/- 0.038   | 0.463          | 0.364        | 25.80      | 52.2      | 9.3 |
| YOLOv11l  | 0.449 +/- 0.034   | 0.436          | 0.360        | 25.33      | 51.2      | **10.1** |

Group A (pollen, 15 classes) is the primary metric; Group B (non-pollen
palynomorphs, 7 classes) is reported separately. Classes absent from a split
(Fagus and Type-128 have no test instances) are excluded from the
corresponding macro-average. YOLOv12l is the best detector overall and is used
for the ablations and robustness benchmark below.

### Leakage Ablation (random vs. depth-based split)

YOLOv12l, identical hyperparameters, 3 seeds, evaluated on the corresponding
test split:

| Split       | Test Group A mAP@0.5 |
|-------------|---------------------:|
| Depth-based (this study) | 0.496 +/- 0.035 |
| Random (pooled, reshuffled, same sizes) | 0.749 +/- 0.023 |
| **Inflation from leakage** | **+0.253 (+51%)** |

A random split lets near-duplicate images from the same slide/depth end up in
both train and test, inflating reported accuracy. The depth-based split
removes this leakage entirely.

### Oversampling Ablation (rare pollen taxa)

YOLOv12l, training images containing rare Group-A classes (<100 train
instances: Rumex, Ulmus, Asteraceae, Cyperaceae, Apiaceae, Fagus, Salix)
duplicated 3x; validation/test unchanged, 3 seeds:

| Metric (test slide 1188)              | Baseline        | Oversampled (3x) | Delta |
|----------------------------------------|----------------:|------------------:|------:|
| Group A macro-mAP@0.5                  | 0.496 +/- 0.035 | 0.494 +/- 0.002   | -0.002 |
| Rare-class mean AP@0.5 (6 classes with test instances) | 0.349 | 0.339 | -0.010 |

Oversampling does not improve rare-class detection — the bottleneck is the
amount of underlying training data, not the sampling strategy.

### Robustness (YOLOv12l, seed 1, Group A mAP@0.5)

| Severity | Blur (sigma) | Noise (std) | Darkness (factor) |
|---------:|----:|----:|----:|
| Clean (0) | 0.536 | 0.536 | 0.536 |
| 1 | 0.536 (sigma=1) | 0.461 (std=5)  | 0.555 (x0.80) |
| 2 | 0.505 (sigma=2) | 0.299 (std=15) | 0.567 (x0.60) |
| 3 | 0.493 (sigma=3) | 0.184 (std=25) | 0.559 (x0.40) |
| 4 | 0.477 (sigma=5) | 0.074 (std=40) | 0.503 (x0.25) |
| 5 | 0.402 (sigma=8) | 0.015 (std=60) | 0.396 (x0.10) |

The model is markedly more sensitive to additive noise (near-total collapse by
severity 5) than to blur or reduced brightness. Mild darkening (severity 1-3)
does not hurt and slightly *improves* mAP, consistent with the source images
being somewhat over-bright.

### Expert Comparison

Model-derived relative pollen abundances vs. expert counts on 3 independent
external slides (1008, 1016, 1020): **Pearson r = 0.86, Spearman rho = 0.80**.
See `analysis/expert_comparison/relative_abundance.csv` and
`analysis/paper_assets/fig_model_vs_expert.png`.

## Installation

From the `repo/` directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt           # inference / scanner only
pip install -r requirements-training.txt  # + training, ablations, analysis figures
```

## Related Prototype: Automated Microscope Stage

`hardware/` and `scanner/` contain an earlier-stage, related component: an
Arduino MEGA + stepper-motor microscope stage with a live detection loop
(`scanner/auto_scan.py`). It uses the legacy 24-class dataset described in
`data/Dataset_info.txt` and the checkpoints in `models/weights/` (Git LFS),
and is independent of the 22-class benchmark above.

```bash
python3 -m scanner.auto_scan
```

Flash **StandardFirmata** to the Arduino MEGA via
`File -> Examples -> Firmata -> StandardFirmata`, and edit `scanner/config.py`
for your serial port, camera index, and model path.

## Citation

If you use this code, please cite the paper (manuscript submitted to
*Journal of Imaging*):

> Bazenova, A., Pál, I., & Magyari, E. K. *Leakage-Free Benchmarking of CNN
> and Transformer Detectors for Automated Fossil Pollen Identification.*
> Journal of Imaging (submitted).

If you use the dataset, please cite:

> Bazenova, A. (2026). *Fossil Pollen Detection Dataset (LIL-DEEP, Lake
> Latoriței), v1* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20549576

## License

Software code is released under the MIT License (see `LICENSE`). The dataset
on Zenodo is released under CC BY 4.0. Hardware files in `hardware/` should be
cited and licensed separately if published as open hardware.
