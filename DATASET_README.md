# Fossil Pollen Detection Dataset (LIL-DEEP, Lake Latorița) — v1

A custom annotated microscopy dataset for object detection of fossil pollen and
non-pollen palynomorphs (NPPs) from the LIL-DEEP sediment core, Lake Latorița,
Southern Carpathians, Romania.

## Contents
- **3,001 microscope images** (400× brightfield, Olympus CX41)
- **8,460 bounding-box annotations** across **22 classes**
- Six stratigraphic depths spanning Late Glacial → early Holocene assemblages
- YOLO format (normalized `class cx cy w h`), with `data.yaml`

## Split (depth-based, leakage-free)
| Split | Depths | Images |
|-------|--------|--------|
| train | 1101, 1102, 1104, 1148 | 2,523 |
| valid | 1012 | 120 |
| test  | 1188 | 358 |

Images are prefixed by depth (e.g. `1101_0001...`). No depth appears in more
than one split, so train/val/test are biologically independent.

## Classes (22)
**Group A — pollen taxa (primary detection target):** Pine, Artemisia, Poaceae,
Betula pendula, Chenopodiaceae, Picea, Alnus viridis, Rumex, Ulmus, Asteraceae,
Cyperaceae, Apiaceae, Fagus, Salix, Other_pollen.

**Group B — non-pollen palynomorphs (reported separately, after van Geel 2001):**
Type-128 (HdV-128), Lycopodium, Charcoal, Pediastrum integrum, Pediastrum
boryanum, Pinus stomata, Other_NPP.

`Other_pollen` and `Other_NPP` aggregate rare taxa (< ~30 instances). Asteraceae
subtypes (Achillea, Aster-type, Cirsium-type, Liguliflorae, Senecio-type) were
merged into a single Asteraceae class.

## Use
```python
from ultralytics import YOLO
model = YOLO("yolov8l.pt")
model.train(data="data.yaml", epochs=100, imgsz=640)
```

## Citation
Bazenova, A. (2026). Fossil Pollen Detection Dataset (LIL-DEEP, Lake Latorița), v1 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20549576

## License
CC BY 4.0
