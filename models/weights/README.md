# Model Weights

Trained checkpoints are kept locally in this folder, but `.pt` and `.pth` files are ignored by git because they are too large for a normal GitHub commit.

Local checkpoints currently used by the project:

- `best_yolo8.pt` - default deployment model for live microscope scanning.
- `best_yolo26.pt` - strongest clean validation mAP50-95 / recall model.
- `best_yolo9.pt`
- `best_yolo10.pt`
- `best_yolo11.pt`
- `best_rtdetr.pt`

For publication, upload these files through GitHub Releases, Git LFS, Zenodo, or another artifact host and link them from the main README.
