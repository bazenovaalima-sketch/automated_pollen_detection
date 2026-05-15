import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

import albumentations as A
import cv2
import yaml
from tqdm import tqdm
from ultralytics import RTDETR


USE_COLAB = True

DATA_ZIP = "mergeee.zip"
DATA_ROOT = Path("/content/pollen_data") if USE_COLAB else Path("./pollen_data")

RARE_CLASS_IDS = [0, 8, 10, 11, 12, 17, 19, 23]

TRAIN_PARAMS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "patience": 20,
    "save": True,
    "project": "pollen_project",
    "name": "rtdetr_v2_balanced",
    "degrees": 180.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.05,
    "hsv_s": 0.8,
    "mixup": 0.1,
}


if USE_COLAB:
    from google.colab import drive, files

    drive.mount("/content/drive")


def unzip_dataset(zip_filename: str, data_root: Path) -> None:
    zip_path = Path("/content/drive/MyDrive") / zip_filename if USE_COLAB else Path(zip_filename)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    data_root.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)


def augment_rare_classes(train_path: Path, multiplier: int = 5) -> None:
    image_dir = train_path / "images"
    label_dir = train_path / "labels"

    if not label_dir.exists():
        return

    transform = A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=40,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    label_files = list(label_dir.glob("*.txt"))

    for label_path in tqdm(label_files, desc="Oversampling rare classes"):
        lines = label_path.read_text().splitlines()

        if not lines:
            continue

        class_ids = [int(line.split()[0]) for line in lines]

        if not any(class_id in RARE_CLASS_IDS for class_id in class_ids):
            continue

        image_path = find_image_file(image_dir, label_path.stem)

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = [list(map(float, line.split()[1:])) for line in lines]

        for index in range(multiplier):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_ids,
                )

                new_image_name = f"aug_{index}_{image_path.name}"
                new_label_name = f"aug_{index}_{label_path.name}"

                output_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_dir / new_image_name), output_image)

                with open(label_dir / new_label_name, "w") as file:
                    for class_id, box in zip(
                        augmented["class_labels"],
                        augmented["bboxes"],
                    ):
                        box_values = " ".join(map(str, box))
                        file.write(f"{class_id} {box_values}\n")

            except Exception:
                continue


def find_image_file(image_dir: Path, stem: str) -> Path | None:
    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        image_path = image_dir / f"{stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def configure_yaml(data_root: Path) -> Path:
    yaml_path = data_root / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    config["train"] = str(data_root / "train" / "images")
    config["val"] = str(data_root / "valid" / "images")
    config["test"] = str(data_root / "test" / "images")

    with open(yaml_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return yaml_path


def backup_to_drive(trainer) -> None:
    if not USE_COLAB:
        return

    backup_dir = Path("/content/drive/MyDrive/pollen_backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(trainer.save_dir)
    weights_dir = save_dir / "weights"

    files_to_backup = [
        weights_dir / "best.pt",
        weights_dir / "last.pt",
        save_dir / "results.csv",
    ]

    for file_path in files_to_backup:
        if file_path.exists():
            shutil.copy(file_path, backup_dir / file_path.name)


def train_model(data_yaml: Path):
    model = RTDETR("rtdetr-l.pt")

    if USE_COLAB:
        model.add_callback("on_train_epoch_end", backup_to_drive)

    return model.train(
        data=str(data_yaml),
        **TRAIN_PARAMS,
    )


def export_results(project: str, name: str) -> None:
    run_dir = Path(project) / name

    if not run_dir.exists():
        return

    if USE_COLAB:
        output_dir = Path("/content/drive/MyDrive/pollen_results_final")
        output_dir.mkdir(parents=True, exist_ok=True)

        archive_path = output_dir / f"{name}_full_backup"
        shutil.make_archive(str(archive_path), "zip", run_dir)

        try:
            files.download(f"{archive_path}.zip")
        except Exception:
            pass

    else:
        output_dir = Path("./training_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(
            run_dir,
            output_dir / name,
            dirs_exist_ok=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR on pollen dataset")

    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip rare-class oversampling",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_PARAMS["epochs"],
        help="Number of training epochs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    TRAIN_PARAMS["epochs"] = args.epochs

    unzip_dataset(DATA_ZIP, DATA_ROOT)

    if not args.skip_augment:
        augment_rare_classes(DATA_ROOT / "train", multiplier=5)

    data_yaml = configure_yaml(DATA_ROOT)

    train_model(data_yaml)

    export_results(
        TRAIN_PARAMS["project"],
        TRAIN_PARAMS["name"],
    )


if __name__ == "__main__":
    main()
