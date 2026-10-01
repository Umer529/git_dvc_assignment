"""Normalize Fashion-MNIST and create reproducible train/validation splits."""

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["preprocess"]


def main() -> None:
    params = load_params()
    train_images = np.load(RAW_DIR / "train_images.npy")
    train_labels = np.load(RAW_DIR / "train_labels.npy")
    test_images = np.load(RAW_DIR / "test_images.npy")
    test_labels = np.load(RAW_DIR / "test_labels.npy")

    # float32 keeps the processed data and training memory footprint compact.
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size=params["validation_size"],
        random_state=params["seed"],
        stratify=train_labels,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": test_images,
        "y_test": test_labels,
    }
    for name, values in arrays.items():
        np.save(PROCESSED_DIR / f"{name}.npy", values)

    print(f"Saved processed data to {PROCESSED_DIR}")
    print(f"  train: {x_train.shape}, validation: {x_val.shape}, test: {test_images.shape}")


if __name__ == "__main__":
    main()

