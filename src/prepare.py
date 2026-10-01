"""Download Fashion-MNIST and persist its original arrays."""

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    np.save(raw_dir / "train_images.npy", train_images)
    np.save(raw_dir / "train_labels.npy", train_labels)
    np.save(raw_dir / "test_images.npy", test_images)
    np.save(raw_dir / "test_labels.npy", test_labels)

    print(f"Saved raw data to {raw_dir}")
    print(f"  train_images: {train_images.shape}, train_labels: {train_labels.shape}")
    print(f"  test_images:  {test_images.shape},  test_labels:  {test_labels.shape}")


if __name__ == "__main__":
    main()
