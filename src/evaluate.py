"""Evaluate the trained model and persist machine-readable metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main() -> None:
    model = tf.keras.models.load_model(MODELS_DIR / "model.h5")
    x_test = np.load(PROCESSED_DIR / "x_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
    with (PROJECT_ROOT / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    figure.tight_layout()
    figure.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
