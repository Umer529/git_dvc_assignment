"""Train a fully connected Fashion-MNIST classifier."""

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["train"]


def build_model(params: dict) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params["dense_units"], activation="relu"),
            tf.keras.layers.Dropout(params["dropout_rate"]),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History, destination: Path) -> None:
    columns = ["epoch", *history.history.keys()]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in range(len(history.history["loss"])):
            writer.writerow(
                {"epoch": epoch + 1}
                | {name: values[epoch] for name, values in history.history.items()}
            )


def main() -> None:
    params = load_params()
    tf.keras.utils.set_random_seed(params["seed"])
    x_train = np.load(PROCESSED_DIR / "x_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    x_val = np.load(PROCESSED_DIR / "x_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    model = build_model(params)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model.h5")
    save_history(history, MODELS_DIR / "history.csv")
    print(f"Saved model and training history to {MODELS_DIR}")


if __name__ == "__main__":
    main()

