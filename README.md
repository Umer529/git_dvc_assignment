# Fashion-MNIST ANN Pipeline

An end-to-end, reproducible MLOps project that trains a fully connected
TensorFlow neural network to classify Fashion-MNIST images. Git versions the
source and pipeline definitions, while DVC versions datasets and model
artifacts.

## Pipeline

1. `prepare` downloads Fashion-MNIST and saves NumPy arrays in `data/raw/`.
2. `preprocess` normalizes images and creates training, validation, and test sets.
3. `train` fits a Flatten/Dense/Dropout/Softmax ANN and saves its history.
4. `evaluate` writes test metrics and a confusion matrix.

## Setup and execution

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
dvc pull
dvc repro
dvc metrics show
```

Hyperparameters are centralized in `params.yaml`. The default DVC remote is a
Google Drive folder; access must be granted by the repository owner.
