# Chest X-Ray Pneumonia Detection

Binary classification of chest X-ray images (Normal vs Pneumonia) using a custom CNN built with TensorFlow/Keras. The model achieves 85.5% accuracy and 0.996 AUC on the held-out test set.

## Dataset

[Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) — 5,856 grayscale images across two classes.

| Class | Count | Proportion |
|---|---|---|
| NORMAL | 1,583 | 27.0% |
| PNEUMONIA | 4,273 | 73.0% |

The original Kaggle split is discarded in favour of a stratified 80/20 re-split to ensure reliable evaluation metrics.

## Model

Custom CNN with 4 convolutional blocks (32→64→128→256 filters), `GlobalAveragePooling2D`, and a two-layer fully-connected head with L2 regularisation and Dropout. Total parameters: ~488K.

Class imbalance is handled via `compute_class_weight('balanced')` applied during training and on-the-fly augmentation (rotation, flip, zoom, brightness shift) applied exclusively to the training generator.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 0.8549 |
| Precision (Pneumonia) | 1.0000 |
| Recall (Pneumonia) | 0.8012 |
| F1-Score (Pneumonia) | 0.8896 |
| ROC-AUC | 0.9959 |

## Project Structure

```
chest-xray/
├── notebook/
│   └── chest_xray_detection.ipynb
├── outputs/
│   └── models/
│       ├── best_model.keras
│       └── chest_xray_cnn_final.keras
├── .gitignore
├── README.md
└── requirements.txt
```

> Data folders (`chest_xray/`, `Dataset-Final/`) and model weights are excluded from version control via `.gitignore`.

## Setup

```bash
pip install -r requirements.txt
```

Open `notebook/chest_xray_detection.ipynb` and run all cells top to bottom. The notebook will download the dataset via the Kaggle API on first run.

## Requirements

- Python 3.11+
- TensorFlow 2.21.0
- See `requirements.txt` for the full dependency list
