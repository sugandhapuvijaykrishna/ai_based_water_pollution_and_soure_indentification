"""
River Detector Trainer
======================

Trains a binary classifier (river vs non-river) on `river_dataset.npz`
and writes predictions for all patches to `outputs/results_for_viz.csv`
for use by the Streamlit dashboard.

Labels:
    1 -> river
    0 -> non-river
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("trainer_river")


ROOT_DIR = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR.parent / "outputs"
DATA_PATH = OUTPUTS_DIR / "river_dataset.npz"
MODEL_DIR = OUTPUTS_DIR / "river_model"
MODEL_H5 = OUTPUTS_DIR / "river_model.h5"
RESULTS_CSV = OUTPUTS_DIR / "results_for_viz.csv"
METRICS_TXT = OUTPUTS_DIR / "river_metrics.txt"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_river_model(patch_shape=(32, 32, 6), meta_dim=2) -> Model:
    """
    Dual-branch architecture:
      - CNN branch on image patch
      - MLP branch on (lat, lon)
      - Sigmoid output for river probability
    """
    # CNN branch
    patch_input = Input(shape=patch_shape, name="patch_input")

    x = Conv2D(32, (3, 3), padding="same")(patch_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    cnn_out = Dense(64, activation="relu")(x)

    # Metadata branch
    meta_input = Input(shape=(meta_dim,), name="meta_input")
    m = Dense(16, activation="relu")(meta_input)
    mlp_out = Dense(8, activation="relu")(m)

    # Fusion
    fused = Concatenate()([cnn_out, mlp_out])
    fused = Dense(32, activation="relu")(fused)
    output = Dense(1, activation="sigmoid", name="river_prob")(fused)

    model = Model(inputs=[patch_input, meta_input], outputs=output)
    return model


def main() -> None:
    if not DATA_PATH.exists():
        log.error("River dataset not found: %s", DATA_PATH)
        log.error("Run build_river_dataset.py first.")
        return

    log.info("Loading river dataset: %s", DATA_PATH)
    # Match build_river_dataset.py which may store object arrays
    data = np.load(DATA_PATH, allow_pickle=True)

    X_patches = data["X_patches"].astype("float32")
    X_meta = data["X_meta"].astype("float32")
    y_river = data["y_river"].astype("float32")  # shape (N,)

    n = len(y_river)
    log.info("Samples: %d", n)
    log.info("Positive (river)   : %d", int(y_river.sum()))
    log.info("Negative (non-river): %d", int(n - y_river.sum()))

    # Train / test split using shared indices so patches and meta stay aligned
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y_river,
    )

    X_p_train = X_patches[train_idx]
    X_p_test = X_patches[test_idx]
    X_m_train = X_meta[train_idx]
    X_m_test = X_meta[test_idx]
    y_train = y_river[train_idx]
    y_test = y_river[test_idx]

    log.info("Train size: %d | Test size: %d", len(X_p_train), len(X_p_test))

    model = build_river_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary(print_fn=lambda s: log.info(s))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(MODEL_H5),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    log.info("Starting training...")
    history = model.fit(
        [X_p_train, X_m_train],
        y_train,
        validation_data=([X_p_test, X_m_test], y_test),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.export(str(MODEL_DIR))
    log.info("Model saved to: %s", MODEL_DIR)
    log.info("Best weights (h5) saved to: %s", MODEL_H5)

    # Evaluation on test set
    log.info("Evaluating on test set...")
    y_prob_test = model.predict([X_p_test, X_m_test], verbose=0).flatten()
    y_pred_test = (y_prob_test >= 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    log.info("Test Accuracy : %.4f", acc)
    log.info("Test Precision: %.4f", prec)
    log.info("Test Recall   : %.4f", rec)
    log.info("Test F1-score : %.4f", f1)

    # Persist metrics to a text file for audit
    with METRICS_TXT.open("w", encoding="utf-8") as f:
        f.write("River Detector Metrics\n")
        f.write("======================\n")
        f.write(f"Dataset       : {DATA_PATH}\n")
        f.write(f"Model (SavedModel): {MODEL_DIR}\n")
        f.write(f"Model (h5)   : {MODEL_H5}\n\n")
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"Precision     : {prec:.4f}\n")
        f.write(f"Recall        : {rec:.4f}\n")
        f.write(f"F1-score      : {f1:.4f}\n")

    # Generate predictions for ALL patches for visualization
    log.info("Generating predictions for all patches for visualization...")
    y_prob_all = model.predict([X_patches, X_meta], verbose=0).flatten()
    y_pred_all = (y_prob_all >= 0.5).astype("int32")

    lats = X_meta[:, 0]
    lons = X_meta[:, 1]

    # Build viz-friendly CSV.
    # NOTE: For compatibility with the existing dashboard, we still
    # populate a `Predicted_NTU` column, but it now represents
    # river probability scaled to [0, 100].
    df = pd.DataFrame(
        {
            "Lat": lats,
            "Lon": lons,
            "River_Prob": y_prob_all,
            "Is_River": y_pred_all,
            "Predicted_NTU": y_prob_all * 100.0,
        }
    )

    # Dummy columns kept for compatibility with the current dashboard.
    df["Flow_U"] = 0.0
    df["Flow_V"] = 0.0
    df["Risk_Level"] = df["Is_River"].map({1: "River", 0: "Non-River"})
    df["Source"] = "RiverDetectorV1"

    df.to_csv(RESULTS_CSV, index=False)
    log.info("Visualization CSV written to: %s", RESULTS_CSV)


if __name__ == "__main__":
    main()

