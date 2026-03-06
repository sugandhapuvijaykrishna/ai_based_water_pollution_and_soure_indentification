"""
Dual-Branch Fusion Trainer
===========================
Trains CNN + MLP fusion model on final_fusion_dataset.npz
Outputs: best_fusion_model (SavedModel format) + results_for_viz.csv
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Flatten, Dense, Concatenate,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── Relative Paths (works on any machine) ─────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR.parent / "outputs"
DATA_PATH   = OUTPUTS_DIR / "final_fusion_dataset.npz"
RESULTS_CSV = OUTPUTS_DIR / "results_for_viz.csv"
MODEL_DIR   = OUTPUTS_DIR / "best_fusion_model"   # SavedModel format (folder)
MODEL_H5    = OUTPUTS_DIR / "best_fusion_model.h5" # h5 backup

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_dual_branch_model(patch_shape=(32, 32, 6), meta_dim=2):
    """
    Dual-Branch Model:
    - Satellite Branch: CNN (3 Conv blocks + GlobalAveragePooling)
    - Metadata Branch:  MLP for Lat/Lon
    - Fusion:           Linear output for NTU prediction
    """
    # CNN Branch
    patch_input = Input(shape=patch_shape, name='patch_input')

    x = Conv2D(32, (3, 3), padding='same')(patch_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)  # Better than Flatten for regression
    x = Dropout(0.3)(x)
    cnn_out = Dense(64, activation='relu')(x)

    # MLP Branch
    meta_input = Input(shape=(meta_dim,), name='meta_input')
    m = Dense(16, activation='relu')(meta_input)
    mlp_out = Dense(8, activation='relu')(m)

    # Fusion
    fused = Concatenate()([cnn_out, mlp_out])
    fused = Dense(32, activation='relu')(fused)
    output = Dense(1, activation='linear', name='ntu_output')(fused)

    model = Model(inputs=[patch_input, meta_input], outputs=output)
    return model


def main():
    # Load dataset
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please run fusion_dataset.py first.")
        return

    print("Loading dataset...")
    data = np.load(DATA_PATH)
    X_patches = data['X_patches'].astype('float32')  # (N, 32, 32, 6)
    X_meta    = data['X_meta'].astype('float32')      # (N, 2)
    y_ntu     = data['y_ntu'].astype('float32')       # (N,)

    total_samples = len(y_ntu)
    print(f"Total samples: {total_samples}")
    print(f"NTU range: {y_ntu.min():.1f} - {y_ntu.max():.1f}")

    # Subsample if large
    subsample_size = min(8000, total_samples)
    if subsample_size < total_samples:
        print(f"Subsampling {subsample_size} from {total_samples} samples...")
        indices = np.random.choice(total_samples, subsample_size, replace=False)
        X_patches = X_patches[indices]
        X_meta    = X_meta[indices]
        y_ntu     = y_ntu[indices]

    # Train/Val split
    X_p_train, X_p_val, X_m_train, X_m_val, y_train, y_val = train_test_split(
        X_patches, X_meta, y_ntu, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_p_train)} | Val: {len(X_p_val)}")

    # Build model
    model = build_dual_branch_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODEL_H5),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train - 50 epochs with early stopping
    print("\nStarting training (max 50 epochs with early stopping)...")
    model.fit(
        [X_p_train, X_m_train], y_train,
        validation_data=([X_p_val, X_m_val], y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Save in SavedModel format (more reliable than h5)
    model.export(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")
    print(f"Model also saved to {MODEL_H5}")

    # Generate predictions + flow vectors for Person 3
    print("\nGenerating predictions and flow vectors...")
    y_pred = model.predict([X_patches, X_meta], verbose=0).flatten()

    lats = X_meta[:, 0]
    lons = X_meta[:, 1]

    # KNN spatial gradient for flow direction
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=15).fit(X_meta)
    indices_nn = nn.kneighbors(X_meta, return_distance=False)

    flow_u = np.zeros_like(y_pred)
    flow_v = np.zeros_like(y_pred)

    for i in range(len(y_pred)):
        neigh_idx = indices_nn[i]
        d_lon = lons[neigh_idx] - lons[i]
        d_lat = lats[neigh_idx] - lats[i]
        d_ntu = y_pred[neigh_idx] - y_pred[i]

        M = np.vstack([d_lon, d_lat]).T
        if np.linalg.cond(M.T @ M) < 1e18:
            coeffs = np.linalg.lstsq(M, d_ntu, rcond=None)[0]
            flow_u[i] = -coeffs[0]
            flow_v[i] = -coeffs[1]

    # Fallback: assign river default flow direction (west to east)
    # for any point that still has zero flow
    zero_mask = (flow_u == 0) & (flow_v == 0)
    if zero_mask.sum() > 0:
        # Use mean flow from non-zero neighbors
        mean_u = flow_u[~zero_mask].mean() if (~zero_mask).sum() > 0 else 0.001
        mean_v = flow_v[~zero_mask].mean() if (~zero_mask).sum() > 0 else 0.0
        flow_u[zero_mask] = mean_u
        flow_v[zero_mask] = mean_v

    # Add risk level for dashboard
    def get_risk_level(ntu):
        if ntu < 100:   return 'Low'
        elif ntu < 300: return 'Moderate'
        elif ntu < 500: return 'High'
        else:           return 'Critical'

    def get_pollution_source(lat, lon):
        # Industrial zone is in eastern river corridor
        IND_LAT, IND_LON = 16.5193, 80.6305
        # Agricultural zone is in western river corridor
        AGR_LAT, AGR_LON = 16.5500, 80.4800
        d_ind = ((lat - IND_LAT)**2 + (lon - IND_LON)**2)**0.5
        d_agr = ((lat - AGR_LAT)**2 + (lon - AGR_LON)**2)**0.5
        if d_ind < d_agr:
            return 'Industrial Zone Upstream'
        return 'Agricultural Runoff'

    results = pd.DataFrame({
        'Lat':           lats,
        'Lon':           lons,
        'Predicted_NTU': y_pred,
        'Flow_U':        flow_u,
        'Flow_V':        flow_v,
        'Risk_Level':    [get_risk_level(n) for n in y_pred],
        'Source':        [get_pollution_source(la, lo) for la, lo in zip(lats, lons)],
    })

    results.to_csv(str(RESULTS_CSV), index=False)
    print(f"Results saved to {RESULTS_CSV}")

    # Print final metrics
    val_pred = model.predict([X_p_val, X_m_val], verbose=0).flatten()
    mae  = np.mean(np.abs(y_val - val_pred))
    rmse = np.sqrt(np.mean((y_val - val_pred)**2))
    print(f"\nFinal Validation MAE:  {mae:.2f} NTU")
    print(f"Final Validation RMSE: {rmse:.2f} NTU")
    print("\nPerson 2 task complete. results_for_viz.csv ready for Person 3.")


if __name__ == "__main__":
    main()

