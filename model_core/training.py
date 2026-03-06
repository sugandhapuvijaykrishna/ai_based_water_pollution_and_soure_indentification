"""
Model training with EarlyStopping and ModelCheckpoint
Memory-efficient batch handling

Extended with:
  train_fusion_model()    – dual-input regression training loop
  NTUGradientCallback     – logs Pearson r(predicted_NTU, longitude)
                            as a proxy for directional pollution gradient
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, Callback
)
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import pearsonr

class TrainingConfig:
    """Training configuration"""

    def __init__(self,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 early_stop_patience: int = 10,
                 reduce_lr_patience: int = 5,
                 model_dir: str = './models',
                 mode: str = 'classification'):   # 'classification' | 'regression'

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_patience = early_stop_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.mode = mode

        print('\n' + '=' * 60)
        print('TRAINING CONFIGURATION')
        print('=' * 60)
        print(f'Mode:                {self.mode}')
        print(f'Epochs:              {self.epochs}')
        print(f'Batch size:          {self.batch_size}')
        print(f'Learning rate:       {self.learning_rate}')
        print(f'Early stop patience: {self.early_stop_patience}')
        print(f'Model directory:     {self.model_dir}')
        print('=' * 60 + '\n')

    def get_callbacks(self) -> list:
        """Create callback list, adapted for classification vs regression."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.mode == 'regression':
            ckpt_monitor, ckpt_mode = 'val_mae',  'min'
        else:
            ckpt_monitor, ckpt_mode = 'val_mae',  'min'   # unified: lower is better

        callbacks = [
            # Early stopping on val_loss (works for both modes)
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),

            # Save best model
            ModelCheckpoint(
                str(self.model_dir / f'best_model_{timestamp}.h5'),
                monitor=ckpt_monitor,
                save_best_only=True,
                verbose=1,
                mode=ckpt_mode
            ),

            # Reduce LR on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard
            TensorBoard(
                log_dir=str(self.model_dir / f'logs_{timestamp}'),
                histogram_freq=0,
                write_graph=True,
                update_freq='epoch'
            ),
        ]

        return callbacks


class NTUGradientCallback(Callback):
    """
    After each epoch, compute Pearson r between the model's predicted NTU
    values and the raw longitude of each validation sample.

    A strong negative r  (r < -0.3) means NTU decreases as longitude
    increases → the model has learned the west→east pollution decay
    along the Krishna River.  This is the directional gradient signal.
    """

    def __init__(self, X_val_patches, X_val_meta, y_val_ntu,
                 X_meta_raw_val):
        """
        Args:
            X_val_patches   : (N,32,32,6) scaled patch array
            X_val_meta      : (N,2) scaled lat/lon for model input
            y_val_ntu       : (N,) ground-truth NTU
            X_meta_raw_val  : (N,2) unscaled [lat, lon] for Pearson r
        """
        super().__init__()
        self.X_val_patches  = X_val_patches
        self.X_val_meta     = X_val_meta
        self.y_val_ntu      = y_val_ntu
        self.lons           = X_meta_raw_val[:, 1]   # column 1 = longitude

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(
            [self.X_val_patches, self.X_val_meta],
            verbose=0
        ).flatten()

        if len(np.unique(y_pred)) > 1:
            r, p = pearsonr(y_pred, self.lons)
            logs['val_ntu_lon_pearson_r'] = float(r)
            direction = (
                'West→East decay detected' if r < -0.3
                else 'East→West decay detected' if r > 0.3
                else 'No strong directional gradient'
            )
            print(f'  [NTUGradient] Pearson r(NTU, lon)={r:.3f}  →  {direction}')


class CustomMetricsCallback(Callback):
    """Custom metrics callback for monitoring per-class performance (classification only)"""

    def __init__(self, X_val, y_val, class_names=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names or ['Clean', 'Moderate', 'High']

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val), axis=1)
        for i, class_name in enumerate(self.class_names):
            mask = self.y_val == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).mean()
                logs[f'val_acc_{class_name.lower()}'] = acc


def train_model(model,
                data: dict,
                config: TrainingConfig,
                data_augmentation: bool = False) -> dict:
    """
    Train a classification model (single input X, integer labels y).

    Args:
        model             : Compiled Keras model
        data              : Data dictionary from DataPipeline.train_test_split()
        config            : TrainingConfig instance
        data_augmentation : Whether to use data augmentation

    Returns:
        Dictionary with history and metrics
    """
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']

    callbacks = config.get_callbacks()

    print('\n' + '=' * 60)
    print('TRAINING CLASSIFICATION MODEL')
    print('=' * 60)
    print(f'Training samples:   {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Batch size:         {config.batch_size}')
    print('=' * 60 + '\n')

    if data_augmentation:
        print('Applying data augmentation...')
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomFlip('vertical'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        X_train_aug = []
        for x in X_train:
            X_train_aug.append(
                augmentation(x[np.newaxis, ...], training=True).numpy()
            )
        X_train = np.concatenate(X_train_aug)
        y_train = np.repeat(y_train, len(X_train) // len(y_train))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
        class_weight={0: 1.0, 1: 1.2, 2: 1.5}   # address class imbalance
    )

    return {
        'history':       history.history,
        'epochs_trained': len(history.history['loss'])
    }


# ───────────────────────────────────────────────────────────────────
# Regression training loop (dual-input)
# ───────────────────────────────────────────────────────────────────
def train_fusion_model(model,
                       data: dict,
                       config: TrainingConfig,
                       X_meta_raw_val: np.ndarray = None) -> dict:
    """
    Train the DualBranchFusionModel on (patches + GPS) → NTU regression.

    Args:
        model           : Compiled DualBranchFusionModel
        data            : Dict from DataPipeline.load_fusion_npz()
        config          : TrainingConfig (mode should be 'regression')
        X_meta_raw_val  : Unscaled (lat, lon) for NTUGradientCallback

    Returns:
        Dict with history and epochs_trained
    """
    Xp_tr  = data['X_patches_train']
    Xm_tr  = data['X_meta_train']
    y_tr   = data['y_ntu_train']

    Xp_val = data['X_patches_val']
    Xm_val = data['X_meta_val']
    y_val  = data['y_ntu_val']

    print('\n' + '=' * 60)
    print('TRAINING DUAL-BRANCH FUSION MODEL  (Regression)')
    print('=' * 60)
    print(f'  Training samples   : {len(y_tr)}')
    print(f'  Validation samples : {len(y_val)}')
    print(f'  Batch size         : {config.batch_size}')
    print(f'  NTU train range    : [{y_tr.min():.1f}, {y_tr.max():.1f}]')
    print('=' * 60 + '\n')

    callbacks = config.get_callbacks()

    # Add the directional gradient monitor if raw meta is provided
    if X_meta_raw_val is not None:
        callbacks.append(
            NTUGradientCallback(
                X_val_patches  = Xp_val,
                X_val_meta     = Xm_val,
                y_val_ntu      = y_val,
                X_meta_raw_val = X_meta_raw_val,
            )
        )

    history = model.fit(
        [Xp_tr, Xm_tr], y_tr,
        validation_data=([Xp_val, Xm_val], y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return {
        'history':        history.history,
        'epochs_trained': len(history.history['loss']),
    }


if __name__ == '__main__':
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stop_patience=10,
        mode='regression',
    )