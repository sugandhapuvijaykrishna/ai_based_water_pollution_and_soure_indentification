"""
Regression Training Pipeline
Trains PollutionCNNRegression model for continuous pollution index prediction
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from model_regression import PollutionCNNRegression, RegressionMetrics


class RegressionTrainingConfig:
    """Configuration for regression model training"""
    
    def __init__(self,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 early_stop_patience: int = 15,
                 reduce_lr_patience: int = 7,
                 model_dir: str = './models/regression',
                 log_dir: str = './logs/regression'):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_patience = early_stop_patience
        self.reduce_lr_patience = reduce_lr_patience
        
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_config()
    
    def _print_config(self):
        """Print training configuration"""
        print("\n" + "="*70)
        print("REGRESSION TRAINING CONFIGURATION")
        print("="*70)
        print(f"Epochs:                  {self.epochs}")
        print(f"Batch size:              {self.batch_size}")
        print(f"Learning rate:           {self.learning_rate}")
        print(f"Early stop patience:     {self.early_stop_patience}")
        print(f"Reduce LR patience:      {self.reduce_lr_patience}")
        print(f"Model directory:         {self.model_dir}")
        print(f"Log directory:           {self.log_dir}")
        print("="*70 + "\n")
    
    def get_callbacks(self, model_name: str = 'pollution_regression') -> list:
        """
        Create callback list for training
        
        Args:
            model_name: Name prefix for saved models
        
        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Early stopping based on validation loss
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
                mode='min',
                min_delta=1e-5
            ),
            
            # Save best model
            ModelCheckpoint(
                self.model_dir / f'{model_name}_best_{timestamp}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            
            # Save checkpoint every N epochs
            ModelCheckpoint(
                self.model_dir / f'{model_name}_checkpoint_{timestamp}.h5',
                monitor='val_loss',
                save_freq='epoch',
                verbose=0,
                period=5
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir / f'run_{timestamp}',
                histogram_freq=0,
                write_graph=True,
                update_freq='epoch',
                profile_batch='500,520'
            )
        ]
        
        return callbacks


class RegressionTrainer:
    """Train regression model"""
    
    def __init__(self, config: RegressionTrainingConfig):
        self.config = config
        self.model = None
        self.history = None
    
    def train(self,
             model: tf.keras.models.Sequential,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             data_augmentation: bool = False) -> Dict:
        """
        Train regression model
        
        Args:
            model: Compiled Keras model
            X_train: Training features (N, 32, 32, 6)
            y_train: Training targets (N, 1) - values 0-1
            X_val: Validation features
            y_val: Validation targets
            data_augmentation: Apply data augmentation
        
        Returns:
            Dictionary with training history
        """
        self.model = model
        
        # Validate data
        assert X_train.shape[1:] == (32, 32, 6), "Invalid feature shape"
        assert y_train.ndim == 1 or y_train.shape[1] == 1, "Invalid target shape"
        assert np.all((y_train >= 0) & (y_train <= 1)), "Target values must be in [0, 1]"
        
        # Reshape targets if needed
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        print("\n" + "="*70)
        print("STARTING REGRESSION TRAINING")
        print("="*70)
        print(f"Training samples:    {len(X_train)}")
        print(f"Validation samples:  {len(X_val)}")
        print(f"Feature shape:       {X_train.shape[1:]}")
        print(f"Target range:        [{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
        print("="*70 + "\n")
        
        # Optional data augmentation
        if data_augmentation:
            print("Applying data augmentation...")
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomFlip("vertical"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])
            
            X_train_aug = augmentation(X_train, training=True)
            X_train = np.concatenate([X_train, X_train_aug], axis=0)
            y_train = np.concatenate([y_train, y_train], axis=0)
            
            print(f"✓ Data augmented. New training size: {len(X_train)}")
        
        # Get callbacks
        callbacks = self.config.get_callbacks()
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=2
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Epochs completed:   {len(self.history.history['loss'])}")
        print(f"Final train loss:   {self.history.history['loss'][-1]:.6f}")
        print(f"Final val loss:     {self.history.history['val_loss'][-1]:.6f}")
        print("="*70 + "\n")
        
        return {
            'history': self.history.history,
            'epochs_trained': len(self.history.history['loss']),
            'callbacks': callbacks
        }


class RegressionEvaluator:
    """Evaluate regression model performance"""
    
    def __init__(self):
        self.metrics = None
    
    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values (N,) or (N, 1) in range [0, 1]
            y_pred: Predicted values (N,) or (N, 1) in range [0, 1]
        
        Returns:
            Dictionary with metrics
        """
        # Flatten if needed
        y_true = np.atleast_1d(y_true).flatten()
        y_pred = np.atleast_1d(y_pred).flatten()
        
        # Scale to pollution index
        y_true_index = y_true * 100
        y_pred_index = y_pred * 100
        
        # Calculate metrics
        mae = RegressionMetrics.calculate_mae(y_true, y_pred)
        mae_index = mae * 100
        
        rmse = RegressionMetrics.calculate_rmse(y_true, y_pred)
        rmse_index = rmse * 100
        
        mse = np.mean((y_true - y_pred) ** 2)
        mse_index = mse * 10000
        
        r_squared = RegressionMetrics.calculate_r_squared(y_true, y_pred)
        mape = RegressionMetrics.calculate_mape(y_true_index, y_pred_index)
        
        self.metrics = {
            # Raw (0-1)
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'r_squared': r_squared,
            
            # Scaled (0-100)
            'mae_index': mae_index,
            'rmse_index': rmse_index,
            'mse_index': mse_index,
            'mape': mape,
            
            # Predictions
            'y_true': y_true,
            'y_pred': y_pred,
            'y_true_index': y_true_index,
            'y_pred_index': y_pred_index
        }
        
        return self.metrics
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed regression report"""
        metrics = self.evaluate(y_true, y_pred)
        
        print("\n" + "="*70)
        print("REGRESSION EVALUATION REPORT")
        print("="*70)
        
        print("\nMetrics (Raw Scale 0-1):")
        print(f"  MAE (Mean Absolute Error):        {metrics['mae']:.6f}")
        print(f"  RMSE (Root Mean Squared Error):   {metrics['rmse']:.6f}")
        print(f"  MSE (Mean Squared Error):         {metrics['mse']:.6f}")
        print(f"  R² (Coefficient of Determination):{metrics['r_squared']:.4f}")
        
        print("\nMetrics (Pollution Index Scale 0-100):")
        print(f"  MAE:                              {metrics['mae_index']:.2f}")
        print(f"  RMSE:                             {metrics['rmse_index']:.2f}")
        print(f"  MAPE (Mean Absolute % Error):     {metrics['mape']:.2f}%")
        
        print("\nPrediction Statistics:")
        print(f"  True values range:   [{np.min(metrics['y_true_index']):.2f}, "
              f"{np.max(metrics['y_true_index']):.2f}]")
        print(f"  Predicted range:     [{np.min(metrics['y_pred_index']):.2f}, "
              f"{np.max(metrics['y_pred_index']):.2f}]")
        print(f"  Mean difference:     {np.mean(metrics['y_true_index'] - metrics['y_pred_index']):.2f}")
        
        print("="*70 + "\n")
        
        return metrics
    
    def plot_results(self, metrics: Dict, figsize: tuple = (15, 5)):
        """Plot regression results"""
        import matplotlib.pyplot as plt
        
        y_true = metrics['y_true_index']
        y_pred = metrics['y_pred_index']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Scatter plot: True vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
        axes[0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('True Pollution Index')
        axes[0].set_ylabel('Predicted Pollution Index')
        axes[0].set_title('Predictions vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([-5, 105])
        axes[0].set_ylim([-5, 105])
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Pollution Index')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.abs(residuals)
        axes[2].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(metrics['mae_index'], color='r', linestyle='--', 
                       linewidth=2, label=f"MAE: {metrics['mae_index']:.2f}")
        axes[2].set_xlabel('Absolute Error')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Error Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Evaluation plots saved to regression_evaluation.png")
        plt.show()


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("REGRESSION TRAINING PIPELINE - DEMONSTRATION")
    print("="*70)
    
    # ===== 1. CREATE MODEL =====
    print("\n[1] Creating regression model...")
    model_builder = PollutionCNNRegression(
        input_shape=(32, 32, 6),
        learning_rate=0.001,
        l2_strength=1e-4,
        dropout_rate=0.5
    )
    model = model_builder.build_model()
    model = model_builder.compile_model()
    model_builder.print_model_summary()
    
    # ===== 2. CREATE DUMMY DATASET =====
    print("\n[2] Creating dummy training dataset...")
    np.random.seed(42)
    
    X_train = np.random.randn(1000, 32, 32, 6).astype(np.float32)
    y_train = np.random.uniform(0, 1, 1000).astype(np.float32)
    
    X_val = np.random.randn(200, 32, 32, 6).astype(np.float32)
    y_val = np.random.uniform(0, 1, 200).astype(np.float32)
    
    X_test = np.random.randn(200, 32, 32, 6).astype(np.float32)
    y_test = np.random.uniform(0, 1, 200).astype(np.float32)
    
    print(f"✓ Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # ===== 3. TRAIN MODEL =====
    print("\n[3] Training model...")
    config = RegressionTrainingConfig(
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        early_stop_patience=5,
        reduce_lr_patience=3
    )
    
    trainer = RegressionTrainer(config)
    train_info = trainer.train(
        model, X_train, y_train, X_val, y_val,
        data_augmentation=False
    )
    
    # ===== 4. EVALUATE =====
    print("\n[4] Evaluating model...")
    y_test_pred = model.predict(X_test, batch_size=32, verbose=0)
    
    evaluator = RegressionEvaluator()
    metrics = evaluator.print_report(y_test, y_test_pred)
    
    # ===== 5. VISUALIZE =====
    print("\n[5] Generating visualizations...")
    evaluator.plot_results(metrics)
    
    print("\n" + "="*70)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*70)