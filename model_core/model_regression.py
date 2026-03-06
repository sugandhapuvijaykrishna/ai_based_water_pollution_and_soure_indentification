"""
CNN Regression Model for Water Pollution Detection
Landsat 8/9 Satellite Patches (32x32x6)
Output: Continuous Pollution Index (0-1, scaled to 0-100)

Architecture:
- 3 Conv blocks with BatchNorm + ReLU + MaxPooling + Dropout
- GlobalAveragePooling2D (instead of Flatten)
- Dense(128) + BatchNorm + ReLU + Dropout
- Dense(1, sigmoid) for regression output

Loss: MSE
Metrics: MAE, RMSE
Optimizer: Adam (lr=0.001)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    BatchNormalization, Dropout, Dense, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import numpy as np


class PollutionCNNRegression:
    """
    CNN Regression Model for Continuous Pollution Index Prediction
    
    Input: 32x32x6 patches (4 bands + NDWI + NDTI)
    Output: Single value between 0-1 (Pollution Index)
    
    Scaled output: Pollution_Index = model_output * 100
    """
    
    def __init__(self, 
                 input_shape: tuple = (32, 32, 6),
                 learning_rate: float = 0.001,
                 l2_strength: float = 1e-4,
                 dropout_rate: float = 0.5):
        """
        Initialize regression model configuration
        
        Args:
            input_shape: Input tensor shape (H, W, C)
            learning_rate: Adam optimizer learning rate
            l2_strength: L2 regularization strength
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.l2_strength = l2_strength
        self.dropout_rate = dropout_rate
        self.model = None
        
        print("\n" + "="*70)
        print("POLLUTION CNN REGRESSION - INITIALIZATION")
        print("="*70)
        print(f"Input shape:        {self.input_shape}")
        print(f"Learning rate:      {self.learning_rate}")
        print(f"L2 regularization:  {self.l2_strength}")
        print(f"Dropout rate:       {self.dropout_rate}")
        print("="*70 + "\n")
    
    def build_model(self) -> Sequential:
        """
        Build regression CNN architecture
        
        Architecture:
        - Conv Block 1: Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - Conv Block 2: Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - Conv Block 3: Conv2D(128) -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - GlobalAveragePooling2D
        - Dense(128) -> BatchNorm -> ReLU -> Dropout
        - Dense(1, sigmoid) [Regression Output]
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential(name='PollutionRegressionCNN')
        
        # ===== CONVOLUTIONAL BLOCK 1 =====
        model.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=l2(self.l2_strength),
            input_shape=self.input_shape,
            name='conv1_1'
        ))
        model.add(BatchNormalization(name='bn1_1'))
        model.add(Activation('relu', name='relu1_1'))
        
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            name='maxpool1'
        ))
        model.add(Dropout(self.dropout_rate, name='dropout1'))
        
        # ===== CONVOLUTIONAL BLOCK 2 =====
        model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=l2(self.l2_strength),
            name='conv2_1'
        ))
        model.add(BatchNormalization(name='bn2_1'))
        model.add(Activation('relu', name='relu2_1'))
        
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            name='maxpool2'
        ))
        model.add(Dropout(self.dropout_rate, name='dropout2'))
        
        # ===== CONVOLUTIONAL BLOCK 3 =====
        model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=l2(self.l2_strength),
            name='conv3_1'
        ))
        model.add(BatchNormalization(name='bn3_1'))
        model.add(Activation('relu', name='relu3_1'))
        
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            name='maxpool3'
        ))
        model.add(Dropout(self.dropout_rate, name='dropout3'))
        
        # ===== GLOBAL AVERAGE POOLING (NOT FLATTEN) =====
        model.add(GlobalAveragePooling2D(name='global_avg_pool'))
        
        # ===== FULLY CONNECTED LAYERS =====
        model.add(Dense(
            units=128,
            kernel_regularizer=l2(self.l2_strength),
            name='dense1'
        ))
        model.add(BatchNormalization(name='bn_dense1'))
        model.add(Activation('relu', name='relu_dense1'))
        model.add(Dropout(self.dropout_rate, name='dropout_dense1'))
        
        # ===== REGRESSION OUTPUT LAYER =====
        # Single neuron with sigmoid activation (output range: 0-1)
        model.add(Dense(
            units=1,
            activation='sigmoid',
            kernel_regularizer=l2(self.l2_strength),
            name='pollution_index'
        ))
        
        self.model = model
        
        return model
    
    def compile_model(self, 
                     loss_function: str = 'mse',
                     metrics: list = None) -> Sequential:
        """
        Compile the regression model
        
        Args:
            loss_function: Loss function ('mse', 'mae', 'msle')
            metrics: List of metrics to monitor
        
        Returns:
            Compiled model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Default metrics if not provided
        if metrics is None:
            metrics = [
                'mae',
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanSquaredError(name='mse')
            ]
        
        # Optimizer
        optimizer = Adam(learning_rate=self.learning_rate, name='adam_opt')
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        
        print("\n" + "="*70)
        print("MODEL COMPILATION")
        print("="*70)
        print(f"Optimizer:     Adam (lr={self.learning_rate})")
        print(f"Loss function: {loss_function}")
        print(f"Metrics:       {[m if isinstance(m, str) else m.name for m in metrics]}")
        print("="*70 + "\n")
        
        return self.model
    
    def print_model_summary(self, verbose: int = 1):
        """
        Print detailed model summary and statistics
        
        Args:
            verbose: Verbosity level (0, 1, or 2)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        
        self.model.summary(verbose=verbose)
        
        # Calculate parameter statistics
        total_params = self.model.count_params()
        trainable_params = sum([
            tf.keras.backend.count_params(w) 
            for w in self.model.trainable_weights
        ])
        non_trainable_params = total_params - trainable_params
        
        print("\n" + "="*70)
        print("PARAMETER STATISTICS")
        print("="*70)
        print(f"Total parameters:          {total_params:,}")
        print(f"Trainable parameters:      {trainable_params:,}")
        print(f"Non-trainable parameters:  {non_trainable_params:,}")
        print("="*70 + "\n")
        
        # Layer breakdown
        print("\n" + "="*70)
        print("LAYER BREAKDOWN")
        print("="*70)
        
        for i, layer in enumerate(self.model.layers):
            params = layer.count_params()
            layer_type = layer.__class__.__name__
            
            if params > 0:
                print(f"{i:2d}. {layer_type:25s} {str(layer.output_shape):30s} "
                      f"Params: {params:>10,}")
            else:
                print(f"{i:2d}. {layer_type:25s} {str(layer.output_shape):30s} "
                      f"Params: {'—':>10s}")
        
        print("="*70 + "\n")
        
        # Model configuration
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        print(f"Input shape:           {self.input_shape}")
        print(f"Output shape:          (None, 1)  [Regression]")
        print(f"Output range:          [0, 1]  (Sigmoid activation)")
        print(f"Pollution Index scale: 0-1 → 0-100")
        print(f"Loss function:         MSE (Mean Squared Error)")
        print(f"Optimizer:             Adam (lr={self.learning_rate})")
        print("="*70 + "\n")
    
    def get_model(self) -> Sequential:
        """Get the compiled model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() and compile_model().")
        return self.model
    
    @staticmethod
    def scale_predictions(predictions: np.ndarray) -> np.ndarray:
        """
        Scale regression output to Pollution Index (0-100)
        
        Args:
            predictions: Raw model predictions (0-1 range)
        
        Returns:
            Scaled Pollution Index (0-100)
        """
        return np.clip(predictions * 100, 0, 100).astype(np.float32)
    
    @staticmethod
    def descale_predictions(pollution_index: np.ndarray) -> np.ndarray:
        """
        Convert Pollution Index back to model output (0-1)
        
        Args:
            pollution_index: Pollution Index (0-100)
        
        Returns:
            Model output (0-1)
        """
        return np.clip(pollution_index / 100.0, 0, 1).astype(np.float32)
    
    def print_prediction_example(self, sample_input: np.ndarray):
        """
        Print example prediction with scaling
        
        Args:
            sample_input: Sample patch array (32, 32, 6) or (1, 32, 32, 6)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() and compile_model().")
        
        # Ensure correct shape
        if len(sample_input.shape) == 3:
            sample_input = sample_input[np.newaxis, ...]
        
        # Get prediction
        raw_pred = self.model.predict(sample_input, verbose=0)
        scaled_pred = self.scale_predictions(raw_pred)
        
        print("\n" + "="*70)
        print("EXAMPLE PREDICTION")
        print("="*70)
        print(f"Input shape:          {sample_input.shape}")
        print(f"Raw model output:     {raw_pred[0, 0]:.6f}")
        print(f"Pollution Index:      {scaled_pred[0, 0]:.2f} / 100")
        print("="*70 + "\n")


# ===== REGRESSION-SPECIFIC UTILITIES =====

class RegressionMetrics:
    """Custom metrics for regression evaluation"""
    
    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination)
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE score (%)
        """
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("POLLUTION CNN REGRESSION - DEMONSTRATION")
    print("="*70)
    
    # ===== 1. CREATE AND BUILD MODEL =====
    print("\n[1] Creating regression model...")
    model_builder = PollutionCNNRegression(
        input_shape=(32, 32, 6),
        learning_rate=0.001,
        l2_strength=1e-4,
        dropout_rate=0.5
    )
    
    # Build architecture
    model = model_builder.build_model()
    print("✓ Model architecture built")
    
    # ===== 2. COMPILE MODEL =====
    print("\n[2] Compiling model...")
    model = model_builder.compile_model(
        loss_function='mse',
        metrics=['mae', 
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                tf.keras.metrics.MeanSquaredError(name='mse')]
    )
    print("✓ Model compiled")
    
    # ===== 3. PRINT SUMMARY =====
    print("\n[3] Model summary...")
    model_builder.print_model_summary(verbose=1)
    
    # ===== 4. TEST WITH DUMMY DATA =====
    print("\n[4] Testing with dummy data...")
    
    # Create dummy patch
    dummy_patch = np.random.randn(1, 32, 32, 6).astype(np.float32)
    
    # Make prediction
    raw_pred = model.predict(dummy_patch, verbose=0)
    pollution_index = PollutionCNNRegression.scale_predictions(raw_pred)
    
    print(f"✓ Sample prediction:")
    print(f"  Raw output:      {raw_pred[0, 0]:.6f}")
    print(f"  Pollution Index: {pollution_index[0, 0]:.2f} / 100")
    
    # ===== 5. TEST SCALING FUNCTIONS =====
    print("\n[5] Testing scaling functions...")
    
    test_indices = np.array([[25.5], [50.0], [75.5]])
    test_raw = PollutionCNNRegression.descale_predictions(test_indices)
    back_scaled = PollutionCNNRegression.scale_predictions(test_raw)
    
    print(f"Original index:   {test_indices.flatten()}")
    print(f"→ to raw (0-1):   {test_raw.flatten()}")
    print(f"→ back to index:  {back_scaled.flatten()}")
    
    print("\n" + "="*70)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*70)