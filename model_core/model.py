"""
CNN Architecture for Water Pollution Classification & Regression
Optimized for 16GB RAM with BatchNorm + Dropout

Models
------
PollutionCNN          : original classification model (3-class softmax)
DualBranchFusionModel : regression model (NTU prediction)
  - CNN branch   : 32x32x6 image patches → spectral features
  - MLP branch   : (Lat, Lon) metadata  → spatial features
  - Fusion head  : combined → single linear NTU output
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    BatchNormalization, Dropout, Flatten, Dense,
    Input, Activation, Add, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError

class PollutionCNN:
    """Build and compile CNN models for pollution classification"""
    
    @staticmethod
    def simple_cnn(input_shape: tuple = (32, 32, 6), 
                   num_classes: int = 3) -> Sequential:
        """
        Simple 3-layer CNN (lightweight)
        Good for 32x32 patches
        
        Args:
            input_shape: Input tensor shape (H, W, C)
            num_classes: Number of output classes (3)
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding='same', 
                   kernel_regularizer=l2(1e-4),
                   input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully connected
            Flatten(),
            Dense(256, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def medium_cnn(input_shape: tuple = (32, 32, 6),
                   num_classes: int = 3) -> Sequential:
        """
        Medium 4-layer CNN with more capacity
        For 64x64 patches or higher precision
        
        Args:
            input_shape: Input tensor shape
            num_classes: Number of output classes
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4),
                   input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully connected
            Flatten(),
            Dense(512, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(256, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def compile_model(model: Sequential, 
                     learning_rate: float = 0.001) -> Sequential:
        """
        Compile model with optimizer and loss
        
        Args:
            model: Keras model
            learning_rate: Adam learning rate
        
        Returns:
            Compiled model
        """
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    @staticmethod
    def print_model_info(model: Sequential):
        """Print model architecture and parameter count"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        model.summary()
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in model.trainable_weights])
        non_trainable = total_params - trainable_params
        
        print("\nPARAMETER SUMMARY:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable:        {non_trainable:,}")
        print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Dual-Branch Fusion Model  (Regression – NTU prediction)
# ─────────────────────────────────────────────────────────────────────────────
class DualBranchFusionModel:
    """
    Dual-branch architecture for turbidity regression.

    Branch 1 – CNN
        Processes 32×32×6 image patches to extract spectral/spatial features.
        Architecture: Conv(32)→BN→ReLU→Pool → Conv(64)→BN→ReLU→Pool
                     → Conv(128)→BN→ReLU→Pool → Flatten → Dense(256)

    Branch 2 – MLP
        Processes (Latitude, Longitude) GPS coordinates to encode directional
        spatial context along the river gradient.
        Architecture: Dense(32)→BN→ReLU → Dense(64)→BN→ReLU → Dense(32)

    Fusion Head
        Concatenate(CNN_out, MLP_out) → Dense(128) → Dropout(0.3)
        → Dense(64) → Dense(1, activation='linear')  [NTU output]

    Loss   : Mean Squared Error  (MSE)
    Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
    """

    @staticmethod
    def build(patch_shape: tuple = (32, 32, 6),
              meta_dim:   int   = 2,
              l2_reg:     float = 1e-4,
              dropout:    float = 0.3) -> Model:
        """
        Build the dual-branch fusion model.

        Args:
            patch_shape : (H, W, C) shape of image patches
            meta_dim    : number of metadata features (default 2: lat, lon)
            l2_reg      : L2 regularization weight
            dropout     : dropout rate in fusion head

        Returns:
            Uncompiled Keras functional Model
        """
        reg = l2(l2_reg)

        # ── Branch 1: CNN ──────────────────────────────────────────────────
        patch_input = Input(shape=patch_shape, name='patch_input')

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(patch_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        cnn_out = Activation('relu', name='cnn_embedding')(x)

        # ── Branch 2: MLP (GPS / spatial) ────────────────────────────────
        meta_input = Input(shape=(meta_dim,), name='meta_input')

        m = Dense(32, kernel_regularizer=reg)(meta_input)
        m = BatchNormalization()(m)
        m = Activation('relu')(m)

        m = Dense(64, kernel_regularizer=reg)(m)
        m = BatchNormalization()(m)
        m = Activation('relu')(m)

        m = Dense(32, kernel_regularizer=reg)(m)
        m = BatchNormalization()(m)
        mlp_out = Activation('relu', name='mlp_embedding')(m)

        # ── Fusion Head ───────────────────────────────────────────────────
        fused = Concatenate(name='fusion')([cnn_out, mlp_out])

        f = Dense(128, kernel_regularizer=reg)(fused)
        f = BatchNormalization()(f)
        f = Activation('relu')(f)
        f = Dropout(dropout)(f)

        f = Dense(64, kernel_regularizer=reg)(f)
        f = BatchNormalization()(f)
        f = Activation('relu')(f)

        # Single linear output = predicted NTU
        ntu_output = Dense(1, activation='linear', name='ntu_output')(f)

        model = Model(
            inputs=[patch_input, meta_input],
            outputs=ntu_output,
            name='DualBranchFusionModel'
        )
        return model

    @staticmethod
    def compile(model: Model,
                learning_rate: float = 5e-4) -> Model:
        """
        Compile with MSE loss and MAE / RMSE metrics.

        Args:
            model         : uncompiled Keras Model
            learning_rate : Adam learning rate

        Returns:
            Compiled model
        """
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=[
                'mae',
                RootMeanSquaredError(name='rmse'),
            ]
        )
        return model

    @staticmethod
    def print_info(model: Model):
        """Print architecture summary and parameter count."""
        print('\n' + '=' * 60)
        print('DUAL-BRANCH FUSION MODEL  (Regression)')
        print('=' * 60)
        model.summary()
        total = model.count_params()
        trainable = sum(tf.keras.backend.count_params(w)
                        for w in model.trainable_weights)
        print(f'\n  Total parameters     : {total:,}')
        print(f'  Trainable parameters : {trainable:,}')
        print('=' * 60 + '\n')


if __name__ == '__main__':
    # ── Classification smoke test ──────────────────────────────────────────
    cls_model = PollutionCNN.simple_cnn(input_shape=(32, 32, 6))
    cls_model = PollutionCNN.compile_model(cls_model)
    PollutionCNN.print_model_info(cls_model)

    # ── Regression smoke test ─────────────────────────────────────────────
    reg_model = DualBranchFusionModel.build(patch_shape=(32, 32, 6), meta_dim=2)
    reg_model = DualBranchFusionModel.compile(reg_model, learning_rate=5e-4)
    DualBranchFusionModel.print_info(reg_model)