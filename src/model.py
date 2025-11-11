"""
CNN Model Architecture
Implements a 1D CNN for capturing local temporal patterns in feature sequences.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel:
    """1D Convolutional Neural Network for time series classification."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 num_classes: int = 3,
                 filters: list = [32, 64, 128],
                 kernel_sizes: list = [3, 3, 3],
                 pool_size: int = 2,
                 dropout_rate: float = 0.3,
                 dense_units: list = [128, 64]):
        """
        Initialize CNN model architecture.
        
        Args:
            input_shape: (timesteps, features) shape
            num_classes: Number of output classes (3 for buy/sell/hold)
            filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            pool_size: Max pooling size
            dropout_rate: Dropout rate for regularization
            dense_units: List of units for dense layers
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model = None
        
    def build(self) -> keras.Model:
        """
        Build the CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building CNN model...")
        
        model = models.Sequential(name='CNN_Trading_Model')
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional blocks
        for i, (filters, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.MaxPooling1D(
                pool_size=self.pool_size,
                name=f'maxpool_{i+1}'
            ))
            model.add(layers.Dropout(
                self.dropout_rate,
                name=f'dropout_conv_{i+1}'
            ))
        
        # Flatten for dense layers
        model.add(layers.Flatten(name='flatten'))
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            model.add(layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            model.add(layers.Dropout(
                self.dropout_rate,
                name=f'dropout_dense_{i+1}'
            ))
        
        # Output layer
        model.add(layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        ))
        
        self.model = model
        
        logger.info(f"Model built successfully with {model.count_params():,} parameters")
        
        return model
    
    def compile(self, 
                learning_rate: float = 0.001,
                class_weights: Dict[int, float] = None):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            class_weights: Optional class weights for imbalanced data
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Use categorical crossentropy with class weights handled in fit()
        # Note: Only using accuracy during training to avoid shape mismatch issues
        # with Precision/Recall metrics on uneven batch sizes in TF 2.20+
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',  # Use sparse since labels are integers
            metrics=['accuracy']
        )

        logger.info(f"Model compiled with learning_rate={learning_rate}")

    def get_model(self) -> keras.Model:
        """Return the Keras model."""
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        self.model.summary()


def create_sequences(X: np.ndarray, 
                     y: np.ndarray, 
                     lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for CNN input.
    
    Args:
        X: Feature array (samples, features)
        y: Label array (samples,)
        lookback: Number of timesteps to look back
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_seq = []
    y_seq = []
    
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def get_callbacks(model_path: str = 'models/best_model.h5',
                  patience: int = 10) -> list:
    """
    Get training callbacks.
    
    Args:
        model_path: Path to save best model
        patience: Patience for early stopping
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    model_builder = CNNModel(
        input_shape=(60, 38),  # 60 timesteps, 38 features
        num_classes=3,
        filters=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_size=2,
        dropout_rate=0.3,
        dense_units=[128, 64]
    )
    
    model = model_builder.build()
    model_builder.compile(learning_rate=0.001)
    model_builder.summary()
