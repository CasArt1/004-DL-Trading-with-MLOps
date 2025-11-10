"""Script to train CNN model with MLFlow tracking."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import CNNModel, ModelTrainer
from configs.config import (
    DATA_DIR, MODELS_DIR, SEQUENCE_LENGTH, CNN_CONFIG,
    TRAIN_CONFIG, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
import mlflow


def load_features(filepath):
    """Load engineered features."""
    features = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return features


def main():
    """Main function."""
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load features
    features_path = os.path.join(DATA_DIR, "engineered_features.csv")
    print(f"Loading features from: {features_path}")
    features = load_features(features_path)
    
    print(f"Loaded {len(features)} rows with {len(features.columns)} features")
    
    # Initialize model
    print("\nInitializing CNN model...")
    
    # We'll determine input shape after preparing sequences
    # For now, create a placeholder
    num_features = len(features.select_dtypes(include=[np.number]).columns)
    input_shape = (SEQUENCE_LENGTH, num_features)
    
    cnn_model = CNNModel(
        input_shape=input_shape,
        num_filters=CNN_CONFIG["num_filters"],
        kernel_size=CNN_CONFIG["kernel_size"],
        num_conv_layers=CNN_CONFIG["num_conv_layers"],
        dropout_rate=CNN_CONFIG["dropout_rate"],
        dense_units=CNN_CONFIG["dense_units"]
    )
    
    cnn_model.compile_model(learning_rate=CNN_CONFIG["learning_rate"])
    
    # Initialize trainer
    trainer = ModelTrainer(cnn_model, experiment_name=EXPERIMENT_NAME)
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X, y = trainer.prepare_sequences(features, sequence_length=SEQUENCE_LENGTH)
    
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    test_size = TRAIN_CONFIG["test_split"]
    val_size = TRAIN_CONFIG["validation_split"]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size), shuffle=False
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining model...")
    model_save_path = os.path.join(MODELS_DIR, "best_cnn_model.keras")
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        model_save_path=model_save_path
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: {model_save_path}")
    print(f"MLFlow tracking URI: {MLFLOW_TRACKING_URI}")
    print("="*50)


if __name__ == "__main__":
    main()
