"""
Training Pipeline with MLFlow Integration
Trains CNN model and tracks experiments with MLFlow.
"""

import pandas as pd
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import classification_report, confusion_matrix

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from model import CNNModel, create_sequences, get_callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Train CNN model with MLFlow tracking."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_builder = None
        self.model = None
        self.history = None
        
    def load_data(self, use_selected_features: bool = True) -> Tuple:
        """
        Load processed and labeled data.
        
        Args:
            use_selected_features: If True, use feature-selected datasets
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, X_val, y_val, feature_names)
        """
        logger.info("Loading labeled data...")
        
        # Choose dataset path based on feature selection
        suffix = "_selected" if use_selected_features else ""
        logger.info(f"Using {'feature-selected' if use_selected_features else 'full'} datasets")
        
        train_df = pd.read_csv(f"data/processed/train_labeled{suffix}.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv(f"data/processed/test_labeled{suffix}.csv", index_col=0, parse_dates=True)
        val_df = pd.read_csv(f"data/processed/val_labeled{suffix}.csv", index_col=0, parse_dates=True)
        
        # Load feature names
        with open("data/processed/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f]
        
        # Extract features and labels
        X_train = train_df[feature_names].values
        y_train = train_df['label'].values
        
        X_test = test_df[feature_names].values
        y_test = test_df['label'].values
        
        X_val = val_df[feature_names].values
        y_val = val_df['label'].values
        
        logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
        
        return X_train, y_train, X_test, y_test, X_val, y_val, feature_names
    
    def prepare_sequences(self, X_train, y_train, X_test, y_test, X_val, y_val, lookback):
        """
        Create sequences for CNN input.
        
        Returns:
            Tuple of sequence data
        """
        logger.info(f"Creating sequences with lookback={lookback}...")
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)
        
        logger.info(f"Sequences created - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}, Val: {X_val_seq.shape}")
        
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq
    
    def build_model(self, input_shape: Tuple[int, int]) -> CNNModel:
        """Build CNN model from config."""
        model_config = self.config['model']
        
        self.model_builder = CNNModel(
            input_shape=input_shape,
            num_classes=model_config['output_classes'],
            filters=model_config['filters'],
            kernel_sizes=model_config['kernel_sizes'],
            pool_size=model_config['pool_size'],
            dropout_rate=model_config['dropout_rate'],
            dense_units=model_config['dense_units']
        )
        
        self.model = self.model_builder.build()
        
        return self.model_builder
    
    def train(self, X_train, y_train, X_val, y_val, class_weights: Dict = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Class weights for imbalanced data
        """
        training_config = self.config['training']
        
        # Compile model
        self.model_builder.compile(
            learning_rate=training_config['learning_rate'],
            class_weights=class_weights
        )
        
        # Get callbacks
        callbacks = get_callbacks(
            model_path='models/best_model.h5',
            patience=training_config['early_stopping_patience']
        )
        
        # Train model
        logger.info("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate loss and accuracy from model
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        # Classification report (for precision, recall, f1 per class)
        report = classification_report(
            y_test, y_pred,
            target_names=['SELL', 'HOLD', 'BUY'],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Extract weighted average precision and recall from classification report
        test_precision = report['weighted avg']['precision']
        test_recall = report['weighted avg']['recall']

        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")

        return metrics
    
    def run_experiment(self, experiment_name: str = None):
        """
        Run full training experiment with MLFlow tracking.
        
        Args:
            experiment_name: Name for MLFlow experiment
        """
        # Set MLFlow experiment
        if MLFLOW_AVAILABLE:
            if experiment_name is None:
                experiment_name = self.config['mlflow']['experiment_name']
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
        
        try:
            # Load data
            X_train, y_train, X_test, y_test, X_val, y_val, feature_names = self.load_data()
            
            # Load class weights
            with open("data/processed/label_stats.json", 'r') as f:
                label_stats = json.load(f)
            class_weights = {int(k): float(v) for k, v in label_stats['class_weights'].items()}
            
            # Create sequences
            lookback = self.config['features']['lookback_window']
            X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq = \
                self.prepare_sequences(X_train, y_train, X_test, y_test, X_val, y_val, lookback)
            
            # Log parameters to MLFlow if available
            if MLFLOW_AVAILABLE:
                mlflow.log_param("asset_symbol", self.config['data']['asset_symbol'])
                mlflow.log_param("lookback_window", lookback)
                mlflow.log_param("num_features", len(feature_names))
                mlflow.log_param("train_samples", len(X_train_seq))
                mlflow.log_param("test_samples", len(X_test_seq))
                mlflow.log_param("val_samples", len(X_val_seq))
                
                # Log model hyperparameters
                for key, value in self.config['model'].items():
                    mlflow.log_param(f"model_{key}", value)
                
                # Log training hyperparameters
                for key, value in self.config['training'].items():
                    if key != 'class_weights':
                        mlflow.log_param(f"training_{key}", value)
                
                mlflow.log_param("class_weights", class_weights)
            
            # Build model
            input_shape = (lookback, len(feature_names))
            self.build_model(input_shape)
            
            # Log model architecture
            self.model_builder.summary()
            
            # Train model
            history = self.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, class_weights)
            
            # Log training metrics to MLFlow if available
            if MLFLOW_AVAILABLE:
                for epoch in range(len(history.history['loss'])):
                    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            
            # Evaluate on test set
            metrics = self.evaluate(X_test_seq, y_test_seq)
            
            # Log test metrics to MLFlow if available
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("test_loss", metrics['test_loss'])
                mlflow.log_metric("test_accuracy", metrics['test_accuracy'])
                mlflow.log_metric("test_precision", metrics['test_precision'])
                mlflow.log_metric("test_recall", metrics['test_recall'])
                
                # Log per-class metrics
                for class_name in ['SELL', 'HOLD', 'BUY']:
                    mlflow.log_metric(f"{class_name}_precision", metrics['classification_report'][class_name]['precision'])
                    mlflow.log_metric(f"{class_name}_recall", metrics['classification_report'][class_name]['recall'])
                    mlflow.log_metric(f"{class_name}_f1", metrics['classification_report'][class_name]['f1-score'])
                
                # Save and log model
                if self.config['mlflow']['log_models']:
                    mlflow.keras.log_model(self.model, "model")
                    logger.info("Model logged to MLFlow")

                # Save metrics to file first, then log to MLFlow
                with open("models/test_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

                mlflow.log_artifact("models/test_metrics.json")

                logger.info(f"Experiment completed! Run ID: {mlflow.active_run().info.run_id}")

            # Metrics already saved above within MLFlow block
            else:
                # Save metrics to file when MLFlow not available
                with open("models/test_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

            logger.info("Training completed successfully!")
            
        finally:
            if MLFLOW_AVAILABLE:
                mlflow.end_run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model with MLFlow tracking')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='MLFlow experiment name')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and run experiment
    trainer = Trainer(config)
    trainer.run_experiment(experiment_name=args.experiment)
