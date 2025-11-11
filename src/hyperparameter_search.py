"""
Hyperparameter Search with MLFlow Tracking

Grid search or random search over hyperparameter space with MLFlow experiment tracking.
"""

import yaml
import numpy as np
import pandas as pd
from itertools import product
import mlflow
from training import Trainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grid_search(base_config_path='config/config.yaml', experiment_name='hyperparameter-search'):
    """
    Perform grid search over hyperparameters.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: MLFlow experiment name
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define hyperparameter grid
    param_grid = {
        'filters': [
            [32, 64, 128],      # Current
            [64, 128, 256],     # Larger
            [16, 32, 64],       # Smaller
            [32, 64, 128, 256], # Deeper
        ],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [24, 36, 48],  # Must divide 2148 evenly
        'lookback_window': [30, 60, 90, 120]
    }
    
    # Set MLFlow experiment
    mlflow.set_experiment(experiment_name)
    
    # Generate all combinations (warning: can be large!)
    # For demo, let's do random search instead
    n_trials = 20
    logger.info(f"Running {n_trials} random search trials...")
    
    results = []
    
    for trial in range(n_trials):
        # Random sample from grid
        trial_params = {
            'filters': np.random.choice([str(f) for f in param_grid['filters']]),
            'dropout_rate': np.random.choice(param_grid['dropout_rate']),
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'batch_size': np.random.choice(param_grid['batch_size']),
            'lookback_window': np.random.choice(param_grid['lookback_window'])
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial + 1}/{n_trials}")
        logger.info(f"Parameters: {trial_params}")
        logger.info(f"{'='*60}\n")
        
        # Update config
        config = base_config.copy()
        config['model']['filters'] = eval(trial_params['filters'])
        config['model']['dropout_rate'] = trial_params['dropout_rate']
        config['training']['learning_rate'] = trial_params['learning_rate']
        config['training']['batch_size'] = trial_params['batch_size']
        config['features']['lookback_window'] = trial_params['lookback_window']
        
        try:
            # Create trainer and run
            trainer = Trainer(config)
            
            with mlflow.start_run(run_name=f"trial_{trial+1}"):
                # Log hyperparameters
                for key, value in trial_params.items():
                    mlflow.log_param(key, value)
                
                # Load and prepare data
                X_train, y_train, X_test, y_test, X_val, y_val, feature_names = trainer.load_data()
                
                # Load class weights
                import json
                with open("data/processed/label_stats.json", 'r') as f:
                    label_stats = json.load(f)
                class_weights = {int(k): float(v) for k, v in label_stats['class_weights'].items()}
                
                # Create sequences
                lookback = config['features']['lookback_window']
                X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq = \
                    trainer.prepare_sequences(X_train, y_train, X_test, y_test, X_val, y_val, lookback)
                
                # Build model
                input_shape = (lookback, len(feature_names))
                trainer.build_model(input_shape)
                
                # Train
                history = trainer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, class_weights)
                
                # Evaluate
                metrics = trainer.evaluate(X_test_seq, y_test_seq)
                
                # Log metrics
                mlflow.log_metric("final_test_accuracy", metrics['test_accuracy'])
                mlflow.log_metric("final_test_loss", metrics['test_loss'])
                
                # Store results
                results.append({
                    **trial_params,
                    'test_accuracy': metrics['test_accuracy'],
                    'test_loss': metrics['test_loss'],
                    'val_accuracy': history.history['val_accuracy'][-1]
                })
                
                logger.info(f"Trial {trial + 1} completed: Test Acc = {metrics['test_accuracy']:.4f}")
        
        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/hyperparameter_search_results.csv", index=False)
    
    # Print best results
    best_trial = results_df.loc[results_df['test_accuracy'].idxmax()]
    logger.info(f"\n{'='*60}")
    logger.info("BEST TRIAL:")
    logger.info(f"{'='*60}")
    for key, value in best_trial.items():
        logger.info(f"{key}: {value}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to base config file')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of random trials')
    parser.add_argument('--experiment', type=str, default='hyperparameter-search',
                       help='MLFlow experiment name')
    args = parser.parse_args()
    
    results = grid_search(args.config, args.experiment)
    print("\nSearch completed! Results saved to models/hyperparameter_search_results.csv")
    print("\nView results in MLFlow UI:")
    print("  mlflow ui --port 5000")
