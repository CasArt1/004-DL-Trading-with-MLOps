"""Configuration file for the trading system."""

import os

# Data settings
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Feature engineering settings
TIMEFRAMES = ['1h', '4h', '1d']
SEQUENCE_LENGTH = 20
PREDICTION_HORIZON = 1

# Model settings
CNN_CONFIG = {
    "num_filters": 64,
    "kernel_size": 3,
    "num_conv_layers": 3,
    "dropout_rate": 0.3,
    "dense_units": 128,
    "learning_rate": 0.001
}

# Training settings
TRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.1
}

# MLFlow settings
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "cnn_trading_model"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "best_cnn_model_scaler.pkl")

# Drift detection settings
DRIFT_CONFIG = {
    "drift_threshold": 0.1,
    "alpha": 0.05,
    "method": "ks"  # kolmogorov-smirnov
}

# Backtesting settings
BACKTEST_CONFIG = {
    "initial_capital": 100000.0,
    "commission_rate": 0.001,  # 0.1%
    "slippage_rate": 0.0005,   # 0.05%
    "position_size": 1.0       # 100% of capital
}

# Data fetching settings
TICKER = "BTC-USD"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INTERVAL = "1h"
