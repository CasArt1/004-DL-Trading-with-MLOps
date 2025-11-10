# Setup Guide

This guide will help you set up the DL Trading with MLOps project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CasArt1/004-DL-Trading-with-MLOps.git
cd 004-DL-Trading-with-MLOps
```

### 2. Create a Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: If you encounter issues with `ta-lib`, you may need to install system dependencies first:

**On Ubuntu/Debian:**
```bash
sudo apt-get install build-essential libta-lib-dev
pip install ta-lib
```

**On macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**On Windows:**
Download and install TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

Alternatively, you can use `pandas-ta` which is a pure Python alternative (already in requirements.txt).

### 4. Verify Installation

```bash
python scripts/verify_installation.py
```

If all checks pass, you're ready to proceed!

## Quick Start

### Step 1: Fetch Data

Fetch historical market data from Yahoo Finance:

```bash
python scripts/01_fetch_data.py
```

This will:
- Download BTC-USD data (configurable in `configs/config.py`)
- Save raw data to `data/raw_data.csv`

### Step 2: Engineer Features

Create technical indicators and multi-timeframe features:

```bash
python scripts/02_engineer_features.py
```

This will:
- Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Create multi-timeframe features (1h, 4h, 1d)
- Save engineered features to `data/engineered_features.csv`

### Step 3: Train Model

Train the CNN model with MLFlow tracking:

```bash
python scripts/03_train_model.py
```

This will:
- Prepare training sequences
- Train CNN model with early stopping
- Log experiments to MLFlow
- Save best model to `models/best_cnn_model.keras`

View MLFlow experiments:
```bash
mlflow ui
```
Then navigate to http://localhost:5000

### Step 4: Monitor Data Drift

Check for data drift in production-like conditions:

```bash
python scripts/05_monitor_drift.py
```

This will:
- Detect drift using statistical tests
- Generate Evidently HTML report
- Save drift history to JSON

View the drift report: `reports/drift_report.html`

### Step 5: Run Backtesting

Test the strategy with realistic trading costs:

```bash
python scripts/06_run_backtest.py
```

This will:
- Generate predictions using the trained model
- Simulate trading with commissions and slippage
- Calculate performance metrics
- Save results to `reports/`

### Step 6: Start API Server

Run the prediction API:

```bash
python scripts/04_run_api.py
```

API will be available at:
- Documentation: http://localhost:8000/docs
- Endpoint: http://localhost:8000/predict

Test the API:
```bash
python scripts/test_api.py
```

## Configuration

Edit `configs/config.py` to customize:

- **Data Settings**: ticker symbol, date range, interval
- **Model Hyperparameters**: layers, filters, dropout rate
- **Training Settings**: epochs, batch size, learning rate
- **API Settings**: host, port
- **Drift Detection**: thresholds, statistical tests
- **Backtesting**: capital, commission, slippage

## Project Structure

```
004-DL-Trading-with-MLOps/
├── src/                      # Source code
│   ├── feature_engineering/  # Technical indicators and features
│   ├── models/              # CNN model and training
│   ├── api/                 # FastAPI application
│   ├── monitoring/          # Data drift detection
│   └── backtesting/         # Trading simulation
├── scripts/                 # Executable scripts
├── configs/                 # Configuration files
├── notebooks/               # Jupyter notebooks
├── data/                    # Data files (created at runtime)
├── models/                  # Saved models (created at runtime)
├── reports/                 # Reports and visualizations
└── mlruns/                  # MLFlow experiments
```

## Troubleshooting

### Issue: ImportError for ta-lib

**Solution**: Use pandas-ta instead (already in requirements.txt) or install ta-lib system dependencies.

### Issue: TensorFlow installation fails

**Solution**: Check your Python version (3.8-3.11 recommended). For Apple Silicon Macs, use:
```bash
pip install tensorflow-macos tensorflow-metal
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `configs/config.py`:
```python
TRAIN_CONFIG = {
    "batch_size": 16,  # Reduced from 32
    ...
}
```

### Issue: Model not found when running API

**Solution**: Train the model first using `scripts/03_train_model.py`

## Support

For issues and questions, please open an issue on GitHub:
https://github.com/CasArt1/004-DL-Trading-with-MLOps/issues

## License

MIT License - See LICENSE file for details
