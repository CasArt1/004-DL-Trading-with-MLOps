# 004-DL-Trading-with-MLOps

A comprehensive systematic trading system using Deep Learning models with MLOps best practices. This project implements end-to-end machine learning workflows for financial trading, including feature engineering, CNN model training with MLFlow tracking, API serving, data drift monitoring, and backtesting with realistic trading costs.

## 🎯 Features

- **Multi-Timeframe Feature Engineering**: Extract technical indicators (RSI, MACD, Bollinger Bands, etc.) across multiple timeframes (1h, 4h, 1d)
- **CNN Model Architecture**: Custom Convolutional Neural Network for time series prediction of trading signals
- **MLFlow Experiment Tracking**: Complete experiment management with parameter tracking, metrics logging, and model versioning
- **REST API**: FastAPI-based prediction endpoint for real-time trading signal generation
- **Data Drift Monitoring**: Production monitoring with Evidently for detecting distribution shifts
- **Realistic Backtesting**: Backtesting engine with commission costs, slippage, and detailed performance metrics

## 🏗️ Project Structure

```
004-DL-Trading-with-MLOps/
├── src/
│   ├── feature_engineering/
│   │   ├── indicators.py          # Technical indicator calculations
│   │   └── multi_timeframe.py     # Multi-timeframe feature engineering
│   ├── models/
│   │   └── cnn_model.py            # CNN model and trainer with MLFlow
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── monitoring/
│   │   └── drift_detector.py       # Data drift detection
│   └── backtesting/
│       └── backtest_engine.py      # Backtesting with realistic costs
├── scripts/
│   ├── 01_fetch_data.py            # Fetch market data from Yahoo Finance
│   ├── 02_engineer_features.py     # Generate engineered features
│   ├── 03_train_model.py           # Train CNN model with MLFlow
│   ├── 04_run_api.py               # Start API server
│   ├── 05_monitor_drift.py         # Monitor data drift
│   └── 06_run_backtest.py          # Run backtesting
├── configs/
│   └── config.py                   # Configuration settings
├── data/                           # Data directory (created at runtime)
├── models/                         # Saved models directory
├── reports/                        # Reports and visualizations
└── requirements.txt                # Python dependencies

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CasArt1/004-DL-Trading-with-MLOps.git
cd 004-DL-Trading-with-MLOps
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Run the complete pipeline:

```bash
# 1. Fetch market data
python scripts/01_fetch_data.py

# 2. Engineer features
python scripts/02_engineer_features.py

# 3. Train CNN model
python scripts/03_train_model.py

# 4. Monitor data drift
python scripts/05_monitor_drift.py

# 5. Run backtesting
python scripts/06_run_backtest.py

# 6. Start API server
python scripts/04_run_api.py
```

## 📊 Components

### 1. Feature Engineering

The feature engineering module calculates technical indicators at multiple timeframes:

- **Trend Indicators**: SMA, EMA
- **Momentum Indicators**: RSI, Stochastic Oscillator, ROC
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume SMA
- **Price Action**: MACD, Price Momentum

Features are extracted from multiple timeframes (1h, 4h, 1d) and combined with lagged features and rolling statistics.

### 2. CNN Model

The CNN architecture is designed for time series prediction:

- Multiple 1D convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Dense layers for final classification
- 3-class output: BUY (1), HOLD (0), SELL (2)

All experiments are tracked with MLFlow for reproducibility.

### 3. API

FastAPI-based REST API with the following endpoints:

- `GET /`: Root endpoint with status
- `GET /health`: Health check
- `POST /predict`: Generate trading predictions from market data
- `POST /load-model`: Load or reload the model
- `GET /model-info`: Get model information

Start the API server:
```bash
python scripts/04_run_api.py
```

Access API documentation at: http://localhost:8000/docs

Example prediction request:
```python
import requests

data = {
    "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
    "open": [100.0, 101.0],
    "high": [102.0, 103.0],
    "low": [99.0, 100.0],
    "close": [101.0, 102.0],
    "volume": [1000000.0, 1100000.0]
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### 4. Data Drift Monitoring

The monitoring system detects distribution shifts in production data using:

- Kolmogorov-Smirnov test
- Chi-squared test
- Evidently reports with visual dashboards
- Statistical comparisons between reference and current data

### 5. Backtesting

Realistic backtesting with:

- Commission costs (configurable, default 0.1%)
- Slippage modeling (configurable, default 0.05%)
- Position sizing
- Performance metrics: Win rate, Sharpe ratio, Maximum drawdown, Profit factor
- Visualization of results

## ⚙️ Configuration

Modify `configs/config.py` to customize:

- Data fetching parameters (ticker, date range, interval)
- Model hyperparameters (filters, layers, dropout)
- Training settings (epochs, batch size)
- API settings (host, port)
- Drift detection thresholds
- Backtesting parameters (capital, commission, slippage)

## 📈 Results

After running the pipeline, you'll find:

- **Models**: Saved in `models/` directory
- **MLFlow Experiments**: Available in `mlruns/` directory (view with `mlflow ui`)
- **Drift Reports**: HTML reports in `reports/drift_report.html`
- **Backtest Results**: 
  - Performance report: `reports/backtest_report.txt`
  - Visualization: `reports/backtest_results.png`
  - Trade history: `reports/backtest_trades.csv`
  - Portfolio history: `reports/portfolio_history.csv`

## 🔬 MLFlow Tracking

View experiment results:
```bash
mlflow ui
```

Then navigate to http://localhost:5000 to explore:
- Model parameters
- Training metrics
- Model artifacts
- Experiment comparisons

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This project is for educational purposes only. Trading financial instruments involves risk. Past performance does not guarantee future results. Always do your own research and consult with financial advisors before making investment decisions.
