# AI Coding Agent Instructions: DL Trading with MLOps

## Project Overview
Systematic trading strategy using CNN models trained on 20+ engineered time series features from 15 years of daily price data. The project delivers experiment tracking with MLFlow, a Streamlit data drift dashboard, and a comprehensive backtesting framework with transaction costs.

## Project Requirements Summary

### Data Specifications
- **Historical Data**: 15 years of daily price data (chosen asset)
- **Data Splits**: 60% Training, 20% Testing, 20% Validation (chronological order, no look-ahead bias)
- **Features**: Minimum 20 features (momentum, volatility, volume indicators) with normalization
- **Target Labels**: Buy/sell/hold signals with class weighting for imbalanced data

### Model Architecture
- **CNN Model**: Capture local temporal patterns in feature sequences
- **Class Balancing**: Apply class weights during training or adjust thresholds (market bias toward "hold")

### MLOps Components
- **MLFlow Tracking**: Compare experiments, track hyperparameters/metrics, register best model
- **Drift Dashboard**: Streamlit app with timeline plots, KS-test statistics, drift detection, top 5 drifted features
- **Backtesting**: Signal generation with SL/TP/position sizing, commission (0.125%), borrow rate (0.25% annualized)

## Architecture & Components

### Core Structure
- **Data Pipeline**: Ingestion, preprocessing, chronological splitting
- **Feature Engineering**: 20+ indicators with normalization pipeline
- **CNN Model**: Deep learning for temporal pattern recognition
- **MLFlow Integration**: Experiment tracking, model versioning, model registry
- **Drift Monitoring**: Streamlit dashboard for distribution analysis and KS-tests
- **Backtesting Engine**: Strategy simulation with realistic transaction costs

## Development Environment

### Python Setup
- Virtual environment: `.venv/` (recommended for Windows PowerShell)
- MLFlow artifacts stored in `mlruns/` directory
- Streamlit for interactive dashboards

### Key Dependencies
- **Deep Learning**: TensorFlow/Keras or PyTorch for CNN implementation
- **Data & Features**: pandas, numpy, scikit-learn, ta-lib or ta (technical indicators)
- **MLOps**: mlflow for experiment tracking and model registry
- **Monitoring**: streamlit, scipy (for KS-test), matplotlib/plotly
- **Data Source**: yfinance, alpaca-trade-api, or similar for historical data
- **Backtesting**: Custom framework (strategy logic, transaction costs)

## Coding Conventions

### Data Flow Pattern
1. Raw market data → Feature engineering pipeline
2. Engineered features → Model training (tracked in MLFlow)
3. Trained models → Backtesting validation
4. Validated models → Production deployment with drift monitoring
5. Production signals → Trading API execution

### Time Series Best Practices
- Avoid look-ahead bias in feature engineering
- Use proper train/validation/test splits (temporal, not random)
- Account for market hours and trading calendars
- Handle missing data and market gaps appropriately

### MLFlow Workflow
- Log all experiments with hyperparameters
- Track metrics: train/val loss, trading metrics (Sharpe, returns, drawdown)
- Register models with version tags and stages (Staging/Production)
- Store feature engineering artifacts with models

## Testing & Validation

### Backtesting Requirements
- Walk-forward validation for realistic performance
- Transaction costs and slippage modeling
- Position sizing and risk management
- Performance metrics: returns, Sharpe ratio, max drawdown, win rate

### Monitoring in Production
- Track data drift for input features
- Monitor model prediction distributions
- Alert on anomalous market conditions
- Log all trades and signals for audit

## Common Commands

```powershell
# Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# MLFlow UI (if tracking server needed)
mlflow ui --port 5000

# Training (convention)
python train.py --config config/experiment.yaml

# Backtesting
python backtest.py --model-uri runs:/<run_id>/model --start-date YYYY-MM-DD

# Monitoring
python monitor_drift.py --production-data data/live_features.csv
```

## Integration Points
- **Trading API**: Authentication, order placement, position management, market data feeds
- **Data Sources**: Market data providers (expect API credentials in `.env`)
- **MLFlow Server**: May use local tracking or remote MLFlow server
- **Monitoring Dashboards**: Potential integration with Grafana/Prometheus or MLFlow metrics

## Risk & Compliance Notes
- Never commit API keys or credentials (use `.env`, already in `.gitignore`)
- Log all trading decisions for regulatory compliance
- Implement position limits and risk controls
- Test thoroughly in paper trading before live deployment
