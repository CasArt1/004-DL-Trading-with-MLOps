# DL Trading with MLOps

A systematic trading strategy using CNN models trained on engineered time series features. The project implements comprehensive MLOps practices including experiment tracking with MLFlow, data drift monitoring with Streamlit, and realistic backtesting with transaction costs.

## Project Overview

This project develops a deep learning-based trading strategy with the following components:

- **Data Pipeline**: 15 years of daily price data with proper chronological splits
- **Feature Engineering**: 20+ technical indicators (momentum, volatility, volume)
- **CNN Model**: Captures local temporal patterns in feature sequences
- **MLFlow Integration**: Experiment tracking and model registry
- **Drift Monitoring**: Streamlit dashboard with KS-test statistics
- **Backtesting**: Realistic strategy validation with commission (0.125%) and borrow rates (0.25%)

## Project Structure

```
004-DL-Trading-with-MLOps/
├── config/
│   └── config.yaml           # Project configuration
├── data/
│   ├── raw/                  # Raw market data splits
│   └── processed/            # Processed features and labels
├── models/                   # Saved model artifacts
├── mlruns/                   # MLFlow tracking data
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── data_collection.py    # Data fetching and splitting
│   ├── feature_engineering.py # Feature generation
│   ├── model.py              # CNN model definition
│   ├── training.py           # Training pipeline
│   ├── backtesting.py        # Backtesting framework
│   └── drift_dashboard.py    # Streamlit dashboard
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script
└── README.md                 # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Windows PowerShell (for Windows users)
- 15+ GB disk space for data and models

### Quick Start

1. **Clone the repository**
   ```powershell
   git clone https://github.com/CasArt1/004-DL-Trading-with-MLOps.git
   cd 004-DL-Trading-with-MLOps
   ```

2. **Run setup script**
   ```powershell
   python setup.py
   ```

3. **Activate virtual environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Configure project**
   - Update `config/config.yaml` with your chosen asset symbol
   - Adjust date ranges, model parameters, and backtesting settings

### Manual Setup (Alternative)

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

## Usage

### 1. Data Collection

Fetch 15 years of historical data and create chronological splits:

```powershell
python src/data_collection.py
```

This creates:
- `data/raw/train_raw.csv` (60% of data)
- `data/raw/test_raw.csv` (20% of data)
- `data/raw/val_raw.csv` (20% of data)

### 2. Feature Engineering

Generate 20+ technical indicators and normalize features:

```powershell
python src/feature_engineering.py
```

Output: `data/processed/` with engineered features and labels

### 3. Model Training

Train CNN model with MLFlow tracking:

```powershell
python src/training.py --config config/config.yaml
```

View experiments:
```powershell
mlflow ui
# Open browser to http://localhost:5000
```

### 4. Data Drift Monitoring

Launch Streamlit dashboard to analyze feature drift:

```powershell
streamlit run src/drift_dashboard.py
```

Features:
- Timeline view of feature distributions
- KS-test statistics table
- Drift detection highlighting
- Top 5 most-drifted features with interpretations

### 5. Backtesting

Run strategy backtest with realistic transaction costs:

```powershell
python src/backtesting.py --model-path models/best_model.h5
```

Metrics calculated:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Transaction costs impact

## Technical Requirements

### Data Specifications
- **Historical Data**: 15 years of daily price data
- **Splits**: 60% Train / 20% Test / 20% Validation (chronological)
- **Features**: Minimum 20 features with normalization
- **Labels**: Buy/sell/hold signals with class weighting

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Purpose**: Capture local temporal patterns in feature sequences
- **Class Balancing**: Handles imbalanced data (market bias toward "hold")

### Backtesting Parameters
- **Commission**: 0.125% per trade
- **Borrow Rate**: 0.25% annualized (for short positions)
- **Position Sizing**: Configurable in `config/config.yaml`
- **Stop Loss / Take Profit**: Configurable thresholds

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  asset_symbol: "AAPL"  # Your chosen asset
  start_date: "2008-01-01"
  end_date: "2023-01-01"

model:
  filters: [32, 64, 128]
  kernel_sizes: [3, 3, 3]
  dropout_rate: 0.3

backtesting:
  initial_capital: 100000
  stop_loss: 0.02  # 2%
  take_profit: 0.05  # 5%
```

## Development Workflow

1. **Exploratory Analysis**: Use `notebooks/` for data exploration
2. **Feature Engineering**: Iterate on feature creation in `src/feature_engineering.py`
3. **Model Experiments**: Track experiments with MLFlow
4. **Drift Analysis**: Monitor feature distributions across splits
5. **Backtesting**: Validate strategy performance
6. **Iteration**: Refine features, model, or strategy based on results

## Deliverables

- ✅ Clean, modular, well-documented code
- ✅ MLFlow experiment tracking
- ✅ Streamlit drift monitoring dashboard
- ✅ Comprehensive backtesting framework
- ✅ README with setup and usage instructions
- ✅ Proper Git version control

## Notes

- **Look-ahead Bias**: All splits are chronological to prevent data leakage
- **Class Imbalance**: Class weights applied during training
- **Transaction Costs**: Realistic commission and borrow rates included
- **API Keys**: Store in `.env` file (not committed to Git)

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue.
