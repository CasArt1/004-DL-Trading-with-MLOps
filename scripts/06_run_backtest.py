"""Script to run backtesting with ML predictions."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from src.backtesting import BacktestEngine, TradingStrategy
from configs.config import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, BACKTEST_CONFIG,
    SEQUENCE_LENGTH
)


def load_data(filepath):
    """Load data."""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data


def generate_predictions(features, model_path, scaler_path):
    """
    Generate predictions using trained model.
    
    Args:
        features: Engineered features DataFrame
        model_path: Path to saved model
        scaler_path: Path to saved scaler
    
    Returns:
        Series with predictions
    """
    print("Loading model and scaler...")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    print("Generating predictions...")
    
    # Prepare features
    numeric_features = features.select_dtypes(include=[np.number]).values
    features_clean = features.dropna()
    numeric_features = features_clean.select_dtypes(include=[np.number]).values
    
    # Scale features
    scaled_features = scaler.transform(numeric_features)
    
    # Generate predictions
    predictions = []
    indices = []
    
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        sequence = scaled_features[i - SEQUENCE_LENGTH:i]
        sequence = np.expand_dims(sequence, axis=0)
        
        pred_probs = model.predict(sequence, verbose=0)[0]
        pred_class = int(np.argmax(pred_probs))
        
        predictions.append(pred_class)
        indices.append(features_clean.index[i])
    
    predictions_series = pd.Series(predictions, index=indices)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction distribution: {np.bincount(predictions)}")
    
    return predictions_series


def main():
    """Main function."""
    # Load data
    raw_data_path = os.path.join(DATA_DIR, "raw_data.csv")
    features_path = os.path.join(DATA_DIR, "engineered_features.csv")
    
    print(f"Loading data from: {raw_data_path}")
    data = load_data(raw_data_path)
    
    print(f"Loading features from: {features_path}")
    features = load_data(features_path)
    
    # Generate predictions
    model_path = os.path.join(MODELS_DIR, "best_cnn_model.keras")
    scaler_path = os.path.join(MODELS_DIR, "best_cnn_model_scaler.pkl")
    
    predictions = generate_predictions(features, model_path, scaler_path)
    
    # Align data with predictions
    data_aligned = data.loc[predictions.index]
    
    # Initialize trading strategy
    print("\nInitializing trading strategy...")
    strategy = TradingStrategy(
        initial_capital=BACKTEST_CONFIG["initial_capital"],
        commission_rate=BACKTEST_CONFIG["commission_rate"],
        slippage_rate=BACKTEST_CONFIG["slippage_rate"],
        position_size=BACKTEST_CONFIG["position_size"]
    )
    
    # Initialize backtest engine
    print("Initializing backtest engine...")
    backtest = BacktestEngine(
        data=data_aligned,
        predictions=predictions,
        strategy=strategy
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtest.run_backtest()
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital:        ${results['initial_capital']:,.2f}")
    print(f"Final Portfolio Value:  ${results['final_value']:,.2f}")
    print(f"Total Return:           {results['total_return_pct']:.2f}%")
    print(f"\nTotal Trades:           {results['num_trades']}")
    print(f"Win Rate:               {results['win_rate']:.2f}%")
    print(f"Profit Factor:          {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {results['max_drawdown']:.2f}%")
    print(f"Total Commissions:      ${results['total_commissions']:,.2f}")
    print("="*60)
    
    # Generate detailed report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "backtest_report.txt")
    backtest.generate_report(report_path)
    
    # Plot results
    plot_path = os.path.join(REPORTS_DIR, "backtest_results.png")
    print(f"\nGenerating plots...")
    backtest.plot_results(plot_path)
    
    # Save trades
    trades_path = os.path.join(REPORTS_DIR, "backtest_trades.csv")
    trades_df = backtest.get_trades_df()
    trades_df.to_csv(trades_path, index=False)
    print(f"Trades saved to: {trades_path}")
    
    # Save portfolio history
    portfolio_path = os.path.join(REPORTS_DIR, "portfolio_history.csv")
    portfolio_df = backtest.get_portfolio_df()
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"Portfolio history saved to: {portfolio_path}")
    
    print("\n" + "="*60)
    print("Backtesting complete!")
    print("="*60)


if __name__ == "__main__":
    main()
