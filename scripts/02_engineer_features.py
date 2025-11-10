"""Script to engineer time series features from raw data."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.feature_engineering import TechnicalIndicators, MultiTimeframeFeatures
from configs.config import DATA_DIR, TIMEFRAMES


def load_data(filepath):
    """Load raw data."""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data


def engineer_features(data, timeframes=TIMEFRAMES):
    """
    Engineer features from raw data.
    
    Args:
        data: Raw OHLCV data
        timeframes: List of timeframes to include
    
    Returns:
        DataFrame with engineered features
    """
    print("Engineering features...")
    
    # Add base technical indicators
    print("Adding technical indicators...")
    indicators = TechnicalIndicators(data)
    data_with_indicators = indicators.add_all_indicators()
    
    # Add multi-timeframe features
    print(f"Adding multi-timeframe features for: {timeframes}")
    mtf = MultiTimeframeFeatures(data_with_indicators)
    features = mtf.create_multi_timeframe_features(timeframes)
    
    # Add lagged features for key indicators
    print("Adding lagged features...")
    key_columns = ['Close', 'Volume', 'RSI_14', 'MACD']
    features = mtf.add_lagged_features(key_columns, lags=[1, 2, 3, 5])
    
    # Add rolling statistics
    print("Adding rolling statistics...")
    features = mtf.add_rolling_statistics(['Close', 'Volume'], windows=[5, 10])
    
    print(f"Feature engineering complete. Total features: {len(features.columns)}")
    
    return features


def main():
    """Main function."""
    # Load raw data
    raw_data_path = os.path.join(DATA_DIR, "raw_data.csv")
    print(f"Loading data from: {raw_data_path}")
    data = load_data(raw_data_path)
    
    print(f"Loaded {len(data)} rows")
    
    # Engineer features
    features = engineer_features(data)
    
    # Save engineered features
    features_path = os.path.join(DATA_DIR, "engineered_features.csv")
    features.to_csv(features_path)
    print(f"\nEngineered features saved to: {features_path}")
    
    # Display summary
    print("\nFeature Summary:")
    print(f"Total features: {len(features.columns)}")
    print(f"Feature names: {list(features.columns[:10])}... (showing first 10)")
    print(f"\nMissing values per column:")
    print(features.isnull().sum().head(20))


if __name__ == "__main__":
    main()
