"""
Data collection and preprocessing module.
Handles fetching historical data and preparing train/test/validation splits.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and preprocesses market data."""
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        """
        Initialize data collector.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data using yfinance.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=self.start_date, end=self.end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        logger.info(f"Fetched {len(self.data)} rows of data")
        return self.data
    
    def handle_missing_data(self) -> pd.DataFrame:
        """
        Handle missing data in the dataset.
        Uses forward fill for missing values.
        
        Returns:
            DataFrame with missing values handled
        """
        if self.data is None:
            raise ValueError("No data to process. Call fetch_data() first.")
        
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values. Applying forward fill.")
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        return self.data
    
    def create_splits(
        self, 
        train_ratio: float = 0.6, 
        test_ratio: float = 0.2, 
        val_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/test/validation splits.
        
        Args:
            train_ratio: Proportion for training set
            test_ratio: Proportion for test set
            val_ratio: Proportion for validation set
            
        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        if self.data is None:
            raise ValueError("No data to split. Call fetch_data() first.")
        
        if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        n = len(self.data)
        train_end = int(n * train_ratio)
        test_end = int(n * (train_ratio + test_ratio))
        
        train_df = self.data.iloc[:train_end].copy()
        test_df = self.data.iloc[train_end:test_end].copy()
        val_df = self.data.iloc[test_end:].copy()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")
        
        return train_df, test_df, val_df
    
    def save_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                  val_df: pd.DataFrame, output_dir: str = "data/raw"):
        """
        Save the split datasets to CSV files.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            val_df: Validation dataframe
            output_dir: Directory to save files
        """
        train_df.to_csv(f"{output_dir}/train_raw.csv")
        test_df.to_csv(f"{output_dir}/test_raw.csv")
        val_df.to_csv(f"{output_dir}/val_raw.csv")
        
        logger.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    import yaml
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect and prepare market data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Override symbol from config')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data configuration
    symbol = args.symbol or config['data']['asset_symbol']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    train_ratio = config['data']['train_split']
    test_ratio = config['data']['test_split']
    val_ratio = config['data']['val_split']
    
    # Run data collection pipeline
    logger.info(f"Starting data collection for {symbol}")
    collector = DataCollector(symbol, start_date, end_date)
    
    data = collector.fetch_data()
    data = collector.handle_missing_data()
    train, test, val = collector.create_splits(train_ratio, test_ratio, val_ratio)
    collector.save_data(train, test, val)
    
    logger.info("Data collection completed successfully!")
    logger.info(f"Data date range: {data.index[0]} to {data.index[-1]}")
