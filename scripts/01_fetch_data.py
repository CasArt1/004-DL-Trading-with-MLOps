"""Script to fetch market data from Yahoo Finance."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
from configs.config import TICKER, START_DATE, END_DATE, INTERVAL, DATA_DIR


def fetch_data(ticker=TICKER, start=START_DATE, end=END_DATE, interval=INTERVAL):
    """
    Fetch historical market data.
    
    Args:
        ticker: Trading symbol
        start: Start date
        end: End date
        interval: Data interval
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching data for {ticker} from {start} to {end} with interval {interval}...")
    
    data = yf.download(ticker, start=start, end=end, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}")
    
    print(f"Fetched {len(data)} rows of data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    return data


def main():
    """Main function."""
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Fetch data
    data = fetch_data()
    
    # Save raw data
    raw_data_path = os.path.join(DATA_DIR, "raw_data.csv")
    data.to_csv(raw_data_path)
    print(f"Raw data saved to: {raw_data_path}")
    
    # Display summary
    print("\nData Summary:")
    print(data.describe())
    print(f"\nColumns: {list(data.columns)}")


if __name__ == "__main__":
    main()
