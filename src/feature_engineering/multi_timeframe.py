"""Multi-timeframe feature engineering module."""

import pandas as pd
import numpy as np
from typing import Dict, List
from .indicators import TechnicalIndicators


class MultiTimeframeFeatures:
    """Engineer features from multiple timeframes."""
    
    def __init__(self, data: pd.DataFrame, base_timeframe: str = '1h'):
        """
        Initialize with base timeframe data.
        
        Args:
            data: DataFrame with datetime index and OHLCV columns
            base_timeframe: Base timeframe string (e.g., '1h', '1d')
        """
        self.data = data.copy()
        self.base_timeframe = base_timeframe
        self.features_data = None
        
    def resample_to_timeframe(self, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            timeframe: Target timeframe (e.g., '4h', '1d')
        
        Returns:
            Resampled DataFrame
        """
        resampled = self.data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return resampled
    
    def add_timeframe_features(self, timeframe: str, prefix: str = None) -> pd.DataFrame:
        """
        Add features from a specific timeframe.
        
        Args:
            timeframe: Timeframe to extract features from
            prefix: Prefix for column names (defaults to timeframe)
        
        Returns:
            DataFrame with merged features
        """
        if prefix is None:
            prefix = timeframe
            
        # Resample to target timeframe
        resampled_data = self.resample_to_timeframe(timeframe)
        
        # Calculate indicators for this timeframe
        indicators = TechnicalIndicators(resampled_data)
        indicators.add_all_indicators()
        tf_features = indicators.get_features()
        
        # Rename columns with prefix
        rename_dict = {col: f'{prefix}_{col}' for col in tf_features.columns}
        tf_features = tf_features.rename(columns=rename_dict)
        
        # Forward fill to match base timeframe
        tf_features = tf_features.reindex(self.data.index, method='ffill')
        
        return tf_features
    
    def create_multi_timeframe_features(
        self, 
        timeframes: List[str] = ['1h', '4h', '1d']
    ) -> pd.DataFrame:
        """
        Create features from multiple timeframes.
        
        Args:
            timeframes: List of timeframes to include
        
        Returns:
            DataFrame with all multi-timeframe features
        """
        # Start with base data
        all_features = self.data.copy()
        
        # Add features from each timeframe
        for tf in timeframes:
            tf_features = self.add_timeframe_features(tf)
            # Only add indicator columns, not the OHLCV columns
            indicator_cols = [col for col in tf_features.columns 
                            if not any(x in col for x in ['_Open', '_High', '_Low', '_Close', '_Volume'])]
            all_features = pd.concat([all_features, tf_features[indicator_cols]], axis=1)
        
        self.features_data = all_features
        return all_features
    
    def add_lagged_features(self, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lagged features
        """
        if self.features_data is None:
            self.features_data = self.data.copy()
            
        for col in columns:
            if col in self.features_data.columns:
                for lag in lags:
                    self.features_data[f'{col}_lag_{lag}'] = self.features_data[col].shift(lag)
        
        return self.features_data
    
    def add_rolling_statistics(
        self, 
        columns: List[str], 
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Add rolling statistics for specified columns.
        
        Args:
            columns: Columns to calculate statistics for
            windows: Rolling window sizes
        
        Returns:
            DataFrame with rolling statistics
        """
        if self.features_data is None:
            self.features_data = self.data.copy()
            
        for col in columns:
            if col in self.features_data.columns:
                for window in windows:
                    self.features_data[f'{col}_roll_mean_{window}'] = \
                        self.features_data[col].rolling(window=window).mean()
                    self.features_data[f'{col}_roll_std_{window}'] = \
                        self.features_data[col].rolling(window=window).std()
        
        return self.features_data
    
    def get_features(self) -> pd.DataFrame:
        """Return the engineered features."""
        if self.features_data is None:
            return self.data
        return self.features_data
    
    def save_features(self, filepath: str):
        """Save engineered features to CSV."""
        if self.features_data is not None:
            self.features_data.to_csv(filepath)
        else:
            self.data.to_csv(filepath)
