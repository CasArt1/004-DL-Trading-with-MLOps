"""Technical indicators calculation module."""

import pandas as pd
import numpy as np
from typing import Dict, List


class TechnicalIndicators:
    """Calculate various technical indicators for time series data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data.copy()
        
    def add_sma(self, periods: List[int] = [10, 20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Average indicators."""
        for period in periods:
            self.data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
        return self.data
    
    def add_ema(self, periods: List[int] = [12, 26]) -> pd.DataFrame:
        """Add Exponential Moving Average indicators."""
        for period in periods:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        return self.data
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return self.data
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        return self.data
    
    def add_bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        self.data['BB_Upper'] = sma + (std * num_std)
        self.data['BB_Middle'] = sma
        self.data['BB_Lower'] = sma - (std * num_std)
        self.data['BB_Width'] = self.data['BB_Upper'] - self.data['BB_Lower']
        return self.data
    
    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.data[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        return self.data
    
    def add_stochastic(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = self.data['Low'].rolling(window=period).min()
        high_max = self.data['High'].rolling(window=period).max()
        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        self.data['Stoch_K'] = self.data['Stoch_K'].rolling(window=smooth_k).mean()
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=smooth_d).mean()
        return self.data
    
    def add_obv(self) -> pd.DataFrame:
        """Add On-Balance Volume."""
        obv = [0]
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + self.data['Volume'].iloc[i])
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - self.data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        self.data['OBV'] = obv
        return self.data
    
    def add_momentum_indicators(self) -> pd.DataFrame:
        """Add various momentum indicators."""
        # Rate of Change
        self.data['ROC_10'] = self.data['Close'].pct_change(periods=10) * 100
        
        # Price momentum
        self.data['Momentum_10'] = self.data['Close'] - self.data['Close'].shift(10)
        
        return self.data
    
    def add_all_indicators(self) -> pd.DataFrame:
        """Add all technical indicators."""
        self.add_sma()
        self.add_ema()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_stochastic()
        self.add_obv()
        self.add_momentum_indicators()
        return self.data
    
    def get_features(self) -> pd.DataFrame:
        """Return the data with all calculated features."""
        return self.data
