"""
Feature Engineering Module
Generates 20+ technical indicators from OHLCV data including:
- Momentum indicators (RSI, MACD, ROC, etc.)
- Volatility indicators (ATR, Bollinger Bands, etc.)
- Volume indicators (OBV, Volume MA, etc.)
- Market context features (VIX, SPY, sector rotation, etc.)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import ta
from market_context_features import MarketContextFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer technical indicators from price data."""
    
    def __init__(self, normalization: str = 'standard'):
        """
        Initialize feature engineer.
        
        Args:
            normalization: Type of normalization ('standard' or 'minmax')
        """
        self.normalization = normalization
        self.scaler = None
        self.feature_names = []
        
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features added
        """
        logger.info("Creating momentum features...")
        
        # RSI - Relative Strength Index (14, 21 periods)
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(close=df['Close'], window=21).rsi()
        
        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ROC - Rate of Change
        df['roc_12'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()
        df['roc_25'] = ta.momentum.ROCIndicator(close=df['Close'], window=25).roc()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], lbp=14
        ).williams_r()
        
        # Awesome Oscillator
        df['ao'] = ta.momentum.AwesomeOscillatorIndicator(
            high=df['High'], low=df['Low']
        ).awesome_oscillator()
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features added
        """
        logger.info("Creating volatility features...")
        
        # ATR - Average True Range
        df['atr_14'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pct'] = bollinger.bollinger_pband()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(
            high=df['High'], low=df['Low'], close=df['Close']
        )
        df['kc_high'] = keltner.keltner_channel_hband()
        df['kc_low'] = keltner.keltner_channel_lband()
        df['kc_mid'] = keltner.keltner_channel_mband()
        
        # Historical Volatility (standard deviation of returns)
        df['returns'] = df['Close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        logger.info("Creating volume features...")
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        
        # Volume Moving Averages
        df['volume_ma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = df['Volume'].pct_change(periods=5)
        
        # Accumulation/Distribution Index
        df['adi'] = ta.volume.AccDistIndexIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        ).acc_dist_index()
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        ).chaikin_money_flow()
        
        # Force Index
        df['force_index'] = ta.volume.ForceIndexIndicator(
            close=df['Close'], volume=df['Volume']
        ).force_index()
        
        # Volume Weighted Average Price (approximation using daily data)
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend features added
        """
        logger.info("Creating trend features...")
        
        # Simple Moving Averages
        df['sma_10'] = ta.trend.SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26).ema_indicator()
        
        # ADX - Average Directional Index
        adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        return df
    
    def create_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price action and pattern-based features (IMPROVEMENT: 5 new features).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price action features added
        """
        logger.info("Creating price action features...")
        
        # 1. Price body size (relative candle body)
        df['body_size'] = abs(df['Close'] - df['Open']) / (df['Open'] + 1e-10)
        
        # 2. Volume-price trend (correlation between volume and price changes)
        df['volume_price_trend'] = df['volume_roc'] * df['returns']
        
        # 3. RSI divergence (momentum vs price divergence)
        df['rsi_divergence'] = df['rsi_14'].diff(5) * np.sign(df['Close'].pct_change(5))
        
        # 4. Price position in Bollinger Bands (normalized 0-1)
        df['price_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # 5. OBV slope (rate of change in OBV)
        df['obv_slope'] = df['obv'].diff(5) / (df['obv'].shift(5).abs() + 1e-10)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Create features
        df = self.create_momentum_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        df = self.create_trend_features(df)
        df = self.create_price_action_features(df)
        
        # Drop rows with NaN values (from rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        # Store feature names (exclude original OHLCV columns)
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'returns']
        self.feature_names = [col for col in df.columns if col not in original_cols]
        
        logger.info(f"Created {len(self.feature_names)} features: {self.feature_names}")
        
        return df
    
    def normalize_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using training set statistics.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            val_df: Validation dataframe
            
        Returns:
            Tuple of normalized dataframes
        """
        logger.info(f"Normalizing features using {self.normalization} scaler...")
        
        # Initialize scaler
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        
        # Fit scaler on training data only
        train_features = train_df[self.feature_names].values
        self.scaler.fit(train_features)
        
        # Transform all splits
        train_df[self.feature_names] = self.scaler.transform(train_features)
        test_df[self.feature_names] = self.scaler.transform(test_df[self.feature_names].values)
        val_df[self.feature_names] = self.scaler.transform(val_df[self.feature_names].values)
        
        logger.info("Normalization complete")
        
        return train_df, test_df, val_df
    
    def save_scaler(self, filepath: str = "models/feature_scaler.pkl"):
        """Save the fitted scaler for later use."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str = "models/feature_scaler.pkl"):
        """Load a previously saved scaler."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {filepath}")


def process_all_splits(config: Dict) -> None:
    """
    Process all data splits with feature engineering.
    
    Args:
        config: Configuration dictionary
    """
    # Load raw data
    logger.info("Loading raw data splits...")
    train_df = pd.read_csv("data/raw/train_raw.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/raw/test_raw.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/raw/val_raw.csv", index_col=0, parse_dates=True)
    
    # Get date ranges for market context
    start_date = min(train_df.index.min(), test_df.index.min(), val_df.index.min())
    end_date = max(train_df.index.max(), test_df.index.max(), val_df.index.max())
    
    # Fetch market context features
    logger.info("Fetching market context features...")
    market_context = MarketContextFeatures(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    market_context.fetch_market_data()
    
    # Initialize feature engineer
    normalization = config['features']['normalization']
    engineer = FeatureEngineer(normalization=normalization)
    
    # Engineer features for each split
    logger.info("Engineering features for training set...")
    train_df = engineer.engineer_features(train_df)
    train_df = market_context.merge_with_stock_data(train_df)
    
    logger.info("Engineering features for test set...")
    test_df = engineer.engineer_features(test_df)
    test_df = market_context.merge_with_stock_data(test_df)
    
    logger.info("Engineering features for validation set...")
    val_df = engineer.engineer_features(val_df)
    val_df = market_context.merge_with_stock_data(val_df)
    
    # Update feature names to include market context
    context_features = market_context.calculate_context_features().columns.tolist()
    engineer.feature_names.extend(context_features)
    
    # Normalize features
    train_df, test_df, val_df = engineer.normalize_features(train_df, test_df, val_df)
    
    # Save processed data
    logger.info("Saving processed data...")
    train_df.to_csv("data/processed/train_features.csv")
    test_df.to_csv("data/processed/test_features.csv")
    val_df.to_csv("data/processed/val_features.csv")
    
    # Save scaler
    engineer.save_scaler()
    
    # Save feature names
    with open("data/processed/feature_names.txt", 'w') as f:
        for feature in engineer.feature_names:
            f.write(f"{feature}\n")
    
    logger.info(f"Feature engineering complete! Total features: {len(engineer.feature_names)}")
    logger.info(f"  - Technical indicators: 38")
    logger.info(f"  - Market context: {len(context_features)}")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Val shape: {val_df.shape}")


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Engineer features from raw data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process all splits
    process_all_splits(config)
