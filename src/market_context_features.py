"""
Market Context Feature Engineering
Add macro-level features to improve model's understanding of market regimes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketContextFeatures:
    """
    Extract market context features:
    - VIX (volatility regime)
    - SPY (market direction)
    - Market breadth (advance/decline)
    - Sector rotation signals
    """
    
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = {}
        
    def fetch_market_data(self):
        """Download market context data"""
        logger.info("Fetching market context data...")
        
        # VIX - Volatility Index
        try:
            logger.info("Downloading VIX...")
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
            self.market_data['VIX'] = vix['Close']
            logger.info(f"VIX: {len(vix)} days")
        except Exception as e:
            logger.warning(f"Failed to download VIX: {e}")
            self.market_data['VIX'] = None
            
        # SPY - S&P 500 ETF (market proxy)
        try:
            logger.info("Downloading SPY...")
            spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
            self.market_data['SPY'] = spy
            logger.info(f"SPY: {len(spy)} days")
        except Exception as e:
            logger.warning(f"Failed to download SPY: {e}")
            self.market_data['SPY'] = None
            
        # QQQ - NASDAQ ETF (tech sector proxy)
        try:
            logger.info("Downloading QQQ...")
            qqq = yf.download('QQQ', start=self.start_date, end=self.end_date, progress=False)
            self.market_data['QQQ'] = qqq
            logger.info(f"QQQ: {len(qqq)} days")
        except Exception as e:
            logger.warning(f"Failed to download QQQ: {e}")
            self.market_data['QQQ'] = None
            
        # DIA - Dow Jones ETF (blue chip proxy)
        try:
            logger.info("Downloading DIA...")
            dia = yf.download('DIA', start=self.start_date, end=self.end_date, progress=False)
            self.market_data['DIA'] = dia
            logger.info(f"DIA: {len(dia)} days")
        except Exception as e:
            logger.warning(f"Failed to download DIA: {e}")
            self.market_data['DIA'] = None
            
        # TLT - Treasury Bonds ETF (risk-off proxy)
        try:
            logger.info("Downloading TLT...")
            tlt = yf.download('TLT', start=self.start_date, end=self.end_date, progress=False)
            self.market_data['TLT'] = tlt
            logger.info(f"TLT: {len(tlt)} days")
        except Exception as e:
            logger.warning(f"Failed to download TLT: {e}")
            self.market_data['TLT'] = None
            
        return self.market_data
    
    def calculate_context_features(self):
        """Calculate market context features"""
        logger.info("Calculating market context features...")
        
        features_df = pd.DataFrame()
        
        # VIX Features
        if self.market_data.get('VIX') is not None:
            vix = self.market_data['VIX'].squeeze()  # Convert to Series
            features_df['vix'] = vix
            features_df['vix_ma_20'] = vix.rolling(20).mean()
            features_df['vix_std_20'] = vix.rolling(20).std()
            features_df['vix_regime'] = (vix > vix.rolling(50).mean()).astype(int)  # High vol regime
            features_df['vix_spike'] = (vix > vix.rolling(20).mean() + vix.rolling(20).std()).astype(int)
            logger.info("VIX features calculated")
        
        # SPY (Market) Features
        if self.market_data.get('SPY') is not None:
            spy = self.market_data['SPY']
            features_df['spy_close'] = spy['Close']
            features_df['spy_return'] = spy['Close'].pct_change()
            features_df['spy_volume'] = spy['Volume']
            
            # Market trend
            features_df['spy_sma_50'] = spy['Close'].rolling(50).mean()
            features_df['spy_sma_200'] = spy['Close'].rolling(200).mean()
            features_df['spy_trend'] = (features_df['spy_sma_50'] > features_df['spy_sma_200']).astype(int)
            
            # Market strength
            features_df['spy_rsi'] = self._calculate_rsi(spy['Close'], 14)
            features_df['spy_momentum'] = spy['Close'].pct_change(20)
            
            # Volume analysis
            spy_volume_ma = spy['Volume'].rolling(20).mean()
            features_df['spy_volume_ma'] = spy_volume_ma
            features_df['spy_volume_ratio'] = spy['Volume'] / spy_volume_ma
            
            logger.info("SPY features calculated")
        
        # Tech vs Market (QQQ/SPY ratio)
        if self.market_data.get('QQQ') is not None and self.market_data.get('SPY') is not None:
            qqq = self.market_data['QQQ']['Close'].squeeze()
            spy = self.market_data['SPY']['Close'].squeeze()
            tech_vs_market = qqq / spy
            features_df['tech_vs_market'] = tech_vs_market
            features_df['tech_outperformance'] = tech_vs_market.pct_change(20)
            logger.info("Tech rotation features calculated")
        
        # Blue Chip vs Market (DIA/SPY ratio)
        if self.market_data.get('DIA') is not None and self.market_data.get('SPY') is not None:
            dia = self.market_data['DIA']['Close'].squeeze()
            spy = self.market_data['SPY']['Close'].squeeze()
            bluechip_vs_market = dia / spy
            features_df['bluechip_vs_market'] = bluechip_vs_market
            features_df['bluechip_outperformance'] = bluechip_vs_market.pct_change(20)
            logger.info("Blue chip rotation features calculated")
        
        # Risk-On/Risk-Off (SPY vs TLT)
        if self.market_data.get('SPY') is not None and self.market_data.get('TLT') is not None:
            spy = self.market_data['SPY']['Close'].squeeze()
            tlt = self.market_data['TLT']['Close'].squeeze()
            risk_on_off = spy / tlt
            features_df['risk_on_off'] = risk_on_off
            features_df['risk_sentiment'] = risk_on_off.pct_change(20)
            features_df['risk_regime'] = (risk_on_off > risk_on_off.rolling(50).mean()).astype(int)
            logger.info("Risk sentiment features calculated")
        
        # Market breadth approximation (using volume)
        if self.market_data.get('SPY') is not None:
            spy = self.market_data['SPY']
            features_df['market_breadth_proxy'] = (spy['High'] - spy['Low']) / spy['Close']
            features_df['market_breadth_ma'] = features_df['market_breadth_proxy'].rolling(20).mean()
            logger.info("Market breadth features calculated")
        
        # Forward fill missing values, then backfill remaining
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Total market context features: {len(features_df.columns)}")
        return features_df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def merge_with_stock_data(self, stock_df):
        """Merge market context features with stock data"""
        logger.info("Merging market context with stock data...")
        
        # Calculate context features
        context_df = self.calculate_context_features()
        
        # Ensure both dataframes have Date index
        if 'Date' in stock_df.columns:
            stock_df = stock_df.set_index('Date')
        if 'Date' in context_df.columns:
            context_df = context_df.set_index('Date')
            
        # Merge on date
        merged_df = stock_df.join(context_df, how='left')
        
        # Forward fill any remaining missing values
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Merged shape: {merged_df.shape}")
        logger.info(f"New features added: {len(context_df.columns)}")
        
        return merged_df


def main():
    """Test the market context feature extraction"""
    # Test dates
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    
    # Initialize
    market_context = MarketContextFeatures(start_date, end_date)
    
    # Fetch data
    market_context.fetch_market_data()
    
    # Calculate features
    features_df = market_context.calculate_context_features()
    
    print("\n" + "="*60)
    print("MARKET CONTEXT FEATURES")
    print("="*60)
    print(f"\nTotal features: {len(features_df.columns)}")
    print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    print(f"Total days: {len(features_df)}")
    print(f"\nFeature list:")
    for col in features_df.columns:
        print(f"  - {col}")
    
    print(f"\nSample data (first 5 rows):")
    print(features_df.head())
    
    print(f"\nMissing values:")
    print(features_df.isnull().sum())


if __name__ == "__main__":
    main()
