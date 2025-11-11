"""
Label Generation Module
Creates target labels (buy/sell/hold) for supervised learning.
Handles class imbalance with configurable thresholds and class weights.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate trading labels from price data."""
    
    def __init__(self, 
                 buy_threshold: float = 0.02,
                 sell_threshold: float = -0.02,
                 forward_days: int = 5):
        """
        Initialize label generator.
        
        Args:
            buy_threshold: Return threshold for buy signal (e.g., 0.02 = 2% gain)
            sell_threshold: Return threshold for sell signal (e.g., -0.02 = 2% loss)
            forward_days: Number of days to look forward for returns
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.forward_days = forward_days
        self.class_weights = None
        
    def create_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward returns for labeling.
        
        Args:
            df: DataFrame with Close prices
            
        Returns:
            DataFrame with forward returns column
        """
        df = df.copy()
        
        # Calculate forward return
        df['forward_return'] = (
            df['Close'].shift(-self.forward_days) / df['Close'] - 1
        )
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell/hold labels based on forward returns.
        
        Labels:
        - 0: SELL (forward return < sell_threshold)
        - 1: HOLD (sell_threshold <= forward return < buy_threshold)
        - 2: BUY (forward return >= buy_threshold)
        
        Args:
            df: DataFrame with forward returns
            
        Returns:
            DataFrame with labels column
        """
        df = df.copy()
        
        # Create labels based on thresholds
        df['label'] = 1  # Default to HOLD
        df.loc[df['forward_return'] >= self.buy_threshold, 'label'] = 2  # BUY
        df.loc[df['forward_return'] < self.sell_threshold, 'label'] = 0  # SELL
        
        # Remove rows where we can't calculate forward return (last N days)
        df = df.dropna(subset=['forward_return'])
        
        return df
    
    def analyze_class_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of labels.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Dictionary with class statistics
        """
        label_counts = df['label'].value_counts().sort_index()
        label_pcts = df['label'].value_counts(normalize=True).sort_index() * 100
        
        stats = {
            'counts': {
                'SELL': int(label_counts.get(0, 0)),
                'HOLD': int(label_counts.get(1, 0)),
                'BUY': int(label_counts.get(2, 0))
            },
            'percentages': {
                'SELL': float(label_pcts.get(0, 0)),
                'HOLD': float(label_pcts.get(1, 0)),
                'BUY': float(label_pcts.get(2, 0))
            },
            'total': len(df)
        }
        
        logger.info(f"Label Distribution:")
        logger.info(f"  SELL (0): {stats['counts']['SELL']} ({stats['percentages']['SELL']:.2f}%)")
        logger.info(f"  HOLD (1): {stats['counts']['HOLD']} ({stats['percentages']['HOLD']:.2f}%)")
        logger.info(f"  BUY  (2): {stats['counts']['BUY']} ({stats['percentages']['BUY']:.2f}%)")
        
        # Check for severe imbalance
        minority_pct = min(stats['percentages'].values())
        if minority_pct < 15:
            logger.warning(f"Severe class imbalance detected! Minority class: {minority_pct:.2f}%")
            logger.warning("Consider adjusting thresholds or using class weights")
        
        return stats
    
    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights to handle imbalanced data.
        
        Args:
            labels: Array of labels
            
        Returns:
            Dictionary mapping class to weight
        """
        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        
        self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        
        logger.info(f"Computed class weights: {self.class_weights}")
        
        return self.class_weights
    
    def adjust_thresholds_for_balance(self, df: pd.DataFrame, 
                                     target_buy_pct: float = 25.0,
                                     target_sell_pct: float = 25.0) -> Tuple[float, float]:
        """
        Automatically adjust thresholds to achieve target class balance.
        
        Args:
            df: DataFrame with forward returns
            target_buy_pct: Target percentage for BUY class
            target_sell_pct: Target percentage for SELL class
            
        Returns:
            Tuple of (buy_threshold, sell_threshold)
        """
        logger.info("Adjusting thresholds for balanced distribution...")
        
        forward_returns = df['forward_return'].dropna()
        
        # Calculate thresholds based on percentiles
        buy_percentile = 100 - target_buy_pct
        sell_percentile = target_sell_pct
        
        new_buy_threshold = np.percentile(forward_returns, buy_percentile)
        new_sell_threshold = np.percentile(forward_returns, sell_percentile)
        
        logger.info(f"Original thresholds - Buy: {self.buy_threshold:.4f}, Sell: {self.sell_threshold:.4f}")
        logger.info(f"Adjusted thresholds - Buy: {new_buy_threshold:.4f}, Sell: {new_sell_threshold:.4f}")
        
        return new_buy_threshold, new_sell_threshold


def process_labels(config: Dict, adjust_thresholds: bool = False) -> None:
    """
    Process all data splits to generate labels.
    
    Args:
        config: Configuration dictionary
        adjust_thresholds: Whether to automatically adjust thresholds for balance
    """
    # Load feature-engineered data
    logger.info("Loading processed feature data...")
    train_df = pd.read_csv("data/processed/train_features.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/processed/test_features.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/val_features.csv", index_col=0, parse_dates=True)
    
    # Initialize label generator
    buy_threshold = config.get('labeling', {}).get('buy_threshold', 0.02)
    sell_threshold = config.get('labeling', {}).get('sell_threshold', -0.02)
    forward_days = config.get('labeling', {}).get('forward_days', 5)
    
    generator = LabelGenerator(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        forward_days=forward_days
    )
    
    # Optionally adjust thresholds
    if adjust_thresholds:
        train_with_returns = generator.create_forward_returns(train_df)
        new_buy, new_sell = generator.adjust_thresholds_for_balance(train_with_returns)
        generator.buy_threshold = new_buy
        generator.sell_threshold = new_sell
    
    # Generate labels for all splits
    logger.info("\nProcessing training set...")
    train_df = generator.create_forward_returns(train_df)
    train_df = generator.generate_labels(train_df)
    train_stats = generator.analyze_class_distribution(train_df)
    
    logger.info("\nProcessing test set...")
    test_df = generator.create_forward_returns(test_df)
    test_df = generator.generate_labels(test_df)
    test_stats = generator.analyze_class_distribution(test_df)
    
    logger.info("\nProcessing validation set...")
    val_df = generator.create_forward_returns(val_df)
    val_df = generator.generate_labels(val_df)
    val_stats = generator.analyze_class_distribution(val_df)
    
    # Compute class weights from training set
    class_weights = generator.compute_class_weights(train_df['label'].values)
    
    # Save labeled data
    logger.info("\nSaving labeled data...")
    train_df.to_csv("data/processed/train_labeled.csv")
    test_df.to_csv("data/processed/test_labeled.csv")
    val_df.to_csv("data/processed/val_labeled.csv")
    
    # Save class weights and statistics
    import json
    stats_summary = {
        'thresholds': {
            'buy_threshold': float(generator.buy_threshold),
            'sell_threshold': float(generator.sell_threshold),
            'forward_days': generator.forward_days
        },
        'class_weights': class_weights,
        'distribution': {
            'train': train_stats,
            'test': test_stats,
            'val': val_stats
        }
    }
    
    with open("data/processed/label_stats.json", 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    logger.info(f"\nLabeling complete!")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Val shape: {val_df.shape}")


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate labels for supervised learning')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--adjust-thresholds', action='store_true',
                       help='Automatically adjust thresholds for balanced distribution')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process labels
    process_labels(config, adjust_thresholds=args.adjust_thresholds)
