"""
Feature Selection Module
Select the most informative features using statistical methods
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select best features using multiple methods."""
    
    def __init__(self, n_features: int = 45):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of features to keep (38 technical + ~7 market context)
        """
        self.n_features = n_features
        self.selected_features = []
        self.feature_scores = {}
        
    def select_by_anova(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features using ANOVA F-statistic.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        logger.info(f"ANOVA F-test: Selecting top {self.n_features} features...")
        
        selector = SelectKBest(f_classif, k=self.n_features)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'pvalue': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        selected = scores.head(self.n_features)['feature'].tolist()
        self.feature_scores['anova'] = scores
        
        logger.info(f"ANOVA: Selected {len(selected)} features")
        return selected
    
    def select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features using Mutual Information.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Mutual Information: Selecting top {self.n_features} features...")
        
        selector = SelectKBest(mutual_info_classif, k=self.n_features)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected = scores.head(self.n_features)['feature'].tolist()
        self.feature_scores['mutual_info'] = scores
        
        logger.info(f"Mutual Info: Selected {len(selected)} features")
        return selected
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Random Forest: Selecting top {self.n_features} features...")
        
        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        scores = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected = scores.head(self.n_features)['feature'].tolist()
        self.feature_scores['random_forest'] = scores
        
        logger.info(f"Random Forest: Selected {len(selected)} features")
        return selected
    
    def select_features_ensemble(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features using ensemble of methods.
        Features that appear in multiple methods get higher priority.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        logger.info("=" * 60)
        logger.info("ENSEMBLE FEATURE SELECTION")
        logger.info("=" * 60)
        
        # Get selections from each method
        anova_features = set(self.select_by_anova(X, y))
        mi_features = set(self.select_by_mutual_info(X, y))
        rf_features = set(self.select_by_importance(X, y))
        
        # Count votes for each feature
        all_features = set(X.columns)
        feature_votes = {}
        
        for feature in all_features:
            votes = 0
            if feature in anova_features:
                votes += 1
            if feature in mi_features:
                votes += 1
            if feature in rf_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Sort by votes, then by average score
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected = []
        for feature, votes in sorted_features:
            if len(selected) < self.n_features:
                selected.append(feature)
        
        self.selected_features = selected
        
        logger.info(f"\nEnsemble Selection: {len(selected)} features")
        logger.info(f"\nVoting Results:")
        logger.info(f"  - 3 votes (all methods): {sum(1 for v in feature_votes.values() if v == 3)} features")
        logger.info(f"  - 2 votes (two methods): {sum(1 for v in feature_votes.values() if v == 2)} features")
        logger.info(f"  - 1 vote (one method): {sum(1 for v in feature_votes.values() if v == 1)} features")
        
        return selected
    
    def analyze_market_context_value(self, X: pd.DataFrame, y: pd.Series):
        """
        Analyze which market context features are most valuable.
        
        Args:
            X: Feature matrix with both technical and market context features
            y: Target labels
        """
        logger.info("\n" + "=" * 60)
        logger.info("MARKET CONTEXT FEATURE ANALYSIS")
        logger.info("=" * 60)
        
        # Identify market context features (those not in original 38)
        technical_features = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_diff',
            'roc_12', 'roc_25', 'stoch_k', 'stoch_d', 'williams_r',
            'ao', 'atr_14', 'bb_high', 'bb_low', 'bb_mid', 'bb_width',
            'bb_pct', 'kc_high', 'kc_low', 'kc_mid', 'volatility_20',
            'volatility_50', 'obv', 'volume_ma_10', 'volume_ma_20',
            'volume_roc', 'adi', 'cmf', 'force_index', 'vwap',
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'adx', 'adx_pos', 'adx_neg'
        ]
        
        market_features = [col for col in X.columns if col not in technical_features]
        
        logger.info(f"\nMarket Context Features: {len(market_features)}")
        
        # Check which market features made it to top selections
        if hasattr(self, 'selected_features'):
            selected_market = [f for f in self.selected_features if f in market_features]
            logger.info(f"Selected Market Features: {len(selected_market)}/{len(market_features)}")
            logger.info(f"\nTop Market Features Selected:")
            for i, feature in enumerate(selected_market, 1):
                logger.info(f"  {i}. {feature}")
        
        # Show market feature rankings from each method
        for method_name, scores_df in self.feature_scores.items():
            market_scores = scores_df[scores_df['feature'].isin(market_features)].head(10)
            logger.info(f"\nTop 10 Market Features by {method_name.upper()}:")
            for idx, row in market_scores.iterrows():
                rank = scores_df[scores_df['feature'] == row['feature']].index[0] + 1
                logger.info(f"  Rank {rank:2d}: {row['feature']:30s} (score: {row.get('score', row.get('importance', 0)):.4f})")
    
    def save_selected_features(self, output_path: str = "data/processed/selected_features.txt"):
        """Save selected feature names to file."""
        if not self.selected_features:
            logger.warning("No features selected yet!")
            return
        
        with open(output_path, 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        logger.info(f"\nSaved {len(self.selected_features)} features to {output_path}")


def main():
    """Run feature selection pipeline."""
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load labeled data
    logger.info("Loading training data...")
    train_df = pd.read_csv("data/processed/train_labeled.csv")
    
    # Separate features and labels
    label_col = 'Label' if 'Label' in train_df.columns else 'label'
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                   'Dividends', 'Stock Splits', label_col]
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X = train_df[feature_cols]
    y = train_df[label_col]
    
    # Handle NaN values
    logger.info(f"Checking for NaN values...")
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in {(nan_counts > 0).sum()} features")
        logger.warning(f"Filling NaN with 0...")
        X = X.fillna(0)
    
    # Handle infinite values
    logger.info(f"Checking for infinite values...")
    X = X.replace([np.inf, -np.inf], 0)
    
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Classes: {y.nunique()}")
    
    # Initialize feature selector (keep 45 features: 38 technical + ~7 market)
    selector = FeatureSelector(n_features=45)
    
    # Run ensemble selection
    selected_features = selector.select_features_ensemble(X, y)
    
    # Analyze market context value
    selector.analyze_market_context_value(X, y)
    
    # Save selected features
    selector.save_selected_features()
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Reduced from {len(feature_cols)} to {len(selected_features)} features")
    logger.info(f"Saved to: data/processed/selected_features.txt")


if __name__ == "__main__":
    main()
