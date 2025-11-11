"""
Apply Feature Selection
Filter datasets to keep only selected features
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_selected_features(filepath: str = "data/processed/selected_features.txt") -> list:
    """Load list of selected features from file."""
    with open(filepath, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def filter_dataset(df: pd.DataFrame, selected_features: list, keep_base_cols: bool = True) -> pd.DataFrame:
    """
    Filter dataset to keep only selected features.
    
    Args:
        df: Full dataset
        selected_features: List of feature names to keep
        keep_base_cols: Whether to keep base columns (Date, OHLCV, etc.)
        
    Returns:
        Filtered dataframe
    """
    # Base columns to always keep
    base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                 'Dividends', 'Stock Splits']
    
    # Build final column list
    if keep_base_cols:
        # Keep base cols that exist in df
        cols_to_keep = [col for col in base_cols if col in df.columns]
        # Add selected features
        cols_to_keep.extend([col for col in selected_features if col in df.columns])
    else:
        # Just selected features
        cols_to_keep = [col for col in selected_features if col in df.columns]
    
    # Add label columns if they exist
    for label_col in ['Label', 'label', 'class_weight']:
        if label_col in df.columns and label_col not in cols_to_keep:
            cols_to_keep.append(label_col)
    
    filtered_df = df[cols_to_keep]
    
    logger.info(f"Filtered from {len(df.columns)} to {len(filtered_df.columns)} columns")
    
    return filtered_df


def main():
    """Apply feature selection to all datasets."""
    
    logger.info("=" * 60)
    logger.info("APPLYING FEATURE SELECTION")
    logger.info("=" * 60)
    
    # Load selected features
    selected_features = load_selected_features()
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    # Process each dataset
    datasets = {
        'train_features': 'data/processed/train_features.csv',
        'test_features': 'data/processed/test_features.csv',
        'val_features': 'data/processed/val_features.csv',
        'train_labeled': 'data/processed/train_labeled.csv',
        'test_labeled': 'data/processed/test_labeled.csv',
        'val_labeled': 'data/processed/val_labeled.csv'
    }
    
    for name, filepath in datasets.items():
        if not os.path.exists(filepath):
            logger.warning(f"Skipping {name} - file not found")
            continue
        
        logger.info(f"\nProcessing {name}...")
        
        # Load dataset
        df = pd.read_csv(filepath)
        logger.info(f"Original shape: {df.shape}")
        
        # Check if this is a labeled dataset
        has_label = 'Label' in df.columns or 'label' in df.columns
        
        # Filter to selected features
        filtered_df = filter_dataset(df, selected_features, keep_base_cols=True)
        logger.info(f"Filtered shape: {filtered_df.shape}")
        
        # Save filtered dataset
        output_path = filepath.replace('.csv', '_selected.csv')
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")
    
    # Update feature_names.txt
    feature_names_path = "data/processed/feature_names.txt"
    with open(feature_names_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    logger.info(f"\nUpdated {feature_names_path} with {len(selected_features)} features")
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SELECTION APPLIED")
    logger.info("=" * 60)
    logger.info(f"All datasets filtered to {len(selected_features)} features")
    logger.info("Use *_selected.csv files for training")


if __name__ == "__main__":
    main()
