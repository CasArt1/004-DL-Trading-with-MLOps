"""Script to monitor data drift."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.monitoring import DriftDetector
from configs.config import DATA_DIR, REPORTS_DIR, DRIFT_CONFIG


def load_data(filepath):
    """Load data."""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data


def main():
    """Main function."""
    # Load engineered features
    features_path = os.path.join(DATA_DIR, "engineered_features.csv")
    print(f"Loading features from: {features_path}")
    features = load_data(features_path)
    
    # Split into reference (training) and current (production-like) data
    split_point = int(len(features) * 0.8)
    reference_data = features[:split_point].dropna()
    current_data = features[split_point:].dropna()
    
    print(f"Reference data: {len(reference_data)} rows")
    print(f"Current data: {len(current_data)} rows")
    
    # Initialize drift detector
    print("\nInitializing drift detector...")
    detector = DriftDetector(
        reference_data=reference_data,
        drift_threshold=DRIFT_CONFIG["drift_threshold"]
    )
    
    # Detect drift
    print("\nDetecting drift across all features...")
    drift_results = detector.detect_drift_all_features(
        current_data,
        method=DRIFT_CONFIG["method"],
        alpha=DRIFT_CONFIG["alpha"]
    )
    
    # Print summary
    print("\n" + "="*50)
    print("DRIFT DETECTION SUMMARY")
    print("="*50)
    print(f"Total features analyzed: {drift_results['summary']['total_features']}")
    print(f"Features with drift: {drift_results['summary']['drifted_features']}")
    print(f"Drift percentage: {drift_results['summary']['drift_percentage']:.2f}%")
    print(f"Overall drift detected: {drift_results['summary']['overall_drift_detected']}")
    
    if drift_results['summary']['drifted_feature_names']:
        print(f"\nDrifted features:")
        for feature in drift_results['summary']['drifted_feature_names'][:10]:
            print(f"  - {feature}")
        if len(drift_results['summary']['drifted_feature_names']) > 10:
            print(f"  ... and {len(drift_results['summary']['drifted_feature_names']) - 10} more")
    
    # Generate Evidently report
    print("\nGenerating Evidently drift report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "drift_report.html")
    detector.generate_evidently_report(current_data, report_path)
    
    # Calculate feature statistics
    print("\nCalculating feature statistics...")
    stats = detector.calculate_feature_statistics(current_data)
    
    # Show statistics for a few key features
    key_features = ['Close', 'Volume', 'RSI_14']
    print("\nStatistics for key features:")
    for feature in key_features:
        if feature in stats:
            print(f"\n{feature}:")
            print(f"  Reference mean: {stats[feature]['reference']['mean']:.4f}")
            print(f"  Current mean: {stats[feature]['current']['mean']:.4f}")
            print(f"  Mean difference: {stats[feature]['drift']['mean_diff']:.4f} ({stats[feature]['drift']['mean_diff_pct']:.2f}%)")
    
    # Save drift history
    drift_history_path = os.path.join(REPORTS_DIR, "drift_history.json")
    detector.save_drift_history(drift_history_path)
    
    # Get alerts
    alerts = detector.get_drift_alerts(current_data)
    if alerts:
        print("\n" + "="*50)
        print("DRIFT ALERTS")
        print("="*50)
        for alert in alerts:
            print(alert)
    
    print("\n" + "="*50)
    print("Drift monitoring complete!")
    print(f"Report saved to: {report_path}")
    print("="*50)


if __name__ == "__main__":
    main()
