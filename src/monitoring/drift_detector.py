"""Data drift detection module for production monitoring."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns
import json
from datetime import datetime


class DriftDetector:
    """Detect data drift in production data."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.1
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference (training) data
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.drift_history = []
        
    def kolmogorov_smirnov_test(
        self,
        current_data: pd.DataFrame,
        feature: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for a single feature.
        
        Args:
            current_data: Current production data
            feature: Feature name to test
            alpha: Significance level
        
        Returns:
            Dictionary with test results
        """
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            return {"error": f"Feature {feature} not found in data"}
        
        ref_values = self.reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()
        
        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        
        return {
            "feature": feature,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < alpha,
            "alpha": alpha
        }
    
    def chi_squared_test(
        self,
        current_data: pd.DataFrame,
        feature: str,
        alpha: float = 0.05,
        bins: int = 10
    ) -> Dict:
        """
        Perform Chi-squared test for a single feature.
        
        Args:
            current_data: Current production data
            feature: Feature name to test
            alpha: Significance level
            bins: Number of bins for discretization
        
        Returns:
            Dictionary with test results
        """
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            return {"error": f"Feature {feature} not found in data"}
        
        ref_values = self.reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()
        
        # Create bins
        all_values = pd.concat([ref_values, cur_values])
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
        
        ref_hist, _ = np.histogram(ref_values, bins=bin_edges)
        cur_hist, _ = np.histogram(cur_values, bins=bin_edges)
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        # Chi-squared test
        statistic, p_value = stats.chisquare(cur_hist + 1e-10, ref_hist + 1e-10)
        
        return {
            "feature": feature,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < alpha,
            "alpha": alpha
        }
    
    def detect_drift_all_features(
        self,
        current_data: pd.DataFrame,
        method: str = "ks",
        alpha: float = 0.05
    ) -> Dict:
        """
        Detect drift across all features.
        
        Args:
            current_data: Current production data
            method: Test method ('ks' or 'chi2')
            alpha: Significance level
        
        Returns:
            Dictionary with drift results for all features
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "alpha": alpha,
            "features": {},
            "summary": {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        drifted_features = []
        
        for feature in numeric_cols:
            if feature in current_data.columns:
                if method == "ks":
                    test_result = self.kolmogorov_smirnov_test(current_data, feature, alpha)
                else:
                    test_result = self.chi_squared_test(current_data, feature, alpha)
                
                results["features"][feature] = test_result
                
                if test_result.get("drift_detected", False):
                    drifted_features.append(feature)
        
        # Summary
        total_features = len(results["features"])
        num_drifted = len(drifted_features)
        drift_percentage = (num_drifted / total_features * 100) if total_features > 0 else 0
        
        results["summary"] = {
            "total_features": total_features,
            "drifted_features": num_drifted,
            "drift_percentage": drift_percentage,
            "drifted_feature_names": drifted_features,
            "overall_drift_detected": drift_percentage > self.drift_threshold * 100
        }
        
        self.drift_history.append(results)
        return results
    
    def generate_evidently_report(
        self,
        current_data: pd.DataFrame,
        report_path: str = "reports/drift_report.html"
    ):
        """
        Generate Evidently drift report.
        
        Args:
            current_data: Current production data
            report_path: Path to save the report
        """
        # Create reports directory
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Generate report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        report.save_html(report_path)
        print(f"Drift report saved to: {report_path}")
        
        return report
    
    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
        max_drift_share: float = 0.3
    ) -> Dict:
        """
        Run drift test suite.
        
        Args:
            current_data: Current production data
            max_drift_share: Maximum allowed share of drifted features
        
        Returns:
            Test results dictionary
        """
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=max_drift_share)
        ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        return test_suite.as_dict()
    
    def calculate_feature_statistics(
        self,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Calculate comparative statistics between reference and current data.
        
        Args:
            current_data: Current production data
        
        Returns:
            Dictionary with comparative statistics
        """
        stats_comparison = {}
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_cols:
            if feature in current_data.columns:
                ref_values = self.reference_data[feature].dropna()
                cur_values = current_data[feature].dropna()
                
                stats_comparison[feature] = {
                    "reference": {
                        "mean": float(ref_values.mean()),
                        "std": float(ref_values.std()),
                        "min": float(ref_values.min()),
                        "max": float(ref_values.max()),
                        "median": float(ref_values.median())
                    },
                    "current": {
                        "mean": float(cur_values.mean()),
                        "std": float(cur_values.std()),
                        "min": float(cur_values.min()),
                        "max": float(cur_values.max()),
                        "median": float(cur_values.median())
                    },
                    "drift": {
                        "mean_diff": float(cur_values.mean() - ref_values.mean()),
                        "std_diff": float(cur_values.std() - ref_values.std()),
                        "mean_diff_pct": float((cur_values.mean() - ref_values.mean()) / ref_values.mean() * 100) if ref_values.mean() != 0 else 0
                    }
                }
        
        return stats_comparison
    
    def save_drift_history(self, filepath: str = "reports/drift_history.json"):
        """Save drift detection history to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        
        print(f"Drift history saved to: {filepath}")
    
    def get_drift_alerts(self, current_data: pd.DataFrame) -> List[str]:
        """
        Get list of drift alerts.
        
        Args:
            current_data: Current production data
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        drift_results = self.detect_drift_all_features(current_data)
        
        if drift_results["summary"]["overall_drift_detected"]:
            alerts.append(
                f"ALERT: Overall drift detected! {drift_results['summary']['drift_percentage']:.2f}% of features drifted."
            )
            
            for feature in drift_results["summary"]["drifted_feature_names"]:
                alerts.append(f"  - Feature '{feature}' shows significant drift")
        
        return alerts
