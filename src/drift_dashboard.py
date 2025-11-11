"""
Data Drift Monitoring Dashboard

Streamlit dashboard for monitoring data drift across train/test/validation splits.
Uses KS-test to detect distribution changes and visualizes feature distributions over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ks_2samp
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor data drift across train/test/validation splits."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize drift monitor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.significance_level = self.config['monitoring']['significance_level']
        self.top_n = self.config['monitoring']['top_n_features']
        
    def load_data(self):
        """Load train, test, and validation datasets."""
        logger.info("Loading datasets...")
        
        train_df = pd.read_csv("data/processed/train_features.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test_features.csv", index_col=0, parse_dates=True)
        val_df = pd.read_csv("data/processed/val_features.csv", index_col=0, parse_dates=True)
        
        # Load feature names
        with open("data/processed/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Val: {val_df.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
        return train_df, test_df, val_df, feature_names
    
    def compute_ks_statistics(self, train_df, test_df, val_df, feature_names):
        """
        Compute KS-test statistics for all features.
        
        Args:
            train_df: Training data
            test_df: Test data
            val_df: Validation data
            feature_names: List of feature names
        
        Returns:
            DataFrame with KS statistics and p-values
        """
        logger.info("Computing KS-test statistics...")
        
        results = []
        
        for feature in feature_names:
            # Train vs Test
            ks_stat_test, p_value_test = ks_2samp(train_df[feature], test_df[feature])
            
            # Train vs Val
            ks_stat_val, p_value_val = ks_2samp(train_df[feature], val_df[feature])
            
            # Test vs Val
            ks_stat_test_val, p_value_test_val = ks_2samp(test_df[feature], val_df[feature])
            
            # Detect drift (p-value < significance level)
            drift_test = p_value_test < self.significance_level
            drift_val = p_value_val < self.significance_level
            drift_test_val = p_value_test_val < self.significance_level
            
            results.append({
                'feature': feature,
                'ks_train_test': ks_stat_test,
                'p_value_train_test': p_value_test,
                'drift_train_test': drift_test,
                'ks_train_val': ks_stat_val,
                'p_value_train_val': p_value_val,
                'drift_train_val': drift_val,
                'ks_test_val': ks_stat_test_val,
                'p_value_test_val': p_value_test_val,
                'drift_test_val': drift_test_val,
                'max_ks_stat': max(ks_stat_test, ks_stat_val, ks_stat_test_val)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('max_ks_stat', ascending=False)
        
        logger.info(f"KS-test completed. Detected drift in {results_df['drift_train_test'].sum()} features (train vs test)")
        
        return results_df
    
    def plot_feature_distribution(self, train_df, test_df, val_df, feature_name):
        """
        Plot distribution comparison for a specific feature.
        
        Args:
            train_df: Training data
            test_df: Test data
            val_df: Validation data
            feature_name: Feature to plot
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add histograms
        fig.add_trace(go.Histogram(
            x=train_df[feature_name],
            name='Train',
            opacity=0.6,
            nbinsx=50,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Histogram(
            x=test_df[feature_name],
            name='Test',
            opacity=0.6,
            nbinsx=50,
            marker_color='orange'
        ))
        
        fig.add_trace(go.Histogram(
            x=val_df[feature_name],
            name='Validation',
            opacity=0.6,
            nbinsx=50,
            marker_color='green'
        ))
        
        fig.update_layout(
            title=f'Distribution Comparison: {feature_name}',
            xaxis_title=feature_name,
            yaxis_title='Frequency',
            barmode='overlay',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_timeline_distributions(self, train_df, test_df, val_df, feature_name):
        """
        Plot feature values over time (timeline view).
        
        Args:
            train_df: Training data
            test_df: Test data
            val_df: Validation data
            feature_name: Feature to plot
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Train data
        fig.add_trace(go.Scatter(
            x=train_df.index,
            y=train_df[feature_name],
            mode='lines',
            name='Train',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))
        
        # Test data
        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df[feature_name],
            mode='lines',
            name='Test',
            line=dict(color='orange', width=1),
            opacity=0.7
        ))
        
        # Validation data
        fig.add_trace(go.Scatter(
            x=val_df.index,
            y=val_df[feature_name],
            mode='lines',
            name='Validation',
            line=dict(color='green', width=1),
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Timeline View: {feature_name}',
            xaxis_title='Date',
            yaxis_title=feature_name,
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_drift_heatmap(self, ks_results_df):
        """
        Create heatmap of KS statistics across features and comparisons.
        
        Args:
            ks_results_df: DataFrame with KS statistics
        
        Returns:
            Plotly figure
        """
        # Select top N features
        top_features = ks_results_df.head(self.top_n * 2)
        
        # Prepare data for heatmap
        heatmap_data = top_features[['feature', 'ks_train_test', 'ks_train_val', 'ks_test_val']].set_index('feature')
        heatmap_data.columns = ['Train vs Test', 'Train vs Val', 'Test vs Val']
        
        fig = px.imshow(
            heatmap_data.T,
            labels=dict(x="Feature", y="Comparison", color="KS Statistic"),
            color_continuous_scale="Reds",
            aspect="auto"
        )
        
        fig.update_layout(
            title=f'Top {len(top_features)} Features by KS Statistic',
            height=300
        )
        
        return fig
    
    def get_drift_interpretation(self, feature_name, ks_stat, p_value, has_drift):
        """
        Generate interpretation text for drift detection.
        
        Args:
            feature_name: Name of the feature
            ks_stat: KS statistic value
            p_value: P-value from KS-test
            has_drift: Boolean indicating if drift detected
        
        Returns:
            Interpretation string
        """
        if has_drift:
            severity = "High" if ks_stat > 0.3 else "Moderate" if ks_stat > 0.15 else "Low"
            return f"⚠️ **{severity} Drift Detected** - KS statistic: {ks_stat:.4f}, p-value: {p_value:.4e}"
        else:
            return f"✅ **No Significant Drift** - KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Data Drift Monitoring",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Data Drift Monitoring Dashboard")
    st.markdown("Monitor distribution drift across train, test, and validation datasets using KS-test")
    
    # Initialize monitor
    monitor = DriftMonitor()
    
    # Load data
    with st.spinner("Loading datasets..."):
        train_df, test_df, val_df, feature_names = monitor.load_data()
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", len(train_df))
        st.caption(f"Period: {train_df.index.min().date()} to {train_df.index.max().date()}")
    with col2:
        st.metric("Test Samples", len(test_df))
        st.caption(f"Period: {test_df.index.min().date()} to {test_df.index.max().date()}")
    with col3:
        st.metric("Validation Samples", len(val_df))
        st.caption(f"Period: {val_df.index.min().date()} to {val_df.index.max().date()}")
    
    st.divider()
    
    # Compute KS statistics
    with st.spinner("Computing KS-test statistics..."):
        ks_results = monitor.compute_ks_statistics(train_df, test_df, val_df, feature_names)
    
    # Summary metrics
    st.subheader("📈 Drift Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drift_count_test = ks_results['drift_train_test'].sum()
        st.metric("Features with Drift (Train vs Test)", drift_count_test)
    
    with col2:
        drift_count_val = ks_results['drift_train_val'].sum()
        st.metric("Features with Drift (Train vs Val)", drift_count_val)
    
    with col3:
        drift_count_test_val = ks_results['drift_test_val'].sum()
        st.metric("Features with Drift (Test vs Val)", drift_count_test_val)
    
    with col4:
        max_ks = ks_results['max_ks_stat'].max()
        st.metric("Max KS Statistic", f"{max_ks:.4f}")
    
    st.divider()
    
    # Drift heatmap
    st.subheader("🔥 Drift Heatmap")
    fig_heatmap = monitor.plot_drift_heatmap(ks_results)
    st.plotly_chart(fig_heatmap, width='stretch')
    
    st.divider()
    
    # Top drifted features table
    st.subheader(f"🔝 Top {monitor.top_n} Drifted Features")
    top_drifted = ks_results.head(monitor.top_n)
    
    # Format table for display
    display_df = top_drifted[['feature', 'ks_train_test', 'p_value_train_test', 'drift_train_test',
                               'ks_train_val', 'p_value_train_val', 'drift_train_val']].copy()
    display_df.columns = ['Feature', 'KS (Train-Test)', 'p-value (Train-Test)', 'Drift?', 
                          'KS (Train-Val)', 'p-value (Train-Val)', 'Drift?']
    
    # Rename second Drift? column to avoid duplicates
    display_df.columns = ['Feature', 'KS (Train-Test)', 'p-value (Train-Test)', 'Drift (T-T)?', 
                          'KS (Train-Val)', 'p-value (Train-Val)', 'Drift (T-V)?']
    
    # Style the dataframe
    st.dataframe(
        display_df.style.highlight_max(subset=['KS (Train-Test)', 'KS (Train-Val)'], color='lightcoral')
                       .format({'KS (Train-Test)': '{:.4f}', 'KS (Train-Val)': '{:.4f}',
                               'p-value (Train-Test)': '{:.4e}', 'p-value (Train-Val)': '{:.4e}'}),
        width='stretch'
    )
    
    st.divider()
    
    # Feature-specific analysis
    st.subheader("🔍 Feature-Specific Analysis")
    
    selected_feature = st.selectbox(
        "Select feature to analyze:",
        feature_names,
        index=feature_names.index(top_drifted.iloc[0]['feature'])
    )
    
    # Get statistics for selected feature
    feature_stats = ks_results[ks_results['feature'] == selected_feature].iloc[0]
    
    # Display interpretations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Train vs Test**")
        interpretation = monitor.get_drift_interpretation(
            selected_feature,
            feature_stats['ks_train_test'],
            feature_stats['p_value_train_test'],
            feature_stats['drift_train_test']
        )
        st.markdown(interpretation)
    
    with col2:
        st.markdown("**Train vs Validation**")
        interpretation = monitor.get_drift_interpretation(
            selected_feature,
            feature_stats['ks_train_val'],
            feature_stats['p_value_train_val'],
            feature_stats['drift_train_val']
        )
        st.markdown(interpretation)
    
    # Timeline plot
    st.markdown("**Timeline View**")
    fig_timeline = monitor.plot_timeline_distributions(train_df, test_df, val_df, selected_feature)
    st.plotly_chart(fig_timeline, width='stretch')
    
    # Distribution plot
    st.markdown("**Distribution Comparison**")
    fig_dist = monitor.plot_feature_distribution(train_df, test_df, val_df, selected_feature)
    st.plotly_chart(fig_dist, width='stretch')
    
    # Statistical summary
    st.markdown("**Statistical Summary**")
    stats_df = pd.DataFrame({
        'Dataset': ['Train', 'Test', 'Validation'],
        'Mean': [train_df[selected_feature].mean(), test_df[selected_feature].mean(), val_df[selected_feature].mean()],
        'Std': [train_df[selected_feature].std(), test_df[selected_feature].std(), val_df[selected_feature].std()],
        'Min': [train_df[selected_feature].min(), test_df[selected_feature].min(), val_df[selected_feature].min()],
        'Max': [train_df[selected_feature].max(), test_df[selected_feature].max(), val_df[selected_feature].max()]
    })
    st.dataframe(stats_df.style.format({'Mean': '{:.4f}', 'Std': '{:.4f}', 'Min': '{:.4f}', 'Max': '{:.4f}'}))
    
    st.divider()
    
    # Full KS statistics table
    with st.expander("📋 View Full KS Statistics Table"):
        st.dataframe(
            ks_results.style.format({
                'ks_train_test': '{:.4f}',
                'p_value_train_test': '{:.4e}',
                'ks_train_val': '{:.4f}',
                'p_value_train_val': '{:.4e}',
                'ks_test_val': '{:.4f}',
                'p_value_test_val': '{:.4e}',
                'max_ks_stat': '{:.4f}'
            }),
            width='stretch'
        )


if __name__ == "__main__":
    main()
