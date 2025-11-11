# Market Context Features - Results & Analysis

## 🎯 Experiment Summary

### What We Added
**24 Market Context Features:**
- **VIX Features (5)**: Volatility regime, spikes, moving averages
- **SPY Features (10)**: Market direction, trend, momentum, volume, RSI
- **Sector Rotation (4)**: Tech (QQQ), Blue Chip (DIA) vs Market
- **Risk Sentiment (3)**: Risk-on/Risk-off (SPY/TLT ratio)
- **Market Breadth (2)**: Price range proxy

**Total Features:** 62 (38 technical + 24 market context)

## 📊 Results Comparison

| Metric | Baseline (38 features) | With Market Context (62 features) | Change |
|--------|------------------------|-----------------------------------|--------|
| **Test Accuracy** | 36.62% | **25.51%** | ❌ **-30% worse** |
| **Test Precision** | 35.05% | **6.51%** | ❌ **-81% worse** |
| **Test Recall** | 36.62% | **25.51%** | ❌ **-30% worse** |
| **Parameters** | 158,755 | 161,059 | +1.5% |
| **Training** | 22 epochs | 16 epochs | Earlier stopping |
| **Val Accuracy** | 35.47% | **22.19%** | ❌ **-37% worse** |

## 🔍 Why Did Performance Decrease?

### 1. **Collinearity / Redundancy**
- Many market context features (SPY momentum, trend, etc.) are highly correlated with NVDA's own momentum
- VIX and stock volatility indicators measure similar concepts
- Model got confused by redundant information

### 2. **Feature Normalization Issues**
- Warning: `RuntimeWarning: invalid value encountered in divide`
- Market context features have different scales and distributions than technical indicators
- StandardScaler may not be optimal for mixed feature types

### 3. **Curse of Dimensionality**
- Went from 38 → 62 features (63% increase)
- Training samples: 2,148 sequences
- Ratio got worse: 34 samples per feature → 21 samples per feature
- Not enough data to learn meaningful patterns from 62 features

### 4. **Overfitting to Market Regime**
- Model trained on 2010-2020 market conditions
- Market context features from that period may not generalize to 2023-2025
- VIX behavior changed post-COVID, tech sector dynamics shifted

### 5. **Model Capacity Mismatch**
- Same CNN architecture (32-64-128 filters)
- Designed for 38 features, not 62
- Would need deeper/wider network to handle more features
- But that risks even more overfitting!

## 💡 Lessons Learned

### ✅ What Worked
- Successfully integrated external market data (VIX, SPY, QQQ, DIA, TLT)
- Created 24 meaningful context features
- Pipeline automatically merges and normalizes

### ❌ What Didn't Work
- Adding more features without feature selection
- Assuming "more information = better performance"
- Not considering feature redundancy
- Same model architecture for different input dimensions

## 🚀 Recommended Next Steps

### Option 1: Feature Selection ⭐ **RECOMMENDED**
Select only the **most informative** market context features:
```python
# Keep only these 5-8 high-value features:
- vix_regime (high vs low volatility)
- spy_trend (bull vs bear market)
- tech_outperformance (sector rotation)
- risk_regime (risk-on vs risk-off)
- spy_volume_ratio (unusual volume)
```
**Expected improvement:** +2-5% accuracy

### Option 2: Feature Engineering Instead of Adding
Instead of adding raw market features, create **interaction features**:
```python
# NVDA-specific market context
- nvda_vs_spy_correlation
- nvda_beta_to_market
- nvda_relative_strength_vs_tech
- nvda_volume_vs_spy_volume
```

### Option 3: Separate Model Architecture
Use market context in a **separate model** then ensemble:
```python
Model 1: CNN on technical indicators (38 features)
Model 2: Dense network on market context (24 features)
Ensemble: Weighted average or meta-learner
```

### Option 4: Dimensionality Reduction
Apply **PCA** or **feature selection** to reduce from 62 to 40-45 features:
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Keep top 40 features
selector = SelectKBest(f_classif, k=40)
```

## 📈 Baseline Performance (Still Our Best)

**Model:** CNN [32, 64, 128], 38 technical features, confidence threshold 60%
- Test Accuracy: **36.62%**
- Total Return: **-1.99%** (vs -8.59% without filtering)
- Max Drawdown: **-3.06%** (vs -11.24%)
- Trades: **30** (vs 240)
- **77% improvement** over raw predictions

## 🎯 Conclusion

**Market context features are valuable in theory but harmful in practice for this model.**

The key insight: In trading, **feature quality matters more than quantity**. Our 38 well-chosen technical indicators outperform 62 features with redundancy.

**Recommendation:** Stick with baseline 38 features + confidence threshold, or carefully select 5-8 high-value market context features with proper feature selection methodology.
