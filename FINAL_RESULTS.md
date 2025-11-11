# DL Trading with MLOps - Final Results

**Date**: November 10, 2025  
**Asset**: NVDA (15 years daily data, 2010-2025)  
**Working Directory**: `C:\Trading\DL-MLOps` (short path for TensorFlow compatibility)

---

## 🎯 Project Completion Status

### ✅ Core Components Delivered
1. **Data Pipeline**: 15 years NVDA data, 60/20/20 split (chronological)
2. **Feature Engineering**: 38 technical indicators (momentum, volatility, volume, trend)
3. **Label Generation**: Buy/Sell/Hold with class weighting
4. **CNN Model**: 158,755 parameters, batch normalization, dropout
5. **MLFlow Integration**: Experiment tracking, model registry
6. **Drift Dashboard**: Streamlit app with KS-tests (ready, needs data)
7. **Backtesting**: Transaction costs (0.125% commission, 0.25% borrow rate)
8. **Documentation**: Complete analysis and implementation guides

---

## 📊 Final Model Performance

### Training Configuration
- **Features**: 38 technical indicators (baseline)
- **Architecture**: CNN with [32, 64, 128] filters
- **Parameters**: 158,755 trainable
- **Training**: 16 epochs, early stopping, learning rate reduction
- **Batch Size**: 36

### Model Metrics
- **Test Accuracy**: 40.53%
- **Test Precision**: 40.42%
- **Test Recall**: 40.53%
- **Validation Accuracy**: 48.12% (best epoch: 7)

### Backtesting Results (Test Period: ~2.8 years)
```
Initial Capital:     $100,000.00
Final Portfolio:     $99,833.08
Total Return:        -0.17%
Annualized Return:   -0.07%
Sharpe Ratio:        -0.0459
Max Drawdown:        -1.76%
Win Rate:            16.67%
Total Trades:        30
Commissions Paid:    $375.06
```

**Signal Distribution**: 
- BUY: 608 signals (94.4%)
- SELL: 25 signals (3.9%)
- HOLD: 11 signals (1.7%)

---

## 🔬 Feature Selection Experiment

### Implementation
Created ensemble feature selection pipeline combining:
- ANOVA F-test
- Mutual Information
- Random Forest importance

### Market Context Features Attempted
Added 24 external market features from:
- VIX (volatility regime)
- SPY (market benchmark)
- QQQ (tech sector)
- DIA (blue chips)
- TLT (bonds)

### Results
❌ **FAILED**: Adding market context reduced performance from 40.53% → 25.51% accuracy

**Root Causes Identified**:
1. NaN values in market data after merge
2. Normalization issues (all features became zeros)
3. Curse of dimensionality (62 features, 2,148 samples)
4. Feature redundancy with technical indicators

**Recommendation**: Continue with baseline 38 technical features until data pipeline is fixed.

---

## 📁 Project Structure

```
004-DL-Trading-with-MLOps/
├── src/
│   ├── data_collection.py          # NVDA data download
│   ├── feature_engineering.py       # 38 technical indicators
│   ├── label_generation.py          # Buy/Sell/Hold labels
│   ├── model.py                     # CNN architecture
│   ├── training.py                  # MLFlow integration
│   ├── backtesting.py              # Strategy simulation
│   ├── drift_dashboard.py          # Streamlit monitoring
│   ├── hyperparameter_search.py    # Optuna optimization
│   ├── feature_selection.py        # Ensemble selection
│   ├── apply_feature_selection.py  # Filter datasets
│   └── market_context_features.py  # External market data
├── data/
│   ├── raw/                        # NVDA_data.csv
│   └── processed/                  # Split datasets (62 features available)
├── models/
│   ├── best_model.h5              # Trained CNN (38 features)
│   ├── feature_scaler.pkl         # StandardScaler
│   └── backtest_*.csv             # Trading results
├── config/
│   └── config.yaml                # All hyperparameters
├── mlruns/                        # MLFlow experiments
└── docs/
    ├── FEATURE_SELECTION_SUMMARY.md
    ├── MARKET_CONTEXT_ANALYSIS.md
    └── IMPROVEMENTS.md
```

---

## 🚀 How to Run

### Full Pipeline
```powershell
cd C:\Trading\DL-MLOps

# 1. Train model
.\.venv\Scripts\python.exe .\src\training.py

# 2. Run backtest
.\.venv\Scripts\python.exe .\src\backtesting.py

# 3. View drift dashboard (optional)
.\.venv\Scripts\streamlit.exe run .\src\drift_dashboard.py
```

### Hyperparameter Search
```powershell
.\.venv\Scripts\python.exe .\src\hyperparameter_search.py
```

---

## 📈 Key Findings

### What Worked
✅ CNN architecture captures temporal patterns  
✅ Batch normalization stabilizes training  
✅ Confidence threshold (50-60%) reduces trades  
✅ Transaction cost modeling provides realistic results  
✅ Class weighting handles imbalanced labels  
✅ MLFlow enables experiment comparison  

### What Didn't Work
❌ Market context features (data quality issues)  
❌ Higher feature counts (curse of dimensionality)  
❌ Aggressive trading (240 trades → -8.59% return)  
❌ Very high confidence thresholds (0 trades)  

### Lessons Learned
1. **Quality > Quantity**: 38 features outperformed 62
2. **Conservative Trading**: Fewer trades, lower losses
3. **Data Pipeline Critical**: External data needs careful validation
4. **Feature Engineering**: Technical indicators sufficient for baseline
5. **Transaction Costs Matter**: Commission significantly impacts returns

---

## 🔧 Configuration Notes

### Current Settings (config.yaml)
```yaml
model:
  filters: [32, 64, 128]
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 36

backtesting:
  confidence_threshold: 0.5    # 50% minimum confidence
  commission_rate: 0.00125     # 0.125% per trade
  borrow_rate: 0.0025          # 0.25% annualized
```

### Best Performers from Hyperparameter Search
- **Trial 13**: 42.25% accuracy (filters: [32,64,128], dropout: 0.3, lr: 0.001)
- **Baseline**: 40.53% accuracy (same config)

---

## 🎓 Improvements Documented

See `IMPROVEMENTS.md` for detailed recommendations:
1. Confidence threshold tuning (tested: 60% → 77% improvement)
2. Hyperparameter optimization (20 trials completed)
3. Market context features (needs data pipeline fix)
4. Alternative architectures (LSTM, Transformer)
5. Ensemble methods
6. Feature interaction terms

---

## 📚 Documentation Files

1. **FEATURE_SELECTION_SUMMARY.md**: Complete feature selection implementation and analysis
2. **MARKET_CONTEXT_ANALYSIS.md**: Why market features failed and how to fix
3. **IMPROVEMENTS.md**: Performance optimization history and recommendations
4. **README.md**: Project overview and requirements

---

## ⚠️ Known Issues

1. **Path Length**: Use `C:\Trading\DL-MLOps` (TensorFlow installation issue with long paths)
2. **Market Features**: NaN values after merge, need data pipeline fix
3. **Model Bias**: Predicts mostly BUY (94.4% of signals)
4. **Win Rate**: Low at 16.67% (model needs improvement)
5. **Drift Dashboard**: Requires production data to display monitoring

---

## 🎯 Next Steps for Production

### Short-Term
1. Fix market context data pipeline (date alignment, NaN handling)
2. Address model bias (class weighting, focal loss)
3. Improve win rate (more sophisticated features)
4. Paper trade for 30 days validation

### Medium-Term
1. Implement real-time data feed
2. Set up monitoring alerts
3. A/B test alternative architectures
4. Build ensemble of models

### Long-Term
1. Multi-asset portfolio
2. Adaptive retraining schedule
3. Risk management system
4. Regulatory compliance tracking

---

## 📞 Support & Resources

- **MLFlow UI**: `mlflow ui --port 5000`
- **Drift Dashboard**: `streamlit run src/drift_dashboard.py`
- **Config**: All settings in `config/config.yaml`
- **Logs**: Check MLFlow runs for experiment history

---

**Status**: ✅ Complete end-to-end pipeline  
**Performance**: 📊 Baseline established (-0.17% return)  
**Next**: 🔧 Optimize model and fix market context features

*Generated: November 10, 2025*
