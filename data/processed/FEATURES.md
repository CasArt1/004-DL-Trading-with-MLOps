# Feature Engineering Summary

## Overview
Successfully created **38 technical indicators** from NVDA OHLCV data.

## Feature Categories

### Momentum Indicators (11 features)
1. `rsi_14` - Relative Strength Index (14-day)
2. `rsi_21` - Relative Strength Index (21-day)
3. `macd` - MACD line
4. `macd_signal` - MACD signal line
5. `macd_diff` - MACD histogram
6. `roc_12` - Rate of Change (12-day)
7. `roc_25` - Rate of Change (25-day)
8. `stoch_k` - Stochastic %K
9. `stoch_d` - Stochastic %D
10. `williams_r` - Williams %R
11. `ao` - Awesome Oscillator

### Volatility Indicators (11 features)
12. `atr_14` - Average True Range
13. `bb_high` - Bollinger Band Upper
14. `bb_low` - Bollinger Band Lower
15. `bb_mid` - Bollinger Band Middle
16. `bb_width` - Bollinger Band Width
17. `bb_pct` - Bollinger Band %B
18. `kc_high` - Keltner Channel Upper
19. `kc_low` - Keltner Channel Lower
20. `kc_mid` - Keltner Channel Middle
21. `volatility_20` - 20-day Historical Volatility
22. `volatility_50` - 50-day Historical Volatility

### Volume Indicators (8 features)
23. `obv` - On-Balance Volume
24. `volume_ma_10` - 10-day Volume Moving Average
25. `volume_ma_20` - 20-day Volume Moving Average
26. `volume_roc` - Volume Rate of Change
27. `adi` - Accumulation/Distribution Index
28. `cmf` - Chaikin Money Flow
29. `force_index` - Force Index
30. `vwap` - Volume Weighted Average Price

### Trend Indicators (8 features)
31. `sma_10` - Simple Moving Average (10-day)
32. `sma_20` - Simple Moving Average (20-day)
33. `sma_50` - Simple Moving Average (50-day)
34. `ema_12` - Exponential Moving Average (12-day)
35. `ema_26` - Exponential Moving Average (26-day)
36. `adx` - Average Directional Index
37. `adx_pos` - ADX Positive Directional Indicator
38. `adx_neg` - ADX Negative Directional Indicator

## Normalization
- **Method**: Standard Scaler (z-score normalization)
- **Fit on**: Training set only (prevents data leakage)
- **Applied to**: Train, Test, and Validation sets
- **Scaler saved**: `models/feature_scaler.pkl`

## Data Shapes
- **Training**: 2,213 samples × 38 features
- **Testing**: 704 samples × 38 features
- **Validation**: 705 samples × 38 features

## Output Files
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `data/processed/val_features.csv`
- `data/processed/feature_names.txt`
- `models/feature_scaler.pkl`

## Notes
- 50 rows dropped per split due to NaN values from rolling window calculations
- All features are properly normalized to prevent scale bias in the CNN model
- No look-ahead bias: scaler fit only on training data
