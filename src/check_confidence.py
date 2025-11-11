"""
Diagnostic script to check model prediction confidence values
"""

import numpy as np
import pandas as pd
from tensorflow import keras
import pickle
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model
print("Loading model...")
model = keras.models.load_model('models/best_model.h5')

# Load scaler
print("Loading scaler...")
with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
print("Loading test data...")
test_data = pd.read_csv('data/processed/test_labeled.csv')
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data = test_data.sort_values('Date')

# Load feature names
with open('data/processed/feature_names.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

print(f"Using {len(feature_cols)} features")
X_test = test_data[feature_cols].values

# Scale features
X_scaled = scaler.transform(X_test)

# Create sequences
lookback = config['features']['lookback_window']
sequences = []
valid_indices = []

for i in range(lookback, len(X_scaled)):
    sequences.append(X_scaled[i-lookback:i])
    valid_indices.append(i)

X_sequences = np.array(sequences)

print(f"\nGenerating predictions for {len(X_sequences)} samples...")
predictions = model.predict(X_sequences, verbose=0)

# Get confidence (max probability for each prediction)
confidences = np.max(predictions, axis=1)
predicted_classes = np.argmax(predictions, axis=1)

# Map class indices to labels
class_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
predicted_labels = [class_map[c] for c in predicted_classes]

# Statistics
print("\n" + "="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)
print(f"Total predictions: {len(confidences)}")
print(f"\nConfidence Statistics:")
print(f"  Mean: {confidences.mean():.4f}")
print(f"  Median: {np.median(confidences):.4f}")
print(f"  Min: {confidences.min():.4f}")
print(f"  Max: {confidences.max():.4f}")
print(f"  Std: {confidences.std():.4f}")

print(f"\nConfidence Distribution:")
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    count = np.sum(confidences >= threshold)
    pct = (count / len(confidences)) * 100
    print(f"  >= {threshold:.1f}: {count:4d} ({pct:5.1f}%)")

print(f"\nPredicted Class Distribution:")
for class_idx, class_name in class_map.items():
    count = np.sum(predicted_classes == class_idx)
    pct = (count / len(predicted_classes)) * 100
    avg_conf = confidences[predicted_classes == class_idx].mean() if count > 0 else 0
    print(f"  {class_name:4s}: {count:4d} ({pct:5.1f}%) - Avg Confidence: {avg_conf:.4f}")

# Show some examples
print(f"\nTop 10 Most Confident Predictions:")
top_indices = np.argsort(confidences)[-10:][::-1]
for idx in top_indices:
    print(f"  {predicted_labels[idx]:4s} - Confidence: {confidences[idx]:.4f}")

print(f"\nBottom 10 Least Confident Predictions:")
bottom_indices = np.argsort(confidences)[:10]
for idx in bottom_indices:
    print(f"  {predicted_labels[idx]:4s} - Confidence: {confidences[idx]:.4f}")

# Check specific probability distributions
print(f"\nSample Probability Distributions (first 5 predictions):")
for i in range(min(5, len(predictions))):
    print(f"  Sample {i+1}: SELL={predictions[i][0]:.4f}, HOLD={predictions[i][1]:.4f}, BUY={predictions[i][2]:.4f}")
