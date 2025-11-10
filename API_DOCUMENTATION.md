# API Documentation

This document describes the REST API endpoints for the DL Trading system.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint

**GET /**

Get basic API information and status.

**Response:**
```json
{
  "message": "DL Trading API",
  "version": "0.1.0",
  "status": "running",
  "model_loaded": true
}
```

### 2. Health Check

**GET /health**

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Load Model

**POST /load-model**

Load or reload the trained model.

**Query Parameters:**
- `model_path` (optional): Path to model file (default: `models/best_cnn_model.keras`)
- `scaler_path` (optional): Path to scaler file (default: `models/best_cnn_model_scaler.pkl`)

**Response:**
```json
{
  "status": "success",
  "message": "Model loaded successfully"
}
```

### 4. Get Predictions

**POST /predict**

Generate trading signal predictions from market data.

**Request Body:**
```json
{
  "timestamp": [
    "2024-01-01 00:00:00",
    "2024-01-01 01:00:00",
    ...
  ],
  "open": [100.0, 101.0, ...],
  "high": [102.0, 103.0, ...],
  "low": [99.0, 100.0, ...],
  "close": [101.0, 102.0, ...],
  "volume": [1000000.0, 1100000.0, ...]
}
```

**Requirements:**
- At least 20 data points (for sequence generation)
- All arrays must have the same length
- Timestamps should be in chronological order

**Response:**
```json
{
  "predictions": ["BUY", "HOLD", "SELL", ...],
  "probabilities": [
    {
      "HOLD": 0.2,
      "BUY": 0.7,
      "SELL": 0.1
    },
    ...
  ],
  "confidence": [0.7, 0.65, 0.8, ...],
  "timestamp": [
    "2024-01-01 00:00:00",
    "2024-01-01 01:00:00",
    ...
  ]
}
```

**Prediction Classes:**
- `BUY`: Model predicts price will increase (signal to enter long position)
- `HOLD`: Model predicts no significant movement
- `SELL`: Model predicts price will decrease (signal to exit position or enter short)

### 5. Model Information

**GET /model-info**

Get information about the loaded model.

**Response:**
```json
{
  "status": "Model loaded",
  "model_type": "CNN",
  "sequence_length": 20,
  "input_shape": [null, 20, 150],
  "output_classes": ["HOLD", "BUY", "SELL"]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Need at least 20 data points, got 15"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error loading model: Model file not found"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded. Please load model first via /load-model endpoint."
}
```

## Usage Examples

### Python

```python
import requests

# Get predictions
market_data = {
    "timestamp": ["2024-01-01 00:00:00", ...],
    "open": [100.0, ...],
    "high": [102.0, ...],
    "low": [99.0, ...],
    "close": [101.0, ...],
    "volume": [1000000.0, ...]
}

response = requests.post(
    "http://localhost:8000/predict",
    json=market_data
)

predictions = response.json()
print(predictions["predictions"])
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @market_data.json
```

### JavaScript

```javascript
const marketData = {
  timestamp: ["2024-01-01 00:00:00", ...],
  open: [100.0, ...],
  high: [102.0, ...],
  low: [99.0, ...],
  close: [101.0, ...],
  volume: [1000000.0, ...]
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(marketData)
})
.then(response => response.json())
.then(data => console.log(data.predictions));
```

## Interactive Documentation

The API includes interactive Swagger documentation at:

```
http://localhost:8000/docs
```

You can test all endpoints directly from the browser using this interface.

## Rate Limiting

Currently, there is no rate limiting implemented. For production use, consider adding rate limiting to prevent abuse.

## Authentication

Currently, the API does not require authentication. For production use, implement proper authentication and authorization.

## Best Practices

1. **Data Quality**: Ensure input data is clean and properly formatted
2. **Sequence Length**: Always provide at least 20 data points for predictions
3. **Error Handling**: Always check response status codes and handle errors appropriately
4. **Caching**: Consider caching predictions to reduce API calls
5. **Monitoring**: Log API calls and predictions for monitoring and debugging

## Support

For API issues or questions, please open an issue on GitHub:
https://github.com/CasArt1/004-DL-Trading-with-MLOps/issues
