"""Test script for the trading API."""

import requests
import json
from datetime import datetime, timedelta


def test_api_endpoints():
    """Test API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing DL Trading API...")
    print("="*60)
    
    # Test root endpoint
    print("\n1. Testing root endpoint (GET /)...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint (GET /health)...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test model info endpoint
    print("\n3. Testing model info endpoint (GET /model-info)...")
    response = requests.get(f"{base_url}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test prediction endpoint with sample data
    print("\n4. Testing prediction endpoint (POST /predict)...")
    
    # Generate sample market data (at least 20 points for sequence)
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    base_price = 100.0
    base_volume = 1000000.0
    
    for i in range(30):
        timestamp = (datetime.now() - timedelta(hours=30-i)).strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(timestamp)
        
        # Simulate price movement
        price = base_price + (i * 0.5) + ((i % 3) - 1) * 0.2
        opens.append(price)
        highs.append(price + 0.5)
        lows.append(price - 0.5)
        closes.append(price + 0.1)
        volumes.append(base_volume + (i * 10000))
    
    market_data = {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=market_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of predictions: {len(result['predictions'])}")
            print(f"Sample predictions (first 5):")
            for i in range(min(5, len(result['predictions']))):
                print(f"  {result['timestamp'][i]}: {result['predictions'][i]} (confidence: {result['confidence'][i]:.4f})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error making prediction request: {str(e)}")
    
    print("\n" + "="*60)
    print("API testing complete!")


if __name__ == "__main__":
    test_api_endpoints()
