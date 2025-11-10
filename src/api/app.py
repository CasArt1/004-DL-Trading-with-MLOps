"""FastAPI application for serving CNN trading predictions."""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow import keras
import joblib


app = FastAPI(
    title="DL Trading API",
    description="API for deep learning trading signal predictions",
    version="0.1.0"
)


class MarketData(BaseModel):
    """Market data input schema."""
    timestamp: List[str] = Field(..., description="List of timestamps")
    open: List[float] = Field(..., description="Opening prices")
    high: List[float] = Field(..., description="High prices")
    low: List[float] = Field(..., description="Low prices")
    close: List[float] = Field(..., description="Closing prices")
    volume: List[float] = Field(..., description="Trading volumes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000.0, 1100000.0]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    predictions: List[str] = Field(..., description="Predicted trading signals")
    probabilities: List[Dict[str, float]] = Field(..., description="Prediction probabilities")
    confidence: List[float] = Field(..., description="Confidence scores")
    timestamp: List[str] = Field(..., description="Corresponding timestamps")


class ModelPredictor:
    """Model predictor singleton."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.sequence_length = 20
        
    def load_model(self, model_path: str = "models/best_cnn_model.keras", 
                   scaler_path: str = "models/best_cnn_model_scaler.pkl"):
        """Load the trained model and scaler."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        try:
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare basic features from raw OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with calculated features
        """
        # Calculate basic indicators
        data['Returns'] = data['Close'].pct_change()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = sma_20 + (std_20 * 2)
        data['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # Volume indicators
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        
        return data
    
    def predict(self, market_data: MarketData) -> PredictionResponse:
        """
        Make predictions on market data.
        
        Args:
            market_data: Market data input
        
        Returns:
            Prediction response
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert input to DataFrame
        df = pd.DataFrame({
            'Open': market_data.open,
            'High': market_data.high,
            'Low': market_data.low,
            'Close': market_data.close,
            'Volume': market_data.volume
        }, index=pd.to_datetime(market_data.timestamp))
        
        # Prepare features
        df_with_features = self.prepare_features(df)
        
        # Select numeric features
        numeric_features = df_with_features.select_dtypes(include=[np.number]).values
        
        # Check if we have enough data
        if len(numeric_features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points, got {len(numeric_features)}")
        
        # Scale features
        scaled_features = self.scaler.transform(numeric_features)
        
        # Create sequences
        predictions = []
        probabilities = []
        confidence_scores = []
        timestamps = []
        
        signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        for i in range(self.sequence_length, len(scaled_features)):
            sequence = scaled_features[i - self.sequence_length:i]
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predict
            pred_probs = self.model.predict(sequence, verbose=0)[0]
            pred_class = int(np.argmax(pred_probs))
            
            predictions.append(signal_map[pred_class])
            probabilities.append({
                "HOLD": float(pred_probs[0]),
                "BUY": float(pred_probs[1]),
                "SELL": float(pred_probs[2])
            })
            confidence_scores.append(float(np.max(pred_probs)))
            timestamps.append(df.index[i].strftime("%Y-%m-%d %H:%M:%S"))
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence_scores,
            timestamp=timestamps
        )


# Global predictor instance
predictor = ModelPredictor()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        predictor.load_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {str(e)}")
        print("Model will need to be loaded manually via /load-model endpoint")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DL Trading API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": predictor.model_loaded
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model_loaded
    }


@app.post("/load-model")
async def load_model(model_path: str = "models/best_cnn_model.keras",
                     scaler_path: str = "models/best_cnn_model_scaler.pkl"):
    """
    Load or reload the model.
    
    Args:
        model_path: Path to the model file
        scaler_path: Path to the scaler file
    
    Returns:
        Status message
    """
    try:
        predictor.load_model(model_path, scaler_path)
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(market_data: MarketData):
    """
    Predict trading signals from market data.
    
    Args:
        market_data: Market data with OHLCV values
    
    Returns:
        Trading signal predictions
    """
    try:
        if not predictor.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please load model first via /load-model endpoint."
            )
        
        response = predictor.predict(market_data)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if not predictor.model_loaded:
        return {"status": "No model loaded"}
    
    return {
        "status": "Model loaded",
        "model_type": "CNN",
        "sequence_length": predictor.sequence_length,
        "input_shape": predictor.model.input_shape,
        "output_classes": ["HOLD", "BUY", "SELL"]
    }
