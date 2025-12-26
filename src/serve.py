"""
Model Serving API using FastAPI

Production-grade REST API for serving ML predictions with:
- Input validation
- Error handling
- Logging
- Health checks
- API documentation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional
import yaml
import json
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global variables for model and scaler
model = None
scaler = None
feature_names = None
model_metadata = None


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_artifacts():
    """Load trained model, scaler, and metadata."""
    global model, scaler, feature_names, model_metadata
    
    config = load_config()
    models_dir = Path(config['paths']['models_dir'])
    algorithm = config['model']['algorithm']
    
    try:
        # Load model
        model_path = models_dir / f"model_{algorithm}.pkl"
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded model from {model_path}")
        
        # Load scaler
        scaler_path = models_dir / "scaler.pkl"
        scaler = joblib.load(scaler_path)
        logger.info(f"✅ Loaded scaler from {scaler_path}")
        
        # Load metadata
        metadata_path = models_dir / f"metadata_{algorithm}.json"
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        # Get feature names from processed data metadata
        data_metadata_path = Path(config['data']['processed_data_path']) / "metadata.json"
        with open(data_metadata_path, 'r') as f:
            data_meta = json.load(f)
            feature_names = data_meta['feature_names']
        
        logger.info(f"✅ Loaded metadata: {model_metadata['metrics']}")
        logger.info(f"✅ Expected features: {len(feature_names)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("🚀 Starting FastAPI application...")
    load_model_and_artifacts()
    logger.info("✅ Application ready to serve predictions")
    yield
    # Shutdown
    logger.info("👋 Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Production-grade API for serving machine learning predictions",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input schema for prediction requests."""
    
    feature_1: float = Field(..., description="Feature 1 value")
    feature_2: float = Field(..., description="Feature 2 value")
    feature_3: float = Field(..., description="Feature 3 value")
    feature_4: float = Field(..., ge=0, le=100, description="Feature 4 value (0-100)")
    feature_5: int = Field(..., ge=1, le=10, description="Feature 5 value (1-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_1": 0.5,
                "feature_2": 1.2,
                "feature_3": -0.3,
                "feature_4": 75.5,
                "feature_5": 5
            }
        }
    
    @validator('feature_4')
    def validate_feature_4(cls, v):
        """Validate feature_4 is within expected range."""
        if v < 0 or v > 100:
            raise ValueError('feature_4 must be between 0 and 100')
        return v


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction requests."""
    
    instances: List[PredictionInput] = Field(..., description="List of prediction inputs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "feature_1": 0.5,
                        "feature_2": 1.2,
                        "feature_3": -0.3,
                        "feature_4": 75.5,
                        "feature_5": 5
                    },
                    {
                        "feature_1": -0.3,
                        "feature_2": 0.8,
                        "feature_3": 0.2,
                        "feature_4": 45.0,
                        "feature_5": 3
                    }
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Probability of positive class")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch prediction responses."""
    
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_algorithm: Optional[str] = Field(None, description="Model algorithm")
    model_accuracy: Optional[float] = Field(None, description="Model test accuracy")
    timestamp: str = Field(..., description="Check timestamp")


def engineer_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as training.
    
    Args:
        input_data: DataFrame with raw features
        
    Returns:
        DataFrame with engineered features
    """
    df = input_data.copy()
    
    # Polynomial features
    df['feature_1_squared'] = df['feature_1'] ** 2
    df['feature_2_squared'] = df['feature_2'] ** 2
    
    # Interaction features
    df['feature_1_x_feature_2'] = df['feature_1'] * df['feature_2']
    df['feature_1_x_feature_4'] = df['feature_1'] * df['feature_4']
    
    # Ratio features
    df['feature_4_per_feature_5'] = df['feature_4'] / (df['feature_5'] + 1)
    
    # Log transforms
    df['feature_4_log'] = np.log1p(df['feature_4'])
    
    # Binning (MUST match training exactly)
    df['feature_4_bin'] = pd.cut(
        df['feature_4'], 
        bins=[0, 25, 50, 75, 100],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # One-hot encode (drop_first=True to match training)
    df = pd.get_dummies(
        df, 
        columns=['feature_4_bin'],
        prefix='bin',
        drop_first=True
    )
    
    return df


def preprocess_input(input_dict: dict) -> np.ndarray:
    """
    Preprocess input data for prediction.
    
    Args:
        input_dict: Dictionary with raw feature values
        
    Returns:
        Preprocessed feature array ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Engineer features
    df_features = engineer_features(df)
    
    # DEBUG: Print feature names
    logger.info(f"🔍 Generated features: {list(df_features.columns)}")
    logger.info(f"🔍 Expected features: {feature_names}")
    
    # Separate numeric and boolean features (scaler was fit only on numeric)
    numeric_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                       'feature_1_squared', 'feature_2_squared', 'feature_1_x_feature_2',
                       'feature_1_x_feature_4', 'feature_4_per_feature_5', 'feature_4_log']
    
    boolean_features = ['bin_medium', 'bin_high', 'bin_very_high']
    
    # Scale only numeric features
    X_numeric = df_features[numeric_features]
    X_numeric_scaled = scaler.transform(X_numeric)
    
    # Get boolean features (convert True/False to 1/0)
    X_boolean = df_features[boolean_features].astype(int).values
    
    # Concatenate: scaled numeric + boolean
    X_final = np.concatenate([X_numeric_scaled, X_boolean], axis=1)
    
    return X_final


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and model information.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_algorithm=model_metadata.get('algorithm') if model_metadata else None,
        model_accuracy=model_metadata.get('metrics', {}).get('test_accuracy') if model_metadata else None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(input_data: PredictionInput, request: Request):
    """
    Make a single prediction.
    
    Args:
        input_data: Input features for prediction
        
    Returns:
        Prediction result with probability
    """
    try:
        # Log request
        logger.info(f"📥 Prediction request from {request.client.host}")
        
        # Preprocess input
        input_dict = input_data.dict()
        X = preprocess_input(input_dict)
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        # Prepare response
        response = PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=model_metadata.get('algorithm', 'unknown'),
            timestamp=datetime.now().isoformat()
        )
        
        # Log response
        logger.info(f"📤 Prediction: {prediction}, Probability: {probability:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(input_data: BatchPredictionInput, request: Request):
    """
    Make batch predictions.
    
    Args:
        input_data: List of input features for predictions
        
    Returns:
        List of prediction results
    """
    try:
        # Log request
        logger.info(f"📥 Batch prediction request from {request.client.host} ({len(input_data.instances)} instances)")
        
        predictions = []
        
        for instance in input_data.instances:
            # Preprocess input
            input_dict = instance.dict()
            X = preprocess_input(input_dict)
            
            # Make prediction
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            
            predictions.append(
                PredictionOutput(
                    prediction=prediction,
                    probability=round(probability, 4),
                    model_version=model_metadata.get('algorithm', 'unknown'),
                    timestamp=datetime.now().isoformat()
                )
            )
        
        # Log response
        logger.info(f"📤 Batch predictions complete: {len(predictions)} predictions")
        
        return BatchPredictionOutput(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata including metrics and configuration
    """
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "algorithm": model_metadata.get('algorithm'),
        "hyperparameters": model_metadata.get('hyperparameters'),
        "metrics": model_metadata.get('metrics'),
        "train_samples": model_metadata.get('train_samples'),
        "test_samples": model_metadata.get('test_samples'),
        "n_features": model_metadata.get('n_features'),
        "timestamp": model_metadata.get('timestamp')
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    config = load_config()
    
    # Run server
    uvicorn.run(
        "serve:app",
        host=config['serving']['host'],
        port=config['serving']['port'],
        reload=True,
        log_level="info"
    )
