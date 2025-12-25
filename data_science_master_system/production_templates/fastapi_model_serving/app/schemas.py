"""
Pydantic Schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from datetime import datetime


class PredictRequest(BaseModel):
    """Single prediction request."""
    model_name: str = Field(..., description="Name of the model to use")
    features: List[float] = Field(..., description="Input features for prediction")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("Features cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "demo_classifier",
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictResponse(BaseModel):
    """Single prediction response."""
    prediction: Any = Field(..., description="Model prediction")
    model_name: str = Field(..., description="Model used for prediction")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    probabilities: Optional[List[float]] = Field(None, description="Class probabilities if available")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "model_name": "demo_classifier",
                "latency_ms": 5.2,
                "probabilities": [0.95, 0.03, 0.02]
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    model_name: str = Field(..., description="Name of the model")
    features_list: List[List[float]] = Field(..., description="List of feature vectors")
    
    @validator('features_list')
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        return v


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[Any]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    metadata: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
