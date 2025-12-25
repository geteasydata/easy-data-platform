"""
FastAPI Model Serving Template.

Production-ready REST API for ML model serving.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"feature1": 1.0, "feature2": 2.0, "feature3": "category_a"}
                ]
            }
        }

class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    latency_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float

# Global state
class ModelState:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.version = "1.0.0"
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0.0

state = ModelState()

# Startup event
@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    try:
        state.model = joblib.load("model.joblib")
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model file not found. Using placeholder.")
        from sklearn.ensemble import RandomForestClassifier
        state.model = RandomForestClassifier(n_estimators=10)

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        version=state.version,
        uptime_seconds=round(time.time() - state.start_time, 2)
    )

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics."""
    avg_latency = state.total_latency / max(1, state.request_count)
    return {
        "requests_total": state.request_count,
        "average_latency_ms": round(avg_latency, 2),
        "model_version": state.version,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make predictions."""
    start = time.time()
    
    try:
        df = pd.DataFrame(request.data)
        
        # Preprocess if needed
        if state.preprocessor:
            df = state.preprocessor.transform(df)
        
        # Handle categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.Categorical(df[col]).codes
        
        predictions = state.model.predict(df).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(state.model, 'predict_proba'):
            probabilities = state.model.predict_proba(df).tolist()
        
        latency = (time.time() - start) * 1000
        state.request_count += 1
        state.total_latency += latency
        
        return PredictResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version=state.version,
            latency_ms=round(latency, 2),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Batch predictions with background processing for large datasets."""
    if len(request.data) > 1000:
        # Process in background for large batches
        background_tasks.add_task(process_batch, request.data)
        return {"status": "processing", "message": "Batch queued for processing"}
    
    return await predict(request)

async def process_batch(data: List[Dict]):
    """Background batch processing."""
    df = pd.DataFrame(data)
    predictions = state.model.predict(df)
    logger.info(f"Processed batch of {len(data)} samples")

# Run with: uvicorn api_template:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
