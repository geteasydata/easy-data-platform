"""
API Routes - Model management and prediction endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from app.schemas import ModelInfo, PredictRequest, PredictResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models", response_model=List[str])
async def list_models():
    """List all available models."""
    from app.main import model_manager
    return model_manager.list_models()


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get model information and metadata."""
    from app.main import model_manager
    
    try:
        info = model_manager.get_model_info(model_name)
        return ModelInfo(**info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{model_name}/reload")
async def reload_model(model_name: str):
    """Hot-reload a specific model."""
    from app.main import model_manager
    
    success = model_manager.reload_model(model_name)
    if success:
        return {"status": "reloaded", "model": model_name}
    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/predict/proba")
async def predict_with_probabilities(request: PredictRequest):
    """Prediction with class probabilities."""
    from app.main import model_manager
    import time
    
    start_time = time.time()
    
    try:
        prediction = model_manager.predict(request.model_name, request.features)
        probabilities = model_manager.predict_proba(request.model_name, request.features)
        
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "model_name": request.model_name,
            "latency_ms": (time.time() - start_time) * 1000
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get model serving metrics."""
    from app.main import model_manager
    
    return {
        "models_count": len(model_manager.models),
        "models": model_manager.list_models()
    }
