"""
FastAPI Model Serving - Main Application

Production-ready ML model serving with health checks, versioning, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
from contextlib import asynccontextmanager

from app.models import ModelManager
from app.schemas import PredictRequest, PredictResponse, BatchPredictRequest, HealthResponse
from app.routes import router
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model manager instance
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Loading models...")
    model_manager.load_models()
    logger.info(f"Models loaded: {list(model_manager.models.keys())}")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready API for ML model inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for K8s probes."""
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_manager.models),
        version="1.0.0"
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe for K8s."""
    if not model_manager.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction endpoint."""
    start_time = time.time()
    
    try:
        prediction = model_manager.predict(
            model_name=request.model_name,
            features=request.features
        )
        
        return PredictResponse(
            prediction=prediction,
            model_name=request.model_name,
            latency_ms=(time.time() - start_time) * 1000
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Batch prediction endpoint."""
    start_time = time.time()
    
    predictions = model_manager.batch_predict(
        model_name=request.model_name,
        features_list=request.features_list
    )
    
    # Log batch request asynchronously
    background_tasks.add_task(
        logger.info,
        f"Batch prediction: {len(request.features_list)} samples"
    )
    
    return {
        "predictions": predictions,
        "count": len(predictions),
        "latency_ms": (time.time() - start_time) * 1000
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
