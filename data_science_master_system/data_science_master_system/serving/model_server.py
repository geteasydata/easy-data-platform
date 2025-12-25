"""
Model Serving Module.

Production-ready model serving with multiple backends:
- FastAPI REST API
- gRPC server
- ONNX Runtime
- TensorFlow Serving
- Triton Inference Server
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class ModelServer:
    """
    Production model serving with REST API.
    
    Example:
        >>> server = ModelServer(model, preprocessor)
        >>> server.start(host='0.0.0.0', port=8000)
    """
    
    def __init__(
        self,
        model: Any,
        preprocessor: Any = None,
        model_name: str = 'model',
        version: str = '1.0.0',
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.version = version
        self.request_count = 0
        self.total_latency = 0.0
    
    def predict(self, data: Union[Dict, List[Dict]]) -> Dict:
        """Make prediction with timing."""
        start = time.time()
        
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        
        predictions = self.model.predict(df)
        
        latency = (time.time() - start) * 1000
        self.request_count += 1
        self.total_latency += latency
        
        return {
            'predictions': predictions.tolist(),
            'model': self.model_name,
            'version': self.version,
            'latency_ms': round(latency, 2)
        }
    
    def predict_proba(self, data: Union[Dict, List[Dict]]) -> Dict:
        """Predict with probabilities."""
        start = time.time()
        
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df)
        else:
            probabilities = self.model.predict(df)
        
        latency = (time.time() - start) * 1000
        
        return {
            'probabilities': probabilities.tolist(),
            'latency_ms': round(latency, 2)
        }
    
    def health(self) -> Dict:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model': self.model_name,
            'version': self.version,
            'requests_served': self.request_count,
            'avg_latency_ms': round(self.total_latency / max(1, self.request_count), 2)
        }
    
    def create_fastapi_app(self):
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            
            app = FastAPI(title=f"{self.model_name} API", version=self.version)
            
            class PredictRequest(BaseModel):
                data: List[Dict]
            
            class PredictResponse(BaseModel):
                predictions: List
                model: str
                version: str
                latency_ms: float
            
            @app.get("/health")
            def health():
                return self.health()
            
            @app.post("/predict", response_model=PredictResponse)
            def predict(request: PredictRequest):
                try:
                    return self.predict(request.data)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/predict_proba")
            def predict_proba(request: PredictRequest):
                return self.predict_proba(request.data)
            
            return app
        except ImportError:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
    
    def start(self, host: str = '0.0.0.0', port: int = 8000):
        """Start the API server."""
        import uvicorn
        app = self.create_fastapi_app()
        uvicorn.run(app, host=host, port=port)


class ONNXServer:
    """Serve models using ONNX Runtime for fast inference."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"ONNX model loaded: {self.model_path}")
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Run inference."""
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        result = self.session.run([self.output_name], {self.input_name: data})
        return result[0]
    
    def benchmark(self, data: np.ndarray, n_runs: int = 100) -> Dict:
        """Benchmark inference speed."""
        # Warmup
        for _ in range(10):
            self.predict(data)
        
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            self.predict(data)
            latencies.append((time.time() - start) * 1000)
        
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        }


class BatchPredictor:
    """Batch prediction with automatic batching."""
    
    def __init__(self, model: Any, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict with automatic batching."""
        predictions = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size]
            batch_preds = self.model.predict(batch)
            predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_async(self, data: pd.DataFrame, callback=None):
        """Async batch prediction."""
        import concurrent.futures
        
        batches = [data.iloc[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.model.predict, batch) for batch in batches]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                if callback:
                    callback(result)
        
        return np.concatenate(results)
