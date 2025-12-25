"""
REST API Generator
Generate FastAPI endpoints for ML models
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pickle
from datetime import datetime


@dataclass 
class APIConfig:
    """API configuration"""
    title: str = "ML Model API"
    description: str = "Auto-generated API for ML predictions"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000


class APIGenerator:
    """
    Generate REST API for ML models
    
    Creates:
    - FastAPI application code
    - Dockerfile
    - Requirements file
    - API documentation
    """
    
    def __init__(self, output_dir: str = "api_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_api(self, 
                     model: Any,
                     feature_names: List[str],
                     model_name: str = "model",
                     config: APIConfig = None) -> Dict[str, str]:
        """
        Generate complete API package
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
            model_name: Name for the model
            config: API configuration
            
        Returns:
            Dict with paths to generated files
        """
        if config is None:
            config = APIConfig()
        
        generated_files = {}
        
        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        generated_files['model'] = str(model_path)
        
        # Generate FastAPI code
        api_code = self._generate_fastapi_code(feature_names, model_name, config)
        api_path = self.output_dir / "main.py"
        with open(api_path, 'w', encoding='utf-8') as f:
            f.write(api_code)
        generated_files['api'] = str(api_path)
        
        # Generate requirements
        req_path = self.output_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write(self._generate_requirements())
        generated_files['requirements'] = str(req_path)
        
        # Generate Dockerfile
        docker_path = self.output_dir / "Dockerfile"
        with open(docker_path, 'w') as f:
            f.write(self._generate_dockerfile())
        generated_files['dockerfile'] = str(docker_path)
        
        # Generate docker-compose
        compose_path = self.output_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(self._generate_docker_compose(config))
        generated_files['docker_compose'] = str(compose_path)
        
        # Generate README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_readme(feature_names, model_name, config))
        generated_files['readme'] = str(readme_path)
        
        return generated_files
    
    def _generate_fastapi_code(self, feature_names: List[str], 
                                model_name: str, 
                                config: APIConfig) -> str:
        """Generate FastAPI application code"""
        
        # Create Pydantic model fields
        pydantic_fields = "\n    ".join([f"{name}: float" for name in feature_names])
        
        code = f'''"""
{config.title}
{config.description}
Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="{config.title}",
    description="{config.description}",
    version="{config.version}"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = Path(__file__).parent / "{model_name}.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

FEATURE_NAMES = {feature_names}


class PredictionInput(BaseModel):
    """Input schema for prediction"""
    {pydantic_fields}
    
    class Config:
        schema_extra = {{
            "example": {{{", ".join([f'"{name}": 0.0' for name in feature_names[:5]])}}}
        }}


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction"""
    data: List[Dict[str, float]]


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: float
    probability: Optional[List[float]] = None


class BatchPredictionOutput(BaseModel):
    """Output schema for batch prediction"""
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    features: List[str]
    n_features: int
    model_type: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {{"message": "{config.title}", "status": "running"}}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {{"status": "healthy"}}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information"""
    return ModelInfo(
        name="{model_name}",
        features=FEATURE_NAMES,
        n_features=len(FEATURE_NAMES),
        model_type=type(model).__name__
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Single prediction endpoint"""
    try:
        # Prepare features
        features = [getattr(input_data, name) for name in FEATURE_NAMES]
        X = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = float(model.predict(X)[0])
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0].tolist()
        
        return PredictionOutput(prediction=prediction, probability=probability)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Batch prediction endpoint"""
    try:
        # Prepare features
        X = pd.DataFrame(input_data.data)[FEATURE_NAMES].values
        
        # Predict
        predictions = model.predict(X).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X).tolist()
        
        return BatchPredictionOutput(predictions=predictions, probabilities=probabilities)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="{config.host}", port={config.port})
'''
        return code
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        return """fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-multipart>=0.0.6
"""
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile"""
        return """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    def _generate_docker_compose(self, config: APIConfig) -> str:
        """Generate docker-compose.yml"""
        return f"""version: '3.8'

services:
  api:
    build: .
    ports:
      - "{config.port}:8000"
    restart: always
    environment:
      - PYTHONUNBUFFERED=1
"""
    
    def _generate_readme(self, feature_names: List[str], 
                         model_name: str, 
                         config: APIConfig) -> str:
        """Generate README.md"""
        return f"""# {config.title}

{config.description}

## Quick Start

### Run locally:
```bash
pip install -r requirements.txt
python main.py
```

### Run with Docker:
```bash
docker-compose up -d
```

## API Endpoints

### Health Check
```
GET /health
```

### Model Info
```
GET /model/info
```

### Single Prediction
```
POST /predict
```

**Request Body:**
```json
{{
{chr(10).join([f'  "{name}": 0.0,' for name in feature_names[:5]])}
}}
```

### Batch Prediction
```
POST /predict/batch
```

**Request Body:**
```json
{{
  "data": [
    {{{", ".join([f'"{name}": 0.0' for name in feature_names[:3]])}}}
  ]
}}
```

## Documentation

- Swagger UI: http://localhost:{config.port}/docs
- ReDoc: http://localhost:{config.port}/redoc

## Model Information

- **Model Name:** {model_name}
- **Features:** {len(feature_names)}
- **Feature Names:** {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}
"""
