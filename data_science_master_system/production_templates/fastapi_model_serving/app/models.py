"""
Model Manager - Load, version, and serve ML models.
"""

import joblib
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models with versioning and hot-reload."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, str] = {}
        self.model_metadata: Dict[str, dict] = {}
    
    def load_models(self):
        """Load all models from the models directory."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            # Load a demo model
            self._load_demo_model()
            return
        
        for model_path in self.models_dir.glob("*.joblib"):
            model_name = model_path.stem
            self._load_model(model_name, model_path)
        
        # Load metadata
        metadata_path = self.models_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.model_metadata = json.load(f)
    
    def _load_model(self, name: str, path: Path):
        """Load a single model."""
        try:
            self.models[name] = joblib.load(path)
            self.model_versions[name] = "1.0.0"
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
    
    def _load_demo_model(self):
        """Load a demo model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        self.models["demo_classifier"] = model
        self.model_versions["demo_classifier"] = "1.0.0"
        logger.info("Loaded demo classifier model")
    
    def predict(self, model_name: str, features: List[float]) -> Any:
        """Make a single prediction."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        X = np.array(features).reshape(1, -1)
        
        prediction = model.predict(X)[0]
        
        # Convert numpy types to Python types
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        return prediction
    
    def predict_proba(self, model_name: str, features: List[float]) -> List[float]:
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        X = np.array(features).reshape(1, -1)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            return proba.tolist()
        return []
    
    def batch_predict(self, model_name: str, features_list: List[List[float]]) -> List[Any]:
        """Make batch predictions."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        X = np.array(features_list)
        
        predictions = model.predict(X)
        return [p.item() if hasattr(p, 'item') else p for p in predictions]
    
    def get_model_info(self, model_name: str) -> dict:
        """Get model metadata."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return {
            "name": model_name,
            "version": self.model_versions.get(model_name, "unknown"),
            "metadata": self.model_metadata.get(model_name, {})
        }
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def reload_model(self, model_name: str) -> bool:
        """Hot-reload a specific model."""
        model_path = self.models_dir / f"{model_name}.joblib"
        if model_path.exists():
            self._load_model(model_name, model_path)
            return True
        return False
