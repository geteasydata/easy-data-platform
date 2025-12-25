"""
Models Module - Machine Learning Models and Pipelines.

Provides:
    - model_factory: Unified model creation
    - traditional: Sklearn, XGBoost, LightGBM, CatBoost
    - deep_learning: TensorFlow, PyTorch wrappers
    - automl: Automated ML
"""

from data_science_master_system.models.model_factory import (
    ModelFactory,
    AutoModelSelector,
)
from data_science_master_system.models.traditional.traditional_ml import (
    TraditionalMLModel,
    ClassificationModel,
    RegressionModel,
)

__all__ = [
    "ModelFactory",
    "AutoModelSelector",
    "TraditionalMLModel",
    "ClassificationModel",
    "RegressionModel",
]
