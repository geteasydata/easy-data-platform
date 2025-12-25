"""
Core module - Foundation of the Data Science Master System.

This module provides:
    - ConfigManager: Unified configuration management (YAML/JSON/ENV)
    - Logger: Structured logging with multiple handlers
    - Exceptions: Custom exception hierarchy
    - Base Classes: Abstract interfaces with design patterns
"""

from data_science_master_system.core.config_manager import ConfigManager
from data_science_master_system.core.logger import Logger, get_logger
from data_science_master_system.core.exceptions import (
    DSMSError,
    DataError,
    ModelError,
    ConfigError,
    ValidationError,
    PipelineError,
    FeatureError,
    DeploymentError,
    EvaluationError,
)
from data_science_master_system.core.base_classes import (
    BaseDataSource,
    BaseProcessor,
    BaseModel,
    BaseTransformer,
    BasePipeline,
    BaseEvaluator,
    BaseVisualizer,
    Singleton,
    Observable,
    Observer,
)

__all__ = [
    # Config
    "ConfigManager",
    # Logging
    "Logger",
    "get_logger",
    # Exceptions
    "DSMSError",
    "DataError",
    "ModelError",
    "ConfigError",
    "ValidationError",
    "PipelineError",
    "FeatureError",
    "DeploymentError",
    "EvaluationError",
    # Base Classes
    "BaseDataSource",
    "BaseProcessor",
    "BaseModel",
    "BaseTransformer",
    "BasePipeline",
    "BaseEvaluator",
    "BaseVisualizer",
    # Design Patterns
    "Singleton",
    "Observable",
    "Observer",
]
