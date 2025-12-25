"""
Data Science Master System
==========================

Enterprise-grade Data Science framework with 30+ years of industry expertise.
Supports the entire ML lifecycle from data ingestion to deployment.

Quick Start:
    >>> from data_science_master_system import Pipeline
    >>> pipeline = Pipeline.auto_detect(data, target="label")
    >>> pipeline.fit()
    >>> predictions = pipeline.predict(new_data)
    
Full Example:
    >>> from data_science_master_system import (
    ...     Pipeline, DataLoader, FeatureFactory, ModelFactory
    ... )
    >>> 
    >>> # Load data
    >>> loader = DataLoader()
    >>> df = loader.read("data.csv")
    >>> 
    >>> # Feature engineering
    >>> factory = FeatureFactory()
    >>> df = factory.auto_generate(df, target="label")
    >>> 
    >>> # Build pipeline
    >>> pipeline = Pipeline.auto_detect(df, target="label")
    >>> pipeline.fit()
    >>> 
    >>> # Evaluate
    >>> metrics = pipeline.evaluate(X_test, y_test)
"""

__version__ = "0.1.0"
__author__ = "Data Science Master System Contributors"
__license__ = "MIT"

# Core components
from data_science_master_system.core import (
    ConfigManager,
    Logger,
    get_logger,
)
from data_science_master_system.core.exceptions import (
    DSMSError,
    ConfigError,
    ValidationError,
    DataIngestionError,
    DataProcessingError,
    FeatureError,
    ModelError,
    PipelineError,
)
from data_science_master_system.core.base_classes import (
    BaseDataSource,
    BaseProcessor,
    BaseModel,
    BaseTransformer,
    BasePipeline,
)

# Data ingestion
from data_science_master_system.data.ingestion import (
    DataLoader,
    FileHandler,
    DatabaseConnector,
    SQLConnector,
    APIClient,
    RESTClient,
)

# Data processing
from data_science_master_system.data.processing import (
    ProcessingEngine,
    PandasProcessor,
    PolarsProcessor,
)

# Feature engineering
from data_science_master_system.features import (
    FeatureFactory,
    FeatureSelector,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
)

# Models
from data_science_master_system.models import (
    ModelFactory,
    AutoModelSelector,
    TraditionalMLModel,
    ClassificationModel,
    RegressionModel,
)

# Evaluation
from data_science_master_system.evaluation import (
    Evaluator,
    ClassificationMetrics,
    RegressionMetrics,
    ModelComparison,
    calculate_metrics,
)

# Visualization
from data_science_master_system.visualization import (
    Plotter,
    plot_distribution,
    plot_correlation,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
)

# Pipeline
from data_science_master_system.pipeline import Pipeline

__all__ = [
    # Version
    "__version__",
    # Core
    "ConfigManager",
    "Logger",
    "get_logger",
    # Base classes
    "BaseDataSource",
    "BaseProcessor",
    "BaseModel",
    "BaseTransformer",
    "BasePipeline",
    # Main Pipeline
    "Pipeline",
]
