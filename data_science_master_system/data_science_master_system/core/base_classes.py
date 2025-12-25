"""
Abstract Base Classes and Design Patterns for Data Science Master System.

This module provides foundational abstractions using:
- Abstract Base Classes (ABC) for interfaces
- Design Patterns: Factory, Strategy, Observer, Singleton
- Mixin classes for common functionality
- Type-safe interfaces with generics

Example:
    >>> class MyModel(BaseModel):
    ...     def fit(self, X, y): ...
    ...     def predict(self, X): ...
    
    >>> class MyProcessor(BaseProcessor):
    ...     def process(self, data): ...
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, Generic, List, Optional, Type, TypeVar, 
    Union, Callable, Iterator, Tuple
)
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

# Type variables for generics
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="BaseModel")
DataT = TypeVar("DataT")
ConfigT = TypeVar("ConfigT")


# =============================================================================
# Enums
# =============================================================================

class ProblemType(Enum):
    """Types of machine learning problems."""
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    TIME_SERIES = auto()
    NLP = auto()
    COMPUTER_VISION = auto()
    RECOMMENDATION = auto()
    ANOMALY_DETECTION = auto()
    DIMENSIONALITY_REDUCTION = auto()
    

class DataFormat(Enum):
    """Supported data formats."""
    PANDAS = auto()
    POLARS = auto()
    NUMPY = auto()
    SPARK = auto()
    DASK = auto()


class ModelState(Enum):
    """Model lifecycle states."""
    CREATED = auto()
    FITTED = auto()
    VALIDATED = auto()
    DEPLOYED = auto()
    DEPRECATED = auto()


# =============================================================================
# Design Patterns
# =============================================================================

class Singleton:
    """
    Singleton metaclass for single-instance classes.
    
    Example:
        >>> class Config(metaclass=Singleton):
        ...     pass
        >>> c1 = Config()
        >>> c2 = Config()
        >>> c1 is c2  # True
    """
    _instances: Dict[type, Any] = {}
    
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]


class Observer(ABC):
    """
    Observer interface for the Observer pattern.
    
    Observers receive updates when the subject's state changes.
    """
    
    @abstractmethod
    def update(self, event: str, data: Any = None) -> None:
        """
        Receive update from observable.
        
        Args:
            event: Event name
            data: Optional event data
        """
        pass


class Observable:
    """
    Observable mixin for the Observer pattern.
    
    Allows objects to notify observers of state changes.
    
    Example:
        >>> class TrainingMonitor(Observable):
        ...     def on_epoch_end(self, metrics):
        ...         self.notify("epoch_end", metrics)
    """
    
    def __init__(self) -> None:
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: Any = None) -> None:
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.update(event, data)


# =============================================================================
# Data Source Interfaces
# =============================================================================

class BaseDataSource(ABC):
    """
    Abstract base class for data sources.
    
    Implementations should handle:
    - Connection management
    - Data reading/writing
    - Schema inference
    - Error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize data source.
        
        Args:
            config: Source-specific configuration
        """
        self.config = config or {}
        self._connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def read(self, **kwargs: Any) -> Any:
        """
        Read data from source.
        
        Returns:
            Data in configured format
        """
        pass
    
    @abstractmethod
    def write(self, data: Any, **kwargs: Any) -> None:
        """
        Write data to source.
        
        Args:
            data: Data to write
        """
        pass
    
    def __enter__(self) -> "BaseDataSource":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if source is connected."""
        return self._connected


# =============================================================================
# Processor Interfaces
# =============================================================================

class BaseProcessor(ABC, Generic[DataT]):
    """
    Abstract base class for data processors.
    
    Processors transform data and can be chained together.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize processor.
        
        Args:
            config: Processor configuration
        """
        self.config = config or {}
        self._fitted = False
    
    @abstractmethod
    def fit(self, data: DataT, **kwargs: Any) -> "BaseProcessor":
        """
        Fit processor to data.
        
        Args:
            data: Training data
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: DataT, **kwargs: Any) -> DataT:
        """
        Transform data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: DataT, **kwargs: Any) -> DataT:
        """
        Fit and transform data.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    @property
    def is_fitted(self) -> bool:
        """Check if processor is fitted."""
        return self._fitted


# =============================================================================
# Transformer Interface (Sklearn-compatible)
# =============================================================================

class BaseTransformer(ABC):
    """
    Abstract base class for sklearn-compatible transformers.
    
    Follows the scikit-learn transformer interface for compatibility
    with sklearn pipelines.
    """
    
    def __init__(self) -> None:
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> "BaseTransformer":
        """
        Fit transformer to data.
        
        Args:
            X: Features
            y: Target (optional)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Transform data.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        pass
    
    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)
    
    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get transformer parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params: Any) -> "BaseTransformer":
        """Set transformer parameters."""
        pass


# =============================================================================
# Model Interfaces
# =============================================================================

class BaseModel(ABC, Observable):
    """
    Abstract base class for all models.
    
    Provides:
    - Unified interface for fitting and prediction
    - State management
    - Serialization interface
    - Observer pattern for training monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize model.
        
        Args:
            config: Model configuration/hyperparameters
        """
        Observable.__init__(self)
        self.config = config or {}
        self._state = ModelState.CREATED
        self._metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """
        Fit model to data.
        
        Args:
            X: Features
            y: Target (optional for unsupervised)
            **kwargs: Additional fit parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Make predictions.
        
        Args:
            X: Features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        """
        Predict class probabilities (for classifiers).
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        raise NotImplementedError("predict_proba not implemented for this model")
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model instance
        """
        pass
    
    @property
    def state(self) -> ModelState:
        """Get model state."""
        return self._state
    
    @state.setter
    def state(self, value: ModelState) -> None:
        """Set model state and notify observers."""
        old_state = self._state
        self._state = value
        self.notify("state_change", {"old": old_state, "new": value})
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._state in (ModelState.FITTED, ModelState.VALIDATED, ModelState.DEPLOYED)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config.copy()
    
    def set_params(self, **params: Any) -> "BaseModel":
        """Set model parameters."""
        self.config.update(params)
        return self


# =============================================================================
# Evaluator Interface
# =============================================================================

class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.
    
    Evaluators compute metrics and generate evaluation reports.
    """
    
    @abstractmethod
    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        pass


# =============================================================================
# Visualizer Interface
# =============================================================================

class BaseVisualizer(ABC):
    """
    Abstract base class for visualizers.
    
    Visualizers create plots and visual representations of data/models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
    
    @abstractmethod
    def plot(self, data: Any, **kwargs: Any) -> Any:
        """
        Create a plot.
        
        Args:
            data: Data to visualize
            
        Returns:
            Plot object (figure, axes, or similar)
        """
        pass
    
    @abstractmethod
    def save(self, path: str, **kwargs: Any) -> None:
        """
        Save current plot to file.
        
        Args:
            path: Path to save plot
        """
        pass


# =============================================================================
# Pipeline Interface
# =============================================================================

@dataclass
class PipelineStep:
    """Configuration for a single pipeline step."""
    name: str
    transformer: Union[BaseProcessor, BaseTransformer, BaseModel]
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class BasePipeline(ABC, Observable):
    """
    Abstract base class for ML pipelines.
    
    Pipelines orchestrate the flow of data through transformers and models.
    """
    
    def __init__(
        self,
        steps: Optional[List[PipelineStep]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize pipeline.
        
        Args:
            steps: List of pipeline steps
            config: Pipeline configuration
        """
        Observable.__init__(self)
        self.steps = steps or []
        self.config = config or {}
        self._fitted = False
    
    def add_step(
        self,
        name: str,
        transformer: Union[BaseProcessor, BaseTransformer, BaseModel],
        **params: Any,
    ) -> "BasePipeline":
        """
        Add a step to the pipeline.
        
        Args:
            name: Step name
            transformer: Transformer/processor/model
            **params: Step parameters
            
        Returns:
            Self for chaining
        """
        step = PipelineStep(name=name, transformer=transformer, params=params)
        self.steps.append(step)
        return self
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "BasePipeline":
        """Fit all pipeline steps."""
        pass
    
    @abstractmethod
    def transform(self, X: Any, **kwargs: Any) -> Any:
        """Transform data through all steps."""
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Make predictions (fit must be called first)."""
        pass
    
    def fit_predict(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> Any:
        """Fit and predict."""
        return self.fit(X, y, **kwargs).predict(X, **kwargs)
    
    def fit_transform(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> Any:
        """Fit and transform."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    
    def __iter__(self) -> Iterator[PipelineStep]:
        """Iterate over pipeline steps."""
        return iter(self.steps)
    
    def __len__(self) -> int:
        """Get number of steps."""
        return len(self.steps)
    
    def __getitem__(self, key: Union[int, str]) -> PipelineStep:
        """Get step by index or name."""
        if isinstance(key, int):
            return self.steps[key]
        for step in self.steps:
            if step.name == key:
                return step
        raise KeyError(f"Step '{key}' not found")


# =============================================================================
# Factory Pattern
# =============================================================================

class BaseFactory(ABC, Generic[T]):
    """
    Abstract factory for creating objects.
    
    Example:
        >>> class ModelFactory(BaseFactory[BaseModel]):
        ...     def create(self, name, **kwargs):
        ...         if name == "xgboost":
        ...             return XGBoostModel(**kwargs)
    """
    
    _registry: Dict[str, Type[T]] = {}
    
    @classmethod
    def register(cls, name: str, item_class: Type[T]) -> None:
        """
        Register a class with the factory.
        
        Args:
            name: Registration name
            item_class: Class to register
        """
        cls._registry[name] = item_class
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> T:
        """
        Create an instance by name.
        
        Args:
            name: Registered name
            **kwargs: Constructor arguments
            
        Returns:
            Instance of registered class
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown type: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available registered names."""
        return list(cls._registry.keys())


# =============================================================================
# Strategy Pattern
# =============================================================================

class BaseStrategy(ABC, Generic[T]):
    """
    Abstract strategy interface.
    
    Strategies encapsulate interchangeable algorithms.
    """
    
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Execute the strategy."""
        pass


class StrategyContext(Generic[T]):
    """
    Context for strategy pattern.
    
    Example:
        >>> context = StrategyContext(MyStrategy())
        >>> result = context.execute_strategy(data)
    """
    
    def __init__(self, strategy: Optional[BaseStrategy[T]] = None) -> None:
        self._strategy = strategy
    
    @property
    def strategy(self) -> Optional[BaseStrategy[T]]:
        """Get current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: BaseStrategy[T]) -> None:
        """Set strategy."""
        self._strategy = strategy
    
    def execute_strategy(self, *args: Any, **kwargs: Any) -> T:
        """Execute current strategy."""
        if self._strategy is None:
            raise ValueError("No strategy set")
        return self._strategy.execute(*args, **kwargs)


# =============================================================================
# Utility Mixins
# =============================================================================

class SerializableMixin:
    """Mixin for serializable objects."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "config": getattr(self, "config", {}),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Create object from dictionary."""
        return cls(config=data.get("config", {}))


class ValidatableMixin:
    """Mixin for objects with validation."""
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate object state.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        # Override in subclasses to add validation logic
        return len(errors) == 0, errors
