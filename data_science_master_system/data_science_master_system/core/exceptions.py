"""
Custom Exception Hierarchy for Data Science Master System.

This module provides a comprehensive exception hierarchy that enables:
- Specific error handling for different failure modes
- Informative error messages with context
- Error codes for programmatic handling
- Graceful degradation in production

Exception Hierarchy:
    DSMSError (base)
    ├── ConfigError
    ├── ValidationError
    ├── DataError
    │   ├── DataIngestionError
    │   ├── DataProcessingError
    │   └── DataQualityError
    ├── FeatureError
    │   ├── FeatureEngineeringError
    │   └── FeatureSelectionError
    ├── ModelError
    │   ├── ModelTrainingError
    │   ├── ModelPredictionError
    │   └── ModelSerializationError
    ├── PipelineError
    └── DeploymentError
        ├── ServingError
        └── MonitoringError

Example:
    >>> try:
    ...     loader.read("missing.csv")
    ... except DataIngestionError as e:
    ...     logger.error(f"Failed to load data: {e}")
    ...     # Handle gracefully
"""

from typing import Any, Dict, Optional


class DSMSError(Exception):
    """
    Base exception for all Data Science Master System errors.
    
    Provides:
        - Error codes for programmatic handling
        - Context dictionary for additional information
        - Formatted error messages
    
    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        context: Additional context about the error
        
    Example:
        >>> raise DSMSError("Something went wrong", error_code="E001", context={"file": "data.csv"})
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or "DSMS000"
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with code and context."""
        formatted = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            formatted += f" | Context: {context_str}"
        return formatted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": self.__class__.__name__,
        }


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigError(DSMSError):
    """
    Raised when configuration is invalid or missing.
    
    Example:
        >>> raise ConfigError("Missing required config key", context={"key": "database_url"})
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "CFG001",
            context=context,
        )


class ValidationError(DSMSError):
    """
    Raised when data or parameter validation fails.
    
    Example:
        >>> raise ValidationError("Invalid parameter value", context={"param": "n_estimators", "value": -1})
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "VAL001",
            context=context,
        )


# =============================================================================
# Data Errors
# =============================================================================

class DataError(DSMSError):
    """Base class for all data-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "DAT001",
            context=context,
        )


class DataIngestionError(DataError):
    """
    Raised when data ingestion fails.
    
    Common causes:
        - File not found
        - Database connection failed
        - Invalid file format
        - Permission denied
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "ING001",
            context=context,
        )


class DataProcessingError(DataError):
    """
    Raised when data processing fails.
    
    Common causes:
        - Type conversion error
        - Memory overflow
        - Invalid operation
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "PRC001",
            context=context,
        )


class DataQualityError(DataError):
    """
    Raised when data quality checks fail.
    
    Common causes:
        - Missing values exceed threshold
        - Data drift detected
        - Schema mismatch
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "QUA001",
            context=context,
        )


# =============================================================================
# Feature Errors
# =============================================================================

class FeatureError(DSMSError):
    """Base class for all feature engineering errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "FEA001",
            context=context,
        )


class FeatureEngineeringError(FeatureError):
    """Raised when feature engineering fails."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "FEN001",
            context=context,
        )


class FeatureSelectionError(FeatureError):
    """Raised when feature selection fails."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "FSL001",
            context=context,
        )


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(DSMSError):
    """Base class for all model-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "MOD001",
            context=context,
        )


class ModelTrainingError(ModelError):
    """
    Raised when model training fails.
    
    Common causes:
        - Convergence failure
        - Invalid hyperparameters
        - Insufficient data
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "TRN001",
            context=context,
        )


class ModelPredictionError(ModelError):
    """
    Raised when model prediction fails.
    
    Common causes:
        - Input shape mismatch
        - Model not fitted
        - Invalid input type
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "PRD001",
            context=context,
        )


class ModelSerializationError(ModelError):
    """
    Raised when model serialization/deserialization fails.
    
    Common causes:
        - Incompatible pickle version
        - Missing dependencies
        - Corrupted file
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "SER001",
            context=context,
        )


# =============================================================================
# Pipeline Errors
# =============================================================================

class PipelineError(DSMSError):
    """
    Raised when pipeline execution fails.
    
    Common causes:
        - Step dependency failure
        - Invalid pipeline configuration
        - Circular dependencies
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "PIP001",
            context=context,
        )


# =============================================================================
# Deployment Errors
# =============================================================================

class DeploymentError(DSMSError):
    """Base class for all deployment-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "DEP001",
            context=context,
        )


class ServingError(DeploymentError):
    """
    Raised when model serving fails.
    
    Common causes:
        - Server startup failure
        - Request handling error
        - Resource exhaustion
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "SRV001",
            context=context,
        )


class MonitoringError(DeploymentError):
    """
    Raised when monitoring operations fail.
    
    Common causes:
        - Metrics collection failure
        - Alert delivery failure
        - Dashboard update failure
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "MON001",
            context=context,
        )


# =============================================================================
# Evaluation Errors
# =============================================================================

class EvaluationError(DSMSError):
    """Base class for evaluation errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code or "EVA001",
            context=context,
        )


# =============================================================================
# Exception Handlers
# =============================================================================

def handle_exception(
    exception: Exception,
    reraise: bool = True,
    default_error_code: str = "UNK001",
) -> Dict[str, Any]:
    """
    Handle any exception and convert to standardized format.
    
    Args:
        exception: The exception to handle
        reraise: Whether to reraise the exception after handling
        default_error_code: Error code for unknown exceptions
        
    Returns:
        Dictionary with error information
        
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     error_info = handle_exception(e, reraise=False)
        ...     logger.error(error_info)
    """
    if isinstance(exception, DSMSError):
        error_info = exception.to_dict()
    else:
        error_info = {
            "error_code": default_error_code,
            "message": str(exception),
            "context": {"original_type": type(exception).__name__},
            "type": "UnhandledException",
        }
    
    if reraise:
        raise exception
    
    return error_info
