"""
Validation Utilities for Data Science Master System.

Provides validators for:
- DataFrames (pandas, polars)
- NumPy arrays
- Data types and schemas
- Missing values and data quality

Example:
    >>> from data_science_master_system.utils import validate_dataframe
    >>> validate_dataframe(df, required_columns=["id", "value"])
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from data_science_master_system.core.exceptions import ValidationError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


def validate_dataframe(
    df: Any,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
    max_rows: Optional[int] = None,
    allow_empty: bool = False,
    dtypes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Validate a DataFrame.
    
    Args:
        df: DataFrame to validate (pandas or polars)
        required_columns: List of required column names
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows (optional)
        allow_empty: Whether to allow empty DataFrame
        dtypes: Expected column data types
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> validate_dataframe(df, required_columns=["id", "name"], min_rows=1)
    """
    # Check if DataFrame-like
    if not hasattr(df, "columns") or not hasattr(df, "shape"):
        raise ValidationError(
            "Input is not a valid DataFrame",
            context={"type": type(df).__name__},
        )
    
    rows, cols = df.shape
    
    # Check empty
    if not allow_empty and rows == 0:
        raise ValidationError(
            "DataFrame is empty",
            context={"rows": rows},
        )
    
    # Check row bounds
    if rows < min_rows:
        raise ValidationError(
            f"DataFrame has fewer rows than required",
            context={"rows": rows, "min_rows": min_rows},
        )
    
    if max_rows is not None and rows > max_rows:
        raise ValidationError(
            f"DataFrame has more rows than allowed",
            context={"rows": rows, "max_rows": max_rows},
        )
    
    # Check required columns
    if required_columns:
        columns = set(df.columns)
        missing = set(required_columns) - columns
        if missing:
            raise ValidationError(
                "Missing required columns",
                context={"missing": list(missing), "available": list(columns)},
            )
    
    # Check dtypes
    if dtypes:
        for col, expected_dtype in dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not _dtype_matches(actual_dtype, expected_dtype):
                    raise ValidationError(
                        f"Column '{col}' has incorrect dtype",
                        context={"expected": str(expected_dtype), "actual": actual_dtype},
                    )
    
    return True


def _dtype_matches(actual: str, expected: Any) -> bool:
    """Check if actual dtype matches expected."""
    actual_lower = actual.lower()
    expected_str = str(expected).lower()
    
    # Handle common dtype aliases
    dtype_aliases = {
        "int": ["int", "int32", "int64", "int8", "int16"],
        "float": ["float", "float32", "float64", "float16"],
        "str": ["object", "string", "str"],
        "bool": ["bool", "boolean"],
        "datetime": ["datetime64", "datetime"],
    }
    
    for dtype_group, aliases in dtype_aliases.items():
        if expected_str in aliases:
            return any(alias in actual_lower for alias in aliases)
    
    return expected_str in actual_lower


def validate_array(
    arr: Any,
    ndim: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_size: int = 0,
    dtype: Optional[Any] = None,
    allow_nan: bool = True,
    allow_inf: bool = True,
) -> bool:
    """
    Validate a NumPy array.
    
    Args:
        arr: Array to validate
        ndim: Expected number of dimensions
        shape: Expected shape
        min_size: Minimum number of elements
        dtype: Expected dtype
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow Inf values
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    # Convert to array if needed
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise ValidationError(
                "Cannot convert input to array",
                context={"error": str(e)},
            )
    
    # Check dimensions
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(
            "Array has incorrect dimensions",
            context={"expected": ndim, "actual": arr.ndim},
        )
    
    # Check shape
    if shape is not None and arr.shape != shape:
        raise ValidationError(
            "Array has incorrect shape",
            context={"expected": shape, "actual": arr.shape},
        )
    
    # Check size
    if arr.size < min_size:
        raise ValidationError(
            "Array is too small",
            context={"size": arr.size, "min_size": min_size},
        )
    
    # Check dtype
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        raise ValidationError(
            "Array has incorrect dtype",
            context={"expected": str(dtype), "actual": str(arr.dtype)},
        )
    
    # Check for NaN
    if not allow_nan and np.issubdtype(arr.dtype, np.floating):
        if np.any(np.isnan(arr)):
            raise ValidationError("Array contains NaN values")
    
    # Check for Inf
    if not allow_inf and np.issubdtype(arr.dtype, np.floating):
        if np.any(np.isinf(arr)):
            raise ValidationError("Array contains Inf values")
    
    return True


def validate_not_empty(
    data: Any,
    name: str = "data",
) -> bool:
    """
    Validate that data is not empty.
    
    Works with DataFrame, array, list, dict, string.
    
    Args:
        data: Data to validate
        name: Name for error messages
        
    Returns:
        True if not empty
        
    Raises:
        ValidationError: If data is empty
    """
    if data is None:
        raise ValidationError(f"{name} is None")
    
    # Check for empty using different methods
    if hasattr(data, "empty"):
        if data.empty:
            raise ValidationError(f"{name} is empty")
    elif hasattr(data, "size"):
        if data.size == 0:
            raise ValidationError(f"{name} is empty")
    elif hasattr(data, "__len__"):
        if len(data) == 0:
            raise ValidationError(f"{name} is empty")
    
    return True


def validate_columns(
    df: Any,
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validate DataFrame columns.
    
    Args:
        df: DataFrame to validate
        required: Required columns
        optional: Optional columns (allowed but not required)
        forbidden: Columns that should not exist
        
    Returns:
        Tuple of (found required columns, found optional columns)
        
    Raises:
        ValidationError: If validation fails
    """
    columns = set(df.columns)
    
    # Check required
    required = set(required or [])
    missing = required - columns
    if missing:
        raise ValidationError(
            "Missing required columns",
            context={"missing": list(missing)},
        )
    
    # Check forbidden
    forbidden = set(forbidden or [])
    found_forbidden = columns & forbidden
    if found_forbidden:
        raise ValidationError(
            "Found forbidden columns",
            context={"found": list(found_forbidden)},
        )
    
    # Find optional columns
    optional = set(optional or [])
    found_optional = columns & optional
    
    return list(required), list(found_optional)


def validate_numeric(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
    name: str = "value",
) -> bool:
    """
    Validate a numeric value.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is allowed
        name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_none:
            return True
        raise ValidationError(f"{name} cannot be None")
    
    if not isinstance(value, (int, float, np.number)):
        raise ValidationError(
            f"{name} must be numeric",
            context={"type": type(value).__name__},
        )
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{name} is below minimum",
            context={"value": value, "min": min_value},
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{name} is above maximum",
            context={"value": value, "max": max_value},
        )
    
    return True


def check_missing_values(
    df: Any,
    threshold: float = 0.0,
    columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Check for missing values in DataFrame.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed missing ratio (0.0 to 1.0)
        columns: Specific columns to check (all if None)
        
    Returns:
        Dictionary of column names to missing ratios
        
    Raises:
        ValidationError: If missing values exceed threshold
    """
    columns = columns or list(df.columns)
    missing_ratios = {}
    violations = []
    
    for col in columns:
        if col in df.columns:
            if hasattr(df[col], "isna"):
                missing = df[col].isna().sum()
            elif hasattr(df[col], "is_null"):
                missing = df[col].is_null().sum()
            else:
                continue
            
            ratio = missing / len(df) if len(df) > 0 else 0.0
            missing_ratios[col] = ratio
            
            if ratio > threshold:
                violations.append((col, ratio))
    
    if violations:
        raise ValidationError(
            "Columns exceed missing value threshold",
            context={
                "threshold": threshold,
                "violations": {col: round(ratio, 4) for col, ratio in violations},
            },
        )
    
    return missing_ratios
