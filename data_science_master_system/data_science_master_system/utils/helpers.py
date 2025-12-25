"""
Helper Functions for Data Science Master System.

General-purpose utilities for:
- Data manipulation
- Memory management
- Random seed control
- Dictionary operations
- Iteration utilities

Example:
    >>> from data_science_master_system.utils import flatten_dict, set_random_seed
    >>> flat = flatten_dict({"a": {"b": 1}})  # {"a.b": 1}
    >>> set_random_seed(42)
"""

import os
import random
import sys
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar, Union
import numpy as np

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def ensure_list(value: Union[T, List[T], None]) -> List[T]:
    """
    Ensure value is a list.
    
    Args:
        value: Single value, list, or None
        
    Returns:
        List containing the value(s)
        
    Example:
        >>> ensure_list("a")      # ["a"]
        >>> ensure_list(["a", "b"])  # ["a", "b"]
        >>> ensure_list(None)     # []
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        separator: Key separator
        
    Returns:
        Flattened dictionary
        
    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items: List[tuple] = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Unflatten a flattened dictionary.
    
    Args:
        d: Flattened dictionary
        separator: Key separator used in flattening
        
    Returns:
        Nested dictionary
        
    Example:
        >>> unflatten_dict({"a.b": 1, "a.c": 2})
        {"a": {"b": 1, "c": 2}}
    """
    result: Dict[str, Any] = {}
    
    for key, value in d.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        
        current[parts[-1]] = value
    
    return result


def chunk_iterable(
    iterable: Iterable[T],
    chunk_size: int,
) -> Generator[List[T], None, None]:
    """
    Split an iterable into chunks.
    
    Args:
        iterable: Iterable to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Lists of elements
        
    Example:
        >>> list(chunk_iterable([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    chunk: List[T] = []
    
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    
    if chunk:
        yield chunk


def get_memory_usage(
    obj: Optional[Any] = None,
    deep: bool = True,
) -> Dict[str, float]:
    """
    Get memory usage information.
    
    Args:
        obj: Specific object to measure (None for process)
        deep: Whether to do deep inspection
        
    Returns:
        Dictionary with memory info in MB
        
    Example:
        >>> info = get_memory_usage(large_dataframe)
        >>> print(f"Memory: {info['object_mb']:.2f} MB")
    """
    import gc
    
    info: Dict[str, float] = {}
    
    # Process memory
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        info["process_mb"] = memory_info.rss / (1024 * 1024)
        info["virtual_mb"] = memory_info.vms / (1024 * 1024)
    except ImportError:
        pass
    
    # Object memory
    if obj is not None:
        # Try pandas method first
        if hasattr(obj, "memory_usage"):
            if callable(obj.memory_usage):
                mem = obj.memory_usage(deep=deep)
                if hasattr(mem, "sum"):
                    info["object_mb"] = mem.sum() / (1024 * 1024)
                else:
                    info["object_mb"] = mem / (1024 * 1024)
        else:
            # Fall back to sys.getsizeof
            info["object_mb"] = sys.getsizeof(obj) / (1024 * 1024)
    
    return info


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python's random module
    - NumPy
    - Optional: TensorFlow, PyTorch
    
    Args:
        seed: Random seed value
        
    Example:
        >>> set_random_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.debug(f"Random seed set to {seed}")


def get_feature_names(
    estimator: Any,
    default: Optional[List[str]] = None,
) -> List[str]:
    """
    Get feature names from sklearn-like estimator.
    
    Args:
        estimator: Fitted estimator
        default: Default names if not found
        
    Returns:
        List of feature names
    """
    # Try different methods
    for attr in ["feature_names_in_", "feature_names", "get_feature_names_out"]:
        if hasattr(estimator, attr):
            value = getattr(estimator, attr)
            if callable(value):
                return list(value())
            return list(value)
    
    return default or []


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Safe division handling divide by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when dividing by zero
        
    Returns:
        Division result or default
    """
    if isinstance(denominator, np.ndarray):
        result = np.divide(
            numerator,
            denominator,
            out=np.full_like(denominator, default, dtype=float),
            where=denominator != 0,
        )
    else:
        result = numerator / denominator if denominator != 0 else default
    
    return result


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    
    parts = []
    
    hours = int(seconds // 3600)
    if hours:
        parts.append(f"{hours}h")
        seconds %= 3600
    
    minutes = int(seconds // 60)
    if minutes:
        parts.append(f"{minutes}m")
        seconds %= 60
    
    if seconds or not parts:
        parts.append(f"{seconds:.1f}s")
    
    return " ".join(parts)


def get_class_name(obj: Any) -> str:
    """
    Get fully qualified class name of object.
    
    Args:
        obj: Object or class
        
    Returns:
        Fully qualified name
    """
    if isinstance(obj, type):
        cls = obj
    else:
        cls = type(obj)
    
    return f"{cls.__module__}.{cls.__name__}"


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries (later ones override earlier).
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result: Dict[str, Any] = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
