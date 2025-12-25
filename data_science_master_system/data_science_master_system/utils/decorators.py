"""
Utility Decorators for Data Science Master System.

Provides powerful decorators for:
- Performance optimization (caching, memoization)
- Error handling and retries
- Validation and type checking
- Logging and timing
- Deprecation warnings
- Rate limiting

Example:
    >>> @retry(max_attempts=3, delay=1.0)
    ... def fetch_data(url):
    ...     return requests.get(url)
    
    >>> @validate_args(x=int, y=float)
    ... def compute(x, y):
    ...     return x * y
"""

import functools
import hashlib
import json
import time
import warnings
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from collections import OrderedDict
from threading import Lock

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Caching Decorators
# =============================================================================

def memoize(maxsize: int = 128) -> Callable[[F], F]:
    """
    Memoization decorator with LRU cache.
    
    Caches function results based on arguments. Uses LRU eviction
    when cache exceeds maxsize.
    
    Args:
        maxsize: Maximum cache size
        
    Example:
        >>> @memoize(maxsize=100)
        ... def expensive_computation(x, y):
        ...     return x ** y
    """
    def decorator(func: F) -> F:
        cache: OrderedDict = OrderedDict()
        lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key
            key = _make_cache_key(args, kwargs)
            
            with lock:
                if key in cache:
                    # Move to end (most recently used)
                    cache.move_to_end(key)
                    return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = result
                # Evict oldest if over size
                while len(cache) > maxsize:
                    cache.popitem(last=False)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}
        
        return wrapper  # type: ignore
    
    return decorator


def _make_cache_key(args: tuple, kwargs: dict) -> str:
    """Create a hashable cache key from arguments."""
    key_parts = [repr(arg) for arg in args]
    key_parts.extend(f"{k}={repr(v)}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached_property(func: Callable[[Any], Any]) -> property:
    """
    Decorator for cached class properties.
    
    Computes the property value once and caches it.
    
    Example:
        >>> class DataLoader:
        ...     @cached_property
        ...     def data(self):
        ...         return load_expensive_data()
    """
    attr_name = f"_cached_{func.__name__}"
    
    @property  # type: ignore
    @functools.wraps(func)
    def wrapper(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


# =============================================================================
# Error Handling Decorators
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None,
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        on_failure: Callback function on final failure
        
    Example:
        >>> @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        ... def fetch_api_data():
        ...     return requests.get(url)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}",
                            error=str(e),
                            next_delay=current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            # All retries failed
            logger.error(
                f"All {max_attempts} attempts failed for {func.__name__}",
                error=str(last_exception),
            )
            
            if on_failure:
                on_failure(last_exception)
            
            raise last_exception  # type: ignore
        
        return wrapper  # type: ignore
    
    return decorator


def fallback(default_value: Any = None, log_error: bool = True) -> Callable[[F], F]:
    """
    Decorator that returns a fallback value on exception.
    
    Args:
        default_value: Value to return on failure
        log_error: Whether to log the error
        
    Example:
        >>> @fallback(default_value=[])
        ... def get_items():
        ...     return database.query_items()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}, using fallback",
                        error=str(e),
                        fallback=default_value,
                    )
                return default_value
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Validation Decorators
# =============================================================================

def validate_args(**validators: Union[Type, Callable]) -> Callable[[F], F]:
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Mapping of arg names to types or validator functions
        
    Example:
        >>> @validate_args(x=int, y=lambda v: v > 0)
        ... def process(x, y):
        ...     return x * y
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    
                    if isinstance(validator, type):
                        if not isinstance(value, validator):
                            raise TypeError(
                                f"Argument '{arg_name}' must be {validator.__name__}, "
                                f"got {type(value).__name__}"
                            )
                    elif callable(validator):
                        if not validator(value):
                            raise ValueError(
                                f"Argument '{arg_name}' failed validation"
                            )
            
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def require_fitted(attr: str = "_fitted") -> Callable[[F], F]:
    """
    Decorator that ensures model is fitted before method call.
    
    Args:
        attr: Name of the fitted flag attribute
        
    Example:
        >>> class Model:
        ...     @require_fitted()
        ...     def predict(self, X):
        ...         return self._model.predict(X)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not getattr(self, attr, False):
                raise RuntimeError(
                    f"Cannot call {func.__name__} before fitting. "
                    f"Call fit() first."
                )
            return func(self, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Timing Decorators
# =============================================================================

def timed(func: F) -> F:
    """
    Decorator that logs function execution time.
    
    Example:
        >>> @timed
        ... def process_data(df):
        ...     return df.apply(expensive_operation)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            f"Function {func.__name__} completed",
            duration_seconds=round(elapsed, 4),
        )
        return result
    
    return wrapper  # type: ignore


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator that limits function execution time.
    
    Note: Only works on Unix-like systems.
    
    Args:
        seconds: Maximum execution time
        
    Example:
        >>> @timeout(30)
        ... def long_running_task():
        ...     # Must complete within 30 seconds
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import signal
            import platform
            
            if platform.system() == "Windows":
                # Windows doesn't support SIGALRM
                logger.warning("Timeout decorator not supported on Windows")
                return func(*args, **kwargs)
            
            def handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))
            
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Deprecation Decorator
# =============================================================================

def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    replacement: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Mark a function as deprecated.
    
    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        replacement: Suggested replacement
        
    Example:
        >>> @deprecated(reason="Use new_function instead", version="2.0")
        ... def old_function():
        ...     pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"Function '{func.__name__}' is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            if replacement:
                message += f". Use '{replacement}' instead"
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Rate Limiting
# =============================================================================

def rate_limit(calls: int, period: float) -> Callable[[F], F]:
    """
    Decorator that limits function call rate.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
        
    Example:
        >>> @rate_limit(calls=10, period=60)  # 10 calls per minute
        ... def api_request():
        ...     return requests.get(url)
    """
    def decorator(func: F) -> F:
        call_times: list = []
        lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                now = time.time()
                # Remove old calls
                call_times[:] = [t for t in call_times if now - t < period]
                
                if len(call_times) >= calls:
                    wait_time = period - (now - call_times[0])
                    if wait_time > 0:
                        logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                        time.sleep(wait_time)
                        call_times.pop(0)
                
                call_times.append(time.time())
            
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Singleton Decorator
# =============================================================================

def singleton(cls: Type) -> Type:
    """
    Singleton decorator for classes.
    
    Example:
        >>> @singleton
        ... class Database:
        ...     pass
        >>> db1 = Database()
        >>> db2 = Database()
        >>> db1 is db2  # True
    """
    instances: Dict[Type, Any] = {}
    lock = Lock()
    
    @functools.wraps(cls, updated=[])
    class Wrapper(cls):  # type: ignore
        def __new__(cls_inner, *args: Any, **kwargs: Any) -> Any:
            with lock:
                if cls not in instances:
                    instances[cls] = super().__new__(cls_inner)
                return instances[cls]
    
    return Wrapper
