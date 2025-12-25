"""
Logging Framework for Data Science Master System.

Provides structured logging with:
- Multiple handlers (console, file, rotating)
- JSON formatting for production
- Colored console output for development
- Context injection (request ID, user, etc.)
- Performance timing utilities
- Integration with monitoring systems

Example:
    >>> from data_science_master_system.core import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started", extra={"dataset": "train.csv", "rows": 10000})
"""

import logging
import logging.handlers
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps
from contextlib import contextmanager


# =============================================================================
# Custom Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    
    Colors:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Bold Red
    """
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[34m{record.name}{self.RESET}"  # Blue
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for production logging.
    
    Outputs structured JSON logs compatible with log aggregation
    systems like ELK Stack, Splunk, or cloud logging services.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, default=str)


# =============================================================================
# Logger Class
# =============================================================================

class Logger:
    """
    Enhanced logger with structured logging capabilities.
    
    Features:
        - Automatic context injection
        - Performance timing
        - Multiple output handlers
        - Log level configuration
        
    Example:
        >>> logger = Logger("my_module")
        >>> logger.info("Processing data", dataset="train.csv", rows=10000)
        >>> with logger.timer("data_loading"):
        ...     load_data()
    """
    
    _instances: Dict[str, "Logger"] = {}
    _configured = False
    
    def __new__(cls, name: str, **kwargs: Any) -> "Logger":
        """Implement singleton pattern per logger name."""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(
        self,
        name: str,
        level: Union[str, int] = logging.INFO,
        log_dir: Optional[Path] = None,
        json_format: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> None:
        """
        Initialize the logger.
        
        Args:
            name: Logger name (usually __name__)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (optional)
            json_format: Use JSON formatting for file logs
            max_bytes: Max size per log file before rotation
            backup_count: Number of backup files to keep
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper()))
        self._logger.handlers = []  # Clear existing handlers
        
        self._setup_console_handler()
        
        if log_dir:
            self._setup_file_handler(log_dir, json_format, max_bytes, backup_count)
        
        self._initialized = True
    
    def _setup_console_handler(self) -> None:
        """Set up colored console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def _setup_file_handler(
        self,
        log_dir: Path,
        json_format: bool,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """Set up rotating file handler."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.name.replace('.', '_')}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            )
        
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Internal logging method with extra fields."""
        extra = {"extra_fields": kwargs} if kwargs else {}
        
        # Format message with kwargs for console
        if kwargs:
            extra_str = " | " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = message + extra_str
        
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, extra={"extra_fields": kwargs})
    
    @contextmanager
    def timer(self, operation: str):
        """
        Context manager for timing operations.
        
        Example:
            >>> with logger.timer("data_loading"):
            ...     data = load_large_dataset()
        """
        start_time = time.perf_counter()
        self.info(f"Starting: {operation}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.info(f"Completed: {operation}", duration_seconds=round(elapsed, 4))
    
    def timed(self, func):
        """
        Decorator for timing function execution.
        
        Example:
            >>> @logger.timed
            ... def process_data(df):
            ...     return df.apply(expensive_operation)
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.timer(func.__name__):
                return func(*args, **kwargs)
        return wrapper


# =============================================================================
# Module-level functions
# =============================================================================

def get_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    json_format: bool = False,
) -> Logger:
    """
    Get or create a logger instance.
    
    This is the recommended way to get a logger in your modules.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level
        log_dir: Directory for log files
        json_format: Use JSON formatting
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return Logger(name, level=level, log_dir=log_dir, json_format=json_format)


def configure_root_logger(
    level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    json_format: bool = False,
) -> None:
    """
    Configure the root logger for the entire application.
    
    Call this once at application startup to set default logging behavior.
    
    Args:
        level: Default log level
        log_dir: Directory for log files
        json_format: Use JSON formatting
    """
    Logger("data_science_master_system", level=level, log_dir=log_dir, json_format=json_format)


# =============================================================================
# Decorators
# =============================================================================

def log_exceptions(logger: Optional[Logger] = None):
    """
    Decorator to log exceptions from functions.
    
    Args:
        logger: Logger instance (creates one if not provided)
        
    Example:
        >>> @log_exceptions()
        ... def risky_operation():
        ...     # ... code that might raise
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}",
                    function=func.__name__,
                    error_type=type(e).__name__,
                )
                raise
        return wrapper
    return decorator
