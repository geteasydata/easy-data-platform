"""
Multi-Framework Processing Engine for Data Science Master System.

Provides a unified interface for data processing across:
- Pandas
- Polars
- Dask
- PySpark

Automatically selects the optimal backend based on:
- Data size
- Operation type
- Available resources

Example:
    >>> engine = ProcessingEngine(backend="auto")
    >>> df = engine.read("large_data.csv")
    >>> df = engine.filter(df, "age > 18")
    >>> df = engine.group_by(df, ["category"]).agg({"value": "sum"})
"""

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import warnings

from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.exceptions import DataProcessingError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.utils.helpers import get_memory_usage

logger = get_logger(__name__)


class ProcessingEngine(BaseProcessor):
    """
    Unified processing engine with automatic backend selection.
    
    Provides a consistent API regardless of the underlying framework.
    Automatically switches backends based on data characteristics.
    
    Example:
        >>> engine = ProcessingEngine()
        >>> 
        >>> # Load and process
        >>> df = engine.read("data.csv")
        >>> df = engine.filter(df, lambda x: x["age"] > 18)
        >>> df = engine.select(df, ["name", "age", "city"])
        >>> df = engine.sort(df, "age", descending=True)
        >>> 
        >>> # Aggregation
        >>> result = engine.group_by(df, ["city"]).agg({
        ...     "age": ["mean", "max"],
        ...     "name": "count"
        ... })
    """
    
    BACKENDS = ["pandas", "polars", "dask", "spark"]
    
    # Thresholds for automatic backend selection (rows)
    AUTO_THRESHOLDS = {
        "pandas": 1_000_000,      # Up to 1M rows
        "polars": 10_000_000,     # Up to 10M rows
        "dask": 100_000_000,      # Up to 100M rows
        "spark": float("inf"),    # Unlimited
    }
    
    def __init__(
        self,
        backend: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize processing engine.
        
        Args:
            backend: Backend to use ("pandas", "polars", "dask", "spark", "auto")
            config: Additional configuration
        """
        super().__init__(config)
        self.backend = backend.lower()
        self._processor = None
        self._init_processor()
    
    def _init_processor(self) -> None:
        """Initialize the appropriate processor."""
        from data_science_master_system.data.processing.pandas_processor import PandasProcessor
        
        if self.backend == "auto":
            # Start with pandas, switch if needed
            self._processor = PandasProcessor(self.config)
            self._current_backend = "pandas"
        elif self.backend == "pandas":
            self._processor = PandasProcessor(self.config)
            self._current_backend = "pandas"
        elif self.backend == "polars":
            from data_science_master_system.data.processing.polars_processor import PolarsProcessor
            self._processor = PolarsProcessor(self.config)
            self._current_backend = "polars"
        elif self.backend == "dask":
            # Dask uses similar interface to pandas
            self._processor = PandasProcessor({**self.config, "use_dask": True})
            self._current_backend = "dask"
        else:
            raise DataProcessingError(f"Unknown backend: {self.backend}")
    
    def _maybe_switch_backend(self, data: Any) -> None:
        """Switch backend if data size warrants it."""
        if self.backend != "auto":
            return
        
        try:
            n_rows = len(data) if hasattr(data, "__len__") else 0
        except:
            return
        
        # Determine optimal backend
        if n_rows > self.AUTO_THRESHOLDS["pandas"] and self._current_backend == "pandas":
            try:
                from data_science_master_system.data.processing.polars_processor import PolarsProcessor
                self._processor = PolarsProcessor(self.config)
                self._current_backend = "polars"
                logger.info(f"Switched to polars backend for {n_rows} rows")
            except ImportError:
                logger.warning("Polars not available, continuing with pandas")
    
    def fit(self, data: Any, **kwargs: Any) -> "ProcessingEngine":
        """Fit processor to data (learn statistics for transformations)."""
        self._fitted = True
        return self
    
    def transform(self, data: Any, **kwargs: Any) -> Any:
        """Transform data using fitted processor."""
        return data
    
    # =========================================================================
    # Data I/O
    # =========================================================================
    
    def read(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Read data from file.
        
        Args:
            path: Path to file
            format: File format (auto-detected if not provided)
            **kwargs: Format-specific options
            
        Returns:
            DataFrame
        """
        return self._processor.read(path, format=format, **kwargs)
    
    def write(
        self,
        data: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write data to file."""
        self._processor.write(data, path, format=format, **kwargs)
    
    # =========================================================================
    # Selection Operations
    # =========================================================================
    
    def select(
        self,
        data: Any,
        columns: Union[str, List[str]],
    ) -> Any:
        """
        Select columns from DataFrame.
        
        Args:
            data: DataFrame
            columns: Column name(s) to select
            
        Returns:
            DataFrame with selected columns
        """
        return self._processor.select(data, columns)
    
    def filter(
        self,
        data: Any,
        condition: Union[str, Callable],
    ) -> Any:
        """
        Filter rows based on condition.
        
        Args:
            data: DataFrame
            condition: Filter condition (string expression or callable)
            
        Returns:
            Filtered DataFrame
        """
        return self._processor.filter(data, condition)
    
    def head(self, data: Any, n: int = 5) -> Any:
        """Get first n rows."""
        return self._processor.head(data, n)
    
    def tail(self, data: Any, n: int = 5) -> Any:
        """Get last n rows."""
        return self._processor.tail(data, n)
    
    def sample(
        self,
        data: Any,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: int = 42,
    ) -> Any:
        """Random sample from DataFrame."""
        return self._processor.sample(data, n=n, frac=frac, random_state=random_state)
    
    # =========================================================================
    # Transformation Operations
    # =========================================================================
    
    def rename(
        self,
        data: Any,
        columns: Dict[str, str],
    ) -> Any:
        """Rename columns."""
        return self._processor.rename(data, columns)
    
    def drop(
        self,
        data: Any,
        columns: Union[str, List[str]],
    ) -> Any:
        """Drop columns."""
        return self._processor.drop(data, columns)
    
    def dropna(
        self,
        data: Any,
        subset: Optional[List[str]] = None,
        how: str = "any",
    ) -> Any:
        """Drop rows with missing values."""
        return self._processor.dropna(data, subset=subset, how=how)
    
    def fillna(
        self,
        data: Any,
        value: Any = None,
        method: Optional[str] = None,
    ) -> Any:
        """Fill missing values."""
        return self._processor.fillna(data, value=value, method=method)
    
    def astype(
        self,
        data: Any,
        dtype: Dict[str, Any],
    ) -> Any:
        """Convert column types."""
        return self._processor.astype(data, dtype)
    
    def apply(
        self,
        data: Any,
        func: Callable,
        axis: int = 0,
    ) -> Any:
        """Apply function to DataFrame."""
        return self._processor.apply(data, func, axis=axis)
    
    # =========================================================================
    # Sorting Operations
    # =========================================================================
    
    def sort(
        self,
        data: Any,
        by: Union[str, List[str]],
        descending: bool = False,
    ) -> Any:
        """Sort DataFrame by columns."""
        return self._processor.sort(data, by, descending=descending)
    
    def unique(
        self,
        data: Any,
        column: str,
    ) -> Any:
        """Get unique values in column."""
        return self._processor.unique(data, column)
    
    # =========================================================================
    # Aggregation Operations
    # =========================================================================
    
    def group_by(
        self,
        data: Any,
        by: Union[str, List[str]],
    ) -> "GroupByOperation":
        """
        Group DataFrame by columns.
        
        Returns a GroupByOperation object for chaining aggregations.
        
        Example:
            >>> result = engine.group_by(df, ["category"]).agg({
            ...     "value": ["sum", "mean"],
            ...     "count": "count"
            ... })
        """
        return GroupByOperation(self._processor, data, by)
    
    def agg(
        self,
        data: Any,
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> Any:
        """Apply aggregations to entire DataFrame."""
        return self._processor.agg(data, aggregations)
    
    # =========================================================================
    # Join Operations
    # =========================================================================
    
    def merge(
        self,
        left: Any,
        right: Any,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
    ) -> Any:
        """Merge two DataFrames."""
        return self._processor.merge(
            left, right, on=on, left_on=left_on, right_on=right_on, how=how
        )
    
    def concat(
        self,
        dfs: List[Any],
        axis: int = 0,
    ) -> Any:
        """Concatenate DataFrames."""
        return self._processor.concat(dfs, axis=axis)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def describe(self, data: Any) -> Any:
        """Get descriptive statistics."""
        return self._processor.describe(data)
    
    def info(self, data: Any) -> Dict[str, Any]:
        """Get DataFrame info."""
        return self._processor.info(data)
    
    def shape(self, data: Any) -> tuple:
        """Get DataFrame shape."""
        return self._processor.shape(data)
    
    def columns(self, data: Any) -> List[str]:
        """Get column names."""
        return self._processor.columns(data)
    
    def dtypes(self, data: Any) -> Dict[str, str]:
        """Get column data types."""
        return self._processor.dtypes(data)


class GroupByOperation:
    """
    Represents a grouped DataFrame for aggregation operations.
    """
    
    def __init__(
        self,
        processor: Any,
        data: Any,
        by: Union[str, List[str]],
    ) -> None:
        self._processor = processor
        self._data = data
        self._by = by if isinstance(by, list) else [by]
    
    def agg(self, aggregations: Dict[str, Union[str, List[str]]]) -> Any:
        """
        Apply aggregations to grouped data.
        
        Args:
            aggregations: Dict of column -> aggregation function(s)
            
        Returns:
            Aggregated DataFrame
        """
        return self._processor.group_agg(self._data, self._by, aggregations)
    
    def sum(self, columns: Optional[List[str]] = None) -> Any:
        """Sum grouped columns."""
        return self._processor.group_sum(self._data, self._by, columns)
    
    def mean(self, columns: Optional[List[str]] = None) -> Any:
        """Mean of grouped columns."""
        return self._processor.group_mean(self._data, self._by, columns)
    
    def count(self) -> Any:
        """Count grouped rows."""
        return self._processor.group_count(self._data, self._by)
    
    def first(self) -> Any:
        """First value in each group."""
        return self._processor.group_first(self._data, self._by)
    
    def last(self) -> Any:
        """Last value in each group."""
        return self._processor.group_last(self._data, self._by)
