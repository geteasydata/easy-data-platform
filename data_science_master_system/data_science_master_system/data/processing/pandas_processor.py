"""
Pandas Processor for Data Science Master System.

High-performance data processing using pandas with:
- Dask integration for parallelization
- Memory-efficient operations
- Chunked processing for large files

Example:
    >>> processor = PandasProcessor()
    >>> df = processor.read("data.csv")
    >>> df = processor.filter(df, "age > 18")
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.exceptions import DataProcessingError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class PandasProcessor(BaseProcessor):
    """
    Pandas-based data processor.
    
    Features:
    - Standard pandas operations
    - Optional Dask parallelization
    - Memory-efficient chunked processing
    - Automatic type optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.use_dask = config.get("use_dask", False) if config else False
        self._dask = None
        
        if self.use_dask:
            try:
                import dask.dataframe as dd
                self._dask = dd
                logger.info("Dask integration enabled")
            except ImportError:
                logger.warning("Dask not available, using pandas")
                self.use_dask = False
    
    def fit(self, data: Any, **kwargs: Any) -> "PandasProcessor":
        """Fit processor to data."""
        self._fitted = True
        return self
    
    def transform(self, data: Any, **kwargs: Any) -> Any:
        """Transform data."""
        return data
    
    # =========================================================================
    # I/O Operations
    # =========================================================================
    
    def read(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read data from file."""
        path = Path(path)
        fmt = format or self._detect_format(path)
        
        readers = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "json": pd.read_json,
            "excel": pd.read_excel,
            "feather": pd.read_feather,
            "pickle": pd.read_pickle,
        }
        
        if fmt not in readers:
            raise DataProcessingError(f"Unsupported format: {fmt}")
        
        df = readers[fmt](path, **kwargs)
        logger.debug(f"Read {len(df)} rows from {path}")
        return df
    
    def write(
        self,
        data: pd.DataFrame,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write data to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format or self._detect_format(path)
        
        writers = {
            "csv": lambda: data.to_csv(path, index=False, **kwargs),
            "parquet": lambda: data.to_parquet(path, **kwargs),
            "json": lambda: data.to_json(path, **kwargs),
            "excel": lambda: data.to_excel(path, index=False, **kwargs),
            "feather": lambda: data.to_feather(path, **kwargs),
            "pickle": lambda: data.to_pickle(path, **kwargs),
        }
        
        if fmt not in writers:
            raise DataProcessingError(f"Unsupported format: {fmt}")
        
        writers[fmt]()
        logger.debug(f"Wrote {len(data)} rows to {path}")
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()
        formats = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".json": "json",
            ".xlsx": "excel",
            ".xls": "excel",
            ".feather": "feather",
            ".pkl": "pickle",
            ".pickle": "pickle",
        }
        return formats.get(suffix, "csv")
    
    # =========================================================================
    # Selection Operations
    # =========================================================================
    
    def select(
        self,
        data: pd.DataFrame,
        columns: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Select columns."""
        if isinstance(columns, str):
            columns = [columns]
        return data[columns]
    
    def filter(
        self,
        data: pd.DataFrame,
        condition: Union[str, Callable],
    ) -> pd.DataFrame:
        """Filter rows."""
        if isinstance(condition, str):
            return data.query(condition)
        elif callable(condition):
            mask = data.apply(condition, axis=1)
            return data[mask]
        else:
            return data[condition]
    
    def head(self, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get first n rows."""
        return data.head(n)
    
    def tail(self, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get last n rows."""
        return data.tail(n)
    
    def sample(
        self,
        data: pd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Random sample."""
        return data.sample(n=n, frac=frac, random_state=random_state)
    
    # =========================================================================
    # Transformation Operations
    # =========================================================================
    
    def rename(
        self,
        data: pd.DataFrame,
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        """Rename columns."""
        return data.rename(columns=columns)
    
    def drop(
        self,
        data: pd.DataFrame,
        columns: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Drop columns."""
        if isinstance(columns, str):
            columns = [columns]
        return data.drop(columns=columns)
    
    def dropna(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        how: str = "any",
    ) -> pd.DataFrame:
        """Drop rows with missing values."""
        return data.dropna(subset=subset, how=how)
    
    def fillna(
        self,
        data: pd.DataFrame,
        value: Any = None,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fill missing values."""
        if method:
            return data.fillna(method=method)
        return data.fillna(value)
    
    def astype(
        self,
        data: pd.DataFrame,
        dtype: Dict[str, Any],
    ) -> pd.DataFrame:
        """Convert column types."""
        return data.astype(dtype)
    
    def apply(
        self,
        data: pd.DataFrame,
        func: Callable,
        axis: int = 0,
    ) -> pd.DataFrame:
        """Apply function."""
        return data.apply(func, axis=axis)
    
    # =========================================================================
    # Sorting Operations
    # =========================================================================
    
    def sort(
        self,
        data: pd.DataFrame,
        by: Union[str, List[str]],
        descending: bool = False,
    ) -> pd.DataFrame:
        """Sort by columns."""
        return data.sort_values(by=by, ascending=not descending)
    
    def unique(
        self,
        data: pd.DataFrame,
        column: str,
    ) -> np.ndarray:
        """Get unique values."""
        return data[column].unique()
    
    # =========================================================================
    # Aggregation Operations
    # =========================================================================
    
    def agg(
        self,
        data: pd.DataFrame,
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> pd.DataFrame:
        """Apply aggregations."""
        return data.agg(aggregations)
    
    def group_agg(
        self,
        data: pd.DataFrame,
        by: List[str],
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> pd.DataFrame:
        """Group by and aggregate."""
        result = data.groupby(by).agg(aggregations)
        # Flatten multi-level columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ["_".join(col).strip() for col in result.columns.values]
        return result.reset_index()
    
    def group_sum(
        self,
        data: pd.DataFrame,
        by: List[str],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Group by and sum."""
        grouped = data.groupby(by)
        if columns:
            return grouped[columns].sum().reset_index()
        return grouped.sum().reset_index()
    
    def group_mean(
        self,
        data: pd.DataFrame,
        by: List[str],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Group by and mean."""
        grouped = data.groupby(by)
        if columns:
            return grouped[columns].mean().reset_index()
        return grouped.mean().reset_index()
    
    def group_count(
        self,
        data: pd.DataFrame,
        by: List[str],
    ) -> pd.DataFrame:
        """Group by and count."""
        return data.groupby(by).size().reset_index(name="count")
    
    def group_first(
        self,
        data: pd.DataFrame,
        by: List[str],
    ) -> pd.DataFrame:
        """Group by and get first."""
        return data.groupby(by).first().reset_index()
    
    def group_last(
        self,
        data: pd.DataFrame,
        by: List[str],
    ) -> pd.DataFrame:
        """Group by and get last."""
        return data.groupby(by).last().reset_index()
    
    # =========================================================================
    # Join Operations
    # =========================================================================
    
    def merge(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
    ) -> pd.DataFrame:
        """Merge DataFrames."""
        return pd.merge(
            left, right, on=on, left_on=left_on, right_on=right_on, how=how
        )
    
    def concat(
        self,
        dfs: List[pd.DataFrame],
        axis: int = 0,
    ) -> pd.DataFrame:
        """Concatenate DataFrames."""
        return pd.concat(dfs, axis=axis, ignore_index=True)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def describe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Descriptive statistics."""
        return data.describe()
    
    def info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DataFrame info."""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": data.isnull().sum().to_dict(),
        }
    
    def shape(self, data: pd.DataFrame) -> tuple:
        """DataFrame shape."""
        return data.shape
    
    def columns(self, data: pd.DataFrame) -> List[str]:
        """Column names."""
        return list(data.columns)
    
    def dtypes(self, data: pd.DataFrame) -> Dict[str, str]:
        """Column dtypes."""
        return {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    # =========================================================================
    # Optimization
    # =========================================================================
    
    def optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        - Downcasts numeric types
        - Converts object columns to category where beneficial
        """
        result = data.copy()
        
        for col in result.columns:
            col_type = result[col].dtype
            
            if col_type == "int64":
                result[col] = pd.to_numeric(result[col], downcast="integer")
            elif col_type == "float64":
                result[col] = pd.to_numeric(result[col], downcast="float")
            elif col_type == "object":
                num_unique = result[col].nunique()
                num_total = len(result[col])
                if num_unique / num_total < 0.5:
                    result[col] = result[col].astype("category")
        
        original_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        optimized_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
        
        logger.info(
            f"Memory optimized",
            original_mb=round(original_mb, 2),
            optimized_mb=round(optimized_mb, 2),
            reduction_pct=round((1 - optimized_mb / original_mb) * 100, 1),
        )
        
        return result
