"""
Polars Processor for Data Science Master System.

High-performance data processing using Polars:
- Fast columnar operations
- Lazy evaluation support
- Memory-efficient processing
- Native parallel execution

Example:
    >>> processor = PolarsProcessor()
    >>> df = processor.read("data.csv")
    >>> df = processor.filter(df, pl.col("age") > 18)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.exceptions import DataProcessingError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    
    # Dummy class for type hinting
    class DummyPl:
        DataFrame = Any
        LazyFrame = Any
        Series = Any
        col = Any
        lit = Any
        all = Any
        Int64 = Any
        Float64 = Any
        Utf8 = Any
        Boolean = Any
    
    if pl is None:
        pl = DummyPl()


class PolarsProcessor(BaseProcessor):
    """
    Polars-based high-performance data processor.
    
    Features:
    - Fast columnar operations
    - Lazy evaluation for query optimization
    - Native parallel execution
    - Memory-efficient processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not POLARS_AVAILABLE:
            raise DataProcessingError(
                "Polars not installed. Install with: pip install polars"
            )
        super().__init__(config)
        self.lazy = config.get("lazy", False) if config else False
    
    def fit(self, data: Any, **kwargs: Any) -> "PolarsProcessor":
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
    ) -> pl.DataFrame:
        """Read data from file."""
        path = Path(path)
        fmt = format or self._detect_format(path)
        
        readers = {
            "csv": pl.read_csv,
            "parquet": pl.read_parquet,
            "json": pl.read_json,
            "ndjson": pl.read_ndjson,
        }
        
        if fmt not in readers:
            raise DataProcessingError(f"Unsupported format for Polars: {fmt}")
        
        df = readers[fmt](path, **kwargs)
        logger.debug(f"Read {len(df)} rows from {path}")
        return df
    
    def write(
        self,
        data: pl.DataFrame,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write data to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format or self._detect_format(path)
        
        writers = {
            "csv": lambda: data.write_csv(path, **kwargs),
            "parquet": lambda: data.write_parquet(path, **kwargs),
            "json": lambda: data.write_json(path, **kwargs),
            "ndjson": lambda: data.write_ndjson(path, **kwargs),
        }
        
        if fmt not in writers:
            raise DataProcessingError(f"Unsupported format for Polars: {fmt}")
        
        writers[fmt]()
        logger.debug(f"Wrote {len(data)} rows to {path}")
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format."""
        suffix = path.suffix.lower()
        formats = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".json": "json",
            ".ndjson": "ndjson",
        }
        return formats.get(suffix, "csv")
    
    # =========================================================================
    # Selection Operations
    # =========================================================================
    
    def select(
        self,
        data: pl.DataFrame,
        columns: Union[str, List[str]],
    ) -> pl.DataFrame:
        """Select columns."""
        if isinstance(columns, str):
            columns = [columns]
        return data.select(columns)
    
    def filter(
        self,
        data: pl.DataFrame,
        condition: Any,
    ) -> pl.DataFrame:
        """Filter rows."""
        if isinstance(condition, str):
            # Parse string expression
            return data.filter(eval(f"pl.{condition}"))
        else:
            return data.filter(condition)
    
    def head(self, data: pl.DataFrame, n: int = 5) -> pl.DataFrame:
        """Get first n rows."""
        return data.head(n)
    
    def tail(self, data: pl.DataFrame, n: int = 5) -> pl.DataFrame:
        """Get last n rows."""
        return data.tail(n)
    
    def sample(
        self,
        data: pl.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: int = 42,
    ) -> pl.DataFrame:
        """Random sample."""
        if frac is not None:
            n = int(len(data) * frac)
        return data.sample(n=n, seed=random_state)
    
    # =========================================================================
    # Transformation Operations
    # =========================================================================
    
    def rename(
        self,
        data: pl.DataFrame,
        columns: Dict[str, str],
    ) -> pl.DataFrame:
        """Rename columns."""
        return data.rename(columns)
    
    def drop(
        self,
        data: pl.DataFrame,
        columns: Union[str, List[str]],
    ) -> pl.DataFrame:
        """Drop columns."""
        if isinstance(columns, str):
            columns = [columns]
        return data.drop(columns)
    
    def dropna(
        self,
        data: pl.DataFrame,
        subset: Optional[List[str]] = None,
        how: str = "any",
    ) -> pl.DataFrame:
        """Drop rows with missing values."""
        if subset:
            if how == "any":
                mask = pl.lit(True)
                for col in subset:
                    mask = mask & pl.col(col).is_not_null()
                return data.filter(mask)
            else:  # how == "all"
                mask = pl.lit(False)
                for col in subset:
                    mask = mask | pl.col(col).is_not_null()
                return data.filter(mask)
        return data.drop_nulls()
    
    def fillna(
        self,
        data: pl.DataFrame,
        value: Any = None,
        method: Optional[str] = None,
    ) -> pl.DataFrame:
        """Fill missing values."""
        if method == "forward":
            return data.fill_null(strategy="forward")
        elif method == "backward":
            return data.fill_null(strategy="backward")
        return data.fill_null(value)
    
    def astype(
        self,
        data: pl.DataFrame,
        dtype: Dict[str, Any],
    ) -> pl.DataFrame:
        """Convert column types."""
        exprs = []
        for col, dt in dtype.items():
            if dt == "int":
                exprs.append(pl.col(col).cast(pl.Int64))
            elif dt == "float":
                exprs.append(pl.col(col).cast(pl.Float64))
            elif dt == "str":
                exprs.append(pl.col(col).cast(pl.Utf8))
            elif dt == "bool":
                exprs.append(pl.col(col).cast(pl.Boolean))
        
        if exprs:
            return data.with_columns(exprs)
        return data
    
    def apply(
        self,
        data: pl.DataFrame,
        func: Callable,
        axis: int = 0,
    ) -> pl.DataFrame:
        """Apply function (Polars uses different paradigm)."""
        if axis == 1:
            return data.map_rows(func)
        else:
            # Column-wise application
            return data.select([pl.col(c).map_elements(func) for c in data.columns])
    
    # =========================================================================
    # Sorting Operations
    # =========================================================================
    
    def sort(
        self,
        data: pl.DataFrame,
        by: Union[str, List[str]],
        descending: bool = False,
    ) -> pl.DataFrame:
        """Sort by columns."""
        return data.sort(by, descending=descending)
    
    def unique(
        self,
        data: pl.DataFrame,
        column: str,
    ) -> pl.Series:
        """Get unique values."""
        return data[column].unique()
    
    # =========================================================================
    # Aggregation Operations
    # =========================================================================
    
    def agg(
        self,
        data: pl.DataFrame,
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> pl.DataFrame:
        """Apply aggregations."""
        exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                if agg == "sum":
                    exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif agg == "mean":
                    exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif agg == "min":
                    exprs.append(pl.col(col).min().alias(f"{col}_min"))
                elif agg == "max":
                    exprs.append(pl.col(col).max().alias(f"{col}_max"))
                elif agg == "count":
                    exprs.append(pl.col(col).count().alias(f"{col}_count"))
                elif agg == "std":
                    exprs.append(pl.col(col).std().alias(f"{col}_std"))
        
        return data.select(exprs)
    
    def group_agg(
        self,
        data: pl.DataFrame,
        by: List[str],
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> pl.DataFrame:
        """Group by and aggregate."""
        exprs = []
        for col, aggs in aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                if agg == "sum":
                    exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif agg == "mean":
                    exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif agg == "min":
                    exprs.append(pl.col(col).min().alias(f"{col}_min"))
                elif agg == "max":
                    exprs.append(pl.col(col).max().alias(f"{col}_max"))
                elif agg == "count":
                    exprs.append(pl.col(col).count().alias(f"{col}_count"))
                elif agg == "std":
                    exprs.append(pl.col(col).std().alias(f"{col}_std"))
                elif agg == "first":
                    exprs.append(pl.col(col).first().alias(f"{col}_first"))
                elif agg == "last":
                    exprs.append(pl.col(col).last().alias(f"{col}_last"))
        
        return data.group_by(by).agg(exprs)
    
    def group_sum(
        self,
        data: pl.DataFrame,
        by: List[str],
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Group by and sum."""
        if columns:
            return data.group_by(by).agg([pl.col(c).sum() for c in columns])
        return data.group_by(by).agg(pl.all().sum())
    
    def group_mean(
        self,
        data: pl.DataFrame,
        by: List[str],
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Group by and mean."""
        if columns:
            return data.group_by(by).agg([pl.col(c).mean() for c in columns])
        return data.group_by(by).agg(pl.all().mean())
    
    def group_count(
        self,
        data: pl.DataFrame,
        by: List[str],
    ) -> pl.DataFrame:
        """Group by and count."""
        return data.group_by(by).count()
    
    def group_first(
        self,
        data: pl.DataFrame,
        by: List[str],
    ) -> pl.DataFrame:
        """Group by and get first."""
        return data.group_by(by).first()
    
    def group_last(
        self,
        data: pl.DataFrame,
        by: List[str],
    ) -> pl.DataFrame:
        """Group by and get last."""
        return data.group_by(by).last()
    
    # =========================================================================
    # Join Operations
    # =========================================================================
    
    def merge(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
    ) -> pl.DataFrame:
        """Merge DataFrames."""
        return left.join(right, on=on, left_on=left_on, right_on=right_on, how=how)
    
    def concat(
        self,
        dfs: List[pl.DataFrame],
        axis: int = 0,
    ) -> pl.DataFrame:
        """Concatenate DataFrames."""
        if axis == 0:
            return pl.concat(dfs)
        else:
            return pl.concat(dfs, how="horizontal")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def describe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Descriptive statistics."""
        return data.describe()
    
    def info(self, data: pl.DataFrame) -> Dict[str, Any]:
        """DataFrame info."""
        return {
            "shape": data.shape,
            "columns": data.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)},
            "null_counts": {col: data[col].null_count() for col in data.columns},
        }
    
    def shape(self, data: pl.DataFrame) -> tuple:
        """DataFrame shape."""
        return data.shape
    
    def columns(self, data: pl.DataFrame) -> List[str]:
        """Column names."""
        return data.columns
    
    def dtypes(self, data: pl.DataFrame) -> Dict[str, str]:
        """Column dtypes."""
        return {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
    
    # =========================================================================
    # Lazy Evaluation
    # =========================================================================
    
    def lazy(self, data: pl.DataFrame) -> pl.LazyFrame:
        """Convert to lazy frame for query optimization."""
        return data.lazy()
    
    def collect(self, data: pl.LazyFrame) -> pl.DataFrame:
        """Execute lazy query and return DataFrame."""
        return data.collect()
