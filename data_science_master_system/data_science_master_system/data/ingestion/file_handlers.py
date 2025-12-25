"""
File Handlers for Data Science Master System.

Provides unified file reading/writing with support for:
- CSV, TSV, and delimited files
- Excel (xls, xlsx)
- JSON and JSON Lines
- Parquet, Avro, ORC
- Feather, Pickle
- XML
- Automatic format detection

Example:
    >>> handler = FileHandler()
    >>> df = handler.read("data.csv")
    >>> handler.write(df, "output.parquet")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io

from data_science_master_system.core.base_classes import BaseDataSource
from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class FileHandler(BaseDataSource):
    """
    Unified file handler supporting multiple formats.
    
    Automatically detects file format from extension and uses
    the appropriate reader. Supports both pandas and polars backends.
    
    Example:
        >>> handler = FileHandler(backend="pandas")
        >>> df = handler.read("data.csv")
        >>> df = handler.read("large_data.parquet", columns=["id", "value"])
    """
    
    SUPPORTED_FORMATS = {
        "csv": [".csv", ".tsv", ".txt"],
        "excel": [".xls", ".xlsx", ".xlsm"],
        "json": [".json", ".jsonl", ".ndjson"],
        "parquet": [".parquet", ".pq"],
        "feather": [".feather", ".ftr"],
        "pickle": [".pkl", ".pickle"],
        "avro": [".avro"],
        "orc": [".orc"],
        "xml": [".xml"],
    }
    
    def __init__(
        self,
        backend: str = "pandas",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize file handler.
        
        Args:
            backend: Processing backend ("pandas", "polars")
            config: Additional configuration
        """
        super().__init__(config)
        self.backend = backend.lower()
        self._connected = True  # Files don't need connection
        
        # Import backend
        if self.backend == "pandas":
            import pandas as pd
            self._pd = pd
        elif self.backend == "polars":
            try:
                import polars as pl
                self._pl = pl
            except ImportError:
                logger.warning("Polars not installed, falling back to pandas")
                self.backend = "pandas"
                import pandas as pd
                self._pd = pd
    
    def connect(self) -> None:
        """No connection needed for files."""
        self._connected = True
    
    def disconnect(self) -> None:
        """No disconnection needed for files."""
        self._connected = False
    
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
            DataFrame (pandas or polars depending on backend)
            
        Example:
            >>> df = handler.read("data.csv", encoding="utf-8")
            >>> df = handler.read("data.parquet", columns=["id", "name"])
        """
        path = Path(path)
        
        if not path.exists():
            raise DataIngestionError(
                f"File not found: {path}",
                context={"path": str(path)},
            )
        
        # Detect format
        file_format = format or self._detect_format(path)
        
        logger.info(f"Reading file", path=str(path), format=file_format, backend=self.backend)
        
        try:
            if self.backend == "pandas":
                return self._read_pandas(path, file_format, **kwargs)
            else:
                return self._read_polars(path, file_format, **kwargs)
        except Exception as e:
            raise DataIngestionError(
                f"Failed to read file: {path}",
                context={"path": str(path), "format": file_format, "error": str(e)},
            )
    
    def write(
        self,
        data: Any,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write data to file.
        
        Args:
            data: DataFrame to write
            path: Output path
            format: Output format (auto-detected from extension)
            **kwargs: Format-specific options
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = format or self._detect_format(path)
        
        logger.info(f"Writing file", path=str(path), format=file_format)
        
        try:
            if self.backend == "pandas":
                self._write_pandas(data, path, file_format, **kwargs)
            else:
                self._write_polars(data, path, file_format, **kwargs)
        except Exception as e:
            raise DataIngestionError(
                f"Failed to write file: {path}",
                context={"path": str(path), "format": file_format, "error": str(e)},
            )
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()
        
        for format_name, extensions in self.SUPPORTED_FORMATS.items():
            if suffix in extensions:
                return format_name
        
        raise DataIngestionError(
            f"Unsupported file format: {suffix}",
            context={"path": str(path), "supported": list(self.SUPPORTED_FORMATS.keys())},
        )
    
    def _read_pandas(self, path: Path, format: str, **kwargs: Any) -> Any:
        """Read file using pandas."""
        pd = self._pd
        
        readers = {
            "csv": lambda: pd.read_csv(path, **kwargs),
            "excel": lambda: pd.read_excel(path, **kwargs),
            "json": lambda: self._read_json_pandas(path, **kwargs),
            "parquet": lambda: pd.read_parquet(path, **kwargs),
            "feather": lambda: pd.read_feather(path, **kwargs),
            "pickle": lambda: pd.read_pickle(path, **kwargs),
            "xml": lambda: pd.read_xml(path, **kwargs),
        }
        
        if format not in readers:
            raise DataIngestionError(f"Unsupported format for pandas: {format}")
        
        return readers[format]()
    
    def _read_json_pandas(self, path: Path, **kwargs: Any) -> Any:
        """Read JSON/JSONL with pandas."""
        pd = self._pd
        
        # Check if JSON Lines
        if path.suffix.lower() in (".jsonl", ".ndjson"):
            return pd.read_json(path, lines=True, **kwargs)
        return pd.read_json(path, **kwargs)
    
    def _write_pandas(self, data: Any, path: Path, format: str, **kwargs: Any) -> None:
        """Write file using pandas."""
        writers = {
            "csv": lambda: data.to_csv(path, index=False, **kwargs),
            "excel": lambda: data.to_excel(path, index=False, **kwargs),
            "json": lambda: data.to_json(path, **kwargs),
            "parquet": lambda: data.to_parquet(path, **kwargs),
            "feather": lambda: data.to_feather(path, **kwargs),
            "pickle": lambda: data.to_pickle(path, **kwargs),
        }
        
        if format not in writers:
            raise DataIngestionError(f"Unsupported format for writing: {format}")
        
        writers[format]()
    
    def _read_polars(self, path: Path, format: str, **kwargs: Any) -> Any:
        """Read file using polars."""
        pl = self._pl
        
        readers = {
            "csv": lambda: pl.read_csv(path, **kwargs),
            "excel": lambda: pl.read_excel(path, **kwargs),
            "json": lambda: pl.read_json(path, **kwargs),
            "parquet": lambda: pl.read_parquet(path, **kwargs),
        }
        
        if format not in readers:
            raise DataIngestionError(f"Unsupported format for polars: {format}")
        
        return readers[format]()
    
    def _write_polars(self, data: Any, path: Path, format: str, **kwargs: Any) -> None:
        """Write file using polars."""
        writers = {
            "csv": lambda: data.write_csv(path, **kwargs),
            "json": lambda: data.write_json(path, **kwargs),
            "parquet": lambda: data.write_parquet(path, **kwargs),
        }
        
        if format not in writers:
            raise DataIngestionError(f"Unsupported format for polars writing: {format}")
        
        writers[format]()
    
    def read_chunked(
        self,
        path: Union[str, Path],
        chunk_size: int = 10000,
        **kwargs: Any,
    ):
        """
        Read file in chunks (generator).
        
        Args:
            path: Path to file
            chunk_size: Rows per chunk
            **kwargs: Additional read options
            
        Yields:
            DataFrame chunks
        """
        path = Path(path)
        file_format = self._detect_format(path)
        
        if file_format != "csv":
            raise DataIngestionError(
                "Chunked reading only supported for CSV files",
                context={"format": file_format},
            )
        
        if self.backend == "pandas":
            for chunk in self._pd.read_csv(path, chunksize=chunk_size, **kwargs):
                yield chunk
        else:
            # Polars lazy reading
            lazy_df = self._pl.scan_csv(path, **kwargs)
            # Manual chunking would need additional implementation
            yield self._pl.read_csv(path, **kwargs)
    
    def get_schema(self, path: Union[str, Path]) -> Dict[str, str]:
        """
        Get file schema (column names and types).
        
        Args:
            path: Path to file
            
        Returns:
            Dictionary of column names to data types
        """
        # Read first few rows to infer schema
        if self.backend == "pandas":
            df = self._pd.read_csv(path, nrows=100) if Path(path).suffix.lower() == ".csv" else self.read(path)
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        else:
            df = self._pl.read_csv(path, n_rows=100) if Path(path).suffix.lower() == ".csv" else self.read(path)
            return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
