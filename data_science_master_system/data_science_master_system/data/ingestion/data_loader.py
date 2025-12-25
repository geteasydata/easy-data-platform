"""
Unified Data Loader for Data Science Master System.

Provides a single interface for loading data from any source:
- Local files
- Databases
- Cloud storage
- APIs
- Streaming sources

Example:
    >>> loader = DataLoader()
    >>> 
    >>> # Load from file
    >>> df = loader.read("data.csv")
    >>> 
    >>> # Load from database
    >>> df = loader.read("postgresql://user:pass@localhost/db", query="SELECT * FROM users")
    >>> 
    >>> # Load from S3
    >>> df = loader.read("s3://bucket/data.parquet")
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.data.ingestion.file_handlers import FileHandler
from data_science_master_system.data.ingestion.database_connectors import (
    SQLConnector,
    MongoDBConnector,
    RedisConnector,
)
from data_science_master_system.data.ingestion.cloud_storage import (
    S3Adapter,
    GCSAdapter,
    AzureBlobAdapter,
)
from data_science_master_system.data.ingestion.api_clients import RESTClient

logger = get_logger(__name__)


class DataLoader:
    """
    Unified data loader with automatic source detection.
    
    Automatically detects the data source type from the path/URL
    and uses the appropriate loader.
    
    Supported sources:
        - Local files: /path/to/file.csv, C:\\data\\file.parquet
        - Databases: postgresql://, mysql://, sqlite://, mongodb://
        - Cloud: s3://, gs://, azure://
        - APIs: http://, https://
    
    Example:
        >>> loader = DataLoader(backend="pandas")
        >>> 
        >>> # Various sources
        >>> df = loader.read("data.csv")
        >>> df = loader.read("postgresql://localhost/db", query="SELECT * FROM t")
        >>> df = loader.read("s3://bucket/path/data.parquet")
        >>> df = loader.read("https://api.example.com/data")
    """
    
    def __init__(
        self,
        backend: str = "pandas",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize data loader.
        
        Args:
            backend: Processing backend ("pandas", "polars")
            config: Default configuration for all loaders
        """
        self.backend = backend
        self.config = config or {}
        
        # Cached loaders
        self._file_handler = None
        self._connectors: Dict[str, Any] = {}
    
    def read(
        self,
        source: str,
        format: Optional[str] = None,
        query: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Read data from any source.
        
        Args:
            source: Data source (path, URL, or connection string)
            format: Optional format override
            query: SQL query (for databases)
            table: Table name (for databases)
            **kwargs: Source-specific options
            
        Returns:
            Data (typically pandas or polars DataFrame)
        """
        source_type, parsed = self._detect_source_type(source)
        
        logger.info(f"Loading data", source_type=source_type, source=source[:100])
        
        if source_type == "file":
            return self._read_file(source, format, **kwargs)
        elif source_type == "sql":
            return self._read_sql(source, query, table, **kwargs)
        elif source_type == "mongodb":
            return self._read_mongodb(source, query, table, **kwargs)
        elif source_type == "redis":
            return self._read_redis(source, **kwargs)
        elif source_type == "s3":
            return self._read_s3(parsed, **kwargs)
        elif source_type == "gcs":
            return self._read_gcs(parsed, **kwargs)
        elif source_type == "azure":
            return self._read_azure(parsed, **kwargs)
        elif source_type == "http":
            return self._read_http(source, **kwargs)
        else:
            raise DataIngestionError(
                f"Unknown source type: {source_type}",
                context={"source": source},
            )
    
    def write(
        self,
        data: Any,
        destination: str,
        format: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write data to any destination.
        
        Args:
            data: Data to write
            destination: Destination (path, URL, or connection string)
            format: Optional format override
            table: Table name (for databases)
            **kwargs: Destination-specific options
        """
        source_type, parsed = self._detect_source_type(destination)
        
        logger.info(f"Writing data", source_type=source_type, destination=destination[:100])
        
        if source_type == "file":
            self._write_file(data, destination, format, **kwargs)
        elif source_type == "sql":
            self._write_sql(data, destination, table, **kwargs)
        elif source_type == "s3":
            self._write_s3(data, parsed, **kwargs)
        elif source_type == "gcs":
            self._write_gcs(data, parsed, **kwargs)
        elif source_type == "azure":
            self._write_azure(data, parsed, **kwargs)
        else:
            raise DataIngestionError(
                f"Cannot write to source type: {source_type}",
                context={"destination": destination},
            )
    
    def _detect_source_type(self, source: str) -> tuple:
        """Detect the type of data source."""
        # Check for URL schemes
        parsed = urlparse(source)
        scheme = parsed.scheme.lower()
        
        # Cloud storage
        if scheme == "s3":
            return "s3", parsed
        elif scheme == "gs":
            return "gcs", parsed
        elif scheme in ("azure", "az", "wasb", "wasbs"):
            return "azure", parsed
        
        # Databases
        elif scheme in ("postgresql", "postgres", "mysql", "sqlite", "mssql", "oracle"):
            return "sql", parsed
        elif scheme == "mongodb":
            return "mongodb", parsed
        elif scheme == "redis":
            return "redis", parsed
        
        # HTTP
        elif scheme in ("http", "https"):
            return "http", parsed
        
        # Local file (no scheme or file://)
        elif scheme in ("", "file"):
            return "file", parsed
        
        else:
            # Try as local file
            if Path(source).exists() or any(
                source.endswith(ext)
                for ext in [".csv", ".parquet", ".json", ".xlsx", ".pickle"]
            ):
                return "file", parsed
            
            raise DataIngestionError(
                f"Cannot determine source type",
                context={"source": source, "scheme": scheme},
            )
    
    @property
    def file_handler(self) -> FileHandler:
        """Get or create file handler."""
        if self._file_handler is None:
            self._file_handler = FileHandler(backend=self.backend)
        return self._file_handler
    
    def _read_file(self, path: str, format: Optional[str], **kwargs: Any) -> Any:
        """Read from local file."""
        return self.file_handler.read(path, format=format, **kwargs)
    
    def _write_file(
        self,
        data: Any,
        path: str,
        format: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Write to local file."""
        self.file_handler.write(data, path, format=format, **kwargs)
    
    def _read_sql(
        self,
        url: str,
        query: Optional[str],
        table: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Read from SQL database."""
        if url not in self._connectors:
            self._connectors[url] = SQLConnector.from_url(url)
        
        conn = self._connectors[url]
        return conn.read(query=query, table=table, **kwargs)
    
    def _write_sql(
        self,
        data: Any,
        url: str,
        table: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Write to SQL database."""
        if url not in self._connectors:
            self._connectors[url] = SQLConnector.from_url(url)
        
        conn = self._connectors[url]
        conn.write(data, table=table or "data", **kwargs)
    
    def _read_mongodb(
        self,
        url: str,
        query: Optional[str],
        table: Optional[str],
        **kwargs: Any,
    ) -> Any:
        """Read from MongoDB."""
        if url not in self._connectors:
            self._connectors[url] = MongoDBConnector.from_url(url)
        
        conn = self._connectors[url]
        collection = table or "data"
        filter_query = {}
        if query:
            import json
            filter_query = json.loads(query)
        
        return conn.find(collection, filter_query, **kwargs)
    
    def _read_redis(self, url: str, **kwargs: Any) -> Any:
        """Read from Redis."""
        if url not in self._connectors:
            self._connectors[url] = RedisConnector.from_url(url)
        
        conn = self._connectors[url]
        key = kwargs.pop("key", None)
        if key:
            return conn.get(key)
        return None
    
    def _read_s3(self, parsed: Any, **kwargs: Any) -> Any:
        """Read from S3."""
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"s3://{bucket}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = S3Adapter(
                bucket=bucket,
                region=kwargs.pop("region", "us-east-1"),
            )
        
        adapter = self._connectors[cache_key]
        return adapter.read(key, **kwargs)
    
    def _write_s3(self, data: Any, parsed: Any, **kwargs: Any) -> None:
        """Write to S3."""
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"s3://{bucket}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = S3Adapter(
                bucket=bucket,
                region=kwargs.pop("region", "us-east-1"),
            )
        
        adapter = self._connectors[cache_key]
        adapter.write(data, key, **kwargs)
    
    def _read_gcs(self, parsed: Any, **kwargs: Any) -> Any:
        """Read from Google Cloud Storage."""
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"gs://{bucket}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = GCSAdapter(
                bucket=bucket,
                project=kwargs.pop("project", None),
            )
        
        adapter = self._connectors[cache_key]
        return adapter.read(key, **kwargs)
    
    def _write_gcs(self, data: Any, parsed: Any, **kwargs: Any) -> None:
        """Write to Google Cloud Storage."""
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"gs://{bucket}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = GCSAdapter(
                bucket=bucket,
                project=kwargs.pop("project", None),
            )
        
        adapter = self._connectors[cache_key]
        adapter.write(data, key, **kwargs)
    
    def _read_azure(self, parsed: Any, **kwargs: Any) -> Any:
        """Read from Azure Blob Storage."""
        container = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"azure://{container}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = AzureBlobAdapter(
                bucket=container,
                connection_string=kwargs.pop("connection_string", None),
            )
        
        adapter = self._connectors[cache_key]
        return adapter.read(key, **kwargs)
    
    def _write_azure(self, data: Any, parsed: Any, **kwargs: Any) -> None:
        """Write to Azure Blob Storage."""
        container = parsed.netloc
        key = parsed.path.lstrip("/")
        
        cache_key = f"azure://{container}"
        if cache_key not in self._connectors:
            self._connectors[cache_key] = AzureBlobAdapter(
                bucket=container,
                connection_string=kwargs.pop("connection_string", None),
            )
        
        adapter = self._connectors[cache_key]
        adapter.write(data, key, **kwargs)
    
    def _read_http(self, url: str, **kwargs: Any) -> Any:
        """Read from HTTP API."""
        import pandas as pd
        
        client = RESTClient(base_url="")
        client.connect()
        
        response = client.get(url, **kwargs)
        
        if isinstance(response, list):
            return pd.DataFrame(response)
        elif isinstance(response, dict):
            return pd.DataFrame([response])
        else:
            return response
    
    def close(self) -> None:
        """Close all cached connections."""
        for connector in self._connectors.values():
            if hasattr(connector, "disconnect"):
                connector.disconnect()
        self._connectors.clear()
        
        if self._file_handler:
            self._file_handler.disconnect()
