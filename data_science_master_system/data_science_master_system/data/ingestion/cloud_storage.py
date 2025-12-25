"""
Cloud Storage Adapters for Data Science Master System.

Provides adapters for:
- AWS S3
- Google Cloud Storage (GCS)
- Azure Blob Storage

Example:
    >>> adapter = S3Adapter(bucket="my-bucket", region="us-east-1")
    >>> df = adapter.read("data/train.csv")
    >>> adapter.write(df, "data/processed.parquet")
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import io
import tempfile

from data_science_master_system.core.base_classes import BaseDataSource
from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class CloudStorageAdapter(BaseDataSource):
    """
    Abstract base class for cloud storage adapters.
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize cloud storage adapter.
        
        Args:
            bucket: Bucket/container name
            prefix: Default prefix for all operations
            config: Additional configuration
        """
        super().__init__(config)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
    
    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key.lstrip('/')}"
        return key.lstrip("/")
    
    @abstractmethod
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket."""
        pass
    
    @abstractmethod
    def download(self, key: str, local_path: str) -> None:
        """Download object to local path."""
        pass
    
    @abstractmethod
    def upload(self, local_path: str, key: str) -> None:
        """Upload local file to bucket."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete object from bucket."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass


class S3Adapter(CloudStorageAdapter):
    """
    AWS S3 adapter.
    
    Example:
        >>> adapter = S3Adapter(
        ...     bucket="my-bucket",
        ...     region="us-east-1",
        ...     aws_access_key_id="...",
        ...     aws_secret_access_key="...",
        ... )
        >>> adapter.connect()
        >>> 
        >>> # List objects
        >>> files = adapter.list_objects("data/")
        >>> 
        >>> # Read CSV directly
        >>> df = adapter.read("data/train.csv")
        >>> 
        >>> # Upload processed data
        >>> adapter.write(df, "data/processed.parquet")
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(bucket, prefix, config)
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self._client = None
        self._s3 = None
    
    def connect(self) -> None:
        """Connect to S3."""
        try:
            import boto3
            
            session_kwargs = {"region_name": self.region}
            if self.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
            if self.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            
            session = boto3.Session(**session_kwargs)
            
            client_kwargs = {}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            
            self._client = session.client("s3", **client_kwargs)
            self._s3 = session.resource("s3", **client_kwargs)
            
            # Test connection
            self._client.head_bucket(Bucket=self.bucket)
            
            self._connected = True
            logger.info(f"Connected to S3", bucket=self.bucket, region=self.region)
            
        except ImportError:
            raise DataIngestionError("boto3 not installed")
        except Exception as e:
            raise DataIngestionError(f"S3 connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from S3."""
        self._client = None
        self._s3 = None
        self._connected = False
    
    def read(self, key: str, **kwargs: Any) -> Any:
        """
        Read data from S3.
        
        Args:
            key: Object key
            **kwargs: Additional read options
            
        Returns:
            pandas DataFrame
        """
        if not self._connected:
            self.connect()
        
        import pandas as pd
        
        full_key = self._full_key(key)
        
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=full_key)
            body = obj["Body"].read()
            
            # Detect format from key
            if key.endswith(".csv"):
                return pd.read_csv(io.BytesIO(body), **kwargs)
            elif key.endswith(".parquet"):
                return pd.read_parquet(io.BytesIO(body), **kwargs)
            elif key.endswith(".json"):
                return pd.read_json(io.BytesIO(body), **kwargs)
            else:
                # Default to CSV
                return pd.read_csv(io.BytesIO(body), **kwargs)
                
        except Exception as e:
            raise DataIngestionError(
                f"Failed to read from S3",
                context={"bucket": self.bucket, "key": full_key, "error": str(e)},
            )
    
    def write(self, data: Any, key: str, **kwargs: Any) -> None:
        """
        Write data to S3.
        
        Args:
            data: pandas DataFrame
            key: Object key
            **kwargs: Additional write options
        """
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        
        try:
            buffer = io.BytesIO()
            
            # Detect format from key
            if key.endswith(".csv"):
                data.to_csv(buffer, index=False, **kwargs)
            elif key.endswith(".parquet"):
                data.to_parquet(buffer, **kwargs)
            elif key.endswith(".json"):
                data.to_json(buffer, **kwargs)
            else:
                data.to_csv(buffer, index=False, **kwargs)
            
            buffer.seek(0)
            self._client.put_object(Bucket=self.bucket, Key=full_key, Body=buffer)
            
            logger.info(f"Wrote to S3", bucket=self.bucket, key=full_key)
            
        except Exception as e:
            raise DataIngestionError(
                f"Failed to write to S3",
                context={"bucket": self.bucket, "key": full_key, "error": str(e)},
            )
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket."""
        if not self._connected:
            self.connect()
        
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        
        return keys
    
    def download(self, key: str, local_path: str) -> None:
        """Download object to local path."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, full_key, local_path)
    
    def upload(self, local_path: str, key: str) -> None:
        """Upload local file to bucket."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        self._client.upload_file(local_path, self.bucket, full_key)
    
    def delete(self, key: str) -> None:
        """Delete object from bucket."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        self._client.delete_object(Bucket=self.bucket, Key=full_key)
    
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except:
            return False


class GCSAdapter(CloudStorageAdapter):
    """
    Google Cloud Storage adapter.
    
    Example:
        >>> adapter = GCSAdapter(bucket="my-bucket", project="my-project")
        >>> adapter.connect()
        >>> df = adapter.read("data/train.csv")
    """
    
    def __init__(
        self,
        bucket: str,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
        prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(bucket, prefix, config)
        self.project = project
        self.credentials_path = credentials_path
        self._client = None
        self._bucket = None
    
    def connect(self) -> None:
        """Connect to GCS."""
        try:
            from google.cloud import storage
            
            if self.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project,
                )
            else:
                self._client = storage.Client(project=self.project)
            
            self._bucket = self._client.bucket(self.bucket)
            
            self._connected = True
            logger.info(f"Connected to GCS", bucket=self.bucket)
            
        except ImportError:
            raise DataIngestionError("google-cloud-storage not installed")
        except Exception as e:
            raise DataIngestionError(f"GCS connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from GCS."""
        self._client = None
        self._bucket = None
        self._connected = False
    
    def read(self, key: str, **kwargs: Any) -> Any:
        """Read data from GCS."""
        if not self._connected:
            self.connect()
        
        import pandas as pd
        
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        content = blob.download_as_bytes()
        
        if key.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content), **kwargs)
        elif key.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(content), **kwargs)
        elif key.endswith(".json"):
            return pd.read_json(io.BytesIO(content), **kwargs)
        else:
            return pd.read_csv(io.BytesIO(content), **kwargs)
    
    def write(self, data: Any, key: str, **kwargs: Any) -> None:
        """Write data to GCS."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        buffer = io.BytesIO()
        
        if key.endswith(".csv"):
            data.to_csv(buffer, index=False, **kwargs)
        elif key.endswith(".parquet"):
            data.to_parquet(buffer, **kwargs)
        elif key.endswith(".json"):
            data.to_json(buffer, **kwargs)
        else:
            data.to_csv(buffer, index=False, **kwargs)
        
        buffer.seek(0)
        blob = self._bucket.blob(full_key)
        blob.upload_from_file(buffer)
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket."""
        if not self._connected:
            self.connect()
        
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        blobs = self._client.list_blobs(self.bucket, prefix=full_prefix)
        return [blob.name for blob in blobs]
    
    def download(self, key: str, local_path: str) -> None:
        """Download object to local path."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        blob.download_to_filename(local_path)
    
    def upload(self, local_path: str, key: str) -> None:
        """Upload local file to bucket."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        blob.upload_from_filename(local_path)
    
    def delete(self, key: str) -> None:
        """Delete object from bucket."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        blob.delete()
    
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob = self._bucket.blob(full_key)
        return blob.exists()


class AzureBlobAdapter(CloudStorageAdapter):
    """
    Azure Blob Storage adapter.
    
    Example:
        >>> adapter = AzureBlobAdapter(
        ...     bucket="my-container",
        ...     connection_string="DefaultEndpointsProtocol=https;...",
        ... )
        >>> adapter.connect()
        >>> df = adapter.read("data/train.csv")
    """
    
    def __init__(
        self,
        bucket: str,  # container name
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(bucket, prefix, config)
        self.connection_string = connection_string
        self.account_name = account_name
        self.account_key = account_key
        self._client = None
        self._container = None
    
    def connect(self) -> None:
        """Connect to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(self.connection_string)
            else:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key,
                )
            
            self._container = self._client.get_container_client(self.bucket)
            
            self._connected = True
            logger.info(f"Connected to Azure Blob Storage", container=self.bucket)
            
        except ImportError:
            raise DataIngestionError("azure-storage-blob not installed")
        except Exception as e:
            raise DataIngestionError(f"Azure Blob connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from Azure Blob Storage."""
        self._client = None
        self._container = None
        self._connected = False
    
    def read(self, key: str, **kwargs: Any) -> Any:
        """Read data from Azure Blob."""
        if not self._connected:
            self.connect()
        
        import pandas as pd
        
        full_key = self._full_key(key)
        blob_client = self._container.get_blob_client(full_key)
        content = blob_client.download_blob().readall()
        
        if key.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content), **kwargs)
        elif key.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(content), **kwargs)
        elif key.endswith(".json"):
            return pd.read_json(io.BytesIO(content), **kwargs)
        else:
            return pd.read_csv(io.BytesIO(content), **kwargs)
    
    def write(self, data: Any, key: str, **kwargs: Any) -> None:
        """Write data to Azure Blob."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        buffer = io.BytesIO()
        
        if key.endswith(".csv"):
            data.to_csv(buffer, index=False, **kwargs)
        elif key.endswith(".parquet"):
            data.to_parquet(buffer, **kwargs)
        elif key.endswith(".json"):
            data.to_json(buffer, **kwargs)
        else:
            data.to_csv(buffer, index=False, **kwargs)
        
        buffer.seek(0)
        blob_client = self._container.get_blob_client(full_key)
        blob_client.upload_blob(buffer, overwrite=True)
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List blobs in container."""
        if not self._connected:
            self.connect()
        
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        blobs = self._container.list_blobs(name_starts_with=full_prefix)
        return [blob.name for blob in blobs]
    
    def download(self, key: str, local_path: str) -> None:
        """Download blob to local path."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob_client = self._container.get_blob_client(full_key)
        
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
    
    def upload(self, local_path: str, key: str) -> None:
        """Upload local file to blob."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob_client = self._container.get_blob_client(full_key)
        
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
    
    def delete(self, key: str) -> None:
        """Delete blob."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob_client = self._container.get_blob_client(full_key)
        blob_client.delete_blob()
    
    def exists(self, key: str) -> bool:
        """Check if blob exists."""
        if not self._connected:
            self.connect()
        
        full_key = self._full_key(key)
        blob_client = self._container.get_blob_client(full_key)
        return blob_client.exists()
