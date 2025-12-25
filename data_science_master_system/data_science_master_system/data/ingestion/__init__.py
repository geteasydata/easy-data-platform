"""
Data Ingestion Module.

Provides connectors and loaders for all data sources:
    - File formats (CSV, Excel, Parquet, JSON, etc.)
    - Databases (SQL, NoSQL)
    - Cloud storage (S3, GCS, Azure)
    - APIs (REST, GraphQL)
    - Streaming (Kafka, RabbitMQ)
"""

from data_science_master_system.data.ingestion.file_handlers import FileHandler
from data_science_master_system.data.ingestion.database_connectors import (
    DatabaseConnector,
    SQLConnector,
    MongoDBConnector,
    RedisConnector,
)
from data_science_master_system.data.ingestion.api_clients import (
    APIClient,
    RESTClient,
    GraphQLClient,
)
from data_science_master_system.data.ingestion.cloud_storage import (
    CloudStorageAdapter,
    S3Adapter,
    GCSAdapter,
    AzureBlobAdapter,
)
from data_science_master_system.data.ingestion.streaming import (
    StreamingConsumer,
    KafkaConsumer,
)
from data_science_master_system.data.ingestion.data_loader import DataLoader

__all__ = [
    # Main loader
    "DataLoader",
    # File handlers
    "FileHandler",
    # Database connectors
    "DatabaseConnector",
    "SQLConnector",
    "MongoDBConnector",
    "RedisConnector",
    # API clients
    "APIClient",
    "RESTClient",
    "GraphQLClient",
    # Cloud storage
    "CloudStorageAdapter",
    "S3Adapter",
    "GCSAdapter",
    "AzureBlobAdapter",
    # Streaming
    "StreamingConsumer",
    "KafkaConsumer",
]
