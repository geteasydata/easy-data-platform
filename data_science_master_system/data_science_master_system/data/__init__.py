"""
Data Module - Data Ingestion, Processing, Storage, and Quality.

Provides comprehensive data handling capabilities:
    - ingestion: Database, file, API, and streaming data sources
    - processing: Multi-framework data processing engines
    - storage: Data storage backends
    - quality: Data validation and quality checks
"""

from data_science_master_system.data.ingestion import (
    DataLoader,
    FileHandler,
    DatabaseConnector,
    APIClient,
    StreamingConsumer,
)
from data_science_master_system.data.processing import (
    ProcessingEngine,
    PandasProcessor,
    PolarsProcessor,
)

__all__ = [
    # Ingestion
    "DataLoader",
    "FileHandler",
    "DatabaseConnector",
    "APIClient",
    "StreamingConsumer",
    # Processing
    "ProcessingEngine",
    "PandasProcessor",
    "PolarsProcessor",
]
