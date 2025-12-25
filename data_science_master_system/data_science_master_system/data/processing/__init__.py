"""
Data Processing Module.

Provides multi-framework data processing engines:
    - ProcessingEngine: Unified interface
    - PandasProcessor: Pandas + Dask
    - PolarsProcessor: High-performance Polars
"""

from data_science_master_system.data.processing.processing_engine import (
    ProcessingEngine,
)
from data_science_master_system.data.processing.pandas_processor import (
    PandasProcessor,
)
from data_science_master_system.data.processing.polars_processor import (
    PolarsProcessor,
)

__all__ = [
    "ProcessingEngine",
    "PandasProcessor",
    "PolarsProcessor",
]
