"""Data loaders module for DQM ML Job.

This module contains classes for loading data from various sources
and protocols. It provides the DataLoader and DataSelection protocols
along with concrete implementations for different file formats.

Classes:
    DataLoader: Protocol for data loader factories.
    DataSelection: Protocol for data subsets.
    ParquetDataLoader: Loader for Parquet files.
    PandasDataLoader: Loader for CSV files using Pandas.
"""

from dqm_ml_job.dataloaders.pandas import PandasDataLoader
from dqm_ml_job.dataloaders.parquet import ParquetDataLoader
from dqm_ml_job.dataloaders.proto import DataLoader, DataSelection

# Registry of supported data loaders
dqml_dataloaders_registry = {"parquet": ParquetDataLoader, "csv": PandasDataLoader}

__all__ = [
    "DataLoader",
    "DataSelection",
    "PandasDataLoader",
    "ParquetDataLoader",
    "dqml_dataloaders_registry",
]
