"""
Data processors module.

This module contains classes for processing data and computing metrics.
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
