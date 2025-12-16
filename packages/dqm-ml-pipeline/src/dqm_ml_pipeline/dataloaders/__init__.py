"""
Data processors module.

This module contains classes for processing data and computing metrics.
"""

from dqm_ml_pipeline.dataloaders.pandas import PandasDataLoader
from dqm_ml_pipeline.dataloaders.parquet import ParquetDataLoader

# Registry of supported data loaders
dqml_dataloaders_registry = {"parquet": ParquetDataLoader, "csv": "PandasDataLoader"}

__all__ = [
    "PandasDataLoader",
    "ParquetDataLoader",
    "dqml_dataloaders_registry",
]
