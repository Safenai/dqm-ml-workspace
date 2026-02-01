"""
Data processors module.

This module contains classes for processing data and computing metrics.
"""

from typing import Any, Protocol, runtime_checkable

from dqm_ml_pipeline.dataloaders.pandas import PandasDataLoader
from dqm_ml_pipeline.dataloaders.parquet import ParquetDataLoader


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for Data Loaders.

    Defines the interface that all data loaders must implement to be used in the pipeline.
    """

    def bootstrap(self, columns_list: list[str]) -> None:
        """
        Initialize the data loader with the required columns.

        Args:
            columns_list: List of column names that need to be loaded.
        """
        ...

    def get_nb_batches(self) -> int:
        """
        Get the total number of batches in the dataset.

        Returns:
            The number of batches.
        """
        ...

    def __iter__(self) -> Any:
        """
        Iterate over the dataset batches.

        Yields:
            pa.RecordBatch: A PyArrow RecordBatch containing a chunk of the data.
        """
        ...


# Registry of supported data loaders
dqml_dataloaders_registry = {"parquet": ParquetDataLoader, "csv": PandasDataLoader}

__all__ = [
    "DataLoader",
    "PandasDataLoader",
    "ParquetDataLoader",
    "dqml_dataloaders_registry",
]
