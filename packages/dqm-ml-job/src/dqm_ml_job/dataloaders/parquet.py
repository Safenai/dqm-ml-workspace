"""Parquet data loader for reading Parquet files.

This module contains the ParquetDataLoader and ParquetDataSelection classes
for loading and iterating over Parquet file data.
"""

import logging
from typing import Any
from typing_extensions import override
# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed

import pyarrow.compute as pc
import pyarrow.parquet as pq

from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


class ParquetDataSelection(DataSelection):
    """A specific selection of data from a Parquet dataset.

    This class represents a filtered subset of a Parquet dataset
    and provides an iterator over PyArrow RecordBatches.

    Attributes:
        name: Name identifier for this selection.
        path: Path to the Parquet file or directory.
        batch_size: Number of rows per batch.
        threads: Number of threads for parallel reading.
        filters_dict: Optional dictionary of column filters to apply.
    """

    def __init__(
        self,
        name: str,
        path: str,
        batch_size: int = 100_000,
        threads: int = 4,
        filters_dict: dict[str, Any] | None = None,
    ):
        """Initialize a Parquet data selection.

        Args:
            name: Name identifier for this selection.
            path: Path to the Parquet file or directory.
            batch_size: Number of rows per batch (default: 100000).
            threads: Number of threads for parallel reading (default: 4).
            filters_dict: Optional dictionary of column filters to apply.
        """
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self.threads = threads
        self.filters_dict = filters_dict
        self.columns_list: list[str] | None = None
        self.dataset: pq.ParquetDataset | None = None
        self.samples_count: int = 0

    @override
    def bootstrap(self, columns_list: list[str]) -> None:
        self.columns_list = columns_list
        filter_expr = None
        if self.filters_dict is not None:
            expr = None
            for col, val in self.filters_dict.items():
                if col not in (self.columns_list):
                    self.columns_list.append(col)
                col_expr = pc.equal(pc.field(col), val)
                expr = col_expr if expr is None else pc.and_(expr, col_expr)
            filter_expr = expr
        self.filter_expr = filter_expr
        self.dataset = pq.ParquetDataset(self.path, filters=filter_expr)
        if len(self.dataset.fragments) > 0:
            self.samples_count = sum(p.count_rows() for p in self.dataset.fragments)
        else:
            self.samples_count = 0

    def __len__(self) -> int:
        return int(self.samples_count)

    @override
    def get_nb_batches(self) -> int:
        return int(len(self) / self.batch_size) + (len(self) % self.batch_size > 0)

    @override
    def __iter__(self) -> Any:
        if self.dataset is None:
            return
        for file in self.dataset.files:
            parquet_file = pq.ParquetFile(file)
            batch_iterator = parquet_file.iter_batches(
                batch_size=self.batch_size, columns=self.columns_list, use_threads=self.threads
            )
            for batch in batch_iterator:
                if self.filter_expr is not None:
                    batch = batch.filter(self.filter_expr)
                if len(batch) == 0:
                    continue
                yield batch

    @override
    def __repr__(self) -> str:
        return f"ParquetSelection(name='{self.name}', path='{self.path}', filters={self.filters_dict})"


class ParquetDataLoader:
    """Data loader for Parquet files that generates one or more DataSelections.

    This loader can read from a single Parquet file or a directory of Parquet
    files, optionally splitting the data by a column value to create multiple
    selections.

    Attributes:
        type: The loader type identifier ("parquet").
    """

    type: str = "parquet"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the Parquet data loader.

        Args:
            name: Unique name for this loader instance.
            config: Configuration dictionary containing:
                - path: Path to Parquet file or directory (required)
                - batch_size: Rows per batch (default: 100000)
                - threads: Number of threads (default: 4)
                - split_by: Column name to split selections by
                - split_values: Specific values to split on
                - filter: Dictionary of column filters

        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")

        self.name = name
        self.config = config
        self.path = config["path"]
        self.batch_size = config.get("batch_size", 100_000)
        self.threads = config.get("threads", 4)
        self.split_by = config.get("split_by")
        self.split_values = config.get("split_values")
        self.filters_dict = config.get("filter", None)

    def get_selections(self) -> list[DataSelection]:
        """Create one or more ParquetDataSelection instances based on configuration.

        Returns:
            A list of DataSelection instances. If split_by is configured,
            returns one selection per unique value. Otherwise, returns a
            single selection for the entire dataset.
        """
        if not self.split_by:
            # Single selection
            return [
                ParquetDataSelection(
                    name=self.name,
                    path=self.path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=self.filters_dict,
                )
            ]

        # Splitting logic
        values = self.split_values
        if values is None:
            # Automatic discovery if split_values not provided
            logger.info(f"Discovering unique values for split_by='{self.split_by}' in {self.path}")
            table = pq.read_table(self.path, columns=[self.split_by])
            values = [str(v) for v in pc.unique(table.column(0)).to_pylist() if v is not None]

        selections: list[DataSelection] = []
        for val in values:
            selection_name = f"{self.name}_{val}"
            # Merge existing filters with the split filter
            merged_filters = (self.filters_dict or {}).copy()
            merged_filters[self.split_by] = val

            selections.append(
                ParquetDataSelection(
                    name=selection_name,
                    path=self.path,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    filters_dict=merged_filters,
                )
            )
        return selections
