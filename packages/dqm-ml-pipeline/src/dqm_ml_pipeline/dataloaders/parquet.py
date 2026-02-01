import logging
from typing import Any, override

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

# -----------------------------
# Data Loader
# -----------------------------
# TODO create abstract class


class ParquetDataLoader:
    """
    Data loader for reading datasets from Parquet files in batches.
    """

    type: str = "parquet"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize a ParquetDataLoader.

        Args:
            name: Unique name for this data loader.
            config: Configuration dictionary with keys:
                - path (str): Path to the parquet file.
                - batch_size (int): Rows per batch (default 100,000).
                - memory_limit (str): Max memory usage hint (default "2GB").
                - threads (int): Number of threads to use (default 4).

        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")

        self.path = config["path"]
        self.batch_size: int = config.get("batch_size", 100_000)
        self.memory_limit: str = config.get("memory_limit", "2GB")
        self.threads: int = config.get("threads", 4)
        self.name = name
        self.filters_dict = config.get("filter", None)

    def bootstrap(self, columns_list: list[str] | None = None) -> None:
        """
        Call before iterating the data loader content, can be used for iterator initialization
        We also pass argument computed from config by pipeline, to allow dala loader optimization

        Args:
            columns_list: Needed columns list in the data to load.
        Raises:
            ValueError: Missing informations.
        """

        self.columns_list = columns_list

        filter_expr = None
        # Building parametric filter expr
        if self.filters_dict is not None:
            expr = None
            for col, val in self.filters_dict.items():
                col_expr = pc.equal(pc.field(col), val)
                expr = col_expr if expr is None else pc.and_(expr, col_expr)
            filter_expr = expr

        self.dataset = pq.ParquetDataset(self.path, filters=filter_expr)

        # If schema was not provided, we read it from the first file
        if len(self.dataset.files) > 0:
            self.schema = pq.read_schema(self.dataset.files[0])
            # self.samples_count = sum(p.metadata.num_rows for p in self.dataset.files)
            self.samples_count = sum(p.count_rows() for p in self.dataset.fragments)
        else:
            self.schema = None
            self.samples_count = 0

    @override
    def __repr__(self) -> str:
        return f"Dataload for {self.path}"

    def __len__(self) -> int:
        """
        Get the total number of rows in the parquet file.

        Returns:
            int: Total number of rows.
        """
        return int(self.samples_count)

    def get_nb_batches(self) -> int:
        """
        Get the total number of batches in the parquet file.

        Returns:
            int: Total number of batches.
        """
        return int(len(self) / self.batch_size) + (len(self) % self.batch_size > 0)

    def __iter__(self) -> pa.RecordBatch:
        """
        Iterate over parquet file in batches.

        Args:
            columns_list: List of column names to load, or None for all.

        Yields:
            pyarrow.RecordBatch: A batch of data.
        """
        for file in self.dataset.files:
            parquet_file = pq.ParquetFile(file)
            batch_iterator = parquet_file.iter_batches(
              batch_size=self.batch_size, columns=self.columns_list, use_threads=self.threads
            )
            yield from batch_iterator
