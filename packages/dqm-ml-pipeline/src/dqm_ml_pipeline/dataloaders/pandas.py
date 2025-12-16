import logging
from typing import Any, override

import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

# -----------------------------
# Data Loader
# -----------------------------
# TODO create abstract class


class PandasDataLoader:
    """
    Data loader for reading datasets from Parquet files in batches.
    """

    type: str = "csv"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize a ParquetDataLoader.

        Args:
            name: Unique name for this data loader.
            config: Configuration dictionary with keys:
                - path (str): Path to the parquet file.
        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")

        self.path = config["path"]

    def bootstrap(self, columns_list: list[str] | None = None) -> None:
        """
        Call before iterating the data loader content, can be used for iterator initialization
        We also pass argument computed from config by pipeline, to allow dala loader optimization

        Args:
            columns_list: Needed columns list in the data to load.
        Raises:
            ValueError: Missing informations.
        """

        self.data = pd.read_csv(self.path, sep=",")

    @override
    def __repr__(self) -> str:
        return f"Dataload for {self.path}"

    def __iter__(self) -> pa.RecordBatch:
        """
        Iterate over parquet csv file with only one batch as panda load everything in memmory.

        Args:
            columns_list: List of column names to load, or None for all.

        Yields:
            pyarrow.RecordBatch: A batch of data.
        """
        yield pa.RecordBatch.from_pandas(self.data)
