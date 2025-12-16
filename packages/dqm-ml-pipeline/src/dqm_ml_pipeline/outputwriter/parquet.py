import logging
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetOutputWriter:
    """
    Output writer that saves processed features to a Parquet file.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize a ParquetOutputWriter.

        Args:
            name: Unique name for this output writer.
            config: Configuration dictionary with keys:
                - path_pattern (str): Output file path format string.
                - columns (List[str]): Columns to save.

        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path_pattern" not in config:
            raise ValueError(f"Configuration for ParquetOutputWriter '{name}' must contain 'path_pattern'")
        if "columns" not in config:
            raise ValueError(f"Configuration for ParquetOutputWriter '{name}' must contain 'columns'")

        self.path_pattern = config["path_pattern"]
        self.columns = config["columns"]
        self.name = name

    def write_table(self, dataloader: str, features_array: dict[str, Any], part: int) -> None:
        """
        Write the processed features to a parquet file.

        Args:
            dataloader: Name of the data loader that produced the data.
            features_array: Dictionary of column name -> pyarrow.Array.
            part: Partition index for splitting large datasets.
        """

        for key in self.columns:
            if key not in features_array:
                logger.error(f"Missing {key} in features for output")

        table = pa.table(features_array)
        filename = self.path_pattern.format(dataloader, part)

        pq.write_table(table, filename)
        logger.info(f"Wrote output table to {filename}")
