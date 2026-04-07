"""Parquet output writer for persisting pipeline results.

This module contains the ParquetOutputWriter class that writes
metrics and features to Parquet files.
"""

import logging
from pathlib import Path
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

    def write_metrics_dict(self, metrics_dict: dict[str, dict[str, Any]]) -> None:
        """Aggregate and write dataset-level metrics for all selections to a Parquet file.

        Args:
            metrics_dict: Map of selection names to their computed metric dictionaries.
        """
        if len(metrics_dict) > 0:
            logger.debug(f"Writing metrics for the {len(metrics_dict)} data selections")

            # We get all the selections computed
            keys = list(metrics_dict.keys())
            metric_names = list(metrics_dict[keys[0]].keys())
            metrics_table = {"selection": pa.array(keys)}

            for metric_name in metric_names:
                metrics_table[metric_name] = pa.array([metrics_dict[key][metric_name] for key in keys])
        self.write_table("", metrics_table)

    def write_table(self, path_pattern: str, features_array: dict[str, Any], part: int | None = None) -> None:
        """
        Write a table of features or metrics to a Parquet file on disk.

        Handles directory creation if the target path doesn't exist.

        Args:
            path_pattern: Identifier for the data detination (used in filename pattern).
            features_array: Map of column names to pyarrow Arrays.
            part: Optional partition index for chunked output.
        """

        for key in self.columns:
            if key not in features_array:
                logger.error(f"Missing {key} in features for output")

        table = pa.table(features_array)
        if part is None:
            filename = self.path_pattern.format(path_pattern, "")
        else:
            filename = self.path_pattern.format(path_pattern, part)

        output_dir = Path(filename).parent
        if not Path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            Path.mkdir(output_dir, parents=True, exist_ok=True)

        pq.write_table(table, filename)
        logger.info(f"Wrote output table to {filename}")
