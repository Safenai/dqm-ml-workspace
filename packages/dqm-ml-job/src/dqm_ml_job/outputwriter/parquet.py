"""Parquet output writer for persisting pipeline results.

This module contains the ParquetOutputWriter class that writes
metrics and features to Parquet files.
"""

import logging
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dqm_ml_job.utils.s3 import get_s3_filesystem

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
                - s3_filesystem (bool or dict, optional): Enable S3 filesystem.
                  If dict, can contain access_key, secret_key, and endpoint_override.

        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path_pattern" not in config:
            raise ValueError(f"Configuration for ParquetOutputWriter '{name}' must contain 'path_pattern'")
        if "columns" not in config:
            raise ValueError(f"Configuration for ParquetOutputWriter '{name}' must contain 'columns'")

        self.path_pattern = config["path_pattern"]
        self.columns = list(config["columns"])
        self.name = name
        self.s3_filesystem = None

        self.add_dataloader_column = config.get("add_dataloader_column", False)
        self.dataloader_column_name = config.get("dataloader_column_name", "dataloader")

        self._accumulate = "{}" not in self.path_pattern
        self._accumulated_features: dict[str, list[pa.Array]] = {}
        s3_fs_config = config.get("s3_filesystem")
        if s3_fs_config:
            if isinstance(s3_fs_config, bool) and s3_fs_config:
                self.s3_filesystem = get_s3_filesystem()
            elif isinstance(s3_fs_config, dict):
                self.s3_filesystem = get_s3_filesystem(
                    access_key=s3_fs_config.get("access_key"),
                    secret_key=s3_fs_config.get("secret_key"),
                    endpoint=s3_fs_config.get("endpoint_override"),
                    region=s3_fs_config.get("region"),
                )

    def write_metrics_dict(self, metrics_dict: dict[str, dict[str, Any]]) -> None:
        """Aggregate and write dataset-level metrics for all selections.

        Args:
            metrics_dict: Map of selection names to their computed
                metric dictionaries.
        """
        if len(metrics_dict) > 0:
            logger.debug(f"Writing metrics for the {len(metrics_dict)} data selections")

            # We get all the selections computed
            keys = list(metrics_dict.keys())
            metric_names = list(metrics_dict[keys[0]].keys())
            metrics_table = {"selection": pa.array(keys)}

            for metric_name in metric_names:
                values = []
                for key in keys:
                    val = metrics_dict[key][metric_name]
                    if isinstance(val, pa.Array):
                        values.extend(val.to_pylist())
                    else:
                        values.append(val)
                metrics_table[metric_name] = pa.array(values)
            self.write_table("", metrics_table)

    def write_table(
        self,
        path_pattern: str,
        features_array: dict[str, Any],
        part: int | None = None,
    ) -> None:
        """Write a table of features or metrics to a Parquet file.

        Handles directory creation if the target path doesn't exist
        (for local writes).

        Args:
            path_pattern: Identifier for the data destination.
            features_array: Map of column names to pyarrow Arrays.
            part: Optional partition index for chunked output.
        """

        for key in self.columns:
            if key not in features_array:
                logger.error(f"Missing {key} in features for output")

        # Accumulate mode: buffer features for a single flush at the end
        if self._accumulate:
            for k, v in features_array.items():
                if isinstance(v, pa.ChunkedArray):
                    v = v.combine_chunks()
                self._accumulated_features.setdefault(k, []).append(v)
            return

        table = pa.table(features_array)
        if part is None:
            filename = self.path_pattern.format(path_pattern, "")
        else:
            filename = self.path_pattern.format(path_pattern, part)

        self._persist_table(table, filename)

    def flush(self) -> None:
        """Write all accumulated features to the output file.

        Called after all selections have been processed in accumulate mode.
        Does nothing if there are no accumulated features or if not
        in accumulate mode.
        """
        if not self._accumulated_features:
            return

        final = {k: pa.concat_arrays(v) for k, v in self._accumulated_features.items()}
        table = pa.table(final)
        filename = self.path_pattern

        self._persist_table(table, filename)

        self._accumulated_features.clear()

    def _persist_table(self, table: pa.Table, filename: str) -> None:
        """Write a PyArrow table to S3 or local disk.

        Args:
            table: The PyArrow table to persist.
            filename: The target file path.
        """
        if self.s3_filesystem is not None:
            s3_path = self._get_s3_path(filename)
            try:
                pq.write_table(table, s3_path, filesystem=self.s3_filesystem)
                logger.info(f"Wrote output table to S3: {s3_path}")
            except Exception as e:
                logger.exception(f"Failed to write to S3: {e}")
                raise
        else:
            output_dir = Path(filename).parent
            if not Path.exists(output_dir):
                logger.info(f"Creating output directory: {output_dir}")
                Path.mkdir(output_dir, parents=True, exist_ok=True)
            pq.write_table(table, filename)
            logger.info(f"Wrote output table to {filename}")

    def _get_s3_path(self, file_path: str) -> str:
        """Construct an S3 path by combining the bucket name with a file path.

        Args:
            file_path: The file path within the bucket.

        Returns:
            str: The full S3 path in format "bucket_name/file_path".
        """
        bucket_name = os.getenv("S3_BUCKET_NAME", "")
        return bucket_name + "/" + file_path
