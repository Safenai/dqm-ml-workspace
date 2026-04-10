"""Output writers module for DQM ML Job.

This module contains classes for writing pipeline results (features
and metrics) to various storage backends.

Classes:
    OutputWriter: Protocol for output writer implementations.
    ParquetOutputWriter: Writer that saves data to Parquet files.
"""

from typing import Any, Protocol, runtime_checkable

from dqm_ml_job.outputwriter.parquet import ParquetOutputWriter


@runtime_checkable
class OutputWriter(Protocol):
    """
    Protocol for Output Writers.

    Defines the interface for writing pipeline results (features or metrics) to storage.
    """

    columns: list[str]
    name: str

    def write_metrics_dict(self, metrics_dict: dict[str, dict[str, Any]]) -> None:
        """ """

    def write_table(self, name: str, table: Any, part_index: int | None = None) -> None:
        """
        Write a table (features or metrics) to the output.

        Args:
            name: Name of the dataset or metric.
            table: The data to write (usually a pyarrow Table or dict of arrays).
            part_index: Index of the data part (for chunked writing).
        """


dqml_outputs_registry = {"parquet": ParquetOutputWriter}


__all__ = ["OutputWriter", "ParquetOutputWriter", "dqml_outputs_registry"]
