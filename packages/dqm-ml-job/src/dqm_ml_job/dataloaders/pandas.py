"""Pandas data loader for reading CSV files.

This module contains the PandasDataLoader and PandasDataSelection classes
for loading and iterating over CSV file data using Pandas.
"""

import logging
from typing import Any

import pandas as pd
import pyarrow as pa

# COMPATIBILITY : from typing import Any, override # When support of 3.10 and 3.11 will be removed
from typing_extensions import override

from dqm_ml_job.dataloaders.proto import DataSelection

logger = logging.getLogger(__name__)


class PandasDataSelection(DataSelection):
    """A selection of data from a CSV file loaded via Pandas.

    This class represents data loaded from a CSV file and provides
    an iterator over PyArrow RecordBatches.

    Attributes:
        name: Name identifier for this selection.
        path: Path to the CSV file.
        data: The loaded pandas DataFrame.
    """

    def __init__(self, name: str, path: str):
        """Initialize a Pandas data selection.

        Args:
            name: Name identifier for this selection.
            path: Path to the CSV file.
        """
        self.name = name
        self.path = path
        self.data: pd.DataFrame | None = None

    @override
    def bootstrap(self, columns_list: list[str] | None = None) -> None:
        # For CSV, we currently load everything
        self.data = pd.read_csv(self.path, sep=",")

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    @override
    def get_nb_batches(self) -> int:
        return 1 if self.data is not None else 0

    @override
    def __iter__(self) -> Any:
        if self.data is not None:
            yield pa.RecordBatch.from_pandas(self.data)

    @override
    def __repr__(self) -> str:
        return f"PandasSelection(name='{self.name}', path='{self.path}')"


class PandasDataLoader:
    """Data loader for CSV files using Pandas.

    This loader reads CSV files and provides DataSelections for
    processing by the DQM pipeline.

    Attributes:
        type: The loader type identifier ("csv").
    """

    type: str = "csv"

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the Pandas data loader.

        Args:
            name: Unique name for this loader instance.
            config: Configuration dictionary containing:
                - path: Path to CSV file (required)

        Raises:
            ValueError: If required config keys are missing.
        """
        if not config or "path" not in config:
            raise ValueError(f"Configuration for dataloader '{name}' must contain 'path'")
        self.name = name
        self.path = config["path"]

    def get_selections(self) -> list[DataSelection]:
        """Create a PandasDataSelection for the CSV file.

        Returns:
            A list containing a single PandasDataSelection instance.
        """
        return [PandasDataSelection(name=self.name, path=self.path)]
